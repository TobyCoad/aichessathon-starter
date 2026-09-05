"""The search compiled: negamax and quiescence as numba kernels over fastboard.

FastEngine.search in agent.py is the reference. Profiled at 8 s it spent 25% of
its time in the Python loop itself and another 25% dispatching make/unmake into
compiled code, so this moves the loop to where the board already lives. The
semantics are the reference's, line for line -- transposition probe and store,
reverse futility, null move, futility at depth 1-2, check extension, killers,
butterfly history, MVV-LVA ordering, delta pruning, the contempt draw score and
repetition against the stack and the game -- so that testing/check_fastsearch
can hold the two to identical scores and best moves at fixed depth.

Differences, deliberate:
  * the transposition table is fixed arrays indexed by the low key bits, not a
    dict that is cleared when full; an entry is replaced when the key matches,
    the stored entry is from an older search, or the new depth is not shallower.
  * no tablebase probing inside the tree (the root still probes); a 4-man
    position in the tree is scored by the net and the search like any other.
  * the clock is read every 256 nodes through objmode; on timeout an abort flag
    is set and every frame unwinds normally, so the board is always consistent.

The root loop -- iterative deepening, the per-root-move calls, the time rules --
stays in Python in FastEngine.choose, unchanged.

Constants below mirror agent.py; testing/check_fastsearch asserts they match.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numba import njit, objmode

import fastboard as fb

MATE = 30_000
DISTANCE_THRESHOLD = 19_000
INFINITY = 1 << 20
OUTPUT_SCALE = 400.0
MVV = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int64)
DELTA_MARGIN = 200
BIG_DELTA = 975
RFP_MAX_DEPTH = 6
RFP_MARGIN = 80
NMP_MIN_DEPTH = 3
NMP_REDUCTION = 2
MAX_PLY = 72  # agent.MAX_PLY: the check-extension limit
FUTILITY_MARGIN = np.array([0, 150, 300], dtype=np.int64)
POLL_MASK = 255

TT_BITS = 22
TT_SIZE = 1 << TT_BITS
TT_MASK = np.uint64(TT_SIZE - 1)

# ctrl slots: search state the Python side sets and reads.
C_NODES, C_ABORT, C_AGE, C_ROOT_SIDE, C_DRAW_ROOT, C_TT_OFF, C_HYGIENE, C_FUTILITY = range(8)
# Stage-3 switches, off = the reference search.
C_PVS, C_LMR, C_LMP, C_SEE = 8, 9, 10, 11
# C_STOP: set from another thread to end a search now (pondering).
C_STOP = 12
# C_NMP_GUARD: forbid a null move directly after a null move. Two nulls
# restore the key (the en passant hash is 0 without an ep square), the stack
# repetition check fires, and the grandchild scores as a draw: null-move
# pruning was inert at every node of depth >= 6.
C_NMP_GUARD = 13
# C_RFP_PHASE: scale the reverse-futility and futility margins by piece count,
# because the static score they trust is 2-6x less accurate below 17 pieces.
# C_IIR: a node of depth >= 4 with no hash move is searched one ply shallower.
C_RFP_PHASE, C_IIR = 14, 15
# Margin scale in percent for the piece bands <= 8, 9-12, 13-16, 17-20 (21+ is
# always 100); 0 turns the pruning off in that band. Set from agent.RFP_PHASE_PERCENT.
C_PH_LE8, C_PH_9_12, C_PH_13_16, C_PH_17_20 = 16, 17, 18, 19
# C_HISTORY2: side-indexed history with gravity and malus, plus counter moves.
C_HISTORY2 = 20
HISTORY_MAX = 16384
# C_TT_KEEP: an entry from an older search is replaced only by one at most 4
# plies shallower, instead of unconditionally. C_QS_CAP: quiescence depth cap
# (8 in the reference). C_SAFE: mate-distance pruning and null-move reduction
# growing with depth.
C_TT_KEEP, C_QS_CAP, C_SAFE = 21, 22, 23
# C_QS_CACHE: quiescence static evaluations memoised by full key (exact: the same
# position always has the same static score). C_SEE_MAIN: in the main search skip
# captures that lose material on the exchange at depth <= 5. C_CHECK_CAP: at most
# this many check extensions along one line (0 = unlimited, the reference).
C_QS_CACHE, C_SEE_MAIN, C_CHECK_CAP = 24, 25, 26
# C_TT_BUCKETS: the table as pairs of slots. The even slot keeps the deeper
# entry (replaced when the key matches, the entry has aged, or the new depth is
# not shallower); the odd slot always takes the store. A probe checks both, so
# a deep entry survives the key traffic that evicts it from a single slot.
C_TT_BUCKETS = 27
EVAL_CACHE_BITS = 20
EVAL_CACHE_SIZE = 1 << EVAL_CACHE_BITS
EVAL_CACHE_MASK = np.uint64(EVAL_CACHE_SIZE - 1)
CTRL_SIZE = 32


@njit(cache=False)
def phase_percent(ctrl: Any, pieces: Any) -> Any:
    if pieces <= 8:
        return ctrl[C_PH_LE8]
    if pieces <= 12:
        return ctrl[C_PH_9_12]
    if pieces <= 16:
        return ctrl[C_PH_13_16]
    if pieces <= 20:
        return ctrl[C_PH_17_20]
    return 100

# LMR: reduction by depth and move number, the usual log-log formula. A quiet
# move that is neither the hash move nor a killer, searched after the first two,
# at depth >= 3, gets its depth cut by this much on a null window and is
# re-searched at full depth only if it beats alpha anyway.
LMR_TABLE = np.zeros((64, 64), dtype=np.int64)
for _d in range(1, 64):
    for _m in range(1, 64):
        LMR_TABLE[_d, _m] = int(0.75 + np.log(_d) * np.log(_m) / 2.25)
# LMP: at depth d, once this many moves have been searched, remaining quiet
# moves are skipped when not in check and not near a mate.
LMP_LIMIT = np.array([0, 5, 8, 13], dtype=np.int64)


def new_eval_cache() -> tuple[Any, ...]:
    """(key, value) arrays for the quiescence static-eval memo."""
    return (np.zeros(EVAL_CACHE_SIZE, dtype=np.uint64), np.zeros(EVAL_CACHE_SIZE, dtype=np.int32))


def new_table() -> tuple[Any, ...]:
    """(key, data) arrays for TT_SIZE entries. data packs, low to high: score
    offset by 2**15 (16 bits), static evaluation offset by 2**15 or 0 for none
    (16), move (16), flag (2), depth (8), age (6). One entry is two cache lines
    at most instead of six."""
    return (np.zeros(TT_SIZE, dtype=np.uint64), np.zeros(TT_SIZE, dtype=np.uint64))


NO_EVAL = -INFINITY


@njit(cache=False)
def pack(score: Any, move: Any, flag: Any, depth: Any, age: Any, static: Any) -> Any:
    if static == NO_EVAL:
        packed_eval = 0
    else:
        packed_eval = static + (1 << 15)
        if packed_eval < 1:
            packed_eval = 1
        elif packed_eval > 0xFFFF:
            packed_eval = 0xFFFF
    return (
        np.uint64(score + (1 << 15))
        | (np.uint64(packed_eval) << np.uint64(16))
        | (np.uint64(move) << np.uint64(32))
        | (np.uint64(flag) << np.uint64(48))
        | (np.uint64(depth) << np.uint64(50))
        | (np.uint64(age & 63) << np.uint64(58))
    )


@njit(cache=False)
def unpack_score(data: Any) -> Any:
    return np.int64(data & np.uint64(0xFFFF)) - (1 << 15)


@njit(cache=False)
def unpack_eval(data: Any) -> Any:
    packed_eval = np.int64((data >> np.uint64(16)) & np.uint64(0xFFFF))
    if packed_eval == 0:
        return NO_EVAL
    return packed_eval - (1 << 15)


@njit(cache=False)
def unpack_move(data: Any) -> Any:
    return np.int64((data >> np.uint64(32)) & np.uint64(0xFFFF))


@njit(cache=False)
def unpack_flag(data: Any) -> Any:
    return np.int64((data >> np.uint64(48)) & np.uint64(3))


@njit(cache=False)
def unpack_depth(data: Any) -> Any:
    return np.int64((data >> np.uint64(50)) & np.uint64(0xFF))


@njit(cache=False)
def unpack_age(data: Any) -> Any:
    return np.int64(data >> np.uint64(58))


@njit(cache=False)
def timed_out(deadline: Any) -> Any:
    with objmode(now="float64"):
        now = time.monotonic()
    return now > deadline


@njit(cache=False)
def draw_score(meta: Any, ctrl: Any) -> Any:
    d = ctrl[C_DRAW_ROOT]
    if d == 0:
        return 0
    return d if meta[fb.SIDE] == ctrl[C_ROOT_SIDE] else -d


@njit(cache=False)
def to_table(score: Any, ply: Any) -> Any:
    if score > DISTANCE_THRESHOLD:
        return score + ply
    if score < -DISTANCE_THRESHOLD:
        return score - ply
    return score


@njit(cache=False)
def from_table(score: Any, ply: Any) -> Any:
    if score > DISTANCE_THRESHOLD:
        return score - ply
    if score < -DISTANCE_THRESHOLD:
        return score + ply
    return score


@njit(cache=False, fastmath=True)
def evaluate(
    meta: Any, white: Any, black: Any, w2t: Any, b2: Any, w3: Any, b3: Any, scratch: Any
) -> Any:
    """agent._eval_bucket_kernel with the side and bucket chosen here."""
    buckets = w2t.shape[0]
    k = (meta[fb.PIECES] - 1) * buckets // 32
    if k < 0:
        k = 0
    elif k >= buckets:
        k = buckets - 1
    if meta[fb.SIDE] == 0:
        own = white
        opponent = black
    else:
        own = black
        opponent = white
    acc = own.shape[0]
    hidden = scratch  # caller-owned: no allocation per evaluation
    for i in range(acc):
        x = own[i]
        x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        hidden[i] = x * x
        y = opponent[i]
        y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)
        hidden[acc + i] = y * y
    out = b3[k, 0]
    for j in range(0, 32, 4):
        t0 = b2[k, j]
        t1 = b2[k, j + 1]
        t2 = b2[k, j + 2]
        t3 = b2[k, j + 3]
        r0 = w2t[k, j]
        r1 = w2t[k, j + 1]
        r2 = w2t[k, j + 2]
        r3 = w2t[k, j + 3]
        for i in range(2 * acc):
            h = hidden[i]
            t0 += h * r0[i]
            t1 += h * r1[i]
            t2 += h * r2[i]
            t3 += h * r3[i]
        if t0 > 0.0:
            out += t0 * w3[k, j, 0]
        if t1 > 0.0:
            out += t1 * w3[k, j + 1, 0]
        if t2 > 0.0:
            out += t2 * w3[k, j + 2, 0]
        if t3 > 0.0:
            out += t3 * w3[k, j + 3, 0]
    return int(float(out) * OUTPUT_SCALE)


@njit(cache=False, nogil=True)
def quiesce(
    bb: Any, sq: Any, meta: Any, undo: Any, keys: Any,
    w1: Any, b1: Any, white: Any, black: Any, astack: Any, zones: Any, king_zones: Any,
    w2t: Any, b2: Any, w3: Any, b3: Any,
    butterfly: Any, moves: Any, scores: Any, ctrl: Any, deadline: Any,
    alpha: Any, beta: Any, depth: Any, ply: Any, scratch: Any,
    ec_key: Any, ec_val: Any, exts: Any,
) -> Any:
    ctrl[C_NODES] += 1
    if (ctrl[C_NODES] & POLL_MASK) == 0 and (ctrl[C_STOP] != 0 or timed_out(deadline)):
        ctrl[C_ABORT] = 1
        return 0

    if ctrl[C_QS_CACHE] != 0:
        qkey = keys[meta[fb.PLY]]
        qslot = np.int64(qkey & EVAL_CACHE_MASK)
        if ec_key[qslot] == qkey:
            standing = np.int64(ec_val[qslot])
        else:
            standing = evaluate(meta, white, black, w2t, b2, w3, b3, scratch)
            ec_key[qslot] = qkey
            ec_val[qslot] = standing
    else:
        standing = evaluate(meta, white, black, w2t, b2, w3, b3, scratch)
    if standing >= beta:
        return standing
    if standing + BIG_DELTA < alpha:
        return standing
    if standing > alpha:
        alpha = standing
    if depth >= ctrl[C_QS_CAP] or ply >= fb.MAX_PLY - 2:
        return standing

    captures = moves[ply]
    n = fb.gen_legal(bb, sq, meta, captures, True)
    sc = scores[ply]
    fb.score_moves(captures, n, sq, 0, 0, 0, butterfly, sc)
    use_see = ctrl[C_SEE] != 0
    for i in range(n):
        move = fb.pick_move(captures, sc, i, n)
        victim = sq[(move >> 6) & 63]
        if victim >= 0 and (move >> 12) == 0 and standing + MVV[victim % 6] + DELTA_MARGIN < alpha:
            continue
        if use_see and (move >> 12) == 0 and fb.see(bb, sq, meta, move) < 0:
            # A capture that loses material on the exchange cannot raise alpha.
            continue
        fb.make_full(
            bb, sq, meta, undo, keys, move, w1, b1, white, black, astack, zones, king_zones
        )
        score = -quiesce(
            bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
            w2t, b2, w3, b3, butterfly, moves, scores, ctrl, deadline,
            -beta, -alpha, depth + 1, ply + 1, scratch, ec_key, ec_val, exts,
        )
        fb.unmake_full(bb, sq, meta, undo, keys, white, black, astack, zones)
        if ctrl[C_ABORT]:
            return 0
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha


@njit(cache=False, nogil=True)
def search(
    bb: Any, sq: Any, meta: Any, undo: Any, keys: Any,
    w1: Any, b1: Any, white: Any, black: Any, astack: Any, zones: Any, king_zones: Any,
    w2t: Any, b2: Any, w3: Any, b3: Any,
    tt_key: Any, tt_data: Any,
    killers: Any, butterfly: Any, moves: Any, scores: Any, rep_keys: Any,
    ctrl: Any, deadline: Any,
    depth: Any, alpha: Any, beta: Any, ply: Any, scratch: Any, counter: Any, quiets: Any,
    ec_key: Any, ec_val: Any, exts: Any,
) -> Any:
    ctrl[C_NODES] += 1
    if (ctrl[C_NODES] & POLL_MASK) == 0 and (ctrl[C_STOP] != 0 or timed_out(deadline)):
        ctrl[C_ABORT] = 1
        return 0

    key = keys[meta[fb.PLY]]
    if ply > 0:
        # No repetition is reachable within four reversible plies, and the game
        # list under REPETITION_TWOFOLD is long: skip the scans until then.
        if meta[fb.HALFMOVE] >= 4:
            for i in range(rep_keys.shape[0]):
                if rep_keys[i] == key:
                    return draw_score(meta, ctrl)
            if fb.repeats(meta, keys):
                return draw_score(meta, ctrl)
        if meta[fb.HALFMOVE] >= 100:
            n = fb.gen_legal(bb, sq, meta, moves[ply], False)
            if n == 0 and fb.in_check(bb, meta):
                return -MATE + ply
            return draw_score(meta, ctrl)
    if ply >= fb.MAX_PLY - 8:
        return evaluate(meta, white, black, w2t, b2, w3, b3, scratch)

    if ctrl[C_SAFE] != 0 and ply > 0:
        # Mate-distance pruning: no line from here can beat a mate already found
        # closer to the root, in either direction.
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    original_alpha = alpha
    hash_move = 0
    slot = np.int64(0)
    cached_eval = NO_EVAL
    if ctrl[C_TT_OFF] == 0:
        slot = np.int64(key & TT_MASK)
        if ctrl[C_TT_BUCKETS] != 0:
            slot = slot & -2
            if tt_key[slot] != key and tt_key[slot + 1] == key:
                slot = slot + 1
        if tt_key[slot] == key:
            data = tt_data[slot]
            stored_depth = unpack_depth(data)
            flag = unpack_flag(data)
            hash_move = unpack_move(data)
            cached_eval = unpack_eval(data)
            stored_score = from_table(unpack_score(data), ply)
            if stored_depth >= depth and ply > 0:
                if flag == 0:
                    return stored_score
                if flag == 1 and stored_score > alpha:
                    alpha = stored_score
                elif flag == 2 and stored_score < beta:
                    beta = stored_score
                if alpha >= beta:
                    return stored_score

    if ctrl[C_IIR] != 0 and depth >= 4 and hash_move == 0 and ctrl[C_TT_OFF] == 0:
        depth -= 1

    in_check = fb.in_check(bb, meta)
    if in_check and ply < MAX_PLY - 8:
        if ctrl[C_CHECK_CAP] == 0 or (ply > 0 and exts[ply - 1] < ctrl[C_CHECK_CAP]):
            depth += 1
            exts[ply] = (exts[ply - 1] if ply > 0 else 0) + 1
        else:
            exts[ply] = exts[ply - 1] if ply > 0 else 0
    else:
        exts[ply] = exts[ply - 1] if ply > 0 else 0

    if depth <= 0:
        return quiesce(
            bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
            w2t, b2, w3, b3, butterfly, moves, scores, ctrl, deadline, alpha, beta, 0, ply,
            scratch, ec_key, ec_val, exts,
        )

    standing = -INFINITY
    percent = 100
    if ctrl[C_RFP_PHASE] != 0:
        percent = phase_percent(ctrl, meta[fb.PIECES])
    if (
        percent != 0
        and depth <= RFP_MAX_DEPTH
        and not in_check
        and (ctrl[C_HYGIENE] == 0 or abs(beta) < DISTANCE_THRESHOLD)
    ):
        if cached_eval != NO_EVAL:
            standing = cached_eval
        else:
            standing = evaluate(meta, white, black, w2t, b2, w3, b3, scratch)
            cached_eval = standing
        if standing - RFP_MARGIN * depth * percent // 100 >= beta:
            return standing

    futile = False
    if (
        ctrl[C_FUTILITY] != 0
        and percent != 0
        and depth <= 2
        and not in_check
        and abs(alpha) < DISTANCE_THRESHOLD
    ):
        if standing == -INFINITY:
            if cached_eval != NO_EVAL:
                standing = cached_eval
            else:
                standing = evaluate(meta, white, black, w2t, b2, w3, b3, scratch)
                cached_eval = standing
        futile = standing + FUTILITY_MARGIN[depth] * percent // 100 <= alpha

    if (
        depth >= NMP_MIN_DEPTH
        and not in_check
        and abs(beta) < DISTANCE_THRESHOLD
        and fb.non_pawn_material(bb, meta[fb.SIDE])
        and (ctrl[C_NMP_GUARD] == 0 or ply == 0 or undo[meta[fb.PLY] - 1, fb.U_MOVE] != 0)
    ):
        null_depth = depth - 1 - NMP_REDUCTION
        if ctrl[C_SAFE] != 0:
            null_depth -= depth // 6  # deeper nodes can afford a bigger reduction
        fb.make_null(bb, meta, undo, keys)
        score = -search(
            bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
            w2t, b2, w3, b3, tt_key, tt_data,
            killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
            null_depth, -beta, -beta + 1, ply + 1, scratch, counter, quiets, ec_key, ec_val, exts,
        )
        fb.unmake_null(meta, undo)
        if ctrl[C_ABORT]:
            return 0
        if score >= beta:
            return beta

    mv = moves[ply]
    n = fb.gen_legal(bb, sq, meta, mv, False)
    if n == 0:
        return -MATE + ply if in_check else 0
    sc = scores[ply]
    history2 = ctrl[C_HISTORY2] != 0
    base = 0
    counter_move = 0
    if history2:
        base = meta[fb.SIDE] * 4096
        prev = undo[meta[fb.PLY] - 1, fb.U_MOVE] if meta[fb.PLY] > 0 else 0
        if prev != 0:
            counter_move = counter[(prev & 63) * 64 + ((prev >> 6) & 63)]
    fb.score_moves(
        mv, n, sq, hash_move, killers[ply, 0], killers[ply, 1], butterfly, sc, counter_move, base
    )

    best_score = -INFINITY
    best_move = 0
    searched = 0
    pvs = ctrl[C_PVS] != 0
    lmr = ctrl[C_LMR] != 0 and depth >= 3 and not in_check
    lmp = ctrl[C_LMP] != 0 and depth <= 3 and not in_check and abs(alpha) < DISTANCE_THRESHOLD
    for i in range(n):
        move = fb.pick_move(mv, sc, i, n)
        quiet = sq[(move >> 6) & 63] < 0
        plain = quiet and (move >> 12) == 0
        if futile and plain:
            continue
        if (
            ctrl[C_SEE_MAIN] != 0
            and not quiet
            and (move >> 12) == 0
            and depth <= 5
            and searched > 0
            and abs(alpha) < DISTANCE_THRESHOLD
            and fb.see(bb, sq, meta, move) < -20 * depth * depth
        ):
            continue
        if lmp and plain and searched >= LMP_LIMIT[depth]:
            continue
        reduction = 0
        if (
            lmr
            and plain
            and searched >= 2
            and move != hash_move
            and move != killers[ply, 0]
            and move != killers[ply, 1]
        ):
            reduction = LMR_TABLE[min(depth, 63), min(searched, 63)]
        fb.make_full(
            bb, sq, meta, undo, keys, move, w1, b1, white, black, astack, zones, king_zones
        )
        if history2 and plain:
            quiets[ply, searched] = move  # every quiet tried at this node, for the malus
        if reduction > 0:
            reduced = depth - 1 - reduction
            if reduced < 1:
                reduced = 1  # never reduce straight into quiescence
            score = -search(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, tt_key, tt_data,
                killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                reduced, -alpha - 1, -alpha, ply + 1, scratch, counter, quiets,
                ec_key, ec_val, exts,
            )
            if score > alpha and ctrl[C_ABORT] == 0:
                # Beat alpha reduced: confirm at full depth. Under PVS a null window
                # first (the full-window re-search below follows if it holds);
                # without PVS the full window straight away, one search not two.
                if pvs:
                    score = -search(
                        bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones,
                        king_zones, w2t, b2, w3, b3, tt_key, tt_data,
                        killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                        depth - 1, -alpha - 1, -alpha, ply + 1, scratch, counter, quiets,
                        ec_key, ec_val, exts,
                    )
                else:
                    score = -search(
                        bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones,
                        king_zones, w2t, b2, w3, b3, tt_key, tt_data,
                        killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                        depth - 1, -beta, -alpha, ply + 1, scratch, counter, quiets,
                        ec_key, ec_val, exts,
                    )
        elif pvs and searched > 0:
            score = -search(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, tt_key, tt_data,
                killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                depth - 1, -alpha - 1, -alpha, ply + 1, scratch, counter, quiets,
                ec_key, ec_val, exts,
            )
        else:
            score = -search(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, tt_key, tt_data,
                killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                depth - 1, -beta, -alpha, ply + 1, scratch, counter, quiets, ec_key, ec_val, exts,
            )
        narrow = pvs and (reduction > 0 or searched > 0)
        if narrow and alpha < score < beta and ctrl[C_ABORT] == 0:
            # The null window said this move beats alpha: find out by how much.
            score = -search(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, tt_key, tt_data,
                killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                depth - 1, -beta, -alpha, ply + 1, scratch, counter, quiets, ec_key, ec_val, exts,
            )
        fb.unmake_full(bb, sq, meta, undo, keys, white, black, astack, zones)
        if ctrl[C_ABORT]:
            return 0
        searched += 1
        if score > best_score:
            best_score = score
            best_move = move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if quiet:
                        if killers[ply, 0] != move:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = move
                        if history2:
                            # gravity: pull toward +MAX for the cutoff move, toward
                            # -MAX for the quiets searched before it at this node
                            bonus = depth * depth
                            if bonus > 1200:
                                bonus = 1200
                            idx = base + (move & 63) * 64 + ((move >> 6) & 63)
                            butterfly[idx] += bonus - butterfly[idx] * bonus // HISTORY_MAX
                            for q in range(searched):
                                other = quiets[ply, q]
                                if other != move and other != 0:
                                    jdx = base + (other & 63) * 64 + ((other >> 6) & 63)
                                    butterfly[jdx] -= bonus + butterfly[jdx] * bonus // HISTORY_MAX
                            prev = undo[meta[fb.PLY] - 1, fb.U_MOVE] if meta[fb.PLY] > 0 else 0
                            if prev != 0:
                                counter[(prev & 63) * 64 + ((prev >> 6) & 63)] = move
                        else:
                            butterfly[(move & 63) * 64 + ((move >> 6) & 63)] += depth * depth
                    break

    if searched == 0:
        # Every move was futility-pruned: the position is at least as bad as the
        # static score says, which is below alpha.
        return standing

    if ctrl[C_TT_OFF] == 0:
        if best_score <= original_alpha:
            flag = 2
        elif best_score >= beta:
            flag = 1
        else:
            flag = 0
        age = ctrl[C_AGE]
        if ctrl[C_TT_BUCKETS] != 0:
            dslot = np.int64(key & TT_MASK) & -2
            old = tt_data[dslot]
            if tt_key[dslot] == key:
                slot = dslot
            elif tt_key[dslot + 1] == key:
                slot = dslot + 1
            elif unpack_age(old) != (age & 63) or depth >= unpack_depth(old):
                slot = dslot
            else:
                slot = dslot + 1
            replace = True
        elif ctrl[C_TT_KEEP] != 0:
            old = tt_data[slot]
            handicap = 4 if unpack_age(old) != (age & 63) else 0
            replace = tt_key[slot] == key or depth + handicap >= unpack_depth(old)
        else:
            old = tt_data[slot]
            replace = (
                tt_key[slot] == key
                or unpack_age(old) != (age & 63)
                or depth >= unpack_depth(old)
            )
        if replace:
            tt_key[slot] = key
            tt_data[slot] = pack(
                to_table(best_score, ply), best_move, flag, depth, age, cached_eval
            )
    return best_score


def warm_up(w1: Any, b1: Any, w2t: Any, b2: Any, w3: Any, b3: Any, king_zones: int) -> None:
    """Compile both kernels now, on a real position, inside the init budget."""
    import chess

    fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QK2R w KQ - 0 8"
    pos = fb.Position(chess.Board(fen))
    acc = w1.shape[1]
    white = np.zeros(acc, dtype=np.float32)
    black = np.zeros(acc, dtype=np.float32)
    astack = np.zeros((fb.MAX_PLY, 2, acc), dtype=np.float32)
    zones = np.zeros(2, dtype=np.int64)
    fb.refresh(pos.bb, pos.sq, pos.meta, w1, b1, white, black, zones, king_zones)
    table = new_table()
    killers = np.zeros((fb.MAX_PLY, 2), dtype=np.int32)
    butterfly = np.zeros(8192, dtype=np.int32)
    counter = np.zeros(4096, dtype=np.int32)
    quiets = np.zeros((fb.MAX_PLY, fb.MOVE_CAP), dtype=np.int32)
    moves = np.zeros((fb.MAX_PLY, fb.MOVE_CAP), dtype=np.int32)
    scores = np.zeros((fb.MAX_PLY, fb.MOVE_CAP), dtype=np.int64)
    scratch = np.zeros(2 * acc, dtype=np.float32)
    rep_keys = np.zeros(0, dtype=np.uint64)
    ec_key, ec_val = new_eval_cache()
    exts = np.zeros(fb.MAX_PLY, dtype=np.int64)
    ctrl = np.zeros(CTRL_SIZE, dtype=np.int64)
    ctrl[C_HYGIENE] = 1
    ctrl[C_FUTILITY] = 1
    ctrl[C_QS_CAP] = 8
    search(  # type: ignore[call-arg]
        pos.bb, pos.sq, pos.meta, pos.undo, pos.keys, w1, b1, white, black, astack, zones,
        king_zones, w2t, b2, w3, b3, *table, killers, butterfly, moves, scores, rep_keys,
        ctrl, time.monotonic() + 60.0, 2, -INFINITY, INFINITY, 0, scratch, counter, quiets,
        ec_key, ec_val, exts,
    )
