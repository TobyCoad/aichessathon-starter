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
CTRL_SIZE = 16

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
def evaluate(meta: Any, white: Any, black: Any, w2t: Any, b2: Any, w3: Any, b3: Any) -> Any:
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
    hidden = np.empty(2 * acc, dtype=np.float32)
    for i in range(acc):
        x = own[i]
        x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        hidden[i] = x * x
        y = opponent[i]
        y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)
        hidden[acc + i] = y * y
    out = b3[k, 0]
    for j in range(32):
        total = b2[k, j]
        for i in range(2 * acc):
            total += hidden[i] * w2t[k, j, i]
        if total > 0.0:
            out += total * w3[k, j, 0]
    return int(float(out) * OUTPUT_SCALE)


@njit(cache=False)
def quiesce(
    bb: Any, sq: Any, meta: Any, undo: Any, keys: Any,
    w1: Any, b1: Any, white: Any, black: Any, astack: Any, zones: Any, king_zones: Any,
    w2t: Any, b2: Any, w3: Any, b3: Any,
    butterfly: Any, moves: Any, scores: Any, ctrl: Any, deadline: Any,
    alpha: Any, beta: Any, depth: Any, ply: Any,
) -> Any:
    ctrl[C_NODES] += 1
    if (ctrl[C_NODES] & POLL_MASK) == 0 and timed_out(deadline):
        ctrl[C_ABORT] = 1
        return 0

    standing = evaluate(meta, white, black, w2t, b2, w3, b3)
    if standing >= beta:
        return standing
    if standing + BIG_DELTA < alpha:
        return standing
    if standing > alpha:
        alpha = standing
    if depth >= 8 or ply >= fb.MAX_PLY - 2:
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
            -beta, -alpha, depth + 1, ply + 1,
        )
        fb.unmake_full(bb, sq, meta, undo, keys, white, black, astack, zones)
        if ctrl[C_ABORT]:
            return 0
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha


@njit(cache=False)
def search(
    bb: Any, sq: Any, meta: Any, undo: Any, keys: Any,
    w1: Any, b1: Any, white: Any, black: Any, astack: Any, zones: Any, king_zones: Any,
    w2t: Any, b2: Any, w3: Any, b3: Any,
    tt_key: Any, tt_data: Any,
    killers: Any, butterfly: Any, moves: Any, scores: Any, rep_keys: Any,
    ctrl: Any, deadline: Any,
    depth: Any, alpha: Any, beta: Any, ply: Any,
) -> Any:
    ctrl[C_NODES] += 1
    if (ctrl[C_NODES] & POLL_MASK) == 0 and timed_out(deadline):
        ctrl[C_ABORT] = 1
        return 0

    key = keys[meta[fb.PLY]]
    if ply > 0:
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
        return evaluate(meta, white, black, w2t, b2, w3, b3)

    original_alpha = alpha
    hash_move = 0
    slot = np.int64(0)
    cached_eval = NO_EVAL
    if ctrl[C_TT_OFF] == 0:
        slot = np.int64(key & TT_MASK)
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

    in_check = fb.in_check(bb, meta)
    if in_check and ply < MAX_PLY - 8:
        depth += 1

    if depth <= 0:
        return quiesce(
            bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
            w2t, b2, w3, b3, butterfly, moves, scores, ctrl, deadline, alpha, beta, 0, ply,
        )

    standing = -INFINITY
    if (
        depth <= RFP_MAX_DEPTH
        and not in_check
        and (ctrl[C_HYGIENE] == 0 or abs(beta) < DISTANCE_THRESHOLD)
    ):
        if cached_eval != NO_EVAL:
            standing = cached_eval
        else:
            standing = evaluate(meta, white, black, w2t, b2, w3, b3)
            cached_eval = standing
        if standing - RFP_MARGIN * depth >= beta:
            return standing

    futile = False
    if ctrl[C_FUTILITY] != 0 and depth <= 2 and not in_check and abs(alpha) < DISTANCE_THRESHOLD:
        if standing == -INFINITY:
            if cached_eval != NO_EVAL:
                standing = cached_eval
            else:
                standing = evaluate(meta, white, black, w2t, b2, w3, b3)
                cached_eval = standing
        futile = standing + FUTILITY_MARGIN[depth] <= alpha

    if (
        depth >= NMP_MIN_DEPTH
        and not in_check
        and abs(beta) < DISTANCE_THRESHOLD
        and fb.non_pawn_material(bb, meta[fb.SIDE])
    ):
        fb.make_null(bb, meta, undo, keys)
        score = -search(
            bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
            w2t, b2, w3, b3, tt_key, tt_data,
            killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
            depth - 1 - NMP_REDUCTION, -beta, -beta + 1, ply + 1,
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
    fb.score_moves(mv, n, sq, hash_move, killers[ply, 0], killers[ply, 1], butterfly, sc)

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
        if reduction > 0:
            score = -search(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, tt_key, tt_data,
                killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                depth - 1 - reduction, -alpha - 1, -alpha, ply + 1,
            )
            if score > alpha and ctrl[C_ABORT] == 0:
                score = -search(
                    bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                    w2t, b2, w3, b3, tt_key, tt_data,
                    killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                    depth - 1, -alpha - 1, -alpha, ply + 1,
                )
        elif pvs and searched > 0:
            score = -search(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, tt_key, tt_data,
                killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                depth - 1, -alpha - 1, -alpha, ply + 1,
            )
        else:
            score = -search(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, tt_key, tt_data,
                killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                depth - 1, -beta, -alpha, ply + 1,
            )
        narrow = reduction > 0 or (pvs and searched > 0)
        if narrow and alpha < score < beta and ctrl[C_ABORT] == 0:
            # The null window said this move beats alpha: find out by how much.
            score = -search(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, tt_key, tt_data,
                killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                depth - 1, -beta, -alpha, ply + 1,
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
        old = tt_data[slot]
        if tt_key[slot] == key or unpack_age(old) != (age & 63) or depth >= unpack_depth(old):
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
    butterfly = np.zeros(4096, dtype=np.int32)
    moves = np.zeros((fb.MAX_PLY, fb.MOVE_CAP), dtype=np.int32)
    scores = np.zeros((fb.MAX_PLY, fb.MOVE_CAP), dtype=np.int64)
    rep_keys = np.zeros(0, dtype=np.uint64)
    ctrl = np.zeros(CTRL_SIZE, dtype=np.int64)
    ctrl[C_HYGIENE] = 1
    ctrl[C_FUTILITY] = 1
    search(  # type: ignore[call-arg]
        pos.bb, pos.sq, pos.meta, pos.undo, pos.keys, w1, b1, white, black, astack, zones,
        king_zones, w2t, b2, w3, b3, *table, killers, butterfly, moves, scores, rep_keys,
        ctrl, time.monotonic() + 60.0, 2, -INFINITY, INFINITY, 0,
    )
