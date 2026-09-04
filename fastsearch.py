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

TT_BITS = 20
TT_SIZE = 1 << TT_BITS
TT_MASK = np.uint64(TT_SIZE - 1)

# ctrl slots: search state the Python side sets and reads.
C_NODES, C_ABORT, C_AGE, C_ROOT_SIDE, C_DRAW_ROOT, C_TT_OFF, C_HYGIENE, C_FUTILITY = range(8)
CTRL_SIZE = 8


def new_table() -> tuple[Any, ...]:
    """(key, depth, flag, move, score, age) arrays for TT_SIZE entries."""
    return (
        np.zeros(TT_SIZE, dtype=np.uint64),
        np.zeros(TT_SIZE, dtype=np.int16),
        np.zeros(TT_SIZE, dtype=np.int8),
        np.zeros(TT_SIZE, dtype=np.int32),
        np.zeros(TT_SIZE, dtype=np.int32),
        np.zeros(TT_SIZE, dtype=np.int32),
    )


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
    fb.order_moves(captures, n, sq, 0, 0, 0, butterfly, scores)
    for i in range(n):
        move = captures[i]
        victim = sq[(move >> 6) & 63]
        if victim >= 0 and (move >> 12) == 0 and standing + MVV[victim % 6] + DELTA_MARGIN < alpha:
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
    tt_key: Any, tt_depth: Any, tt_flag: Any, tt_move: Any, tt_score: Any, tt_age: Any,
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
    if ctrl[C_TT_OFF] == 0:
        slot = np.int64(key & TT_MASK)
        if tt_key[slot] == key:
            stored_depth = tt_depth[slot]
            flag = tt_flag[slot]
            hash_move = tt_move[slot]
            stored_score = from_table(tt_score[slot], ply)
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
        standing = evaluate(meta, white, black, w2t, b2, w3, b3)
        if standing - RFP_MARGIN * depth >= beta:
            return standing

    futile = False
    if ctrl[C_FUTILITY] != 0 and depth <= 2 and not in_check and abs(alpha) < DISTANCE_THRESHOLD:
        if standing == -INFINITY:
            standing = evaluate(meta, white, black, w2t, b2, w3, b3)
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
            w2t, b2, w3, b3, tt_key, tt_depth, tt_flag, tt_move, tt_score, tt_age,
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
    fb.order_moves(mv, n, sq, hash_move, killers[ply, 0], killers[ply, 1], butterfly, scores)

    best_score = -INFINITY
    best_move = 0
    searched = 0
    for i in range(n):
        move = mv[i]
        quiet = sq[(move >> 6) & 63] < 0
        if futile and quiet and (move >> 12) == 0:
            continue
        fb.make_full(
            bb, sq, meta, undo, keys, move, w1, b1, white, black, astack, zones, king_zones
        )
        score = -search(
            bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
            w2t, b2, w3, b3, tt_key, tt_depth, tt_flag, tt_move, tt_score, tt_age,
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
        if tt_key[slot] == key or tt_age[slot] != age or depth >= tt_depth[slot]:
            tt_key[slot] = key
            tt_depth[slot] = depth
            tt_flag[slot] = flag
            tt_move[slot] = best_move
            tt_score[slot] = to_table(best_score, ply)
            tt_age[slot] = age
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
    scores = np.zeros(fb.MOVE_CAP, dtype=np.int64)
    rep_keys = np.zeros(0, dtype=np.uint64)
    ctrl = np.zeros(CTRL_SIZE, dtype=np.int64)
    ctrl[C_HYGIENE] = 1
    ctrl[C_FUTILITY] = 1
    search(  # type: ignore[call-arg]
        pos.bb, pos.sq, pos.meta, pos.undo, pos.keys, w1, b1, white, black, astack, zones,
        king_zones, w2t, b2, w3, b3, *table, killers, butterfly, moves, scores, rep_keys,
        ctrl, time.monotonic() + 60.0, 2, -INFINITY, INFINITY, 0,
    )
