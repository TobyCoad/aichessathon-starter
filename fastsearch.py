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

import re
import time
from pathlib import Path
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
NMP_VERIFY_DEPTH = 10
MAX_PLY = 72  # agent.MAX_PLY: the check-extension limit
FUTILITY_MARGIN = np.array([0, 150, 300], dtype=np.int64)
RAZOR_MAX_DEPTH = 3
# RAZOR (search.md #11): indexed by depth, cp below alpha at which a node is
# assumed unrescuable by a quiet move and verified with a quiescence search.
RAZOR_MARGIN = np.array([0, 500, 700, 900], dtype=np.int64)
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
# C_LMR_AGGR: reduce quiet moves from the second one searched (not the third)
# with the steeper LMR_TABLE_AGGR, adjusted by butterfly history: one ply less
# above +8000, one more below -8000, never below zero and never into quiescence.
# agent.py turns PVS on alongside it -- null-window re-searches are what make
# the deeper reductions cheap.
C_LMR_AGGR = 28
# C_LAZY_ACC: defer the accumulator update from make to the first evaluate on
# the line (exact: same nodes, same scores). C_ACC_PLY holds the ply the
# accumulators currently represent; sync_acc replays the pending moves from the
# undo stack. A king move that crosses a zone boundary still updates eagerly
# (the rebuild needs the board of its own ply), so a pending stretch never
# contains a zone change. agent.FastEngine.root_search resets C_ACC_PLY to the
# board's ply before every kernel call; the root makes stay eager.
C_LAZY_ACC, C_ACC_PLY = 29, 30
# C_PRUNE2: prune plain quiet moves harder at low depth, on top of C_FUTILITY.
# After the first move at a node, when not in check and not near a mate:
# (a) futility to depth 4 -- skip when the static eval plus FUTILITY_MARGIN2
# cannot reach alpha; (b) skip a quiet whose butterfly history (the
# C_HISTORY2 side-to-move band) is below -HIST_PRUNE_SLOPE * depth.
C_PRUNE2 = 31
FUTILITY_MARGIN2 = np.array([0, 100, 200, 300, 400], dtype=np.int64)
HIST_PRUNE_SLOPE = 1500
# C_SINGULAR: singular extensions. At depth >= SINGULAR_MIN_DEPTH with a hash
# move whose stored bound is exact or a lower bound at depth >= depth - 3, the
# node is searched again without that move at (depth - 1) // 2 with the window
# (s - 1, s), s = stored - 2 * depth. If nothing else reaches s the hash move
# is singular and is searched one ply deeper. The excluded move travels in
# C_EXCL_MOVE / C_EXCL_PLY (read once at node entry); the excluded search
# neither probes nor stores the table at that node and skips the null move.
# SINGULAR_EXT_CAP bounds check + singular extensions along one line.
C_SINGULAR, C_EXCL_MOVE, C_EXCL_PLY = 32, 33, 34
SINGULAR_MIN_DEPTH = 7
SINGULAR_EXT_CAP = 6
# C_HMC_DRAW: the halfmove-clock value at which a line scores as a draw
# (normally 100 = the fifty-move rule; <= 0 also means 100). agent.ADJUDICATION
# lowers it at the root when the side to move is losing the referee's ply-300
# material adjudication and a fifty-move draw is reachable before the cap:
# any stretch of non-zeroing plies then reads as draw-reaching.
# C_HIST2_FIX: zero quiets[ply, searched] for non-quiet moves so the history
# malus never punishes a stale entry from an earlier node at the same ply.
# C_KILLER_CLEAR: clear killers[ply + 2] on node entry (grandchild killers are
# stale once this node's subtree is done).
C_HMC_DRAW, C_HIST2_FIX, C_KILLER_CLEAR = 35, 36, 37
# C_CONT_HIST: 1-ply continuation history. conthist1[(prev_piece*64 + prev_to)*768
# + piece*64 + to] scores a quiet by how it fared after the previous move: added
# to the quiet ordering score (stays below killers/counter), folded into the LMR
# history term (continuous hist // CONT_LMR_DIV, clamped +/-2, replacing the
# +/-8000 step) and into the prune2 history test; updated with the butterfly
# gravity formula on a cutoff (skipped in the SINGULAR excluded-move search).
# The previous move's piece is sq[prev_to] at node entry; a null move (prev == 0)
# reads and updates nothing.
C_CONT_HIST = 38
CONT_LMR_DIV = 6000
# C_IMPROVING (v10 search.md 3.3): static_eval(ply) > static_eval(ply - 2).
# The eval is computed at every non-check node that reaches the move loop and
# stored in exts[MAX_PLY + ply] (sentinel -INFINITY in check; the SINGULAR
# excluded-move re-search must not overwrite its own ply's slot). Not improving:
# RFP margin uses depth - improving, prune2 futility FUTILITY_MARGIN2[depth -
# improving], LMR reduction += 1. Default improving at ply < 2 and after a
# sentinel (never over-prune the first two plies).
# C_CUTNODE (same source): is this node expected to fail high? Passed down as
# the kernel's one new parameter: the null-move child is always a cut node, a
# null-window child is a cut node iff its parent was not, a full-window child
# of a PV node is a PV node. Use: LMR reduction += 1 at cut nodes.
C_IMPROVING, C_CUTNODE = 39, 40
# C_NMP_V2 (V10_PLAN #6): dynamic null-move reduction R = 3 + depth//4 +
# min((standing - beta) // 200, 3), tried only when the static eval stands at
# or above beta, and skipped when the TT already holds an upper bound below
# beta (the node is expected to fail low, so the null search is wasted nodes).
# Verification search at depth >= 10 deferred to a follow-up (NMP_V2B).
C_NMP_V2 = 41
# C_CAPTURE_ORDER (V10_PLAN #7): rescore non-promotion captures after
# score_moves. SEE < 0 drops the capture below every quiet (band -(1 << 21) +
# see*16), SEE >= 0 keeps the MVV-LVA band; both add a capture-history tiebreak.
# The capture history lives in the FIRST 4608 entries of the conthist1 buffer,
# indexed (attacker_piece*64 + to)*6 + victim%6 -- CONT_HIST is closed/rejected
# so the buffer is free (agent.py refuses both switches on together), and
# reusing it keeps the kernel signature unchanged. Gravity bonus on a capture
# cutoff, no malus (v1); decayed >>= 1 per move under HYGIENE like butterfly.
C_CAPTURE_ORDER = 42
# C_QS_TT (V10_PLAN #9): probe and store the main transposition table in
# quiescence. Probe before the static eval (any depth suffices for a QS bound:
# exact returns, lower >= beta and upper <= alpha cut). Stores are depth 0,
# move 0, eval NO_EVAL, at the stand-pat cutoff, the capture-loop cutoff and
# the final return only (delta-pruned and QS_CAP returns are evals, not
# bounds); a store never evicts a same-key or current-age entry of depth > 0,
# so main-search entries and their hash moves survive.
C_QS_TT = 43
# C_SEE_QUIET (next step after V10_PLAN #7): skip a late quiet move at depth
# <= 6 when SEE says the moved piece is lost on its destination square
# (see < -30 * depth * depth). fb.see handles quiets: victim value 0, then
# both sides exchange on the target square. Never in check, never the first
# move of the node, never near a mate score.
C_SEE_QUIET = 44
# C_NMP_V2B (follow-up to C_NMP_V2, which PROMOTED as 143-nmp): on a null-move
# cutoff at depth >= NMP_VERIFY_DEPTH, verify with a reduced-depth real search
# at the same node before trusting the cutoff (zugzwang guard at the depths
# where a wrong null cutoff poisons the whole tree). C_NMP_MIN_PLY is state,
# not a switch: while nonzero, null-move pruning is disabled at plies below it
# (Stockfish's nmpMinPly), so the verification subtree cannot re-cut with
# another null near its root; it is set around the verification search and
# restored to 0 after, and stays 0 whenever C_NMP_V2B is off (exact).
C_NMP_V2B = 45
C_NMP_MIN_PLY = 46
# C_EG_SHRINK (V10_PLAN #11, overnight/eval/v10/endgame_shrink.md): below
# EG_HI pieces, blend the net eval toward pure material (agent._MATERIAL
# values, side-to-move POV) inside evaluate -- the blended value is a pure
# function of the position, so QS_EVAL_CACHE and the TT's stored static eval
# stay consistent. w/256 on the net ramps 256 at EG_HI down to C_EG_WMIN at
# EG_LO (continuous at EG_HI: the formula yields 256 there); the correction
# is clamped to +/- C_EG_CAP cp (0 = uncapped) and never touches mate-range
# scores. A damper for the documented large wrong endgame evals (games.md
# 475 cp at 11-16 pieces), not a cure -- that is NET_V10.
C_EG_SHRINK = 47
C_EG_WMIN = 48
C_EG_CAP = 49
# C_SING_EXT2 (search.md #10, the follow-up to C_SINGULAR which shipped in
# v8.5): grade the singular verification result instead of treating it as a
# yes/no. When the hash move beats the rest by more than
# SINGULAR_DOUBLE_MARGIN cp at a non-PV node it is extended TWO plies, not
# one; when it is not singular at all but the table already says the node
# fails high (tt_score >= beta), the hash move is searched one ply SHALLOWER
# -- the cutoff is coming anyway, so the ply is better spent elsewhere. Both
# arms only ever fire inside the existing C_SINGULAR block, so the entry
# guards (depth >= SINGULAR_MIN_DEPTH, a usable hash entry, the exts[] line
# cap) are unchanged; the double arm additionally needs two spare slots under
# SINGULAR_EXT_CAP so a line cannot extend further than it can today.
C_SING_EXT2 = 50
# RAZOR: fail-low shortcut at depth <= RAZOR_MAX_DEPTH (see agent.RAZOR).
C_RAZOR = 51
SINGULAR_DOUBLE_MARGIN = 25
EG_HI = 17
EG_LO = 6
EG_VALUES = np.array([100, 300, 300, 500, 900], dtype=np.int64)
EVAL_CACHE_BITS = 20
EVAL_CACHE_SIZE = 1 << EVAL_CACHE_BITS
EVAL_CACHE_MASK = np.uint64(EVAL_CACHE_SIZE - 1)
CTRL_SIZE = 52

# INIT_FOLD (agent.INIT_FOLD is the switch): compile the settled switches as
# constants. The values are scanned from agent.py next to this file, so a sed
# that flips a flag there is mirrored here on the next import. Every folded
# read is written `_F_X if _FOLD else ctrl[C_X] != 0`: numba prunes the dead
# arm of a ternary on a constant global before typing, so with _FOLD off the
# kernel is byte-for-byte today's ctrl-reading one (testing/check_fastsearch
# runs that arm against the reference with a zeroed ctrl), and with _FOLD on
# the settled branches vanish from the compile, cutting fs.warm_up ~18%
# (overnight/eval/v10/speed.md section 2). Slots still under test (C_QS_CACHE,
# C_HIST2_FIX, C_KILLER_CLEAR, C_CONT_HIST) and all value/state slots stay
# live reads. agent.py asserts ctrl matches FOLDED at engine init when folded.


def _scan_agent_flags() -> dict[str, bool]:
    try:
        src = Path(__file__).with_name("agent.py").read_text(encoding="utf-8")
    except OSError:
        return {}
    pattern = r"^([A-Z][A-Z0-9_]*): Final = (True|False)$"
    return {m.group(1): m.group(2) == "True" for m in re.finditer(pattern, src, re.MULTILINE)}


_AGENT_FLAGS = _scan_agent_flags()
_FOLD = _AGENT_FLAGS.get("INIT_FOLD", False)
_F_HYGIENE = _AGENT_FLAGS.get("HYGIENE", False)
_F_FUTILITY = _AGENT_FLAGS.get("FUTILITY", False)
_F_PVS = _AGENT_FLAGS.get("PVS", False) or _AGENT_FLAGS.get("LMR_AGGRESSIVE", False)
_F_LMR = _AGENT_FLAGS.get("LMR", False)
_F_LMP = _AGENT_FLAGS.get("LMP", False)
_F_SEE = _AGENT_FLAGS.get("SEE", False)
_F_NMP_GUARD = _AGENT_FLAGS.get("NMP_GUARD", False)
_F_RFP_PHASE = _AGENT_FLAGS.get("RFP_PHASE", False)
_F_IIR = _AGENT_FLAGS.get("IIR", False)
_F_HISTORY2 = _AGENT_FLAGS.get("HISTORY2", False)
_F_TT_KEEP = _AGENT_FLAGS.get("TT_KEEP", False)
_F_SAFE = _AGENT_FLAGS.get("SAFE_BITS", False)
_F_SEE_MAIN = _AGENT_FLAGS.get("SEE_MAIN", False)
_F_TT_BUCKETS = _AGENT_FLAGS.get("TT_BUCKETS", False)
_F_LMR_AGGR = _AGENT_FLAGS.get("LMR_AGGRESSIVE", False)
_F_LAZY_ACC = _AGENT_FLAGS.get("LAZY_ACC", False)
_F_PRUNE2 = _AGENT_FLAGS.get("PRUNE_V2", False)
_F_SINGULAR = _AGENT_FLAGS.get("SINGULAR", False)

FOLDED = {
    C_HYGIENE: _F_HYGIENE, C_FUTILITY: _F_FUTILITY, C_PVS: _F_PVS, C_LMR: _F_LMR,
    C_LMP: _F_LMP, C_SEE: _F_SEE, C_NMP_GUARD: _F_NMP_GUARD, C_RFP_PHASE: _F_RFP_PHASE,
    C_IIR: _F_IIR, C_HISTORY2: _F_HISTORY2, C_TT_KEEP: _F_TT_KEEP, C_SAFE: _F_SAFE,
    C_SEE_MAIN: _F_SEE_MAIN, C_TT_BUCKETS: _F_TT_BUCKETS, C_LMR_AGGR: _F_LMR_AGGR,
    C_LAZY_ACC: _F_LAZY_ACC, C_PRUNE2: _F_PRUNE2, C_SINGULAR: _F_SINGULAR,
}


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
# The steeper table C_LMR_AGGR uses: cuts about one ply more from the same spot.
LMR_TABLE_AGGR = np.zeros((64, 64), dtype=np.int64)
for _d in range(1, 64):
    for _m in range(1, 64):
        LMR_TABLE_AGGR[_d, _m] = int(0.5 + np.log(_d) * np.log(_m) / 1.8)
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


@njit(cache=False)
def simple_eval(bb: Any, meta: Any) -> Any:
    """Pure material, agent._MATERIAL values, side-to-move POV."""
    us = meta[fb.SIDE] * 6
    them = 6 - us
    total = 0
    for p in range(5):
        total += EG_VALUES[p] * (fb.popcount(bb[us + p]) - fb.popcount(bb[them + p]))
    return total


@njit(cache=False, fastmath=True)
def evaluate(
    bb: Any, meta: Any, white: Any, black: Any, w2t: Any, b2: Any, w3: Any, b3: Any,
    scratch: Any, ctrl: Any,
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
    score = int(float(out) * OUTPUT_SCALE)
    if (
        ctrl[C_EG_SHRINK] == 0
        or meta[fb.PIECES] >= EG_HI
        or score >= DISTANCE_THRESHOLD
        or score <= -DISTANCE_THRESHOLD
    ):
        return score
    wmin = ctrl[C_EG_WMIN]
    pieces = meta[fb.PIECES]
    w = wmin if pieces <= EG_LO else wmin + (256 - wmin) * (pieces - EG_LO) // (EG_HI - EG_LO)
    delta = (256 - w) * (simple_eval(bb, meta) - score) // 256
    cap = ctrl[C_EG_CAP]
    if cap > 0:
        delta = cap if delta > cap else (-cap if delta < -cap else delta)
    return score + delta


@njit(cache=False)
def sync_acc(
    undo: Any, w1: Any, white: Any, black: Any, astack: Any, zones: Any, ctrl: Any, cur_ply: Any
) -> Any:
    """Replay the deferred accumulator updates up to the board's current ply.

    Each pending ply first saves the accumulators to astack (unmake_move restores
    from there), then applies its move's deltas from the undo stack. Null moves
    (U_MOVE == 0) change nothing. The zone offsets are constant across a pending
    stretch: crossing king moves always go the eager road in make_move.
    """
    a = ctrl[C_ACC_PLY]
    off_w = zones[0] * fb.FEATURES
    off_b = zones[1] * fb.FEATURES
    width = white.shape[0]
    while a < cur_ply:
        for i in range(width):
            astack[a, 0, i] = white[i]
            astack[a, 1, i] = black[i]
        undo[a, fb.U_ZONE_W] = zones[0]
        undo[a, fb.U_ZONE_B] = zones[1]
        move = undo[a, fb.U_MOVE]
        if move != 0:
            frm = move & 63
            to = (move >> 6) & 63
            promo = (move >> 12) & 7
            code = undo[a, fb.U_MOVER]
            us = code // 6
            piece = code - us * 6
            captured = undo[a, fb.U_CAPTURED]
            fb._acc_row(w1, white, black, frm, code, off_w, off_b, -1)
            if captured >= 0:
                fb._acc_row(w1, white, black, to, captured, off_w, off_b, -1)
            elif captured == -2:  # en passant: the pawn sat behind the target
                behind = to - 8 if us == 0 else to + 8
                fb._acc_row(w1, white, black, behind, (1 - us) * 6, off_w, off_b, -1)
            landing = code if promo == 0 else us * 6 + promo
            fb._acc_row(w1, white, black, to, landing, off_w, off_b, 1)
            if piece == 5 and (to - frm == 2 or frm - to == 2):
                rook = us * 6 + 3
                if to > frm:
                    rfrom, rto = to + 1, to - 1
                else:
                    rfrom, rto = to - 2, to + 1
                fb._acc_row(w1, white, black, rfrom, rook, off_w, off_b, -1)
                fb._acc_row(w1, white, black, rto, rook, off_w, off_b, 1)
        a += 1
    ctrl[C_ACC_PLY] = cur_ply


@njit(cache=False)
def make_move(
    bb: Any, sq: Any, meta: Any, undo: Any, keys: Any, move: Any,
    w1: Any, b1: Any, white: Any, black: Any, astack: Any, zones: Any, king_zones: Any,
    ctrl: Any,
) -> Any:
    """make_full, or under C_LAZY_ACC make_light with the accumulator deferred."""
    if (_F_LAZY_ACC if _FOLD else ctrl[C_LAZY_ACC] != 0):
        frm = move & 63
        code = sq[frm]
        us = meta[fb.SIDE]
        crossing = False
        if code - us * 6 == 5 and king_zones > 1:
            to = (move >> 6) & 63
            if fb.zone_of(to if us == 0 else to ^ 56, king_zones) != zones[us]:
                crossing = True
        if not crossing:
            undo[meta[fb.PLY], fb.U_MOVER] = code
            fb.make_light(bb, sq, meta, undo, keys, move)
            return
        sync_acc(undo, w1, white, black, astack, zones, ctrl, meta[fb.PLY])
        fb.make_full(
            bb, sq, meta, undo, keys, move, w1, b1, white, black, astack, zones, king_zones
        )
        ctrl[C_ACC_PLY] = meta[fb.PLY]  # make_full advanced the ply; the acc is current
        return
    fb.make_full(bb, sq, meta, undo, keys, move, w1, b1, white, black, astack, zones, king_zones)


@njit(cache=False)
def unmake_move(
    bb: Any, sq: Any, meta: Any, undo: Any, keys: Any,
    white: Any, black: Any, astack: Any, zones: Any, ctrl: Any,
) -> Any:
    if (_F_LAZY_ACC if _FOLD else ctrl[C_LAZY_ACC] != 0):
        fb.unmake_light(bb, sq, meta, undo, keys)
        ply = meta[fb.PLY]
        if ctrl[C_ACC_PLY] > ply:
            # The accumulators were synced past this ply: restore the snapshot.
            width = white.shape[0]
            for i in range(width):
                white[i] = astack[ply, 0, i]
                black[i] = astack[ply, 1, i]
            zones[0] = undo[ply, fb.U_ZONE_W]
            zones[1] = undo[ply, fb.U_ZONE_B]
            ctrl[C_ACC_PLY] = ply
        return
    fb.unmake_full(bb, sq, meta, undo, keys, white, black, astack, zones)


@njit(cache=False, nogil=True)
def qs_tt_store(
    tt_key: Any, tt_data: Any, key: Any, score: Any, flag: Any, ply: Any, ctrl: Any
) -> None:
    age = ctrl[C_AGE]
    slot = np.int64(key & TT_MASK)
    if (_F_TT_BUCKETS if _FOLD else ctrl[C_TT_BUCKETS] != 0):
        dslot = slot & -2
        if tt_key[dslot] == key:
            slot = dslot
        elif tt_key[dslot + 1] == key:
            slot = dslot + 1
        elif unpack_age(tt_data[dslot]) != (age & 63):
            slot = dslot
        else:
            slot = dslot + 1
    old = tt_data[slot]
    # A same-key deeper entry keeps its hash move; a current-age deeper entry
    # of another position is worth more than a depth-0 bound.
    if unpack_depth(old) > 0 and (tt_key[slot] == key or unpack_age(old) == (age & 63)):
        return
    tt_key[slot] = key
    tt_data[slot] = pack(to_table(score, ply), 0, flag, 0, age, NO_EVAL)


@njit(cache=False, nogil=True)
def quiesce(
    bb: Any, sq: Any, meta: Any, undo: Any, keys: Any,
    w1: Any, b1: Any, white: Any, black: Any, astack: Any, zones: Any, king_zones: Any,
    w2t: Any, b2: Any, w3: Any, b3: Any,
    butterfly: Any, moves: Any, scores: Any, ctrl: Any, deadline: Any,
    alpha: Any, beta: Any, depth: Any, ply: Any, scratch: Any,
    ec_key: Any, ec_val: Any, exts: Any, tt_key: Any, tt_data: Any,
) -> Any:
    ctrl[C_NODES] += 1
    if (ctrl[C_NODES] & POLL_MASK) == 0 and (ctrl[C_STOP] != 0 or timed_out(deadline)):
        ctrl[C_ABORT] = 1
        return 0

    use_qtt = ctrl[C_QS_TT] != 0 and ctrl[C_TT_OFF] == 0
    original_alpha = alpha
    if use_qtt:
        tkey = keys[meta[fb.PLY]]
        tslot = np.int64(tkey & TT_MASK)
        if (_F_TT_BUCKETS if _FOLD else ctrl[C_TT_BUCKETS] != 0):
            tslot = tslot & -2
            if tt_key[tslot] != tkey and tt_key[tslot + 1] == tkey:
                tslot = tslot + 1
        if tt_key[tslot] == tkey:
            data = tt_data[tslot]
            tflag = unpack_flag(data)
            tscore = from_table(unpack_score(data), ply)
            if tflag == 0:
                return tscore
            if tflag == 1 and tscore >= beta:
                return tscore
            if tflag == 2 and tscore <= alpha:
                return tscore

    if ctrl[C_QS_CACHE] != 0:
        qkey = keys[meta[fb.PLY]]
        qslot = np.int64(qkey & EVAL_CACHE_MASK)
        if ec_key[qslot] == qkey:
            standing = np.int64(ec_val[qslot])
        else:
            if (_F_LAZY_ACC if _FOLD else ctrl[C_LAZY_ACC] != 0):
                sync_acc(undo, w1, white, black, astack, zones, ctrl, meta[fb.PLY])
            standing = evaluate(bb, meta, white, black, w2t, b2, w3, b3, scratch, ctrl)
            ec_key[qslot] = qkey
            ec_val[qslot] = standing
    else:
        if (_F_LAZY_ACC if _FOLD else ctrl[C_LAZY_ACC] != 0):
            sync_acc(undo, w1, white, black, astack, zones, ctrl, meta[fb.PLY])
        standing = evaluate(bb, meta, white, black, w2t, b2, w3, b3, scratch, ctrl)
    if standing >= beta:
        if use_qtt:
            qs_tt_store(tt_key, tt_data, keys[meta[fb.PLY]], standing, 1, ply, ctrl)
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
    use_see = _F_SEE if _FOLD else ctrl[C_SEE] != 0
    for i in range(n):
        move = fb.pick_move(captures, sc, i, n)
        victim = sq[(move >> 6) & 63]
        if victim >= 0 and (move >> 12) == 0 and standing + MVV[victim % 6] + DELTA_MARGIN < alpha:
            continue
        if use_see and (move >> 12) == 0 and fb.see(bb, sq, meta, move) < 0:
            # A capture that loses material on the exchange cannot raise alpha.
            continue
        make_move(
            bb, sq, meta, undo, keys, move, w1, b1, white, black, astack, zones, king_zones, ctrl
        )
        score = -quiesce(
            bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
            w2t, b2, w3, b3, butterfly, moves, scores, ctrl, deadline,
            -beta, -alpha, depth + 1, ply + 1, scratch, ec_key, ec_val, exts,
            tt_key, tt_data,
        )
        unmake_move(bb, sq, meta, undo, keys, white, black, astack, zones, ctrl)
        if ctrl[C_ABORT]:
            return 0
        if score >= beta:
            if use_qtt:
                qs_tt_store(tt_key, tt_data, keys[meta[fb.PLY]], score, 1, ply, ctrl)
            return score
        if score > alpha:
            alpha = score
    if use_qtt and ctrl[C_ABORT] == 0:
        qflag = 0 if alpha > original_alpha else 2
        qs_tt_store(tt_key, tt_data, keys[meta[fb.PLY]], alpha, qflag, ply, ctrl)
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
    ec_key: Any, ec_val: Any, exts: Any, conthist1: Any, cutnode: Any,
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
        hmc_draw = ctrl[C_HMC_DRAW]
        if hmc_draw <= 0:
            hmc_draw = 100
        if meta[fb.HALFMOVE] >= hmc_draw:
            n = fb.gen_legal(bb, sq, meta, moves[ply], False)
            if n == 0 and fb.in_check(bb, meta):
                return -MATE + ply
            return draw_score(meta, ctrl)
    if ply >= fb.MAX_PLY - 8:
        if (_F_LAZY_ACC if _FOLD else ctrl[C_LAZY_ACC] != 0):
            sync_acc(undo, w1, white, black, astack, zones, ctrl, meta[fb.PLY])
        return evaluate(bb, meta, white, black, w2t, b2, w3, b3, scratch, ctrl)
    if ctrl[C_KILLER_CLEAR] != 0:
        killers[ply + 2, 0] = 0
        killers[ply + 2, 1] = 0

    if (_F_SAFE if _FOLD else ctrl[C_SAFE] != 0) and ply > 0:
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
    excluded = 0
    if (_F_SINGULAR if _FOLD else ctrl[C_SINGULAR] != 0) and ctrl[C_EXCL_PLY] == ply:
        excluded = ctrl[C_EXCL_MOVE]
    tt_depth = -1
    tt_flag = 2
    tt_score = 0
    if ctrl[C_TT_OFF] == 0 and excluded == 0:
        slot = np.int64(key & TT_MASK)
        if (_F_TT_BUCKETS if _FOLD else ctrl[C_TT_BUCKETS] != 0):
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
            tt_depth = stored_depth
            tt_flag = flag
            tt_score = stored_score
            if stored_depth >= depth and ply > 0:
                if flag == 0:
                    return stored_score
                if flag == 1 and stored_score > alpha:
                    alpha = stored_score
                elif flag == 2 and stored_score < beta:
                    beta = stored_score
                if alpha >= beta:
                    return stored_score

    if (
        (_F_IIR if _FOLD else ctrl[C_IIR] != 0)
        and depth >= 4
        and hash_move == 0
        and ctrl[C_TT_OFF] == 0
        and excluded == 0
    ):
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
            scratch, ec_key, ec_val, exts, tt_key, tt_data,
        )

    standing = -INFINITY
    improving = 1  # ply < 2 and sentinel ancestors default to improving (never over-prune)
    if ctrl[C_IMPROVING] != 0:
        if in_check:
            if excluded == 0:
                exts[fb.MAX_PLY + ply] = -INFINITY  # sentinel: no usable eval at this ply
            improving = 0
        else:
            if cached_eval == NO_EVAL:
                if (_F_LAZY_ACC if _FOLD else ctrl[C_LAZY_ACC] != 0):
                    sync_acc(undo, w1, white, black, astack, zones, ctrl, meta[fb.PLY])
                cached_eval = evaluate(bb, meta, white, black, w2t, b2, w3, b3, scratch, ctrl)
            if excluded == 0:
                # The SINGULAR re-search re-enters this ply: writing there would
                # flip the grandchildren's improving flag mid-node.
                exts[fb.MAX_PLY + ply] = cached_eval
            if ply >= 2:
                prev2 = exts[fb.MAX_PLY + ply - 2]
                if prev2 != -INFINITY and cached_eval <= prev2:
                    improving = 0
    percent = 100
    if _F_RFP_PHASE if _FOLD else ctrl[C_RFP_PHASE] != 0:
        percent = phase_percent(ctrl, meta[fb.PIECES])
    if (
        percent != 0
        and depth <= RFP_MAX_DEPTH
        and not in_check
        and (((not _F_HYGIENE) if _FOLD else ctrl[C_HYGIENE] == 0)
             or abs(beta) < DISTANCE_THRESHOLD)
    ):
        if cached_eval != NO_EVAL:
            standing = cached_eval
        else:
            if (_F_LAZY_ACC if _FOLD else ctrl[C_LAZY_ACC] != 0):
                sync_acc(undo, w1, white, black, astack, zones, ctrl, meta[fb.PLY])
            standing = evaluate(bb, meta, white, black, w2t, b2, w3, b3, scratch, ctrl)
            cached_eval = standing
        rfp_depth = depth - improving if ctrl[C_IMPROVING] != 0 else depth
        if standing - RFP_MARGIN * rfp_depth * percent // 100 >= beta:
            return standing

    if (
        ctrl[C_RAZOR] != 0
        and percent != 0
        and depth <= RAZOR_MAX_DEPTH
        and not in_check
        and excluded == 0
        and beta - alpha <= 1
        and abs(alpha) < DISTANCE_THRESHOLD
    ):
        if standing == -INFINITY:
            if cached_eval != NO_EVAL:
                standing = cached_eval
            else:
                if (_F_LAZY_ACC if _FOLD else ctrl[C_LAZY_ACC] != 0):
                    sync_acc(undo, w1, white, black, astack, zones, ctrl, meta[fb.PLY])
                standing = evaluate(bb, meta, white, black, w2t, b2, w3, b3, scratch, ctrl)
                cached_eval = standing
        if standing + RAZOR_MARGIN[depth] * percent // 100 <= alpha:
            razored = quiesce(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, butterfly, moves, scores, ctrl, deadline, alpha, beta, 0, ply,
                scratch, ec_key, ec_val, exts, tt_key, tt_data,
            )
            if razored <= alpha:
                return razored

    futile = False
    if (
        (_F_FUTILITY if _FOLD else ctrl[C_FUTILITY] != 0)
        and percent != 0
        and depth <= 2
        and not in_check
        and abs(alpha) < DISTANCE_THRESHOLD
    ):
        if standing == -INFINITY:
            if cached_eval != NO_EVAL:
                standing = cached_eval
            else:
                if (_F_LAZY_ACC if _FOLD else ctrl[C_LAZY_ACC] != 0):
                    sync_acc(undo, w1, white, black, astack, zones, ctrl, meta[fb.PLY])
                standing = evaluate(bb, meta, white, black, w2t, b2, w3, b3, scratch, ctrl)
                cached_eval = standing
        futile = standing + FUTILITY_MARGIN[depth] * percent // 100 <= alpha

    if (
        depth >= NMP_MIN_DEPTH
        and not in_check
        and abs(beta) < DISTANCE_THRESHOLD
        and fb.non_pawn_material(bb, meta[fb.SIDE])
        and (((not _F_NMP_GUARD) if _FOLD else ctrl[C_NMP_GUARD] == 0)
             or ply == 0 or undo[meta[fb.PLY] - 1, fb.U_MOVE] != 0)
        and excluded == 0
        and (ctrl[C_NMP_MIN_PLY] == 0 or ply >= ctrl[C_NMP_MIN_PLY])
    ):
        nmp2 = ctrl[C_NMP_V2] != 0
        do_null = True
        if nmp2:
            if tt_depth >= 0 and tt_flag == 2 and tt_score < beta:
                do_null = False  # stored upper bound already puts this node below beta
            else:
                if standing == -INFINITY:
                    if cached_eval != NO_EVAL:
                        standing = cached_eval
                    else:
                        if (_F_LAZY_ACC if _FOLD else ctrl[C_LAZY_ACC] != 0):
                            sync_acc(undo, w1, white, black, astack, zones, ctrl, meta[fb.PLY])
                        standing = evaluate(bb, meta, white, black, w2t, b2, w3, b3, scratch, ctrl)
                        cached_eval = standing
                do_null = standing >= beta
        if do_null:
            if nmp2:
                bonus = (standing - beta) // 200
                if bonus > 3:
                    bonus = 3
                null_depth = depth - 3 - depth // 4 - bonus
            else:
                null_depth = depth - 1 - NMP_REDUCTION
                if (_F_SAFE if _FOLD else ctrl[C_SAFE] != 0):
                    null_depth -= depth // 6  # deeper nodes can afford a bigger reduction
            fb.make_null(bb, meta, undo, keys)
            score = -search(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, tt_key, tt_data,
                killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                null_depth, -beta, -beta + 1, ply + 1, scratch, counter, quiets,
                ec_key, ec_val, exts, conthist1, 1,
            )
            fb.unmake_null(meta, undo)
            if (_F_LAZY_ACC if _FOLD else ctrl[C_LAZY_ACC] != 0) and ctrl[C_ACC_PLY] > meta[fb.PLY]:
                # A sync inside the null search labelled the accumulators with the
                # null ply; the null move left the board unchanged, so they are
                # equally current here -- relabel, or the next lazy make at this
                # ply would sit behind C_ACC_PLY and never be replayed.
                ctrl[C_ACC_PLY] = meta[fb.PLY]
            if ctrl[C_ABORT]:
                return 0
            if score >= beta:
                if (
                    ctrl[C_NMP_V2B] != 0
                    and depth >= NMP_VERIFY_DEPTH
                    and ctrl[C_NMP_MIN_PLY] == 0
                ):
                    # Verify the null cutoff with a reduced-depth real search at
                    # this node; null stays off below min_ply so the verification
                    # cannot re-cut with another null near its root.
                    ctrl[C_NMP_MIN_PLY] = ply + 3 * null_depth // 4
                    score = search(
                        bb, sq, meta, undo, keys, w1, b1, white, black, astack,
                        zones, king_zones, w2t, b2, w3, b3, tt_key, tt_data,
                        killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                        null_depth, beta - 1, beta, ply, scratch, counter, quiets,
                        ec_key, ec_val, exts, conthist1, cutnode,
                    )
                    ctrl[C_NMP_MIN_PLY] = 0
                    if ctrl[C_ABORT]:
                        return 0
                    if score >= beta:
                        return beta
                else:
                    return beta

    extend_hash = 0
    if (
        (_F_SINGULAR if _FOLD else ctrl[C_SINGULAR] != 0)
        and excluded == 0
        and ply > 0
        and depth >= SINGULAR_MIN_DEPTH
        and hash_move != 0
        and tt_flag != 2
        and tt_depth >= depth - 3
        and abs(tt_score) < DISTANCE_THRESHOLD
        and exts[ply] < SINGULAR_EXT_CAP
    ):
        sbeta = tt_score - 2 * depth
        ctrl[C_EXCL_MOVE] = hash_move
        ctrl[C_EXCL_PLY] = ply
        value = search(
            bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
            w2t, b2, w3, b3, tt_key, tt_data,
            killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
            (depth - 1) // 2, sbeta - 1, sbeta, ply, scratch, counter, quiets,
            ec_key, ec_val, exts, conthist1, cutnode,
        )
        ctrl[C_EXCL_PLY] = -1
        if ctrl[C_ABORT]:
            return 0
        sing2 = ctrl[C_SING_EXT2] != 0
        if value < sbeta:
            extend_hash = 1
            if (
                sing2
                and beta - alpha <= 1
                and value < sbeta - SINGULAR_DOUBLE_MARGIN
                and exts[ply] + 2 <= SINGULAR_EXT_CAP
            ):
                extend_hash = 2
        elif sing2 and beta - alpha <= 1 and tt_score >= beta:
            extend_hash = -1

    mv = moves[ply]
    n = fb.gen_legal(bb, sq, meta, mv, False)
    if n == 0:
        return -MATE + ply if in_check else 0
    sc = scores[ply]
    history2 = _F_HISTORY2 if _FOLD else ctrl[C_HISTORY2] != 0
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
    conthist_on = ctrl[C_CONT_HIST] != 0
    ch_base = -1
    if conthist_on:
        prev = undo[meta[fb.PLY] - 1, fb.U_MOVE] if meta[fb.PLY] > 0 else 0
        if prev != 0:
            prev_to = (prev >> 6) & 63
            prev_piece = sq[prev_to]  # the mover is still on its target square
            if prev_piece >= 0:
                ch_base = (prev_piece * 64 + prev_to) * 768
        if ch_base >= 0:
            # Quiet ordering: butterfly + conthist1. Quiet scores stay within
            # +/-2*HISTORY_MAX, far below the killer/counter/capture bands.
            for j in range(n):
                m = mv[j]
                to2 = (m >> 6) & 63
                if (
                    (m >> 12) == 0
                    and sq[to2] < 0
                    and m != hash_move
                    and m != killers[ply, 0]
                    and m != killers[ply, 1]
                    and m != counter_move
                ):
                    sc[j] += conthist1[ch_base + sq[m & 63] * 64 + to2]
    capture_order = ctrl[C_CAPTURE_ORDER] != 0
    if capture_order:
        # Rescore non-promotion captures (EP stays a quiet: sq[to] < 0, same as
        # score_moves). SEE-losing captures fall below every quiet score
        # (quiets stay within +/-2*HISTORY_MAX); winning/equal captures keep
        # the MVV-LVA band with a capture-history tiebreak on top.
        for j in range(n):
            m = mv[j]
            if m == hash_move or (m >> 12) != 0:
                continue
            to2 = (m >> 6) & 63
            victim = sq[to2]
            if victim < 0:
                continue
            chv = conthist1[(sq[m & 63] * 64 + to2) * 6 + victim % 6]
            sv = fb.see(bb, sq, meta, m)
            if sv < 0:
                sc[j] = -(1 << 21) + sv * 16 + chv // 16
            else:
                sc[j] = fb.CAPTURE_BONUS + MVV[victim % 6] * 16 - MVV[sq[m & 63] % 6] + chv // 16

    best_score = -INFINITY
    best_move = 0
    searched = 0
    # Children (search.md 3.3): a null-window child is a cut node iff this node
    # was not; a full-window child of a PV node is a PV node. Read only when
    # C_CUTNODE is on, so the values cost nothing with the switch off.
    scout_cut = 1 - cutnode
    full_cut = 0 if beta - alpha > 1 else scout_cut
    pvs = _F_PVS if _FOLD else ctrl[C_PVS] != 0
    lmr = (_F_LMR if _FOLD else ctrl[C_LMR] != 0) and depth >= 3 and not in_check
    aggr = _F_LMR_AGGR if _FOLD else ctrl[C_LMR_AGGR] != 0
    lmp = (
        (_F_LMP if _FOLD else ctrl[C_LMP] != 0)
        and depth <= 3 and not in_check and abs(alpha) < DISTANCE_THRESHOLD
    )
    prune2 = (
        (_F_PRUNE2 if _FOLD else ctrl[C_PRUNE2] != 0)
        and depth <= 4
        and not in_check
        and abs(alpha) < DISTANCE_THRESHOLD
        and standing != -INFINITY
    )
    for i in range(n):
        move = fb.pick_move(mv, sc, i, n)
        if move == excluded:
            continue
        quiet = sq[(move >> 6) & 63] < 0
        plain = quiet and (move >> 12) == 0
        if futile and plain:
            continue
        if prune2 and plain and searched > 0:
            f2_depth = depth - improving if ctrl[C_IMPROVING] != 0 else depth
            if standing + FUTILITY_MARGIN2[f2_depth] <= alpha:
                continue
            hist = butterfly[base + (move & 63) * 64 + ((move >> 6) & 63)]
            if ch_base >= 0:
                hist += conthist1[ch_base + sq[move & 63] * 64 + ((move >> 6) & 63)]
            if hist < -HIST_PRUNE_SLOPE * depth:
                continue
        if (
            (_F_SEE_MAIN if _FOLD else ctrl[C_SEE_MAIN] != 0)
            and not quiet
            and (move >> 12) == 0
            and depth <= 5
            and searched > 0
            and abs(alpha) < DISTANCE_THRESHOLD
            and fb.see(bb, sq, meta, move) < -20 * depth * depth
        ):
            continue
        if (
            ctrl[C_SEE_QUIET] != 0
            and plain
            and depth <= 6
            and searched > 0
            and not in_check
            and abs(alpha) < DISTANCE_THRESHOLD
            and fb.see(bb, sq, meta, move) < -30 * depth * depth
        ):
            continue
        if lmp and plain and searched >= LMP_LIMIT[depth]:
            continue
        reduction = 0
        if (
            lmr
            and plain
            and searched >= (1 if aggr else 2)
            and move != hash_move
            and move != killers[ply, 0]
            and move != killers[ply, 1]
        ):
            if aggr:
                reduction = LMR_TABLE_AGGR[min(depth, 63), min(searched, 63)]
                hist = butterfly[base + (move & 63) * 64 + ((move >> 6) & 63)]
                if conthist_on:
                    if ch_base >= 0:
                        hist += conthist1[ch_base + sq[move & 63] * 64 + ((move >> 6) & 63)]
                    adj = hist // CONT_LMR_DIV
                    if adj > 2:
                        adj = 2
                    elif adj < -2:
                        adj = -2
                    reduction -= adj
                elif hist > 8000:
                    reduction -= 1
                elif hist < -8000:
                    reduction += 1
                if reduction < 0:
                    reduction = 0
            else:
                reduction = LMR_TABLE[min(depth, 63), min(searched, 63)]
            if ctrl[C_IMPROVING] != 0 and improving == 0:
                reduction += 1
            if ctrl[C_CUTNODE] != 0 and cutnode != 0:
                reduction += 1
        make_move(
            bb, sq, meta, undo, keys, move, w1, b1, white, black, astack, zones, king_zones, ctrl
        )
        if (history2 or conthist_on) and plain:
            quiets[ply, searched] = move  # every quiet tried at this node, for the malus
        elif (history2 or conthist_on) and ctrl[C_HIST2_FIX] != 0:
            quiets[ply, searched] = 0  # else the malus reads a stale move from a prior node
        if reduction > 0:
            reduced = depth - 1 - reduction
            if reduced < 1:
                reduced = 1  # never reduce straight into quiescence
            score = -search(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, tt_key, tt_data,
                killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                reduced, -alpha - 1, -alpha, ply + 1, scratch, counter, quiets,
                ec_key, ec_val, exts, conthist1, scout_cut,
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
                        ec_key, ec_val, exts, conthist1, scout_cut,
                    )
                else:
                    score = -search(
                        bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones,
                        king_zones, w2t, b2, w3, b3, tt_key, tt_data,
                        killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                        depth - 1, -beta, -alpha, ply + 1, scratch, counter, quiets,
                        ec_key, ec_val, exts, conthist1, full_cut,
                    )
        elif pvs and searched > 0:
            score = -search(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, tt_key, tt_data,
                killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                depth - 1, -alpha - 1, -alpha, ply + 1, scratch, counter, quiets,
                ec_key, ec_val, exts, conthist1, scout_cut,
            )
        else:
            ext = 0
            if extend_hash != 0 and move == hash_move:
                ext = extend_hash  # 1, or 2 / -1 under C_SING_EXT2
                if ext > 0:
                    exts[ply] += ext  # the child reads exts[ply] as its line's count
            score = -search(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, tt_key, tt_data,
                killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                depth - 1 + ext, -beta, -alpha, ply + 1, scratch, counter, quiets,
                ec_key, ec_val, exts, conthist1, full_cut,
            )
            if ext > 0:
                exts[ply] -= ext
        narrow = pvs and (reduction > 0 or searched > 0)
        if narrow and alpha < score < beta and ctrl[C_ABORT] == 0:
            # The null window said this move beats alpha: find out by how much.
            score = -search(
                bb, sq, meta, undo, keys, w1, b1, white, black, astack, zones, king_zones,
                w2t, b2, w3, b3, tt_key, tt_data,
                killers, butterfly, moves, scores, rep_keys, ctrl, deadline,
                depth - 1, -beta, -alpha, ply + 1, scratch, counter, quiets,
                ec_key, ec_val, exts, conthist1, 0,
            )
        unmake_move(bb, sq, meta, undo, keys, white, black, astack, zones, ctrl)
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
                        if conthist_on and ch_base >= 0 and excluded == 0 and plain:
                            # Same gravity as butterfly. The board is restored, so
                            # sq[from] is each quiet's mover again. Skipped in the
                            # excluded-move search: it re-enters this node's ply.
                            bonus2 = depth * depth
                            if bonus2 > 1200:
                                bonus2 = 1200
                            cdx = ch_base + sq[move & 63] * 64 + ((move >> 6) & 63)
                            conthist1[cdx] += bonus2 - conthist1[cdx] * bonus2 // HISTORY_MAX
                            for q2 in range(searched):
                                other = quiets[ply, q2]
                                if other != move and other != 0:
                                    op = sq[other & 63]
                                    if op >= 0:
                                        odx = ch_base + op * 64 + ((other >> 6) & 63)
                                        conthist1[odx] -= (
                                            bonus2 + conthist1[odx] * bonus2 // HISTORY_MAX
                                        )
                    elif capture_order and (move >> 12) == 0 and excluded == 0:
                        # Board is restored: sq[from] is the attacker again and
                        # sq[to] the victim. Gravity bonus for the cutting
                        # capture (no malus in v1).
                        victim2 = sq[(move >> 6) & 63]
                        if victim2 >= 0:
                            cbonus = depth * depth
                            if cbonus > 1200:
                                cbonus = 1200
                            kdx = (sq[move & 63] * 64 + ((move >> 6) & 63)) * 6 + victim2 % 6
                            conthist1[kdx] += cbonus - conthist1[kdx] * cbonus // HISTORY_MAX
                    break

    if searched == 0:
        if excluded != 0:
            # The excluded move was the only one: nothing else reaches the window.
            return alpha
        # Every move was futility-pruned: the position is at least as bad as the
        # static score says, which is below alpha.
        return standing

    if ctrl[C_TT_OFF] == 0 and excluded == 0:
        if best_score <= original_alpha:
            flag = 2
        elif best_score >= beta:
            flag = 1
        else:
            flag = 0
        age = ctrl[C_AGE]
        if (_F_TT_BUCKETS if _FOLD else ctrl[C_TT_BUCKETS] != 0):
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
        elif _F_TT_KEEP if _FOLD else ctrl[C_TT_KEEP] != 0:
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
    exts = np.zeros(4 * fb.MAX_PLY, dtype=np.int64)  # 4 lanes, see agent.FastEngine
    conthist1 = np.zeros(768 * 768, dtype=np.int32)
    ctrl = np.zeros(CTRL_SIZE, dtype=np.int64)
    ctrl[C_HYGIENE] = 1
    ctrl[C_FUTILITY] = 1
    ctrl[C_QS_CAP] = 8
    search(  # type: ignore[call-arg]
        pos.bb, pos.sq, pos.meta, pos.undo, pos.keys, w1, b1, white, black, astack, zones,
        king_zones, w2t, b2, w3, b3, *table, killers, butterfly, moves, scores, rep_keys,
        ctrl, time.monotonic() + 60.0, 2, -INFINITY, INFINITY, 0, scratch, counter, quiets,
        ec_key, ec_val, exts, conthist1, 0,
    )
