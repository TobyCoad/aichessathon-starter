"""The submission entrypoint. The platform imports this file and calls get_move.

An iterative-deepening alpha-beta searcher with a transposition table, MVV-LVA
move ordering, quiescence search and a learned evaluation: a (768 -> 512)x2 -> 32 -> 1
network whose first layer is maintained incrementally across make and unmake.

Where the time actually goes, measured per *node* rather than per call -- the
distinction matters, and getting it wrong sent this project after the wrong
bottleneck for a while:

    NNUE evaluate        29.4%    7.08 us x 0.617 calls/node
    accumulator push/pop 15.4%    3.62 us x 0.630
    board push/pop       14.4%    3.06 us x 0.630
    move generation      13.4%   23.30 us x 0.151
    everything else      27.4%

These proportions were measured on the 256-wide net and have not been re-measured
since the accumulator was widened to 512, which roughly doubles the evaluate and
push/pop rows in absolute terms. Treat the ordering as current and the percentages
as indicative. Node rate is about 98 knps single-process on one idle core; figures
near 29 knps that appear in older commit messages were measured on a contended
machine and understate it by roughly three times.

Move generation is the most expensive thing per call and only the fourth largest
per node, because most nodes fall straight through to quiescence or are cut by the
transposition table or reverse futility before any moves are generated. The
evaluation is four times cheaper per call and runs four times as often. So the
evaluation, not the move generator, is the thing worth making fast.

At the depths this reaches, a node doubling is worth roughly 120 Elo (range
80-190), given a measured effective branching factor near 3.

Three python-chess specifics that this file depends on, all measured:

  * `board._transposition_key()` costs 0.46 us. `chess.polyglot.zobrist_hash()`
    costs 12 us and `board.fen()` 23 us, so neither can be a transposition key.
  * `generate_legal_captures()` is ~3x cheaper than full generation, which is what
    makes quiescence affordable.
  * `can_claim_threefold_repetition()` costs ~150 us -- 5x a full move generation.
    It must never appear inside the search; repetition is tracked by hand below.

The rules require that a learned model materially drives move selection. It does:
every leaf score in the search, and so every move chosen, comes from the network in
`weights/net.npz`, which was trained from Lichess positions annotated by an existing
engine -- permitted explicitly, since the ban covers only what ships and runs inside
the submission. No engine, wrapper, or third-party weights are present.
"""

# ruff: noqa: E402
#   The thread-limit variables below are read by OpenMP/BLAS when their shared
#   libraries load, which happens on `import numpy`. Setting them afterwards is
#   silently ignored, so they have to precede the imports and the imports are
#   therefore not at the top of the file. This is the one place that ordering
#   matters more than the convention.
import os

# Pin the maths libraries to one thread each, before numpy is imported -- after
# import the setting is ignored. A referee that runs several games at once puts
# many agents on the same cores, and a BLAS that helpfully spawns a thread per core
# in each of them turns a fast engine into a flagging one. The search is
# single-threaded by design; nothing here wants a thread pool.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import random
import time
from collections.abc import Hashable, Iterator
from pathlib import Path
from typing import Any, Final

import chess
import chess.polyglot
import chess.syzygy
import numpy as np
import numpy.typing as npt

# --------------------------------------------------------------------------------
# The learned evaluation
# --------------------------------------------------------------------------------
# A (768 -> 256)x2 -> 32 -> 1 network, trained on Lichess positions annotated by an
# existing engine -- which the rules permit explicitly: "Training data: unrestricted,
# including positions annotated by an existing engine; the ban covers only what ships
# and runs inside the submission."
#
# Inference is hand-written numpy, not ONNX Runtime. At batch 1, which is all a
# depth-first search ever asks for, numpy measured ~4x faster: ORT carries a fixed
# ~12 us dispatch cost that dominates a network this small, and only wins when
# batching, which alpha-beta cannot do without giving up move ordering.
#
# Weights are float32. int16 measured *slower* in numpy because integer paths miss
# BLAS; quantisation is a C++/SIMD trick that inverts in Python.

_WEIGHTS = np.load(Path(__file__).with_name("weights") / "net.npz")
W1: Final = np.ascontiguousarray(_WEIGHTS["W1"], dtype=np.float32)   # (768, 256)
B1: Final = np.ascontiguousarray(_WEIGHTS["b1"], dtype=np.float32)   # (256,)
W2: Final = np.ascontiguousarray(_WEIGHTS["W2"], dtype=np.float32)   # (512, 32)
B2: Final = np.ascontiguousarray(_WEIGHTS["b2"], dtype=np.float32)   # (32,)
W3: Final = np.ascontiguousarray(_WEIGHTS["W3"], dtype=np.float32)   # (32, 1)
B3: Final = np.ascontiguousarray(_WEIGHTS["b3"], dtype=np.float32)   # (1,)
ACC_SIZE: Final = W1.shape[1]

# The network predicts a win-probability logit; centipawns are that times 400.
# Getting this constant wrong scales the whole evaluation silently.
OUTPUT_SCALE: Final = 400.0

# --------------------------------------------------------------------------------
# Endgame tablebase
# --------------------------------------------------------------------------------
# The complete 3- and 4-man Syzygy set, 70 files and 4.35 MB. Explicitly permitted:
# Books and tablebases are permitted as shipped data within the size cap, and
# chess.polyglot and chess.syzygy are in the base image. The repo's own docs give
# that cap as 200 MB; a review reading the live rules put it at 50 MB. The exact
# wording is not reproduced here because it could not be confirmed from a source in
# this repo -- and it does not matter, because the whole submission is 15.9 MB.
# Original note, retained for context:
# "Books and tablebases: permitted as shipped data within the cap;
# chess.polyglot and chess.syzygy are in the base image."
#
# This is here because the network cannot convert won endgames. It scores four very
# different KQvK positions at +1260, +1175, +1144 and +1241, so the search has no
# gradient to follow and shuffles until the referee claims a draw -- it drew KQ vs K
# in testing. No amount of further training fixes that; exact data does.
#
# Five men is deliberately not shipped: 378 MB of WDL alone, nearly twice the whole
# cap, for a published gain of roughly +2 Elo even to Stockfish.
_TABLEBASE: chess.syzygy.Tablebase | None = None
try:
    _syzygy_path = Path(__file__).with_name("weights") / "syzygy"
    if _syzygy_path.is_dir():
        _TABLEBASE = chess.syzygy.open_tablebase(str(_syzygy_path))
except Exception:
    _TABLEBASE = None

TB_MEN: Final = 4
# Above any evaluation the net can produce, below MATE_THRESHOLD so a tablebase win
# is never mistaken for a forced mate the search actually found.
TB_WIN: Final = 20_000

# --------------------------------------------------------------------------------
# Opening book
# --------------------------------------------------------------------------------
# Twenty plies of human opening moves, counted by frequency from Lichess games and
# stored as Polyglot. Permitted as shipped data alongside the tablebase, and
# `chess.polyglot` is in the base image.
#
# The clock is the main reason it is here. `_budget` allocates by expected moves
# remaining, which front-loads: roughly 40 seconds of a 120 second clock goes into
# the first ten moves, a phase where theory already has the answer and a depth-6
# search is guessing. The book answers instantly and banks that time for the
# middlegame, which is close to a node doubling where games are actually decided.
#
# Moves are chosen weighted-random rather than always-best. Two agents playing
# deterministically from the standard position replay one identical game, so a
# repeat pairing would repeat the result; sampling by popularity keeps the opening
# sound while making the games different.
_BOOK: chess.polyglot.MemoryMappedReader | None = None
try:
    _book_path = Path(__file__).with_name("weights") / "book.bin"
    if _book_path.is_file():
        _BOOK = chess.polyglot.open_reader(str(_book_path))
except Exception:  # a missing or broken book must never stop play
    _BOOK = None

# Ignore moves played far less often than the position's best: frequency data has a
# long tail, and the rare end of it is other people's mistakes.
BOOK_MIN_SHARE: Final = 0.08
# Anything at or above this is a distance-carrying score and must be rebased when
# it crosses the transposition table. That includes tablebase scores, not just
# mates -- they are ply-relative for exactly the same reason.
DISTANCE_THRESHOLD: Final = 19_000

_RANDOM: Final = random.Random()

MATE: Final = 30_000
MATE_THRESHOLD: Final = MATE - 1_000
INFINITY: Final = 1 << 20

# MVV-LVA: order captures by the value of the victim, tie-broken by the cheapness of
# the attacker. Measured at 8-29x fewer nodes depending on depth -- close to two free
# plies, and the single highest-value item in the whole engine per line of code.
_MVV: Final = (100, 320, 330, 500, 900, 20000)
CAPTURE_BONUS: Final = 1 << 20
# Below every capture, above every history-scored quiet move.
KILLER_FIRST: Final = (1 << 20) - 1
KILLER_SECOND: Final = (1 << 20) - 2
PROMOTION_BONUS: Final = 1 << 19

# Delta pruning margins. The per-capture margin is a minor piece: enough slack to
# cover a positional swing, not so much that the test never fires. It was a queen's
# worth (975) and was measured firing on 2 of 15,540 capture candidates -- inert.
# At 200 the same trace prunes 11.7%.
DELTA_MARGIN: Final = 200
# The node-level test: if even winning a queen outright cannot reach alpha, the
# whole capture search is hopeless and the standing evaluation stands.
BIG_DELTA: Final = 975


def _key(board: chess.Board) -> Hashable:
    """The position's transposition key.

    `_transposition_key` is private, but it costs 0.46 us where
    `chess.polyglot.zobrist_hash` costs 12 us and `board.fen()` 23 us. At tens of
    thousands of nodes per second nothing else is affordable.

    It returns a tuple of bitboards plus turn, castling rights and the en passant
    square, not an int -- a fine dict key, but never treat it as a number.
    """
    return board._transposition_key()



def _feature(square: int, piece_type: int, colour: chess.Color, white_pov: bool) -> int:
    """Feature index, matching training/features.py exactly.

    A disagreement here is silent: the net loads, the engine runs, and it plays
    badly. training/check_nnue.py asserts the two agree.
    """
    rel = square if white_pov else square ^ 56
    own = (colour == chess.WHITE) if white_pov else (colour == chess.BLACK)
    return (0 if own else 384) + (piece_type - 1) * 64 + rel



# --------------------------------------------------------------------------------
# Compiled evaluation kernels
# --------------------------------------------------------------------------------
# The evaluation is the hot path, not move generation: measured per node, the
# network forward pass is 29.4% of search time and the accumulator another 15.4%,
# against 13.4% for generating moves. numba is preinstalled on the platform and the
# organisers name it as the supported way to make Python fast here.
#
# Eager signatures, so compilation happens at import inside the 60 second budget
# rather than on the clock at move one. fastmath IS enabled, and this comment used
# to claim the opposite. The concern behind the original wording was real --
# fastmath lets the compiler reassociate floating-point arithmetic, so the
# evaluation is not bit-identical across hardware. It was enabled anyway because
# the alternative was measured and it cost too much: without it a float reduction
# cannot vectorise at all, and the engine ran at 75 knps against 103. The accuracy
# it buys back is one mismatch of 1cp in 7,808 positions, in either direction. That
# is a good trade for a 37% node rate, but it is a trade, not a free lunch.
#
# If numba is unavailable or fails to compile, the pure-numpy path below is used
# instead. An unguarded import here would raise at module load, which the platform
# records as an init failure -- and that loses every game, not one.
_COMPILED = False
try:
    from numba import float32, int32, int64, njit
    from numba import types as _nbt

    _W2T = np.ascontiguousarray(W2.T)

    @njit(
        float32(float32[:], float32[:], float32[:, ::1], float32[:], float32[:, ::1], float32[:]),
        cache=False,
        fastmath=True,
    )
    def _eval_kernel(
        own: npt.NDArray[np.float32],
        opponent: npt.NDArray[np.float32],
        w2t: npt.NDArray[np.float32],
        b2: npt.NDArray[np.float32],
        w3: npt.NDArray[np.float32],
        b3: npt.NDArray[np.float32],
    ) -> np.float32:
        hidden = np.empty(2 * ACC_SIZE, dtype=np.float32)
        for i in range(ACC_SIZE):
            x = own[i]
            x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
            hidden[i] = x * x
            y = opponent[i]
            y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)
            hidden[ACC_SIZE + i] = y * y
        out = b3[0]
        for j in range(32):
            total = b2[j]
            row = w2t[j]
            for i in range(2 * ACC_SIZE):
                total += hidden[i] * row[i]
            if total > 0.0:
                out += total * w3[j, 0]
        # numba infers this as float32; the annotation says so but mypy cannot see
        # through the decorator, so the accumulation reads as Any to it.
        result: np.float32 = out
        return result

    @njit(
        _nbt.void(
            float32[:], float32[:], float32[:, :, ::1], int64,
            float32[:, ::1], int32[:], int64, int32[:], int64,
        ),
        cache=False,
        fastmath=True,
    )
    def _push_kernel(
        white: npt.NDArray[np.float32],
        black: npt.NDArray[np.float32],
        stack: npt.NDArray[np.float32],
        depth: int,
        w1: npt.NDArray[np.float32],
        added: npt.NDArray[np.int32],
        n_added: int,
        removed: npt.NDArray[np.int32],
        n_removed: int,
    ) -> None:
        for i in range(ACC_SIZE):
            stack[depth, 0, i] = white[i]
            stack[depth, 1, i] = black[i]
        for k in range(n_added):
            rw = w1[added[2 * k]]
            rb = w1[added[2 * k + 1]]
            for i in range(ACC_SIZE):
                white[i] += rw[i]
                black[i] += rb[i]
        for k in range(n_removed):
            rw = w1[removed[2 * k]]
            rb = w1[removed[2 * k + 1]]
            for i in range(ACC_SIZE):
                white[i] -= rw[i]
                black[i] -= rb[i]

    @njit(_nbt.void(float32[:], float32[:], float32[:, :, ::1], int64), cache=False, fastmath=True)
    def _pop_kernel(
        white: npt.NDArray[np.float32],
        black: npt.NDArray[np.float32],
        stack: npt.NDArray[np.float32],
        depth: int,
    ) -> None:
        for i in range(ACC_SIZE):
            white[i] = stack[depth, 0, i]
            black[i] = stack[depth, 1, i]

    # Warm every kernel now, so no compilation lands on the game clock.
    _warm_a = B1.copy()
    _warm_b = B1.copy()
    _warm_stack = np.zeros((2, 2, ACC_SIZE), dtype=np.float32)
    _warm_idx = np.zeros(8, dtype=np.int32)
    _eval_kernel(_warm_a, _warm_b, _W2T, B2, W3, B3)
    _push_kernel(_warm_a, _warm_b, _warm_stack, 0, W1, _warm_idx, 1, _warm_idx, 1)
    _pop_kernel(_warm_a, _warm_b, _warm_stack, 0)
    _COMPILED = True
except Exception:
    _COMPILED = False


class Accumulator:
    """Both perspectives' first-layer sums, maintained incrementally.

    A full refresh costs ~5-10 us; an incremental update is ~0.6 us, and the search
    does one per node. `push` stores the previous vectors so `pop` is a restore
    rather than a recompute.
    """

    __slots__ = ("added", "black", "depth", "fast", "removed", "stack", "white")

    def __init__(self) -> None:
        self.white = B1.copy()
        self.black = B1.copy()
        self.fast = _COMPILED
        if self.fast:
            # One preallocated buffer instead of a fresh pair of arrays per node.
            # ndarray here, list below: fixed at construction, never mixed.
            self.stack: Any = np.zeros((MAX_PLY + 16, 2, ACC_SIZE), dtype=np.float32)
            self.depth = 0
            self.added = np.zeros(8, dtype=np.int32)
            self.removed = np.zeros(8, dtype=np.int32)
        else:
            self.stack = []
            self.depth = 0
            self.added = np.zeros(8, dtype=np.int32)
            self.removed = np.zeros(8, dtype=np.int32)

    def refresh(self, board: chess.Board) -> None:
        white = B1.copy()
        black = B1.copy()
        for piece_type in range(1, 7):
            for colour in (chess.WHITE, chess.BLACK):
                mask = board.pieces_mask(piece_type, colour)
                while mask:
                    square = (mask & -mask).bit_length() - 1
                    white += W1[_feature(square, piece_type, colour, True)]
                    black += W1[_feature(square, piece_type, colour, False)]
                    mask &= mask - 1
        self.white = white
        self.black = black
        if self.fast:
            self.depth = 0
        else:
            self.stack.clear()

    def _add(self, square: int, piece_type: int, colour: chess.Color) -> None:
        self.white += W1[_feature(square, piece_type, colour, True)]
        self.black += W1[_feature(square, piece_type, colour, False)]

    def _remove(self, square: int, piece_type: int, colour: chess.Color) -> None:
        self.white -= W1[_feature(square, piece_type, colour, True)]
        self.black -= W1[_feature(square, piece_type, colour, False)]

    def push(self, board: chess.Board, move: chess.Move) -> None:
        """Apply a move's feature deltas. Must be called *before* board.push.

        The deltas are collected into two small index arrays first, so the compiled
        kernel can apply them in one call. At most two features are added (the moved
        piece, and a rook when castling) and three removed (the from-square, a
        capture, and the castling rook), each occupying two slots -- one per
        perspective.
        """
        added = self.added
        removed = self.removed
        n_added = 0
        n_removed = 0

        mover = board.turn
        piece_type = board.piece_type_at(move.from_square)
        if piece_type is not None:
            removed[0] = _feature(move.from_square, piece_type, mover, True)
            removed[1] = _feature(move.from_square, piece_type, mover, False)
            n_removed = 1

            captured = board.piece_type_at(move.to_square)
            if captured is not None:
                removed[2] = _feature(move.to_square, captured, not mover, True)
                removed[3] = _feature(move.to_square, captured, not mover, False)
                n_removed = 2
            elif piece_type == chess.PAWN and move.to_square == board.ep_square:
                # En passant: the captured pawn is not on the destination square.
                behind = move.to_square + (-8 if mover == chess.WHITE else 8)
                removed[2] = _feature(behind, chess.PAWN, not mover, True)
                removed[3] = _feature(behind, chess.PAWN, not mover, False)
                n_removed = 2

            landing = move.promotion or piece_type
            added[0] = _feature(move.to_square, landing, mover, True)
            added[1] = _feature(move.to_square, landing, mover, False)
            n_added = 1

            if piece_type == chess.KING and abs(move.to_square - move.from_square) == 2:
                rank = 0 if mover == chess.WHITE else 56
                if move.to_square > move.from_square:
                    rook_from, rook_to = 7 + rank, 5 + rank
                else:
                    rook_from, rook_to = 0 + rank, 3 + rank
                removed[2 * n_removed] = _feature(rook_from, chess.ROOK, mover, True)
                removed[2 * n_removed + 1] = _feature(rook_from, chess.ROOK, mover, False)
                n_removed += 1
                added[2 * n_added] = _feature(rook_to, chess.ROOK, mover, True)
                added[2 * n_added + 1] = _feature(rook_to, chess.ROOK, mover, False)
                n_added += 1

        if self.fast:
            # The compiled kernel does no bounds checking -- numba's eager
            # signatures compile with boundscheck off, so overrunning the stack
            # corrupts memory and crashes rather than raising. Search depth is
            # bounded well below this, but a crash loses the game outright, so the
            # buffer grows instead of trusting that.
            if self.depth >= len(self.stack):
                grown = np.zeros((len(self.stack) * 2, 2, ACC_SIZE), dtype=np.float32)
                grown[: len(self.stack)] = self.stack
                self.stack = grown
            _push_kernel(
                self.white, self.black, self.stack, self.depth,
                W1, added, n_added, removed, n_removed,
            )
            self.depth += 1
            return

        self.stack.append((self.white.copy(), self.black.copy()))
        for k in range(n_added):
            self.white += W1[added[2 * k]]
            self.black += W1[added[2 * k + 1]]
        for k in range(n_removed):
            self.white -= W1[removed[2 * k]]
            self.black -= W1[removed[2 * k + 1]]

    def pop(self) -> None:
        if self.fast:
            self.depth -= 1
            _pop_kernel(self.white, self.black, self.stack, self.depth)
            return
        self.white, self.black = self.stack.pop()

    def evaluate(self, turn: chess.Color) -> int:
        """Centipawns from the side to move's point of view."""
        if turn == chess.WHITE:
            own, opponent = self.white, self.black
        else:
            own, opponent = self.black, self.white
        if self.fast:
            compiled: float = float(_eval_kernel(own, opponent, _W2T, B2, W3, B3))
            return int(compiled * OUTPUT_SCALE)
        hidden = np.concatenate((own, opponent))
        np.clip(hidden, 0.0, 1.0, out=hidden)
        hidden *= hidden  # SCReLU
        second = np.maximum(hidden @ W2 + B2, 0.0)
        return int(float((second @ W3 + B3)[0]) * OUTPUT_SCALE)


def _to_table(score: int, ply: int) -> int:
    """Make a mate score independent of where in the tree it was found.

    Search returns mate scores relative to the root: being mated at `ply` scores
    `-MATE + ply`, so a later mate is a better one. The transposition table
    outlives the search -- it persists across our moves within a game -- so a
    score stored at one ply is read back at another, and the root has shifted by
    two plies by the next move. Storing verbatim corrupts mate *distance*, which
    is exactly the signal needed to shorten a mate rather than shuffle.
    """
    if score > DISTANCE_THRESHOLD:
        return score + ply
    if score < -DISTANCE_THRESHOLD:
        return score - ply
    return score


def _from_table(score: int, ply: int) -> int:
    """Undo `_to_table`, putting a stored mate score back on this node's clock."""
    if score > DISTANCE_THRESHOLD:
        return score - ply
    if score < -DISTANCE_THRESHOLD:
        return score + ply
    return score


class Timeout(Exception):
    """Raised to unwind the search when the hard time limit passes."""


# The table persists for the whole game, and a full 120 s + 0.5 s game is about
# 160 s of thinking. Measured here: 3,919 new entries per second at 752 bytes each,
# so an unbounded table reaches ~627,000 entries and 0.47 GB -- a quarter of the
# 2 GB cap, before python-chess, numpy and the tablebase. Every SPRT so far ran at
# 8 s controls, where it only reaches ~39,000 entries, so this has never been
# exercised at the control the agent will actually play. Cheap insurance.
MAX_TABLE: Final = 400_000

# Reverse futility pruning. If the static evaluation is far enough above beta that
# even a sizeable positional swing could not bring it below, the node is assumed to
# fail high and is cut without searching. Measured at +145.83 +/- 24.41 in one
# engine's SPRT series and +57.1 +/- 16.9 in another's, and it fires at shallow
# depth -- which is all this engine has.
#
# It is also the first consumer of evaluation quality outside quiescence leaves.
# Until now a better network had almost nowhere to deposit its improvement, which
# is a candidate explanation for the 4x wider net measuring +13 +/- 21.
RFP_MAX_DEPTH: Final = 6
RFP_MARGIN: Final = 80

# Null-move pruning: give the opponent a free move; if the position still fails
# high, the real move would too. Measured +51.4 +/- 14.6 and +116.0 +/- 25.2 in two
# independent engines. Requires non-pawn material, because in a pawn endgame
# zugzwang makes the null-move assumption false.
NMP_MIN_DEPTH: Final = 3
NMP_REDUCTION: Final = 2
MAX_PLY: Final = 72

# --------------------------------------------------------------------------------
# Experiment switches
# --------------------------------------------------------------------------------
# One change each, off by default, so the champion plays exactly as the promoted
# build did. overnight/night.sh builds a challenger by turning a single switch on,
# runs the gauntlet, and on a PASS the switch stays on in the champion. Once every
# switch has a verdict the losing branches are deleted.
#
# TIME_V2        stop deepening when the next iteration is predicted to overrun the
#                soft budget, cap one move at 12% of the clock instead of 35%, and
#                hold back a reserve. The shipped budget measured 98% of a 120 s game
#                spent and single moves at 4x their soft budget.
# QS_EVASIONS    in quiescence, a side in check searches its evasions instead of
#                standing pat on an evaluation that was never trained on checks.
# STAGED_MOVEGEN try the hash move before generating any moves: legal generation is
#                28 us, about half of a node, and a hash move cuts most nodes.
# HYGIENE        halve the history table each move, record the position after our
#                move for repetition detection, and never let reverse futility
#                answer a mate-bound window.
TIME_V2: Final = False
QS_EVASIONS: Final = False
STAGED_MOVEGEN: Final = False
HYGIENE: Final = False

# TIME_V2: the clock is never allowed below this fraction of its starting value,
# which is inferred as the largest time_left_ms seen in the game. 12 s at 120 s.
RESERVE_FRACTION: Final = 0.10
_MAX_CLOCK_MS: float = 0.0
# How often the search looks at the clock. time.monotonic() costs well under a
# microsecond, so polling four times as often under TIME_V2 is free and quarters
# the worst-case overrun past a deadline.
_POLL_MASK: Final = 255 if TIME_V2 else 1023


class Engine:
    """Search state that persists for the lifetime of one game.

    The platform starts one process per game and keeps it alive between moves, so
    the transposition table and the repetition history survive from one of our moves
    to the next. They do not survive to the next game, which is why this is built
    per process rather than at module scope.
    """

    __slots__ = (
        "acc",
        "butterfly",
        "deadline",
        "history",
        "killers",
        "nodes",
        "root_key",
        "table",
    )

    def __init__(self) -> None:
        # key -> (depth, score, flag, best_move); flag 0 exact, 1 lower, 2 upper.
        self.table: dict[Hashable, tuple[int, int, int, chess.Move | None]] = {}
        # Transposition keys of positions we have actually been asked about, so the
        # search can recognise a repetition without paying 150 us to ask python-chess.
        self.history: dict[Hashable, int] = {}
        self.deadline = 0.0
        self.nodes = 0
        self.root_key: Hashable = None
        self.acc = Accumulator()
        # Two quiet moves per ply that last caused a beta cutoff there. They carry
        # information the position alone does not, and cost nothing to try first.
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        # from-square x to-square, credited by depth squared on a cutoff: a deeper
        # cutoff is stronger evidence that a move is generally good.
        self.butterfly: list[list[int]] = [[0] * 64 for _ in range(64)]

    # -- evaluation ---------------------------------------------------------------

    def evaluate(self, board: chess.Board) -> int:
        """The learned evaluation, from the incrementally maintained accumulator."""
        return self.acc.evaluate(board.turn)

    # -- move ordering ------------------------------------------------------------

    def _order(
        self, board: chess.Board, moves: list[chess.Move], best: chess.Move | None, ply: int
    ) -> None:
        """Sort moves in place: transposition move, then captures by MVV-LVA.

        A transposition table earns about +100 Elo used this way and only about +40
        used purely for cutoffs, so the previous iteration's best move going first
        matters more than the table's stored bounds.
        """
        piece_type_at = board.piece_type_at
        killers = self.killers[ply] if ply < MAX_PLY else [None, None]
        butterfly = self.butterfly

        def score(move: chess.Move) -> int:
            if best is not None and move == best:
                return 1 << 30
            value = 0
            victim = piece_type_at(move.to_square)
            if victim is not None:
                attacker = piece_type_at(move.from_square)
                value = (
                    CAPTURE_BONUS
                    + _MVV[victim - 1] * 16
                    - (_MVV[attacker - 1] if attacker is not None else 0)
                )
            elif move == killers[0]:
                value = KILLER_FIRST
            elif move == killers[1]:
                value = KILLER_SECOND
            else:
                value = butterfly[move.from_square][move.to_square]
            if move.promotion is not None:
                value += PROMOTION_BONUS + move.promotion * 100
            return value

        moves.sort(key=score, reverse=True)

    # -- quiescence ---------------------------------------------------------------

    def quiesce(
        self, board: chess.Board, alpha: int, beta: int, depth: int = 0, ply: int = 0
    ) -> int:
        """Search captures only, so evaluation never lands mid-exchange.

        This is the largest single measured feature in the engine literature -- an
        independent test put it at +347 Elo -- because without it every leaf score is
        taken halfway through a trade and is simply wrong.
        """
        self.nodes += 1
        if not self.nodes & _POLL_MASK and time.monotonic() > self.deadline:
            raise Timeout

        if QS_EVASIONS and depth < 8 and board.is_check():
            # In check there is no "do nothing", so standing pat is a fiction, and the
            # evaluation is untrained here besides: the packer drops every in-check
            # position. Search the evasions; none at all is mate, on the true ply so
            # it compares correctly with mates the main search found.
            evasions = list(board.legal_moves)
            if not evasions:
                return -MATE + ply
            self._order(board, evasions, None, 0)
            best = -INFINITY
            for move in evasions:
                self.acc.push(board, move)
                board.push(move)
                try:
                    score = -self.quiesce(board, -beta, -alpha, depth + 1, ply + 1)
                finally:
                    board.pop()
                    self.acc.pop()
                if score > best:
                    best = score
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            break
            return best

        standing = self.evaluate(board)
        if standing >= beta:
            return standing
        # If the best imaginable capture still falls short of alpha, nothing in this
        # subtree can matter. One test, before generating any moves at all.
        if standing + BIG_DELTA < alpha:
            return standing
        if standing > alpha:
            alpha = standing
        if depth >= 8:
            return standing

        captures = list(board.generate_legal_captures())
        self._order(board, captures, None, 0)
        for move in captures:
            # Delta pruning: skip a capture that cannot reach alpha even generously.
            victim = board.piece_type_at(move.to_square)
            if (
                victim is not None
                and move.promotion is None
                and standing + _MVV[victim - 1] + DELTA_MARGIN < alpha
            ):
                continue
            self.acc.push(board, move)
            board.push(move)
            try:
                score = -self.quiesce(board, -beta, -alpha, depth + 1, ply + 1)
            finally:
                board.pop()
                self.acc.pop()
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return alpha

    def _staged(
        self, board: chess.Board, hash_move: chess.Move | None, ply: int
    ) -> Iterator[chess.Move]:
        """Yield the hash move first, and only then generate the rest.

        Legal move generation costs ~28 us, about half of a node's time, and at a
        node with a hash move that move produces the cutoff most of the time -- so
        the list is only built if the search comes back for a second move. The key
        identifies the position exactly, so a stored move is legal here; `is_legal`
        is the guard against ever handing python-chess a move it would reject.
        """
        if hash_move is not None and board.is_legal(hash_move):
            yield hash_move
            moves = [move for move in board.legal_moves if move != hash_move]
        else:
            moves = list(board.legal_moves)
        self._order(board, moves, None, ply)
        yield from moves

    # -- main search --------------------------------------------------------------

    def search(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        """Fail-soft negamax with alpha-beta and a transposition table."""
        self.nodes += 1
        if not self.nodes & _POLL_MASK and time.monotonic() > self.deadline:
            raise Timeout

        # A position repeated inside the search, or one already seen in the game, is
        # a draw we can claim -- the referee claims threefold automatically, so a
        # winning side that shuffles will have the win taken away from it.
        key = _key(board)
        # A count of 1 means the position occurred once, which is not a draw. In-tree
        # repetitions are caught by is_repetition(2); a pre-root position needs two
        # prior sightings before a third occurrence here would let the referee claim.
        if ply and (self.history.get(key, 0) >= 2 or board.is_repetition(2)):
            return 0

        # Exact result for small material. WDL is 26-75 us warm, roughly two move
        # generations, so it is affordable at every node once the board is small
        # enough. `get_wdl` returns None rather than raising for a table we did not
        # ship, and a crash is a lost game.
        #
        # Two subtleties. The fifty-move counter is checked first, because a
        # theoretically won position whose clock has already expired is a draw and
        # the referee will claim it. And Syzygy reports +/-1 for a *cursed* win --
        # one that exists on the board but cannot be converted within fifty moves --
        # which is likewise a draw in play, so only +/-2 counts.
        if ply and board.halfmove_clock >= 100:
            # Checkmate outranks the clock: a mate delivered on the hundredth
            # halfmove is a win. `is_checkmate` is expensive, so it is only asked
            # in this rare branch rather than at every node.
            return -MATE + ply if board.is_checkmate() else 0
        if _TABLEBASE is not None and ply and chess.popcount(board.occupied) <= TB_MEN:
            wdl = _TABLEBASE.get_wdl(board)
            if wdl is not None:
                if wdl > 1:
                    return TB_WIN - ply
                if wdl < -1:
                    return -TB_WIN + ply
                return 0

        original_alpha = alpha
        stored = self.table.get(key)
        best_move = None
        if stored is not None:
            stored_depth, raw_score, flag, best_move = stored
            stored_score = _from_table(raw_score, ply)
            if stored_depth >= depth and ply:
                if flag == 0:
                    return stored_score
                if flag == 1 and stored_score > alpha:
                    alpha = stored_score
                elif flag == 2 and stored_score < beta:
                    beta = stored_score
                if alpha >= beta:
                    return stored_score

        in_check = board.is_check()
        # Check extension. A position in check has a tiny, forcing move list, so the
        # extra ply is cheap, and resolving the check is exactly where tactics live.
        # Measured +55.7 +/- 14.9 -- and about +1 Elo in Stockfish, because it is a
        # shallow-depth feature, which is all this engine has.
        if in_check and ply < MAX_PLY - 8:
            depth += 1

        if depth <= 0:
            return self.quiesce(board, alpha, beta, 0, ply)

        # Reverse futility pruning. Not in check, because the evaluation of a
        # position in check is unreliable -- the training data drops those
        # positions entirely, so the network has never seen one. Under HYGIENE,
        # not against a mate-bound window either: a static score can never answer
        # "is this better than being mated", and returning one there hides mates.
        if (
            depth <= RFP_MAX_DEPTH
            and not in_check
            and (not HYGIENE or abs(beta) < DISTANCE_THRESHOLD)
        ):
            standing = self.evaluate(board)
            if standing - RFP_MARGIN * depth >= beta:
                return standing

        # Null-move pruning. A null move leaves every piece where it is, so the
        # accumulator does not change -- only whose perspective is "own", which
        # evaluate() already takes from board.turn. Nothing to push or pop.
        if (
            depth >= NMP_MIN_DEPTH
            and not in_check
            and abs(beta) < DISTANCE_THRESHOLD
            and _has_non_pawn_material(board, board.turn)
        ):
            board.push(chess.Move.null())
            try:
                score = -self.search(board, depth - 1 - NMP_REDUCTION, -beta, -beta + 1, ply + 1)
            finally:
                board.pop()
            if score >= beta:
                return beta

        hash_move = best_move
        candidates: Iterator[chess.Move]
        if STAGED_MOVEGEN:
            candidates = self._staged(board, hash_move, ply)
        else:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply if board.is_check() else 0
            self._order(board, moves, hash_move, ply)
            candidates = iter(moves)

        best_score = -INFINITY
        best_move = None
        searched = 0
        for move in candidates:
            searched += 1
            self.acc.push(board, move)
            board.push(move)
            try:
                score = -self.search(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
                self.acc.pop()
            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        # A quiet move that causes a cutoff is worth remembering:
                        # here at this ply, and generally by from/to square.
                        if board.piece_type_at(move.to_square) is None:
                            slot = self.killers[ply] if ply < MAX_PLY else None
                            if slot is not None and slot[0] != move:
                                slot[1] = slot[0]
                                slot[0] = move
                            self.butterfly[move.from_square][move.to_square] += depth * depth
                        break
        if not searched:
            # Only reachable on the staged path, which generates lazily.
            return -MATE + ply if board.is_check() else 0

        if len(self.table) >= MAX_TABLE:
            # Always-replace with a hard ceiling. Dropping the table costs a little
            # re-search; running out of memory costs the game.
            self.table.clear()
        if best_score <= original_alpha:
            flag = 2
        elif best_score >= beta:
            flag = 1
        else:
            flag = 0
        self.table[key] = (depth, _to_table(best_score, ply), flag, best_move)
        return best_score

    # -- driver -------------------------------------------------------------------

    def choose(self, board: chess.Board, soft_limit: float, hard_limit: float) -> chess.Move:
        """Iteratively deepen, keeping the best move from the last completed depth.

        Iterative deepening is what makes the clock safe: there is always a legal
        move to return the moment the budget runs out, and each pass orders the next.
        """
        self.deadline = hard_limit
        moves = list(board.legal_moves)
        best = moves[0]

        if HYGIENE:
            # Halve the history each move. It is never otherwise reset during a game,
            # so without decay early preferences keep outranking what works now.
            for row in self.butterfly:
                for index in range(64):
                    row[index] >>= 1

        started = time.monotonic()
        for depth in range(1, 64):
            iteration_started = time.monotonic()
            try:
                score = -INFINITY
                alpha = -INFINITY
                self._order(board, moves, best, 0)
                iteration_best = moves[0]
                for move in moves:
                    self.acc.push(board, move)
                    board.push(move)
                    try:
                        value = -self.search(board, depth - 1, -INFINITY, -alpha, 1)
                    finally:
                        board.pop()
                        self.acc.pop()
                    if value > score:
                        score = value
                        iteration_best = move
                        if value > alpha:
                            alpha = value
                best = iteration_best
            except Timeout:
                break

            # A mate is found; deeper search cannot improve on it.
            if score > MATE_THRESHOLD or score < -MATE_THRESHOLD:
                break
            now = time.monotonic()
            if TIME_V2:
                # The next iteration costs about one effective branching factor more
                # than this one, so starting it merely because the soft limit has not
                # passed yet means finishing it at the hard limit almost every move:
                # measured, a mean spend of 3x the soft budget. Start it only if it is
                # predicted to end within one and a half soft budgets.
                elapsed = now - started
                predicted = (now - iteration_started) * 2.5
                if elapsed + predicted > 1.5 * (soft_limit - started):
                    break
            # Starting a further iteration that cannot finish only wastes clock.
            elif now > soft_limit:
                break

        return best


_ENGINE = Engine()


def _book_move(board: chess.Board) -> chess.Move | None:
    """A book move for this position, sampled by how often humans played it."""
    if _BOOK is None:
        return None
    try:
        entries = [entry for entry in _BOOK.find_all(board) if entry.weight > 0]
    except Exception:  # a corrupt book must not cost the game
        return None
    if not entries:
        return None
    best = max(entry.weight for entry in entries)
    viable = [entry for entry in entries if entry.weight >= best * BOOK_MIN_SHARE]
    total = sum(entry.weight for entry in viable)
    pick = _RANDOM.randrange(total)
    for entry in viable:
        pick -= entry.weight
        if pick < 0:
            return entry.move
    return viable[0].move


def _tablebase_move(board: chess.Board) -> chess.Move | None:
    """The best move when the whole position is tabulated.

    WDL says who wins. DTZ says how many plies until the next pawn move or capture,
    which keeps play safe against the fifty-move rule -- but it is *not* a distance
    to mate, and that distinction is the whole difficulty. In KPvK every winning
    move reports the same DTZ, so DTZ alone leaves the choice arbitrary and the
    engine shuffles: measured, it drew a won KPvK while the halfmove clock climbed
    to 18 without progress.

    Worse, minimising DTZ actively blocks the winning plan. Promoting a pawn raises
    DTZ, because after a queen appears the next capture is far away -- so a
    DTZ-first ranking marched the a-pawn to a7 and then refused to queen it,
    shuffling the king instead until the game was drawn.

    A zeroing move resets the fifty-move clock, which makes the DTZ of the position
    it leads to irrelevant. So zeroing ranks *above* DTZ: among moves that keep the
    win, prefer to reset the clock, then keep DTZ small, then drive the defending
    king toward a corner and bring our own king closer. The last two are the classic
    mate-driver, and this order converges for every ending in a 4-man set.
    """
    if _TABLEBASE is None or chess.popcount(board.occupied) > TB_MEN:
        return None

    best: chess.Move | None = None
    best_key: tuple[int, ...] | None = None
    for move in board.legal_moves:
        zeroing = board.is_zeroing(move)
        board.push(move)
        try:
            if board.is_checkmate():
                key: tuple[int, ...] = (-3, 0, 0, 0, 0)
            else:
                # After our move the opponent is to move, so a negative wdl here
                # means they are lost and we are winning.
                wdl = _TABLEBASE.get_wdl(board)
                dtz = _TABLEBASE.get_dtz(board)
                if wdl is None or dtz is None:
                    return None
                defender = board.king(board.turn)
                attacker = board.king(not board.turn)
                if defender is None or attacker is None:
                    return None
                corner = min(
                    chess.square_distance(defender, c)
                    for c in (chess.A1, chess.A8, chess.H1, chess.H8)
                )
                key = (
                    wdl,
                    0 if zeroing else 1,
                    abs(dtz) if wdl < 0 else -abs(dtz),
                    corner,
                    chess.square_distance(defender, attacker),
                )
        finally:
            board.pop()
        if best_key is None or key < best_key:
            best_key, best = key, move
    return best


def _has_non_pawn_material(board: chess.Board, colour: chess.Color) -> bool:
    """Whether `colour` has a piece other than king and pawns.

    Null-move pruning assumes that having the move is worth something. In a king
    and pawn endgame that is false -- zugzwang means the obligation to move can
    itself be losing -- so the pruning is disabled there.
    """
    return bool(
        board.pieces_mask(chess.KNIGHT, colour)
        | board.pieces_mask(chess.BISHOP, colour)
        | board.pieces_mask(chess.ROOK, colour)
        | board.pieces_mask(chess.QUEEN, colour)
    )


def _note_clock(time_left_ms: int) -> None:
    """Track the largest clock seen: the starting clock, which is never passed in."""
    global _MAX_CLOCK_MS
    if time_left_ms > _MAX_CLOCK_MS:
        _MAX_CLOCK_MS = float(time_left_ms)


def _budget_v2(board: chess.Board, time_left_ms: int) -> tuple[float, float]:
    """TIME_V2 deadlines. The soft budget is what a move should cost on average;
    `choose` now stops deepening when the next iteration would overrun it, so the
    hard limit is a genuine emergency stop rather than the usual spend.

    Three bounds on one move: 12% of the clock, three soft budgets, and whatever
    keeps the clock above the reserve. The reserve floor is `soft`, not zero --
    once the clock is inside the reserve the right move is a normal one, not a
    panicked 20 ms one that loses the game a different way.
    """
    now = time.monotonic()
    remaining = max(time_left_ms - 400.0, 50.0) / 1000.0  # 400 ms for the watchdog

    expected = max(20.0, 40.0 - board.fullmove_number * 0.5)
    increment = 0.5 if remaining > 5.0 else 0.0
    soft = remaining / expected + 0.5 * increment
    hard = min(remaining * 0.12, soft * 3.0)
    reserve = _MAX_CLOCK_MS * RESERVE_FRACTION / 1000.0
    if reserve > 0.0:
        hard = min(hard, max(soft, remaining - reserve))
    hard = max(hard, 0.02)
    soft = min(soft, hard)
    return now + soft, now + hard


def _budget(board: chess.Board, time_left_ms: int) -> tuple[float, float]:
    """Return (soft, hard) monotonic deadlines for this move.

    A flag is a full point and it is the most common self-inflicted loss in this
    format, so the hard limit is deliberately conservative. The referee measures
    wall time and applies the increment only *after* the move, so the increment
    cannot be spent in advance; it is counted at a discount.
    """
    if TIME_V2:
        return _budget_v2(board, time_left_ms)
    now = time.monotonic()
    remaining = max(time_left_ms - 300.0, 50.0) / 1000.0  # 300 ms for the watchdog

    # Expect fewer moves left as the game goes on, but never fewer than a floor:
    # running out of estimated moves is how engines talk themselves into flagging.
    expected = max(18.0, 42.0 - board.fullmove_number * 0.5)
    soft = remaining / expected + 0.35 * (0.5 if remaining > 5.0 else 0.0)
    hard = min(remaining * 0.35, soft * 4.0)
    soft = min(soft, hard)
    return now + soft, now + hard


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion
    """
    board = chess.Board(fen)

    # Remember every position we have been asked about. The referee claims threefold
    # repetition automatically, so an engine that is winning and shuffling can have a
    # won game turned into a draw without ever being told.
    # A position with no legal moves is checkmate or stalemate, and the referee is
    # not supposed to ask about one. If it ever does, every path below raises --
    # `moves[0]` in the search, `next(iter(...))` in the fallback -- and an exception
    # here forfeits the game. UCI's null move is the honest answer.
    if not board.legal_moves:
        return "0000"

    if TIME_V2:
        _note_clock(time_left_ms)

    key = _key(board)
    _ENGINE.history[key] = _ENGINE.history.get(key, 0) + 1

    # Book first: it is instant, and the clock it saves is worth more in the
    # middlegame than the search would be worth here.
    try:
        opening = _book_move(board)
    except Exception:  # never let the book cost a game
        opening = None
    if opening is not None:
        return opening.uci()

    # Exact play once the position is small enough. This is what converts a won
    # endgame; the search alone shuffles because the evaluation is flat there.
    try:
        exact = _tablebase_move(board)
    except Exception:
        exact = None
    if exact is not None:
        return exact.uci()

    # refresh() and _budget() were outside this guard, so an exception in either was
    # a crash rather than a fallback -- and a crash is a lost game where a legal move
    # would only have been a bad one. There is no failure in here worth a full point.
    try:
        _ENGINE.acc.refresh(board)
        soft, hard = _budget(board, time_left_ms)
        move = _ENGINE.choose(board, soft, hard)
    except Exception:
        return next(iter(board.legal_moves)).uci()

    if HYGIENE:
        # The position after our move counts toward a threefold claim just as much
        # as the ones we are asked about, and nothing else ever records it.
        try:
            board.push(move)
            after = _key(board)
            _ENGINE.history[after] = _ENGINE.history.get(after, 0) + 1
            board.pop()
        except Exception:
            pass
    return move.uci()
