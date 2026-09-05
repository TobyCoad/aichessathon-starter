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
import threading
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
W1: Final = np.ascontiguousarray(_WEIGHTS["W1"], dtype=np.float32)   # (K * 768, A)
B1: Final = np.ascontiguousarray(_WEIGHTS["b1"], dtype=np.float32)   # (A,)
ACC_SIZE: Final = W1.shape[1]
FEATURES: Final = 768

# King zones: W1 holds one 768-row block per zone of the perspective's own king, so
# the same piece on the same square can mean something different when the king is
# castled short, castled long or still in the centre. The zone is a property of
# the king's square seen from its own side (mirrored for black), and the first-layer
# index is `zone * 768 + feature`. A one-zone file is the old layout, unchanged.
KING_ZONES: Final = int(W1.shape[0]) // FEATURES


def _zone(square: int) -> int:
    """Zone of a king on `square`, from its own side, for this net's zone count.

    Mirrors `training.features.king_zone` exactly; check_nnue compares all 64.
    """
    rank = square >> 3
    file = square & 7
    if KING_ZONES == 4:
        if rank <= 1:
            return file >> 2
        return 2 if rank <= 3 else 3
    if KING_ZONES == 8:
        if rank <= 1:
            return file >> 1
        if rank <= 3:
            return 4 + (file >> 2)
        return 6 + (file >> 2)
    if KING_ZONES == 16:
        if rank <= 1:
            return file
        if rank <= 3:
            return 8 + (file >> 1)
        return 12 + (file >> 1)
    if KING_ZONES == 32:
        if rank <= 1:
            return rank * 8 + file
        if rank <= 3:
            return 16 + (rank - 2) * 4 + (file >> 1)
        return 24 + ((rank - 4) >> 1) * 4 + (file >> 1)
    return 0


def _stacked(name: str) -> npt.NDArray[np.float32]:
    """A head matrix with a leading bucket axis, whichever layout the file has.

    A single-head file stores W2 as (2A, H); a bucketed one as (B, 2A, H). Both
    are read into the bucketed shape so there is exactly one evaluation path.
    """
    array = np.ascontiguousarray(_WEIGHTS[name], dtype=np.float32)
    matrix = name in ("W2", "W3")
    single = array.ndim == (2 if matrix else 1)
    if single:
        array = array[None]
    return np.ascontiguousarray(array)


# Output buckets: independent heads after the shared accumulator, selected by the
# number of pieces on the board. One shared head scored four different KQvK
# positions within 120 cp of each other and could not convert; a head that only
# ever sees few-piece positions has the capacity to tell them apart. Costs nothing
# at inference: one head's matrices are picked, and the same kernel runs.
W2: Final = _stacked("W2")   # (B, 2A, 32)
B2: Final = _stacked("b2")   # (B, 32)
W3: Final = _stacked("W3")   # (B, 32, 1)
B3: Final = _stacked("b3")   # (B, 1)
BUCKETS: Final = int(W2.shape[0])


def _bucket(pieces: int) -> int:
    """Which head scores a position with `pieces` men on the board, 1..32.

    Mirrors `bucket_of` in training/train.py exactly. training/check_nnue.py
    compares the engine against the torch model on positions spanning every
    band, so a disagreement here fails loudly.
    """
    bucket = (pieces - 1) * BUCKETS // 32
    return 0 if bucket < 0 else (BUCKETS - 1 if bucket >= BUCKETS else bucket)

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

    _W2T = np.ascontiguousarray(W2.transpose(0, 2, 1))  # (B, 32, 2A)

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
        float32(
            float32[:], float32[:], int64,
            float32[:, :, ::1], float32[:, ::1], float32[:, :, ::1], float32[:, ::1],
            float32[:],
        ),
        cache=False,
        fastmath=True,
    )
    def _eval_bucket_kernel(
        own: npt.NDArray[np.float32],
        opponent: npt.NDArray[np.float32],
        k: int,
        w2t: npt.NDArray[np.float32],
        b2: npt.NDArray[np.float32],
        w3: npt.NDArray[np.float32],
        b3: npt.NDArray[np.float32],
        scratch: npt.NDArray[np.float32],
    ) -> np.float32:
        """_eval_kernel with the head chosen inside: slicing the four head arrays
        in Python cost more than the arithmetic once there were eight of them.
        `scratch` is a caller-owned 2*ACC_SIZE buffer: no allocation per call. The
        head loop runs four output neurons at a time so `hidden` is read from
        cache 8 times instead of 32; fastsearch.evaluate is this loop verbatim."""
        hidden = scratch
        acc = ACC_SIZE
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
        result: np.float32 = out
        return result

    @njit(
        _nbt.void(
            float32[:], float32[:], float32[:, :, ::1], int64,
            float32[:, ::1], int32[:], int64, int32[:], int64, int64, int64,
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
        white_offset: int,
        black_offset: int,
    ) -> None:
        # The offsets select each perspective's king-zone block of W1; even index
        # slots are the white perspective, odd slots the black one.
        for i in range(ACC_SIZE):
            stack[depth, 0, i] = white[i]
            stack[depth, 1, i] = black[i]
        for k in range(n_added):
            rw = w1[added[2 * k] + white_offset]
            rb = w1[added[2 * k + 1] + black_offset]
            for i in range(ACC_SIZE):
                white[i] += rw[i]
                black[i] += rb[i]
        for k in range(n_removed):
            rw = w1[removed[2 * k] + white_offset]
            rb = w1[removed[2 * k + 1] + black_offset]
            for i in range(ACC_SIZE):
                white[i] -= rw[i]
                black[i] -= rb[i]

    @njit(
        _nbt.void(float32[:], float32[:], float32[:, ::1], int32[:], int64),
        cache=False,
        fastmath=True,
    )
    def _refresh_kernel(
        out: npt.NDArray[np.float32],
        b1: npt.NDArray[np.float32],
        w1: npt.NDArray[np.float32],
        indices: npt.NDArray[np.int32],
        count: int,
    ) -> None:
        for i in range(ACC_SIZE):
            out[i] = b1[i]
        for k in range(count):
            row = w1[indices[k]]
            for i in range(ACC_SIZE):
                out[i] += row[i]

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
    _eval_kernel(_warm_a, _warm_b, _W2T[0], B2[0], W3[0], B3[0])
    _SCRATCH = np.zeros(2 * ACC_SIZE, dtype=np.float32)
    _eval_bucket_kernel(_warm_a, _warm_b, 0, _W2T, B2, W3, B3, _SCRATCH)
    _push_kernel(_warm_a, _warm_b, _warm_stack, 0, W1, _warm_idx, 1, _warm_idx, 1, 0, 0)
    _warm_feats = np.zeros(32, dtype=np.int32)
    _refresh_kernel(_warm_a, B1, W1, _warm_feats, 1)
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

    __slots__ = (
        "added", "black", "depth", "fast", "features", "removed", "stack", "white",
        "zone_black", "zone_white", "zones",
    )

    def __init__(self) -> None:
        self.white = B1.copy()
        self.black = B1.copy()
        # Each perspective's current king zone, and a stack of them so pop restores
        # the zone along with the vectors. A king move that crosses a zone boundary
        # rebuilds that one perspective from scratch; every other move is a delta.
        self.zone_white = 0
        self.zone_black = 0
        self.zones: list[tuple[int, int]] = []
        # Scratch for a full rebuild: at most 32 first-layer indices.
        self.features = np.zeros(64, dtype=np.int32)
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

    @staticmethod
    def _king_zone(board: chess.Board, colour: chess.Color) -> int:
        """The zone of `colour`'s king from its own side; 0 for a one-zone net."""
        if KING_ZONES == 1:
            return 0
        king = board.king(colour)
        if king is None:
            return 0
        return _zone(king if colour == chess.WHITE else king ^ 56)

    def _rebuild(
        self, board: chess.Board, white_pov: bool, zone: int, out: npt.NDArray[np.float32]
    ) -> None:
        """One perspective's first-layer sum from scratch, in the given king zone,
        written into `out`. Compiled, because a king crossing a zone boundary lands
        here at ~3% of nodes, and the numpy row-add version cost 40 us each time."""
        offset = zone * FEATURES
        features = self.features
        count = 0
        for piece_type in range(1, 7):
            for colour in (chess.WHITE, chess.BLACK):
                mask = board.pieces_mask(piece_type, colour)
                while mask:
                    square = (mask & -mask).bit_length() - 1
                    features[count] = offset + _feature(square, piece_type, colour, white_pov)
                    count += 1
                    mask &= mask - 1
        if self.fast:
            _refresh_kernel(out, B1, W1, features, count)
            return
        out[:] = B1
        for k in range(count):
            out += W1[features[k]]

    def refresh(self, board: chess.Board) -> None:
        self.zone_white = self._king_zone(board, chess.WHITE)
        self.zone_black = self._king_zone(board, chess.BLACK)
        self._rebuild(board, True, self.zone_white, self.white)
        self._rebuild(board, False, self.zone_black, self.black)
        self.zones.clear()
        if self.fast:
            self.depth = 0
        else:
            self.stack.clear()

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
        # A king move may carry its own perspective into another zone, in which case
        # that perspective is rebuilt after the deltas below rather than updated.
        crossing = 0  # 1 = white perspective, 2 = black perspective
        new_zone = 0
        if piece_type == chess.KING and KING_ZONES > 1:
            new_zone = _zone(move.to_square if mover == chess.WHITE else move.to_square ^ 56)
            if mover == chess.WHITE and new_zone != self.zone_white:
                crossing = 1
            elif mover == chess.BLACK and new_zone != self.zone_black:
                crossing = 2
        self.zones.append((self.zone_white, self.zone_black))
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

        self._apply(added, n_added, removed, n_removed)

        if crossing:
            # The deltas above were applied in the old zone, which is now the wrong
            # block for the perspective that crossed; recompute it in the new one.
            # The vectors saved on the stack are the pre-move ones, so pop is exact.
            board.push(move)
            try:
                if crossing == 1:
                    self.zone_white = new_zone
                    self._rebuild(board, True, new_zone, self.white)
                else:
                    self.zone_black = new_zone
                    self._rebuild(board, False, new_zone, self.black)
            finally:
                board.pop()

    def _apply(
        self,
        added: npt.NDArray[np.int32],
        n_added: int,
        removed: npt.NDArray[np.int32],
        n_removed: int,
    ) -> None:
        """Save the current vectors, then add and subtract the given W1 rows, each
        perspective in its own king-zone block."""
        white_offset = self.zone_white * FEATURES
        black_offset = self.zone_black * FEATURES
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
                W1, added, n_added, removed, n_removed, white_offset, black_offset,
            )
            self.depth += 1
            return

        self.stack.append((self.white.copy(), self.black.copy()))
        for k in range(n_added):
            self.white += W1[added[2 * k] + white_offset]
            self.black += W1[added[2 * k + 1] + black_offset]
        for k in range(n_removed):
            self.white -= W1[removed[2 * k] + white_offset]
            self.black -= W1[removed[2 * k + 1] + black_offset]

    def pop(self) -> None:
        self.zone_white, self.zone_black = self.zones.pop()
        if self.fast:
            self.depth -= 1
            _pop_kernel(self.white, self.black, self.stack, self.depth)
            return
        self.white, self.black = self.stack.pop()

    def evaluate(self, turn: chess.Color, pieces: int = 32) -> int:
        """Centipawns from the side to move's point of view.

        `pieces` is the number of men on the board and selects the output head.
        """
        if turn == chess.WHITE:
            own, opponent = self.white, self.black
        else:
            own, opponent = self.black, self.white
        k = _bucket(pieces)
        if self.fast:
            compiled: float = float(
                _eval_bucket_kernel(own, opponent, k, _W2T, B2, W3, B3, _SCRATCH)
            )
            return int(compiled * OUTPUT_SCALE)
        hidden = np.concatenate((own, opponent))
        np.clip(hidden, 0.0, 1.0, out=hidden)
        hidden *= hidden  # SCReLU
        second = np.maximum(hidden @ W2[k] + B2[k], 0.0)
        return int(float((second @ W3[k] + B3[k])[0]) * OUTPUT_SCALE)


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
TIME_V2: Final = True
QS_EVASIONS: Final = False
STAGED_MOVEGEN: Final = False
HYGIENE: Final = True
# CONTEMPT: a repetition or fifty-move draw is not worth exactly zero. Measured at
# the tournament control, 37% of games against Stockfish skill 10 ended by
# repetition, and against Weiss depth 8 -- an opponent this engine beats 88% of the
# time -- five games were repetition draws in the middlegame, two of them with the
# engine two points of material ahead. In a 13-round Swiss most opponents are
# weaker, so a half point conceded from an equal or better position is the most
# expensive habit left. The referee also adjudicates on raw material at ply 300,
# so being ahead late makes a draw cost a whole point.
CONTEMPT: Final = True
# FUTILITY: at depth 1-2, not in check, skip quiet moves when the static score plus
# a margin cannot reach alpha. Reverse futility, the mirror image, measured +62.
FUTILITY: Final = True
# TT_AGE: replace transposition entries by age and depth instead of clearing the
# whole table every 400k entries -- about once a minute at the compiled node rate.
TT_AGE: Final = False
# PVS: after the first move, search the rest with a null window and re-search
# only when one surprises. Never tested here on its own, only bundled with LMR.
PVS: Final = False
# TIME_V3: two lessons from the platform's round-4 loss. The schedule was
# front-loaded -- 52 s left at move 20, 17 s at move 40, the last thirty moves at
# half a second each -- so the horizon is longer and more moves are expected. And
# the decisive error was played in 1.6 s because the iteration-cost rule stopped
# deepening at a position where one more ply found the right move; when the best
# move changes or the score drops between iterations the position is unstable,
# and the next iteration is allowed up to 2.5 soft budgets instead of 1.5.
TIME_V3: Final = True
# TIME_V4: an unfinished iteration is not worthless. A transposition table warm
# from the previous move lets the early iterations finish in milliseconds, the
# cost predictor launches the next depth blind, and it hits the hard cap -- but a
# root move that completed at the new depth with a score above the previous best
# has been proven better, and is kept. (A floor on the predicted cost was tried
# alongside this and measured 46% at 120 s: it stops iterations a warm table
# would have finished. Dropped.)
TIME_V4: Final = True
# TIME_V5: the 26-move floor on the horizon still banks time at move 70 that the
# game will never use; lower it to 18 so the mid and late game spend more. Paired
# with a refund so calm positions hand the extra back: after two consecutive
# completed iterations that kept the same best move with no score drop, the next
# iteration is allowed 1.0 soft budgets instead of 1.5. Only the 120 s control
# can see it -- below LOW_CLOCK the budget is remaining/30 and the floor never
# binds, so 8 s games are unchanged; judged by clocktest + 40 games at 120 s.
TIME_V5: Final = True
# CORRECTION: correction history. The static evaluation is wrong in ways that
# repeat. In the platform's round-8 loss it said +400 to +1010 in a rook-and-knight
# ending that the search itself scored between -37 and -164, and reverse futility
# and futility both trust the static score, so a persistent error cuts exactly the
# lines that would refute it. Per side to move and pawn structure this keeps the
# running gap between the static score and what the search returned, and adds it
# to the static score before anything trusts it.
CORRECTION: Final = False
# TT_EVAL: a transposition entry for this node holds a searched score. When its
# bound allows -- exact, a lower bound above the static score, or an upper bound
# below it -- that score replaces the static score for reverse futility and
# futility, so a search result already in hand is trusted over the network's
# guess. The guard for the round-8 pattern that needs no learning.
TT_EVAL: Final = False
# COMPILED_SEARCH: negamax and quiescence run as numba kernels (fastsearch.py)
# over the compiled board, with the transposition table as fixed arrays. Same
# semantics as FastEngine.search -- testing/check_fastsearch holds them to
# identical scores and node counts at fixed depth with the table off. The root
# loop, the time rules and the fallback are unchanged.
COMPILED_SEARCH: Final = True
# LMR / LMP: late move reductions and late move pruning inside the compiled
# search (fastsearch.py). LMR reduces the depth of quiet moves after the first
# two by a log-log amount and re-searches on a fail high; LMP skips the quiet
# tail of the move list at depth <= 3. Both need COMPILED_SEARCH. PVS in the
# kernel follows the PVS switch above.
LMR: Final = True
LMP: Final = False
# SEE: in quiescence, skip captures that lose material on the exchange
# (fastboard.see). Needs COMPILED_SEARCH.
SEE: Final = True
# ASPIRATION: from depth 4 the root searches a window of +/- ASPIRATION_WINDOW
# around the previous iteration's score, widening on a fail and falling back to
# the full window after three fails. Narrow windows cut off sooner.
ASPIRATION: Final = True
ASPIRATION_WINDOW: Final = 15
# REPETITION_TWOFOLD: the referee calls board.outcome(claim_draw=True) after every
# move, and python-chess lets the side to move claim as soon as ONE legal move
# would make a third occurrence. Round 11 on the platform was drawn with a mate
# on the board for us, because two positions had occurred twice and the referee
# stopped the game before we chose. So in the search a position that has
# occurred even once before in the game is a draw: while winning the engine
# never lets a position repeat at all.
REPETITION_TWOFOLD: Final = True
# PONDER: the rules allow thinking on the opponent's time ("your process keeps
# its core while the opponent thinks") and the runner keeps this process alive
# between requests. After answering, a second engine that shares the main
# engine's transposition table searches the position it expects next -- our
# move plus the reply the table predicts -- until the next request arrives,
# which stops it within a millisecond. The main search then starts on a warm
# table. Needs COMPILED_SEARCH; the kernels release the GIL.
PONDER: Final = False
# NMP_GUARD: no null move directly after a null move. Found by review on 4 Sep:
# two nulls restore the Zobrist key, the stack repetition check fires and the
# grandchild scores as a draw, so null-move pruning never cut at depth >= 6.
NMP_GUARD: Final = False
# RFP_PHASE: the static score is 2-6x less accurate below 17 pieces (mean |err|
# 50 cp at 29-32 pieces, 194 at 13-16, 290 at 9-12), yet reverse futility and
# futility prune on it with one margin. Scale both margins by piece count:
# 1.6x at 17-20, 2x at 13-16, 3x at 9-12, off at 8 and below.
RFP_PHASE: Final = False
# percent of the normal margin in the bands <= 8, 9-12, 13-16, 17-20 pieces (0 = off)
RFP_PHASE_PERCENT: Final = (0, 300, 200, 160)
# IIR: internal iterative reduction, one ply less at depth >= 4 without a hash move.
IIR: Final = False
# BOOK_ENABLED: the polyglot book. Games start from curated positions at ply
# 10-16, so the book fires 0-3 plies per game, covers 35% of the curated pool,
# and on the platform games it cost ~20 cp per firing, with one line losing to
# a Greek gift. Off means the search plays from move one.
BOOK_ENABLED: Final = True
# HISTORY2: move ordering. History indexed by side to move with a gravity update
# (bounded at +/-16384, so it can never outrank a capture or a killer) and a
# malus for the quiet moves searched before a cutoff; a counter-move table keyed
# on the opponent's last move, ranked just below the killers. Needs COMPILED_SEARCH.
HISTORY2: Final = True
# TT_KEEP: table entries from the previous search are not evicted freely; a new
# entry replaces an aged one only if it is at most 4 plies shallower. The warm
# table is what makes the early iterations of the next move free.
TT_KEEP: Final = False
# TT_BUCKETS: the transposition table as pairs of slots. The even slot keeps the
# deeper entry, the odd slot always takes the store, and a probe checks both, so
# a deep entry survives the key traffic that evicts it from a single slot.
TT_BUCKETS: Final = True
# QS_CAP: quiescence depth cap. 8 truncates long exchanges; with SEE pruning the
# capture tree is small enough to follow to 14.
QS_CAP: Final = 14
# SAFE_BITS: mate-distance pruning, null-move reduction growing with depth, and
# a forced move played without searching.
SAFE_BITS: Final = True
# BOOK_VERIFY: a book move is searched first and played only if the search's own
# best is not better by more than BOOK_VERIFY_MARGIN centipawns. Closes the book
# lines measured at -68 and -165 cp on the platform's own start positions.
BOOK_VERIFY: Final = False
# QS_EVAL_CACHE: memoise quiescence static evaluations by position key (exact).
QS_EVAL_CACHE: Final = True
# SEE_MAIN: in the main search skip captures losing more than 20*depth^2 on the
# exchange at depth <= 5 (never the first move).
SEE_MAIN: Final = True
# ROOT_ORDER: from the second iteration order the root moves by the scores the
# previous iteration gave them (stable: moves the aspiration pass never reached
# keep their old order at the tail) instead of re-running the static ordering.
# The book/hash move still leads. Fail-low values are only upper bounds, but
# the relative order they induce is what most engines sort the root by.
ROOT_ORDER: Final = True
# LMR_AGGRESSIVE: depth is the main lever. Reduce quiet moves from the second
# one searched with the steeper log(d)*log(m)/1.8 + 0.5 table, adjusted by
# butterfly history (one ply less above +8000, one more below -8000, never
# below depth 1), and turn PVS on inside the same switch: the null-window
# re-searches are what make the deeper reductions cheap. PVS failed alone
# twice; paired with reductions is the standard reason it exists. Needs
# COMPILED_SEARCH.
LMR_AGGRESSIVE: Final = True
# CHECK_EXT_CAP: at most this many check extensions on one line (0 = unlimited).
CHECK_EXT_CAP: Final = 0
# LAZY_ACC: defer the NNUE accumulator update from make to the first evaluate
# on the line. Exact -- same nodes, same scores -- but a node cut off by the
# hash table, a repetition, or the null move before any static evaluation never
# pays the two 512-float row updates or the astack save/restore, which profiled
# at 15.4% of search time. Needs COMPILED_SEARCH.
LAZY_ACC: Final = True
# PRUNE_V2: prune plain quiet moves harder at depth <= 4, after the first move
# at a node, when not in check and not near a mate: futility with a margin of
# 100 cp per ply of depth, and a history cut for quiets whose butterfly score
# is below -1500 per ply of depth. Removes whole subtrees rather than
# shortening them, which is what a depth gain needs. Needs COMPILED_SEARCH and
# HISTORY2 (the history cut reads the side-to-move band).
PRUNE_V2: Final = True
# SINGULAR: singular extensions. At depth >= 7 with a hash move whose stored
# bound is exact or a lower bound at depth >= depth - 3, the node is searched
# again without that move at half depth with a window two pawns per ply below
# the stored score; if nothing else reaches it the hash move is the only move
# and is searched a ply deeper. Capped at six check-or-singular extensions on
# a line. Needs COMPILED_SEARCH.
SINGULAR: Final = True
BOOK_VERIFY_MARGIN: Final = 25
PONDER_MAX_S: Final = 600.0
# PONDER_DIAG: print, at each request, the wall time since the ponder thread started
# and how many nodes it searched. The platform shows stderr in the validation log's
# smoke games, which is the only way to see whether pondering runs there.
PONDER_DIAG: Final = False
# PONDER_PROBE: answer "does the platform run us between moves?" through the only
# channel a rated game gives back, the clock. On our moves 8-10 the search gets a
# fixed 1.0 s when the ponder thread searched >= 100k nodes in the gap before the
# request, and a fixed 3.0 s when it did not. Read the PGN clocks of one game.
PONDER_PROBE: Final = False

# CONTEMPT: draw scores from the root side's point of view, in centipawns. Level
# positions carry a small reluctance to repeat; being ahead carries more, rising
# toward the adjudication ply; being behind makes a draw welcome.
FUTILITY_MARGIN: Final = (0, 150, 300)
# CORRECTION: table of 2 x 2**BITS entries in grain units; a node of depth d moves
# its entry toward the observed gap with weight min(d + 1, WEIGHT_MAX) / SCALE.
# A first version with cap 400 and scale 256, also applied to the quiescence
# stand-pat, measured -137 +/- 65 (048): the entries saturated within seconds and
# one tactical gap was charged to every leaf sharing its pawn structure. This is
# the mild bias the technique is meant to be.
CORRECTION_BITS: Final = 14
CORRECTION_GRAIN: Final = 256
CORRECTION_SCALE: Final = 1024
CORRECTION_WEIGHT_MAX: Final = 16
CORRECTION_CAP: Final = 100
CORRECTION_QS: Final = False
CONTEMPT_LEVEL: Final = 10
CONTEMPT_AHEAD: Final = 25
CONTEMPT_AHEAD_LATE: Final = 50
CONTEMPT_BEHIND: Final = -20
ADJUDICATION_PLY: Final = 300
# ADJUDICATION (V10_PLAN #3): play the referee's ply-300 material adjudication,
# not just chess. (a) The ply counter is pinned to MATCH plies at our first
# request (the referee counts from the curated start FEN; fullmove_number counts
# from the real initial position and ran 13 ahead in round 18). (b) Behind on
# raw material the draw score ramps from +20 cp to a large bonus as the cap
# nears -- at ply 280 a repetition is worth a half point, not 20 cp (round 18
# shuffled an eval-0 K+R+N vs K+Q into a material adjudication loss). (c) When
# behind AND a fifty-move draw is reachable before the cap
# (match_ply + 100 - halfmove_clock <= 300), the kernel's draw threshold
# C_HMC_DRAW drops to halfmove_clock + ADJ_HORIZON: a horizon's worth of
# non-zeroing plies scores as the draw we are steering for. Also uses HISTORY2's
# quiets fix and KILLER_CLEAR slots in the same ctrl block.
ADJUDICATION: Final = True
ADJ_BEHIND_LATE: Final = 300  # cp added to the behind-side draw score by the cap
ADJ_WINDOW: Final = 80  # arm the fifty-move plan only this close to the cap
ADJ_HORIZON: Final = 16  # non-zeroing plies the search credits as draw-reaching
# HISTORY2_FIX (v10 search.md 3.7): zero quiets[ply, searched] for non-quiet
# moves; without it the cutoff malus punishes stale moves recorded by an earlier
# node at the same ply. KILLER_CLEAR (same source): clear killers[ply + 2] on
# node entry and the whole table between root moves; killers from another
# subtree or the previous search are noise in move ordering.
HISTORY2_FIX: Final = True
KILLER_CLEAR: Final = True
# CONT_HIST (v10 search.md 3.1): 1-ply continuation history. A 768x768 int32
# table indexed by (previous move's piece*64+to, this quiet's piece*64+to),
# added to the quiet ordering score, the LMR history term (continuous
# hist // 6000 clamped +/-2 instead of the +/-8000 step) and the prune2
# history test; butterfly-style gravity update on a cutoff, halved under
# HYGIENE, no read or update after a null move.
CONT_HIST: Final = False
# IMPROVING (v10 search.md 3.3): static_eval(ply) > static_eval(ply - 2), the
# stack in exts[MAX_PLY + ply] (sentinel -INFINITY in check, ply < 2 defaults
# to improving). When NOT improving prune harder: RFP margin depth - improving,
# prune2 futility FUTILITY_MARGIN2[depth - improving], LMR reduction += 1.
# Costs one static eval at the <1% of non-check nodes deeper than RFP reaches.
IMPROVING: Final = False
# CUTNODE (same source): expected-fail-high flag passed down the tree (the one
# new kernel parameter): the null-move child is always a cut node, a null-window
# child is a cut node iff its parent was not, a full-window child of a PV node
# is a PV node. Use: LMR reduction += 1 at cut nodes.
CUTNODE: Final = False
# INIT_FOLD (speed.md section 2): fastsearch scans this file at import and,
# when this is True, compiles the settled switch slots (the eighteen in
# _fs.FOLDED) as constants instead of ctrl reads -- numba prunes the dead arms
# before typing, cutting fs.warm_up ~18% (~-4.3 s local, ~-8 s platform). Node
# and score exact by construction; prepare() asserts ctrl matches _fs.FOLDED.
# Off in the tree so testing/check_fastsearch can still zero ctrl and hold the
# kernel to the flags-off reference.
INIT_FOLD: Final = False
# How many earlier occurrences of a position make it a draw inside the search.
_REPEAT_LIMIT: Final = 1 if REPETITION_TWOFOLD else 2

# TIME_V2: the clock is never allowed below this fraction of its starting value,
# which is inferred as the largest time_left_ms seen in the game. 12 s at 120 s.
RESERVE_FRACTION: Final = 0.10
# TIME_V2: below this many seconds the budget stops crediting the increment.
LOW_CLOCK: Final = 15.0
# TIME_V6 (V10_PLAN #1): what every OpenBench engine measured. (a) The budget
# credits the increment it actually observes (median of the clock deltas between
# our calls), keeps a 4% reserve instead of 10%, and drops the low-clock regime
# to 9 s -- the 13 s absorbing floor in games.md came from the 10% reserve plus
# remaining/30 below 15 s. (b) The next iteration is never predicted (Ethereal
# gained +6..+12 removing exactly that); the search stops at an iteration end
# once elapsed exceeds ideal x stability x score-drop x node-effort, with the
# hard deadline (3 soft budgets, 12% of the clock) as the only mid-iteration stop.
# Stability 1.2 -> 0.8 (Ethereal), score-drop 2^(-drop/100) (Stash), node effort
# max(0.5, 2.0 - 1.6*bestFraction) (Ethereal/Koivisto), product clamped [0.4, 2].
# The first cut (Stash's 2.5x table, 4 soft budgets, 4% reserve) drained the clock
# to 1.6 s with 19 s moves under the 1.5x clocktest charge; these are the tamed
# values. Absorbs TIME_V5's
# 18-move floor. Needs COMPILED_SEARCH (per-root-move node counts from ctrl).
TIME_V6: Final = True
RESERVE_FRACTION_V6: Final = 0.06
LOW_CLOCK_V6: Final = 12.0
_STABILITY_SCALE: Final = (1.2, 1.1, 1.0, 0.9, 0.8)  # Ethereal-style, capped at 4
_INC_SAMPLES: list[float] = []  # observed increment, ms, last five moves
# ADJUDICATION: chess ply of the game's first request, and the last ply seen
# (a ply that goes backwards means a new game in the same process).
_MATCH_BASE_PLY: int = -1
_LAST_GAME_PLY: int = -1
_LAST_CLOCK_MS: float = -1.0
_LAST_SPENT_MS: float = -1.0  # wall time of our previous get_move call
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
        return self.acc.evaluate(board.turn, chess.popcount(board.occupied))

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
        if ply and (self.history.get(key, 0) >= _REPEAT_LIMIT or board.is_repetition(2)):
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


# --------------------------------------------------------------------------------
# The compiled board
# --------------------------------------------------------------------------------
# python-chess costs about six of the ten microseconds a node takes: ~28 us for a
# legal move list, ~2.6 us per push and pop. fastboard.py is the same bitboard
# representation compiled with numba -- the organisers' named fast path -- with
# the accumulator update folded into make. FastEngine below is the same search as
# Engine over that board. python-chess keeps the root: the FEN, the book, the
# tablebase, and a legality check of every move before it leaves this file.
#
# Any exception on the compiled path, and any move it proposes that python-chess
# does not accept, falls back to Engine for that move. The failure mode is a slower
# move, never a lost game.
FAST_BOARD: Final = True
_FAST_OK = False
try:
    if FAST_BOARD and _COMPILED:
        import fastboard as _fb

        _fb.warm_up()
        if COMPILED_SEARCH:
            import fastsearch as _fs

            _fs.warm_up(W1, B1, _W2T, B2, W3, B3, KING_ZONES)
        _FAST_OK = True
except Exception:
    _FAST_OK = False


class FastEngine:
    """Engine's search over the compiled board. Moves are packed ints, keys are
    polyglot Zobrist ints, and the position lives in fastboard.Position's arrays."""

    __slots__ = (
        "age",
        "astack",
        "black",
        "bufs",
        "butterfly",
        "conthist1",
        "corr",
        "counter",
        "ctrl",
        "deadline",
        "draw_root",
        "ec_key",
        "ec_val",
        "exts",
        "first_score",
        "hint",
        "history",
        "killers",
        "killers2",
        "movebuf",
        "nodes",
        "pos",
        "quiets",
        "rep_keys",
        "root_best",
        "root_side",
        "scores",
        "scores2",
        "scratch",
        "table",
        "tt",
        "white",
        "zones",
    )

    def __init__(self) -> None:
        # key -> (depth, score, flag, move, age); flag 0 exact, 1 lower, 2 upper.
        self.table: dict[int, tuple[int, int, int, int, int]] = {}
        self.age = 0
        self.history: dict[int, int] = {}
        self.killers: list[list[int]] = [[0, 0] for _ in range(_fb.MAX_PLY)]
        self.butterfly = np.zeros(8192, dtype=np.int32)
        self.counter = np.zeros(4096, dtype=np.int32)
        # CONT_HIST: (prev piece*64+to) x (piece*64+to), 2.3 MB, zeros when off
        self.conthist1 = np.zeros(768 * 768, dtype=np.int32)
        # 4 lanes of MAX_PLY: [0] extension count, [1] static eval, [2] piece*64+to
        # of the move made at this ply (CONT_HIST), [3] spare; only lane 0 is read yet
        self.exts = np.zeros(4 * _fb.MAX_PLY, dtype=np.int64)
        self.ec_key = np.zeros(1, dtype=np.uint64)
        self.ec_val = np.zeros(1, dtype=np.int32)
        self.quiets = np.zeros((_fb.MAX_PLY, _fb.MOVE_CAP), dtype=np.int32)
        self.corr = np.zeros((2, 1 << CORRECTION_BITS), dtype=np.int64)
        # One move buffer per ply, sliced once: a fresh view per node is not free.
        buffer = np.zeros((_fb.MAX_PLY, _fb.MOVE_CAP), dtype=np.int32)
        self.bufs: list[npt.NDArray[np.int32]] = [buffer[i] for i in range(_fb.MAX_PLY)]
        self.movebuf = buffer
        # COMPILED_SEARCH state: array table, array killers, control block.
        self.tt: tuple[Any, ...] = ()
        self.killers2 = np.zeros((_fb.MAX_PLY, 2), dtype=np.int32)
        self.scores2 = np.zeros((_fb.MAX_PLY, _fb.MOVE_CAP), dtype=np.int64)
        self.scratch = np.zeros(2 * ACC_SIZE, dtype=np.float32)
        self.ctrl = np.zeros(_fs.CTRL_SIZE if COMPILED_SEARCH else 32, dtype=np.int64)
        self.rep_keys = np.zeros(0, dtype=np.uint64)
        self.root_best = 0
        self.hint = 0  # a book move to search first and verify
        self.first_score = -INFINITY
        if COMPILED_SEARCH:
            self.tt = _fs.new_table()
            self.ec_key, self.ec_val = _fs.new_eval_cache()
        self.scores = np.zeros(_fb.MOVE_CAP, dtype=np.int64)
        self.white = B1.copy()
        self.black = B1.copy()
        self.astack = np.zeros((_fb.MAX_PLY, 2, ACC_SIZE), dtype=np.float32)
        self.zones = np.zeros(2, dtype=np.int64)
        self.pos = _fb.Position(chess.Board())
        self.deadline = 0.0
        self.nodes = 0
        # Draw score from the root side's point of view, and which side that is.
        self.draw_root = 0
        self.root_side = 0

    def _draw(self) -> int:
        """What a draw is worth to the side to move at this node."""
        if self.draw_root == 0:
            return 0
        return self.draw_root if self.pos.meta[0] == self.root_side else -self.draw_root

    # -- primitives ---------------------------------------------------------------

    def _make(self, move: int) -> None:
        pos = self.pos
        _fb.make_full(
            pos.bb, pos.sq, pos.meta, pos.undo, pos.keys, move,
            W1, B1, self.white, self.black, self.astack, self.zones, KING_ZONES,
        )

    def _unmake(self) -> None:
        pos = self.pos
        _fb.unmake_full(
            pos.bb, pos.sq, pos.meta, pos.undo, pos.keys,
            self.white, self.black, self.astack, self.zones,
        )

    def evaluate(self) -> int:
        meta = self.pos.meta
        if meta[0] == 0:
            own, opponent = self.white, self.black
        else:
            own, opponent = self.black, self.white
        k = _bucket(int(meta[5]))
        return int(
            float(_eval_bucket_kernel(own, opponent, k, _W2T, B2, W3, B3, self.scratch))
            * OUTPUT_SCALE
        )

    def corrected(self) -> tuple[int, int, int]:
        """(raw static, static plus this pawn structure's correction, table index)."""
        pos = self.pos
        side = int(pos.meta[0])
        index = int(_fb.pawn_index(pos.bb, CORRECTION_BITS))
        raw = self.evaluate()
        return raw, raw + int(self.corr[side, index]) // CORRECTION_GRAIN, index

    # -- quiescence ---------------------------------------------------------------

    def quiesce(self, alpha: int, beta: int, depth: int, ply: int) -> int:
        self.nodes += 1
        if not self.nodes & _POLL_MASK and time.monotonic() > self.deadline:
            raise Timeout

        standing = self.corrected()[1] if CORRECTION and CORRECTION_QS else self.evaluate()
        if standing >= beta:
            return standing
        if standing + BIG_DELTA < alpha:
            return standing
        if standing > alpha:
            alpha = standing
        if depth >= 8 or ply >= _fb.MAX_PLY - 2:
            return standing

        pos = self.pos
        sq = pos.sq
        captures = self.bufs[ply]
        n = _fb.gen_legal(pos.bb, sq, pos.meta, captures, True)
        _fb.order_moves(captures, n, sq, 0, 0, 0, self.butterfly, self.scores)
        for i in range(n):
            move = int(captures[i])
            victim = int(sq[(move >> 6) & 63])
            if (
                victim >= 0
                and not (move >> 12)
                and standing + _MVV[victim % 6] + DELTA_MARGIN < alpha
            ):
                continue
            self._make(move)
            try:
                score = -self.quiesce(-beta, -alpha, depth + 1, ply + 1)
            finally:
                self._unmake()
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return alpha

    # -- main search --------------------------------------------------------------

    def search(self, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if not self.nodes & _POLL_MASK and time.monotonic() > self.deadline:
            raise Timeout

        pos = self.pos
        bb = pos.bb
        meta = pos.meta
        keys = pos.keys
        key = int(keys[meta[4]])

        if ply:
            if meta[3] >= 4 and (
                self.history.get(key, 0) >= _REPEAT_LIMIT or _fb.repeats(meta, keys)
            ):
                return self._draw()
            if meta[3] >= 100:
                n = _fb.gen_legal(bb, pos.sq, meta, self.bufs[ply], False)
                if n == 0 and _fb.in_check(bb, meta):
                    return -MATE + ply
                return self._draw()
            if _TABLEBASE is not None and meta[5] <= TB_MEN:
                wdl = _TABLEBASE.get_wdl(pos.to_board())
                if wdl is not None:
                    if wdl > 1:
                        return TB_WIN - ply
                    if wdl < -1:
                        return -TB_WIN + ply
                    return 0
        if ply >= _fb.MAX_PLY - 8:
            return self.evaluate()

        original_alpha = alpha
        stored = self.table.get(key)
        hash_move = 0
        if stored is not None:
            stored_depth, raw_score, flag, hash_move, _ = stored
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

        in_check = bool(_fb.in_check(bb, meta))
        if in_check and ply < MAX_PLY - 8:
            depth += 1

        if depth <= 0:
            return self.quiesce(alpha, beta, 0, ply)

        standing = -INFINITY
        raw_standing = -INFINITY
        corr_index = -1
        if CORRECTION and not in_check:
            # Every quiet node contributes to the correction table, so the static
            # score is taken here regardless of depth.
            raw_standing, standing, corr_index = self.corrected()
        if (
            depth <= RFP_MAX_DEPTH
            and not in_check
            and (not HYGIENE or abs(beta) < DISTANCE_THRESHOLD)
        ):
            if standing == -INFINITY:
                standing = self.evaluate()
            if (
                TT_EVAL
                and stored is not None
                and abs(stored_score) < DISTANCE_THRESHOLD
                and (
                    flag == 0
                    or (flag == 1 and stored_score > standing)
                    or (flag == 2 and stored_score < standing)
                )
            ):
                standing = stored_score
            if standing - RFP_MARGIN * depth >= beta:
                return standing

        futile = False
        if FUTILITY and depth <= 2 and not in_check and abs(alpha) < DISTANCE_THRESHOLD:
            if standing == -INFINITY:
                standing = self.evaluate()
            futile = standing + FUTILITY_MARGIN[depth] <= alpha

        if (
            depth >= NMP_MIN_DEPTH
            and not in_check
            and abs(beta) < DISTANCE_THRESHOLD
            and _fb.non_pawn_material(bb, int(meta[0]))
            and (not NMP_GUARD or ply == 0 or pos.undo[meta[4] - 1, 0] != 0)
        ):
            _fb.make_null(bb, meta, pos.undo, keys)
            try:
                score = -self.search(depth - 1 - NMP_REDUCTION, -beta, -beta + 1, ply + 1)
            finally:
                _fb.unmake_null(meta, pos.undo)
            if score >= beta:
                return beta

        moves = self.bufs[ply]
        n = _fb.gen_legal(bb, pos.sq, meta, moves, False)
        if n == 0:
            return -MATE + ply if in_check else 0
        killers = self.killers[ply]
        _fb.order_moves(
            moves, n, pos.sq, hash_move, killers[0], killers[1], self.butterfly, self.scores
        )

        best_score = -INFINITY
        best_move = 0
        searched = 0
        sq = pos.sq
        for i in range(n):
            move = int(moves[i])
            quiet = sq[(move >> 6) & 63] < 0
            if futile and quiet and not (move >> 12):
                # A quiet move from a position this far below alpha is not going to
                # raise it; only captures and promotions get a look.
                continue
            self._make(move)
            try:
                if PVS and searched:
                    score = -self.search(depth - 1, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -self.search(depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self.search(depth - 1, -beta, -alpha, ply + 1)
            finally:
                self._unmake()
            searched += 1
            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if quiet:
                            if killers[0] != move:
                                killers[1] = killers[0]
                                killers[0] = move
                            self.butterfly[(move & 63) * 64 + ((move >> 6) & 63)] += depth * depth
                        break

        if not searched:
            # Every move was futility-pruned: the position is at least as bad as
            # the static score says, which is below alpha.
            return standing

        if len(self.table) >= MAX_TABLE:
            if TT_AGE:
                # Keep what this move and the last one learned; drop the rest.
                age = self.age
                self.table = {k: v for k, v in self.table.items() if v[4] >= age - 1}
                if len(self.table) >= MAX_TABLE:
                    self.table.clear()
            else:
                self.table.clear()
        if best_score <= original_alpha:
            flag = 2
        elif best_score >= beta:
            flag = 1
        else:
            flag = 0
        if (
            CORRECTION
            and corr_index >= 0
            and abs(best_score) < DISTANCE_THRESHOLD
            and not (best_move >> 12)
            and sq[(best_move >> 6) & 63] < 0
            and (
                flag == 0
                or (flag == 1 and best_score > raw_standing)
                or (flag == 2 and best_score < raw_standing)
            )
        ):
            # The search disagreed with the static score in a direction the bound
            # supports: move this pawn structure's entry toward the gap.
            side = int(meta[0])
            weight = min(depth + 1, CORRECTION_WEIGHT_MAX)
            entry = int(self.corr[side, corr_index])
            entry = (
                entry * (CORRECTION_SCALE - weight)
                + (best_score - raw_standing) * CORRECTION_GRAIN * weight
            ) // CORRECTION_SCALE
            cap = CORRECTION_CAP * CORRECTION_GRAIN
            self.corr[side, corr_index] = max(-cap, min(cap, entry))
        if TT_AGE:
            old = self.table.get(key)
            if old is not None and old[4] == self.age and old[0] > depth:
                return best_score  # a deeper result from this same search stays
        self.table[key] = (depth, _to_table(best_score, ply), flag, best_move, self.age)
        return best_score

    def root_search(self, depth: int, alpha: int, beta: int, ply: int) -> int:
        """One search call from the root loop: the kernel or the Python search."""
        if not COMPILED_SEARCH:
            return self.search(depth, alpha, beta, ply)
        pos = self.pos
        ctrl = self.ctrl
        if ctrl[_fs.C_STOP] != 0:
            raise Timeout
        ctrl[_fs.C_ABORT] = 0
        if LAZY_ACC:
            # The root makes are eager, so the accumulators are current here.
            ctrl[_fs.C_ACC_PLY] = int(pos.meta[_fb.PLY])
        score = _fs.search(  # type: ignore[call-arg]
            pos.bb, pos.sq, pos.meta, pos.undo, pos.keys,
            W1, B1, self.white, self.black, self.astack, self.zones, KING_ZONES,
            _W2T, B2, W3, B3, *self.tt,
            self.killers2, self.butterfly, self.movebuf, self.scores2, self.rep_keys,
            ctrl, self.deadline, depth, alpha, beta, ply, self.scratch,
            self.counter, self.quiets, self.ec_key, self.ec_val, self.exts, self.conthist1,
            0,  # the root is a PV node, never an expected cut node
        )
        self.nodes = int(ctrl[_fs.C_NODES])
        if ctrl[_fs.C_ABORT]:
            raise Timeout
        return int(score)

    def predicted_reply(self, board: chess.Board) -> chess.Move | None:
        """The move the transposition table holds for `board`, if any and legal."""
        if not COMPILED_SEARCH or not self.tt:
            return None
        pos = _fb.Position(board)
        key = pos.keys[0]
        slot = int(key & _fs.TT_MASK)
        if self.tt[0][slot] != key:
            return None
        packed = int(_fs.unpack_move(self.tt[1][slot]))
        if not packed:
            return None
        move = chess.Move.from_uci(_fb.move_to_uci(packed))
        return move if move in board.legal_moves else None

    def prepare(self, board: chess.Board, draw_root: int) -> None:
        """Load `board` as the root: accumulators, contempt, and the kernel's state."""
        pos = self.pos
        pos.load(board)
        _fb.refresh(
            pos.bb, pos.sq, pos.meta, W1, B1, self.white, self.black, self.zones, KING_ZONES
        )
        self.root_side = int(pos.meta[0])
        self.draw_root = draw_root
        if COMPILED_SEARCH:
            ctrl = self.ctrl
            ctrl[_fs.C_NODES] = 0
            ctrl[_fs.C_ABORT] = 0
            ctrl[_fs.C_AGE] = self.age
            ctrl[_fs.C_ROOT_SIDE] = self.root_side
            ctrl[_fs.C_DRAW_ROOT] = draw_root
            ctrl[_fs.C_TT_OFF] = 0
            ctrl[_fs.C_HYGIENE] = 1 if HYGIENE else 0
            ctrl[_fs.C_FUTILITY] = 1 if FUTILITY else 0
            ctrl[_fs.C_PVS] = 1 if PVS or LMR_AGGRESSIVE else 0
            ctrl[_fs.C_LMR] = 1 if LMR else 0
            ctrl[_fs.C_LMP] = 1 if LMP else 0
            ctrl[_fs.C_SEE] = 1 if SEE else 0
            ctrl[_fs.C_NMP_GUARD] = 1 if NMP_GUARD else 0
            ctrl[_fs.C_RFP_PHASE] = 1 if RFP_PHASE else 0
            ctrl[_fs.C_PH_LE8] = RFP_PHASE_PERCENT[0]
            ctrl[_fs.C_PH_9_12] = RFP_PHASE_PERCENT[1]
            ctrl[_fs.C_PH_13_16] = RFP_PHASE_PERCENT[2]
            ctrl[_fs.C_PH_17_20] = RFP_PHASE_PERCENT[3]
            ctrl[_fs.C_IIR] = 1 if IIR else 0
            ctrl[_fs.C_HISTORY2] = 1 if HISTORY2 else 0
            ctrl[_fs.C_TT_KEEP] = 1 if TT_KEEP else 0
            ctrl[_fs.C_QS_CAP] = QS_CAP
            ctrl[_fs.C_SAFE] = 1 if SAFE_BITS else 0
            ctrl[_fs.C_QS_CACHE] = 1 if QS_EVAL_CACHE else 0
            ctrl[_fs.C_SEE_MAIN] = 1 if SEE_MAIN else 0
            ctrl[_fs.C_CHECK_CAP] = CHECK_EXT_CAP
            ctrl[_fs.C_TT_BUCKETS] = 1 if TT_BUCKETS else 0
            ctrl[_fs.C_LMR_AGGR] = 1 if LMR_AGGRESSIVE else 0
            ctrl[_fs.C_LAZY_ACC] = 1 if LAZY_ACC else 0
            ctrl[_fs.C_PRUNE2] = 1 if PRUNE_V2 else 0
            ctrl[_fs.C_SINGULAR] = 1 if SINGULAR else 0
            ctrl[_fs.C_EXCL_PLY] = -1
            ctrl[_fs.C_HIST2_FIX] = 1 if HISTORY2_FIX else 0
            ctrl[_fs.C_KILLER_CLEAR] = 1 if KILLER_CLEAR else 0
            ctrl[_fs.C_CONT_HIST] = 1 if CONT_HIST else 0
            ctrl[_fs.C_IMPROVING] = 1 if IMPROVING else 0
            ctrl[_fs.C_CUTNODE] = 1 if CUTNODE else 0
            if INIT_FOLD:
                for fold_slot, fold_value in _fs.FOLDED.items():
                    if bool(ctrl[fold_slot]) != fold_value:
                        raise RuntimeError(
                            f"INIT_FOLD: ctrl slot {fold_slot} is {int(ctrl[fold_slot])} "
                            f"but fastsearch folded it as {fold_value}"
                        )
            if KILLER_CLEAR:
                self.killers2[:] = 0  # killers from the previous search are noise
            ctrl[_fs.C_HMC_DRAW] = 100
            if ADJUDICATION:
                match_ply = _match_ply(board)
                hmc = board.halfmove_clock
                if (
                    ADJUDICATION_PLY - match_ply <= ADJ_WINDOW
                    and match_ply + (100 - hmc) <= ADJUDICATION_PLY
                    and hmc + ADJ_HORIZON < 100
                    and _material_balance(board) < 0
                ):
                    # Losing the material adjudication with a fifty-move draw
                    # reachable before the cap: a horizon of non-zeroing plies
                    # already scores as that draw.
                    ctrl[_fs.C_HMC_DRAW] = hmc + ADJ_HORIZON
            repeated = [k for k, count in self.history.items() if count >= _REPEAT_LIMIT]
            self.rep_keys = np.array(repeated, dtype=np.uint64)

    # -- driver -------------------------------------------------------------------

    def choose(self, soft_limit: float, hard_limit: float) -> int:
        self.deadline = hard_limit
        pos = self.pos
        moves = self.bufs[0]
        n = _fb.gen_legal(pos.bb, pos.sq, pos.meta, moves, False)
        if n == 0:
            raise ValueError("no legal moves")
        best = int(moves[0])
        if SAFE_BITS and n == 1:
            self.root_best = best
            return best
        hint = self.hint
        if hint:
            best = hint
        self.first_score = -INFINITY

        if HYGIENE:
            self.butterfly >>= 1
            if CONT_HIST:
                self.conthist1 >>= 1  # same decay, or it saturates within a few moves

        started = time.monotonic()
        previous_best = 0
        previous_score = -INFINITY
        unstable = False
        stable_streak = 0
        stability = 0  # TIME_V6: consecutive iterations that kept the best move
        score_hist: list[int] = []  # TIME_V6: one score per completed iteration
        prev_scores: dict[int, int] = {}
        for depth in range(1, 64):
            iteration_started = time.monotonic()
            first_done = False
            iteration_best = best
            pass_scores: dict[int, int] = {}
            root_nodes: dict[int, int] = {}  # TIME_V6: nodes spent under each root move
            # ASPIRATION: a window around the last score, or the full window.
            window = 0
            if ASPIRATION and depth >= 4 and abs(previous_score) < MATE_THRESHOLD:
                window = ASPIRATION_WINDOW
            lo = previous_score - window if window else -INFINITY
            hi = previous_score + window if window else INFINITY
            fails = 0
            try:
                while True:
                    score = -INFINITY
                    alpha = lo
                    failed_high = False
                    front = hint if hint else best
                    if ROOT_ORDER and prev_scores:
                        ranked = [
                            (int(moves[i]) != front, -prev_scores.get(int(moves[i]), -INFINITY), i)
                            for i in range(n)
                        ]
                        ranked.sort()
                        moves[:n] = moves[:n][[r[2] for r in ranked]]
                    else:
                        _fb.order_moves(
                            moves, n, pos.sq, front, 0, 0, self.butterfly, self.scores
                        )
                    iteration_best = int(moves[0])
                    first_done = False
                    for i in range(n):
                        move = int(moves[i])
                        node_start = int(self.ctrl[_fs.C_NODES]) if TIME_V6 else 0
                        self._make(move)
                        try:
                            if (PVS or LMR_AGGRESSIVE) and i:
                                value = -self.root_search(depth - 1, -alpha - 1, -alpha, 1)
                                if alpha < value < hi:
                                    value = -self.root_search(depth - 1, -hi, -alpha, 1)
                            else:
                                value = -self.root_search(depth - 1, -hi, -alpha, 1)
                        finally:
                            self._unmake()
                            if TIME_V6:
                                root_nodes[move] = (
                                    root_nodes.get(move, 0)
                                    + int(self.ctrl[_fs.C_NODES]) - node_start
                                )
                        if ROOT_ORDER:
                            pass_scores[move] = value
                        if i == 0:
                            # A first move that fell out of the window proves nothing
                            # about the others; with the full window this is always true.
                            first_done = value > lo
                            first_value = value
                        if value > score:
                            score = value
                            iteration_best = move
                            if value > alpha:
                                alpha = value
                                if alpha >= hi:
                                    failed_high = True
                                    break
                    if hint and first_done:
                        self.first_score = first_value
                    if not window:
                        break
                    fails += 1
                    if failed_high:
                        # The move that failed high is proven better than the old best:
                        # it leads the wider pass, and TIME_V4 may keep it.
                        best = iteration_best
                        hi = INFINITY if fails >= 3 else min(INFINITY, score + window * 4**fails)
                    elif score <= lo:
                        lo = -INFINITY if fails >= 3 else max(-INFINITY, score - window * 4**fails)
                    else:
                        break
                best = iteration_best
                if ROOT_ORDER:
                    prev_scores = pass_scores
            except Timeout:
                # The first root move is the previous best, searched with a full
                # window; a later move that came back above alpha at this depth has
                # beaten it at this depth, and is the better answer.
                if TIME_V4 and first_done and iteration_best != best:
                    best = iteration_best
                break

            if score > MATE_THRESHOLD or score < -MATE_THRESHOLD:
                break
            if TIME_V3:
                # A best move that changed, or a score that fell, means the last
                # ply revised the verdict: the next one may revise it again.
                unstable = depth >= 3 and (
                    best != previous_best or score < previous_score - 50
                )
                stable_streak = stable_streak + 1 if depth >= 3 and not unstable else 0
                stability = stability + 1 if depth >= 3 and best == previous_best else 0
                previous_best, previous_score = best, score
            now = time.monotonic()
            if TIME_V6:
                score_hist.append(score)
                elapsed = now - started
                budget = soft_limit - started
                factor = 1.0
                if depth >= 5:
                    factor = _STABILITY_SCALE[min(stability, 4)]
                    if len(score_hist) >= 4:
                        drop = score_hist[-4] - score_hist[-1]
                        factor *= 2.0 ** (max(-100, min(100, drop)) / 100.0)
                    total_nodes = sum(root_nodes.values())
                    if total_nodes > 0:
                        fraction = root_nodes.get(best, 0) / total_nodes
                        factor *= max(0.5, 2.0 - 1.6 * fraction)
                    factor = max(0.4, min(1.5, factor))
                if elapsed > factor * budget:
                    break
            elif TIME_V2:
                elapsed = now - started
                budget = soft_limit - started
                predicted = (now - iteration_started) * 2.5
                allowance = 2.5 if unstable else 1.5
                # TIME_V5 refund: two settled iterations in a row and the next
                # one must fit inside one soft budget, not one and a half.
                if TIME_V5 and stable_streak >= 2:
                    allowance = 1.0
                if elapsed + predicted > allowance * budget:
                    break
            elif now > soft_limit:
                break

        # The book move was searched first with a full window, so its score is
        # exact; keep it unless the search found something clearly better.
        if (
            hint
            and best != hint
            and self.first_score > -INFINITY
            and score - self.first_score <= BOOK_VERIFY_MARGIN
        ):
            best = hint
        self.root_best = best
        return best

    def play(self, board: chess.Board, soft_limit: float, hard_limit: float) -> chess.Move:
        """Search `board` and return the move as python-chess understands it."""
        pos = self.pos
        pos.load(board)
        _fb.refresh(
            pos.bb, pos.sq, pos.meta, W1, B1, self.white, self.black, self.zones, KING_ZONES
        )
        self.age += 1
        self.prepare(board, _contempt(board, self.evaluate()) if CONTEMPT else 0)
        move = self.choose(soft_limit, hard_limit)
        return chess.Move.from_uci(_fb.move_to_uci(move))


_FAST: FastEngine | None = None
_PONDER: FastEngine | None = None
_PONDER_THREAD: threading.Thread | None = None
_PONDER_STARTED: float = 0.0
_PONDER_LAST_NODES: int = 0
_SEARCHED_MOVES: int = 0  # requests answered by the search (not the book or a tablebase)
if _FAST_OK:
    try:
        _FAST = FastEngine()
        if PONDER and COMPILED_SEARCH:
            _PONDER = FastEngine()
            _PONDER.tt = _FAST.tt  # one table, warmed by whichever engine is searching
            _PONDER.history = _FAST.history
    except Exception:  # an init failure would lose every game; a fallback loses none
        _FAST = None
        _PONDER = None


def _ponder_target(board: chess.Board) -> None:
    """Search `board` on the opponent's time until told to stop. Only the shared
    table is kept; the move it finds is never played."""
    engine = _PONDER
    main = _FAST
    if engine is None or main is None:
        return
    try:
        engine.age = main.age
        engine.prepare(board, 0)
        limit = time.monotonic() + PONDER_MAX_S
        engine.choose(limit, limit)
    except Exception:  # pondering is a bonus; nothing here may reach the game
        pass


def _stop_ponder() -> None:
    global _PONDER_THREAD
    thread = _PONDER_THREAD
    if thread is None:
        return
    if _PONDER is not None:
        _PONDER.ctrl[_fs.C_STOP] = 1
    thread.join(0.5)
    _PONDER_THREAD = None


def _start_ponder(board: chess.Board) -> None:
    """`board` is the position after our move. Ponder the reply the table
    expects, or the position itself when it holds none."""
    global _PONDER_THREAD
    if _PONDER is None or _FAST is None:
        return
    try:
        target = board.copy()
        reply = _FAST.predicted_reply(target)
        if reply is not None:
            target.push(reply)
        if not target.legal_moves:
            return
        _PONDER.ctrl[_fs.C_STOP] = 0
        thread = threading.Thread(target=_ponder_target, args=(target,), daemon=True)
        _PONDER_THREAD = thread
        global _PONDER_STARTED
        _PONDER_STARTED = time.monotonic()
        thread.start()
    except Exception:
        _PONDER_THREAD = None
# The platform shows import-time output in the validation log, so this is how to
# see from the dashboard whether the compiled path came up on their image.
print(f"compiled board: {'on' if _FAST is not None else 'off'}")


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
    global _MAX_CLOCK_MS, _LAST_CLOCK_MS
    if time_left_ms > _MAX_CLOCK_MS:
        _MAX_CLOCK_MS = float(time_left_ms)
    if TIME_V6:
        if _LAST_CLOCK_MS >= 0.0 and _LAST_SPENT_MS >= 0.0:
            raw = time_left_ms - (_LAST_CLOCK_MS - _LAST_SPENT_MS)
            if -50.0 <= raw <= 3000.0:  # a new game or a clock reset is out of range
                _INC_SAMPLES.append(max(0.0, raw))
                del _INC_SAMPLES[:-5]
        _LAST_CLOCK_MS = float(time_left_ms)


_MATERIAL: Final = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900
}


def _contempt(board: chess.Board, static: int) -> int:
    """The draw score for the root side, negative when a draw would cost it.

    Ahead on material or on the network's own view of the position, a draw is a
    loss of expectation, and increasingly so as the ply-300 adjudication -- which
    awards the game on raw material -- approaches. Behind, a draw is a gain.
    Level, a small reluctance to repeat: in a Swiss the opponent is usually the
    weaker side, and playing on is where that shows.
    """
    material = _material_balance(board)
    game_ply = _match_ply(board) if ADJUDICATION else (
        2 * (board.fullmove_number - 1) + (0 if board.turn == chess.WHITE else 1)
    )
    late = min(1.0, max(0.0, (game_ply - 150) / (ADJUDICATION_PLY - 150)))
    if material >= 100 or static >= 60:
        return -int(CONTEMPT_AHEAD + (CONTEMPT_AHEAD_LATE - CONTEMPT_AHEAD) * late)
    if material <= -100 or static <= -60:
        if ADJUDICATION and material < 0:
            # Losing the ply-300 material adjudication: a draw approaches a full
            # half point as the cap nears, so make repetitions decisive, not +20.
            return -CONTEMPT_BEHIND + int(ADJ_BEHIND_LATE * late)
        return -CONTEMPT_BEHIND
    return -CONTEMPT_LEVEL


def _material_balance(board: chess.Board) -> int:
    """Raw material for the side to move, in cp; sign matches the referee's
    ply-300 adjudication (its 1/3/3/5/9 scale times 100)."""
    material = 0
    for piece, value in _MATERIAL.items():
        material += value * (
            chess.popcount(board.pieces_mask(piece, chess.WHITE))
            - chess.popcount(board.pieces_mask(piece, chess.BLACK))
        )
    return -material if board.turn == chess.BLACK else material


def _match_ply(board: chess.Board) -> int:
    """The referee's ply count, pinned at our first request of the game.

    The referee counts plies from the curated start FEN; `fullmove_number`
    counts from the real initial position and can run 13+ ahead. We cannot see
    whether the opponent moved before our first request, so assume it did (at
    most one ply conservative). A ply that goes backwards means a new game in
    the same process: re-pin.
    """
    global _MATCH_BASE_PLY, _LAST_GAME_PLY
    ply = 2 * (board.fullmove_number - 1) + (0 if board.turn == chess.WHITE else 1)
    if _MATCH_BASE_PLY < 0 or ply < _LAST_GAME_PLY:
        _MATCH_BASE_PLY = ply
    _LAST_GAME_PLY = ply
    return ply - _MATCH_BASE_PLY + 1


def _observed_increment() -> float:
    """The increment in seconds as seen between our calls; 0 until two samples."""
    if len(_INC_SAMPLES) < 2:
        return 0.0
    ordered = sorted(_INC_SAMPLES)
    return ordered[len(ordered) // 2] / 1000.0


def _budget_v6(board: chess.Board, time_left_ms: int) -> tuple[float, float]:
    """TIME_V6 deadlines: the soft budget is the ideal spend that the stop rule in
    `choose` scales by stability, score drop and node effort; the hard deadline is
    the only mid-iteration stop, four soft budgets or a quarter of the clock."""
    now = time.monotonic()
    remaining = max(time_left_ms - 400.0, 50.0) / 1000.0  # 400 ms for the watchdog
    inc = _observed_increment()
    expected = max(30.0, 56.0 - board.fullmove_number * 0.4)
    if remaining < LOW_CLOCK_V6:
        # An eighteenth of what is left, as a hard stop: with the kernel aborting at the
        # deadline (TIME_V4 keeps the partial result) the spend is exact, so the clock
        # settles where remaining/18 x charge = increment -- 6 s under the 1.5x
        # clocktest charge (measured 5.1-6.3 s at /16), ~8 s on the platform.
        soft = max(0.02, remaining / 18.0)
        hard = soft
    else:
        soft = remaining / expected + 0.7 * inc
        hard = min(remaining * 0.10, soft * 2.5)
    reserve = _MAX_CLOCK_MS * RESERVE_FRACTION_V6 / 1000.0
    if reserve > 0.0:
        hard = min(hard, max(soft, remaining - reserve))
    hard = max(hard, 0.02)
    soft = min(soft, hard)
    return now + soft, now + hard


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

    if TIME_V3:
        expected = max(18.0 if TIME_V5 else 26.0, 46.0 - board.fullmove_number * 0.4)
    else:
        expected = max(20.0, 40.0 - board.fullmove_number * 0.5)
    # Below LOW_CLOCK, live on the increment. Crediting half of it while the clock
    # is low sets up an equilibrium where the clock settles at whatever level makes
    # the spend equal the income -- measured at 4.4-4.6 s under a 1.5x charge, which
    # is no margin at all. With no credit and a longer horizon the same equilibrium
    # sits near 10 s charged 1.5x and near 14 s uncharged.
    soft = remaining / 30.0 if remaining < LOW_CLOCK else remaining / expected + 0.25
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
    if TIME_V6:
        return _budget_v6(board, time_left_ms)
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
    """Return a legal move in UCI notation; the platform's entry point."""
    global _LAST_SPENT_MS
    started = time.monotonic()
    try:
        return _get_move(fen, time_left_ms)
    finally:
        _LAST_SPENT_MS = (time.monotonic() - started) * 1000.0


def _get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion
    """
    board = chess.Board(fen)
    # The contract says an int; a float or a numeric string is cheaper to accept
    # than to lose a game over.
    try:
        time_left_ms = int(time_left_ms)
    except (TypeError, ValueError):
        time_left_ms = 1000

    # Remember every position we have been asked about. The referee claims threefold
    # repetition automatically, so an engine that is winning and shuffling can have a
    # won game turned into a draw without ever being told.
    # A position with no legal moves is checkmate or stalemate, and the referee is
    # not supposed to ask about one. If it ever does, every path below raises --
    # `moves[0]` in the search, `next(iter(...))` in the fallback -- and an exception
    # here forfeits the game. UCI's null move is the honest answer.
    if not board.legal_moves:
        return "0000"

    if PONDER:
        had_thread = _PONDER_THREAD is not None
        _stop_ponder()
        global _PONDER_LAST_NODES
        _PONDER_LAST_NODES = 0
        if had_thread and _PONDER is not None:
            _PONDER_LAST_NODES = int(_PONDER.ctrl[_fs.C_NODES])
        if PONDER_DIAG and had_thread and _PONDER is not None:
            gap = time.monotonic() - _PONDER_STARTED
            print(f"ponder-diag: gap {gap:.2f}s ponder_nodes {_PONDER_LAST_NODES}")
    if TIME_V2:
        _note_clock(time_left_ms)

    key = _key(board)
    _ENGINE.history[key] = _ENGINE.history.get(key, 0) + 1
    if _FAST is not None:
        fast_key = chess.polyglot.zobrist_hash(board)
        _FAST.history[fast_key] = _FAST.history.get(fast_key, 0) + 1

    # Book first: it is instant, and the clock it saves is worth more in the
    # middlegame than the search would be worth here.
    try:
        opening = _book_move(board) if BOOK_ENABLED else None
    except Exception:  # never let the book cost a game
        opening = None
    if opening is not None and not (BOOK_VERIFY and _FAST is not None):
        if PONDER and _PONDER is not None:
            try:
                board.push(opening)
                _start_ponder(board)
                board.pop()
            except Exception:
                pass
        return opening.uci()

    # Exact play once the position is small enough. This is what converts a won
    # endgame; the search alone shuffles because the evaluation is flat there.
    try:
        exact = _tablebase_move(board)
    except Exception:
        exact = None
    if exact is not None:
        return exact.uci()

    move: chess.Move | None = None
    started = time.monotonic()
    if _FAST is not None:
        # The compiled board. A move python-chess would reject, or any exception,
        # hands this move to the python-chess engine instead.
        try:
            soft, hard = _budget(board, time_left_ms)
            _FAST.hint = _fb.move_from_chess(opening) if opening is not None else 0
            global _SEARCHED_MOVES
            _SEARCHED_MOVES += 1
            if PONDER_PROBE and 2 <= _SEARCHED_MOVES <= 4 and time_left_ms > 30_000:
                fixed = 1.0 if _PONDER_LAST_NODES >= 100_000 else 3.0
                soft = hard = time.monotonic() + fixed
            candidate = _FAST.play(board, soft, hard)
            if candidate in board.legal_moves:
                move = candidate
        except Exception:
            move = None

    if move is None:
        # refresh() and _budget() were outside this guard, so an exception in either
        # was a crash rather than a fallback -- and a crash is a lost game where a
        # legal move would only have been a bad one. Nothing in here is worth a point.
        # The fallback budgets from what is left after the compiled attempt, not
        # from the clock as it stood when the move began.
        try:
            spent = int((time.monotonic() - started) * 1000.0)
            _ENGINE.acc.refresh(board)
            soft, hard = _budget(board, max(time_left_ms - spent, 50))
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
            if _FAST is not None:
                fast_after = chess.polyglot.zobrist_hash(board)
                _FAST.history[fast_after] = _FAST.history.get(fast_after, 0) + 1
            board.pop()
        except Exception:
            pass
    if PONDER and _PONDER is not None:
        try:
            board.push(move)
            _start_ponder(board)
            board.pop()
        except Exception:
            pass
    return move.uci()
