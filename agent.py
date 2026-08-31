"""The submission entrypoint. The platform imports this file and calls get_move.

An iterative-deepening alpha-beta searcher with a transposition table, MVV-LVA
move ordering, quiescence search and a learned evaluation: a (768 -> 256)x2 -> 32 -> 1
network whose first layer is maintained incrementally across make and unmake.

The design follows from one measurement: in Python the move generator, not the
evaluation, is the bottleneck. `list(board.legal_moves)` costs ~25 us; a small
neural evaluation with an incremental accumulator costs ~3 us. So strength comes
from generating fewer moves -- ordering and pruning -- rather than from evaluating
faster. At the depths this reaches, one extra ply is worth roughly 150 Elo.

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

import time
from collections.abc import Hashable
from pathlib import Path
from typing import Final

import chess
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

MATE: Final = 30_000
MATE_THRESHOLD: Final = MATE - 1_000
INFINITY: Final = 1 << 20

# MVV-LVA: order captures by the value of the victim, tie-broken by the cheapness of
# the attacker. Measured at 8-29x fewer nodes depending on depth -- close to two free
# plies, and the single highest-value item in the whole engine per line of code.
_MVV: Final = (100, 320, 330, 500, 900, 20000)
CAPTURE_BONUS: Final = 1 << 20
PROMOTION_BONUS: Final = 1 << 19

# Delta pruning margin: a capture that cannot drag the static evaluation back to
# alpha even after winning a queen is not worth searching.
DELTA_MARGIN: Final = 975


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


class Accumulator:
    """Both perspectives' first-layer sums, maintained incrementally.

    A full refresh costs ~5-10 us; an incremental update is ~0.6 us, and the search
    does one per node. `push` stores the previous vectors so `pop` is a restore
    rather than a recompute.
    """

    __slots__ = ("black", "stack", "white")

    def __init__(self) -> None:
        self.white = B1.copy()
        self.black = B1.copy()
        self.stack: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]] = []

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
        self.stack.clear()

    def _add(self, square: int, piece_type: int, colour: chess.Color) -> None:
        self.white += W1[_feature(square, piece_type, colour, True)]
        self.black += W1[_feature(square, piece_type, colour, False)]

    def _remove(self, square: int, piece_type: int, colour: chess.Color) -> None:
        self.white -= W1[_feature(square, piece_type, colour, True)]
        self.black -= W1[_feature(square, piece_type, colour, False)]

    def push(self, board: chess.Board, move: chess.Move) -> None:
        """Apply a move's feature deltas. Must be called *before* board.push."""
        self.stack.append((self.white.copy(), self.black.copy()))
        mover = board.turn
        piece_type = board.piece_type_at(move.from_square)
        if piece_type is None:  # should not happen; leave the accumulator stale
            return

        self._remove(move.from_square, piece_type, mover)

        captured = board.piece_type_at(move.to_square)
        if captured is not None:
            self._remove(move.to_square, captured, not mover)
        elif piece_type == chess.PAWN and move.to_square == board.ep_square:
            # En passant: the captured pawn is not on the destination square.
            behind = move.to_square + (-8 if mover == chess.WHITE else 8)
            self._remove(behind, chess.PAWN, not mover)

        self._add(move.to_square, move.promotion or piece_type, mover)

        if piece_type == chess.KING and abs(move.to_square - move.from_square) == 2:
            rank = 0 if mover == chess.WHITE else 56
            if move.to_square > move.from_square:
                rook_from, rook_to = 7 + rank, 5 + rank
            else:
                rook_from, rook_to = 0 + rank, 3 + rank
            self._remove(rook_from, chess.ROOK, mover)
            self._add(rook_to, chess.ROOK, mover)

    def pop(self) -> None:
        self.white, self.black = self.stack.pop()

    def evaluate(self, turn: chess.Color) -> int:
        """Centipawns from the side to move's point of view."""
        if turn == chess.WHITE:
            own, opponent = self.white, self.black
        else:
            own, opponent = self.black, self.white
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
    if score > MATE_THRESHOLD:
        return score + ply
    if score < -MATE_THRESHOLD:
        return score - ply
    return score


def _from_table(score: int, ply: int) -> int:
    """Undo `_to_table`, putting a stored mate score back on this node's clock."""
    if score > MATE_THRESHOLD:
        return score - ply
    if score < -MATE_THRESHOLD:
        return score + ply
    return score


class Timeout(Exception):
    """Raised to unwind the search when the hard time limit passes."""


class Engine:
    """Search state that persists for the lifetime of one game.

    The platform starts one process per game and keeps it alive between moves, so
    the transposition table and the repetition history survive from one of our moves
    to the next. They do not survive to the next game, which is why this is built
    per process rather than at module scope.
    """

    __slots__ = ("acc", "deadline", "history", "nodes", "root_key", "table")

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

    # -- evaluation ---------------------------------------------------------------

    def evaluate(self, board: chess.Board) -> int:
        """The learned evaluation, from the incrementally maintained accumulator."""
        return self.acc.evaluate(board.turn)

    # -- move ordering ------------------------------------------------------------

    @staticmethod
    def _order(board: chess.Board, moves: list[chess.Move], best: chess.Move | None) -> None:
        """Sort moves in place: transposition move, then captures by MVV-LVA.

        A transposition table earns about +100 Elo used this way and only about +40
        used purely for cutoffs, so the previous iteration's best move going first
        matters more than the table's stored bounds.
        """
        piece_type_at = board.piece_type_at

        def score(move: chess.Move) -> int:
            if move == best:
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
            if move.promotion is not None:
                value += PROMOTION_BONUS + move.promotion * 100
            return value

        moves.sort(key=score, reverse=True)

    # -- quiescence ---------------------------------------------------------------

    def quiesce(self, board: chess.Board, alpha: int, beta: int, depth: int = 0) -> int:
        """Search captures only, so evaluation never lands mid-exchange.

        This is the largest single measured feature in the engine literature -- an
        independent test put it at +347 Elo -- because without it every leaf score is
        taken halfway through a trade and is simply wrong.
        """
        self.nodes += 1
        if not self.nodes & 1023 and time.monotonic() > self.deadline:
            raise Timeout

        standing = self.evaluate(board)
        if standing >= beta:
            return standing
        if standing > alpha:
            alpha = standing
        if depth >= 8:
            return standing

        captures = list(board.generate_legal_captures())
        self._order(board, captures, None)
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
                score = -self.quiesce(board, -beta, -alpha, depth + 1)
            finally:
                board.pop()
                self.acc.pop()
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return alpha

    # -- main search --------------------------------------------------------------

    def search(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        """Fail-soft negamax with alpha-beta and a transposition table."""
        self.nodes += 1
        if not self.nodes & 1023 and time.monotonic() > self.deadline:
            raise Timeout

        # A position repeated inside the search, or one already seen in the game, is
        # a draw we can claim -- the referee claims threefold automatically, so a
        # winning side that shuffles will have the win taken away from it.
        key = _key(board)
        if ply and (self.history.get(key, 0) or board.is_repetition(2)):
            return 0
        if ply and board.halfmove_clock >= 100:
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

        if depth <= 0:
            return self.quiesce(board, alpha, beta)

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if board.is_check() else 0

        self._order(board, moves, best_move)

        best_score = -INFINITY
        best_move = None
        for move in moves:
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
                        break

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

        for depth in range(1, 64):
            try:
                score = -INFINITY
                alpha = -INFINITY
                self._order(board, moves, best)
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
            # Starting a further iteration that cannot finish only wastes clock.
            if time.monotonic() > soft_limit:
                break

        return best


_ENGINE = Engine()


def _budget(board: chess.Board, time_left_ms: int) -> tuple[float, float]:
    """Return (soft, hard) monotonic deadlines for this move.

    A flag is a full point and it is the most common self-inflicted loss in this
    format, so the hard limit is deliberately conservative. The referee measures
    wall time and applies the increment only *after* the move, so the increment
    cannot be spent in advance; it is counted at a discount.
    """
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
    key = _key(board)
    _ENGINE.history[key] = _ENGINE.history.get(key, 0) + 1

    _ENGINE.acc.refresh(board)
    soft, hard = _budget(board, time_left_ms)
    try:
        move = _ENGINE.choose(board, soft, hard)
    except Exception:
        return next(iter(board.legal_moves)).uci()
    return move.uci()
