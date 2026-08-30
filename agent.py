"""The submission entrypoint. The platform imports this file and calls get_move.

An iterative-deepening alpha-beta searcher with a transposition table, MVV-LVA
move ordering, quiescence search and a tapered piece-square evaluation.

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

Note for the competition rules: the event requires that a learned model materially
drives move selection. This evaluation is hand-crafted, so this file is a baseline
and a fallback, not a final submission. The learned evaluation replaces `evaluate`.
"""

import time
from typing import Final, cast

import chess

# --------------------------------------------------------------------------------
# Evaluation tables
# --------------------------------------------------------------------------------
# PeSTO's tapered piece-square tables (Ronald Friederich, rofChade), the standard
# public set documented at chessprogramming.org/PeSTO%27s_Evaluation_Function.
# A tapered, tuned piece-square evaluation is worth around +250 Elo over plain
# material, and is the one evaluation feature that carries fully into games against
# unfamiliar opponents rather than only into self-play.
#
# Tables are written with a8 first, so a white piece on `square` indexes
# `table[square ^ 56]` and a black piece indexes `table[square]`.

MG_VALUE: Final = (82, 337, 365, 477, 1025, 0)
EG_VALUE: Final = (94, 281, 297, 512, 936, 0)
PHASE_WEIGHT: Final = (0, 1, 1, 2, 4, 0)
TOTAL_PHASE: Final = 24

MG_PAWN: Final = (
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
)  # fmt: skip
EG_PAWN: Final = (
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
)  # fmt: skip
MG_KNIGHT: Final = (
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
)  # fmt: skip
EG_KNIGHT: Final = (
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
)  # fmt: skip
MG_BISHOP: Final = (
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
)  # fmt: skip
EG_BISHOP: Final = (
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
)  # fmt: skip
MG_ROOK: Final = (
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
)  # fmt: skip
EG_ROOK: Final = (
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
)  # fmt: skip
MG_QUEEN: Final = (
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
)  # fmt: skip
EG_QUEEN: Final = (
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
)  # fmt: skip
MG_KING: Final = (
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
)  # fmt: skip
EG_KING: Final = (
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
)  # fmt: skip

_MG_TABLES: Final = (MG_PAWN, MG_KNIGHT, MG_BISHOP, MG_ROOK, MG_QUEEN, MG_KING)
_EG_TABLES: Final = (EG_PAWN, EG_KNIGHT, EG_BISHOP, EG_ROOK, EG_QUEEN, EG_KING)


def _build(tables: tuple[tuple[int, ...], ...], values: tuple[int, ...]) -> tuple[list[int], ...]:
    """Fold material value into each square, once, at import time.

    Returns twelve 64-entry lists indexed [piece_type - 1 + 6 * is_black][square],
    so evaluation is one list lookup per piece with no arithmetic on the hot path.
    """
    built: list[list[int]] = []
    for piece in range(6):
        built.append([values[piece] + tables[piece][square ^ 56] for square in range(64)])
    for piece in range(6):
        built.append([values[piece] + tables[piece][square] for square in range(64)])
    return tuple(built)


MG_TABLE: Final = _build(_MG_TABLES, MG_VALUE)
EG_TABLE: Final = _build(_EG_TABLES, EG_VALUE)

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


def _key(board: chess.Board) -> int:
    """The position's transposition key.

    `_transposition_key` is private, but it costs 0.46 us where
    `chess.polyglot.zobrist_hash` costs 12 us and `board.fen()` 23 us. At tens of
    thousands of nodes per second nothing else is affordable. It is typed as
    Hashable and is in fact an int.
    """
    return cast(int, board._transposition_key())


class Timeout(Exception):
    """Raised to unwind the search when the hard time limit passes."""


class Engine:
    """Search state that persists for the lifetime of one game.

    The platform starts one process per game and keeps it alive between moves, so
    the transposition table and the repetition history survive from one of our moves
    to the next. They do not survive to the next game, which is why this is built
    per process rather than at module scope.
    """

    __slots__ = ("deadline", "history", "nodes", "root_key", "table")

    def __init__(self) -> None:
        # key -> (depth, score, flag, best_move); flag 0 exact, 1 lower, 2 upper.
        self.table: dict[int, tuple[int, int, int, chess.Move | None]] = {}
        # Transposition keys of positions we have actually been asked about, so the
        # search can recognise a repetition without paying 150 us to ask python-chess.
        self.history: dict[int, int] = {}
        self.deadline = 0.0
        self.nodes = 0
        self.root_key = 0

    # -- evaluation ---------------------------------------------------------------

    def evaluate(self, board: chess.Board) -> int:
        """Tapered piece-square evaluation, from the side to move's point of view.

        Iterating the twelve piece bitboards with a bit-scan is roughly four times
        cheaper than `board.piece_map()`, which allocates a dict of Piece objects.
        """
        middlegame = 0
        endgame = 0
        phase = 0
        occupied_co = board.occupied_co
        for piece in range(1, 7):
            mask = board.pieces_mask(piece, chess.WHITE) & occupied_co[chess.WHITE]
            white_table_mg = MG_TABLE[piece - 1]
            white_table_eg = EG_TABLE[piece - 1]
            weight = PHASE_WEIGHT[piece - 1]
            while mask:
                square = (mask & -mask).bit_length() - 1
                middlegame += white_table_mg[square]
                endgame += white_table_eg[square]
                phase += weight
                mask &= mask - 1

            mask = board.pieces_mask(piece, chess.BLACK) & occupied_co[chess.BLACK]
            black_table_mg = MG_TABLE[piece + 5]
            black_table_eg = EG_TABLE[piece + 5]
            while mask:
                square = (mask & -mask).bit_length() - 1
                middlegame -= black_table_mg[square]
                endgame -= black_table_eg[square]
                phase += weight
                mask &= mask - 1

        if phase > TOTAL_PHASE:
            phase = TOTAL_PHASE  # early promotions can exceed the starting material
        blended = middlegame * phase + endgame * (TOTAL_PHASE - phase)
        # Truncate toward zero, not toward negative infinity: floor division would
        # make the evaluation asymmetric, scoring a position and its mirror image
        # differently by a point, which shows up as a phantom colour bias.
        score = blended // TOTAL_PHASE if blended >= 0 else -(-blended // TOTAL_PHASE)
        return score if board.turn == chess.WHITE else -score

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
            board.push(move)
            try:
                score = -self.quiesce(board, -beta, -alpha, depth + 1)
            finally:
                board.pop()
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
            stored_depth, stored_score, flag, best_move = stored
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
            board.push(move)
            try:
                score = -self.search(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
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
        self.table[key] = (depth, best_score, flag, best_move)
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
                    board.push(move)
                    try:
                        value = -self.search(board, depth - 1, -INFINITY, -alpha, 1)
                    finally:
                        board.pop()
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

    soft, hard = _budget(board, time_left_ms)
    try:
        move = _ENGINE.choose(board, soft, hard)
    except Exception:
        return next(iter(board.legal_moves)).uci()
    return move.uci()
