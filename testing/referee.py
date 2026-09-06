"""A referee that plays the platform's game, starting from a given position.

`harness/` mirrors the platform and must not be edited -- but our copy of it is
STALE, and this file deliberately diverges from it (6 Sep, verified twice against
the canonical rules the harness names, https://aichessathon.com/docs/rules.md):

  * the ply cap is 600, not `harness.rules.PLY_CAP`'s 300, and the opening
    position counts toward those 600;
  * a game still running at the cap is a DRAW. Material is never considered --
    the harness copy's raw-material adjudication is not a platform rule, and
    every verdict measured under it rewarded a premise the platform does not
    hold (see NOTES.md, "THE PLY CAP IS 600");
  * the init budget is 90 s, not `harness.rules.INIT_BUDGET_S`'s 60 s. The
    platform-init safety margin is the release gate instead (a clean-unzip cold
    import under 45 s here, against their ~1.8x slower box); a 60 s referee
    budget only manufactured init failures under gauntlet load.

Everything else -- the clock accounting, legality check, draw claiming and PGN
output -- is the harness's, so results stay comparable to a real game.
"""

import time
from dataclasses import dataclass
from typing import Literal

import chess
import chess.pgn

from harness.sandbox import Agent, AgentFailure

# The platform's numbers, not the stale ones in `harness/rules.py`.
PLATFORM_PLY_CAP = 600
PLATFORM_INIT_BUDGET_S = 90.0
RESULT_HEADERS = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2", "void": "*"}
FAILED_TERMINATIONS = frozenset({"crash", "illegal", "flag", "init", "both_failed"})

Result = Literal["white", "black", "draw", "void"]
Decision = Literal["white", "black", "draw"]


@dataclass(frozen=True)
class Outcome:
    result: Result
    termination: str
    pgn: str
    plies: int


def play_match(
    white: Agent,
    black: Agent,
    base_ms: int,
    increment_ms: int,
    ply_cap: int = PLATFORM_PLY_CAP,
    start_fen: str | None = None,
) -> Outcome:
    try:
        return _play(white, black, base_ms, increment_ms, ply_cap, start_fen)
    finally:
        white.stop()
        black.stop()


def _play(
    white: Agent,
    black: Agent,
    base_ms: int,
    increment_ms: int,
    ply_cap: int,
    start_fen: str | None,
) -> Outcome:
    board = chess.Board() if start_fen is None else chess.Board(start_fen)
    root = board.copy(stack=False)
    agents = {chess.WHITE: white, chess.BLACK: black}

    white_failure = _start(white)
    black_failure = _start(black)
    if white_failure is not None and black_failure is not None:
        return _outcome(board, root, "void", "both_failed")
    if white_failure is not None:
        return _outcome(board, root, "black", white_failure)
    if black_failure is not None:
        return _outcome(board, root, "white", black_failure)

    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}

    while True:
        finish = board.outcome(claim_draw=True)
        if finish is not None:
            return _outcome(board, root, _decide(finish), finish.termination.name.lower())
        if len(board.move_stack) >= ply_cap:
            return _outcome(board, root, "draw", "ply_cap")

        mover = board.turn
        started_at = time.monotonic()
        try:
            uci = agents[mover].move(board.fen(), int(clock[mover]))
        except AgentFailure as failure:
            return _outcome(board, root, _opponent_wins(mover), failure.reason)
        clock[mover] -= (time.monotonic() - started_at) * 1000.0
        if clock[mover] < 0:
            return _outcome(board, root, _opponent_wins(mover), "flag")

        move = _legal_move(board, uci)
        if move is None:
            return _outcome(board, root, _opponent_wins(mover), "illegal")
        board.push(move)
        clock[mover] += increment_ms


def _start(agent: Agent) -> str | None:
    try:
        agent.start(PLATFORM_INIT_BUDGET_S)
    except AgentFailure as failure:
        return failure.reason
    return None


def _legal_move(board: chess.Board, uci: str) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(uci)
    except chess.InvalidMoveError:
        return None
    return move if move in board.legal_moves else None


def _opponent_wins(mover: chess.Color) -> Decision:
    return "black" if mover == chess.WHITE else "white"


def _decide(finish: chess.Outcome) -> Decision:
    if finish.winner is None:
        return "draw"
    return "white" if finish.winner == chess.WHITE else "black"


def _outcome(board: chess.Board, root: chess.Board, result: Result, termination: str) -> Outcome:
    game = chess.pgn.Game.from_board(board)
    if root.fen() != chess.STARTING_FEN:
        game.headers["FEN"] = root.fen()
        game.headers["SetUp"] = "1"
    game.headers["Result"] = RESULT_HEADERS[result]
    game.headers["Termination"] = termination
    return Outcome(
        result=result, termination=termination, pgn=str(game), plies=len(board.move_stack)
    )
