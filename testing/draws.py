"""Classify the drawn games in a PGN directory: who was ahead when they ended.

A repetition draw is only a leak when it was taken from a position the engine was
not losing. For every drawn game this reports the material balance and the
engine's own static evaluation of the final position, from our side, so a
contempt experiment can be judged on the draws it exists to prevent rather than
on the total.

Run: .venv\\Scripts\\python.exe -m testing.draws overnight/pgn/<match-dir> [--agent-name starter]
"""

import argparse
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

import chess
import chess.pgn

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def load_agent(directory: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("draws_agent", directory / "agent.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["draws_agent"] = module
    spec.loader.exec_module(module)
    return module


def material(board: chess.Board, colour: chess.Color) -> int:
    total = 0
    for piece, value in VALUES.items():
        total += value * (
            chess.popcount(board.pieces_mask(piece, colour))
            - chess.popcount(board.pieces_mask(piece, not colour))
        )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Who was ahead when the draws happened.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--agent", type=Path, default=Path("."), help="engine used to evaluate")
    parser.add_argument("--agent-name", default="starter", help="substring naming our side")
    arguments = parser.parse_args()

    agent = load_agent(arguments.agent.resolve())
    engine = agent.Engine()
    rows: list[tuple[str, str, int, int, int, str]] = []
    totals = {"games": 0, "draws": 0, "repetition": 0, "ahead": 0, "level": 0, "behind": 0}
    for path in sorted(arguments.directory.glob("*.pgn")):
        game = chess.pgn.read_game(io.StringIO(path.read_text(encoding="utf-8")))
        if game is None:
            continue
        totals["games"] += 1
        if game.headers.get("Result") != "1/2-1/2":
            continue
        totals["draws"] += 1
        termination = game.headers.get("Termination", "")
        if termination == "threefold_repetition":
            totals["repetition"] += 1
        us = (
            chess.WHITE
            if arguments.agent_name in game.headers.get("White", "").lower()
            else chess.BLACK
        )
        board = game.end().board()
        mat = material(board, us)
        engine.acc.refresh(board)
        static = engine.evaluate(board)
        if board.turn != us:
            static = -static
        verdict = (
            "ahead"
            if (mat >= 1 or static >= 60)
            else ("behind" if (mat <= -1 or static <= -60) else "level")
        )
        totals[verdict] += 1
        rows.append(
            (path.stem, termination, len(list(game.mainline_moves())), mat, static, verdict)
        )

    for stem, termination, plies, mat, static, verdict in rows:
        print(
            f"  {stem:<40} {termination:<22} {plies:4d} plies"
            f"  material {mat:+d}  eval {static:+5d}  {verdict}"
        )
    print(
        f"\n{arguments.directory.name}: {totals['games']} games, {totals['draws']} draws "
        f"({totals['repetition']} by repetition): ahead {totals['ahead']}, "
        f"level {totals['level']}, behind {totals['behind']}"
    )


if __name__ == "__main__":
    main()
