"""Endgame move quality: the instrument for changes the 8 s gauntlet cannot see.

Our move quality matches the field's in the middlegame and is 2-2.4x worse below
16 pieces (platform games, rounds 1-14), and the self-play gauntlet rarely reaches
those positions. This suite samples positions with 5-16 pieces from our own game
records, labels each with Stockfish, and scores an agent's move choice at a fixed
budget: mean centipawn loss and how often it plays the reference move.

  build   .venv\\Scripts\\python.exe -m testing.endgame_suite build --count 400 --depth 18
  run     .venv\\Scripts\\python.exe -m testing.endgame_suite run --agent overnight/challengers/060-v6 --seconds 2.5

The suite file caches every Stockfish result, so re-running an agent only pays for
the moves it chooses that have not been scored before.
"""

import argparse
import glob
import importlib.util
import io
import json
import random
import sys
import time
from pathlib import Path
from types import ModuleType

import chess
import chess.engine
import chess.pgn

STOCKFISH = Path("engines/stockfish/stockfish-windows-x86-64-avx2.exe")
SUITE = Path("overnight/eval/endgame_suite.json")
MATE_CP = 2000


def load_agent(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("suite_agent", path / "agent.py")
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cp(score: chess.engine.PovScore, colour: chess.Color) -> int:
    s = score.pov(colour)
    if s.is_mate():
        return MATE_CP if (s.mate() or 0) > 0 else -MATE_CP
    return max(-MATE_CP, min(MATE_CP, s.score() or 0))


def sample_positions(count: int, seed: int, lo: int, hi: int) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    candidates: list[str] = []
    for path in sorted(glob.glob("overnight/pgn/**/*.pgn", recursive=True)):
        try:
            game = chess.pgn.read_game(io.StringIO(Path(path).read_text(encoding="utf-8", errors="replace")))
        except Exception:
            continue
        if game is None:
            continue
        board = game.board()
        for index, move in enumerate(game.mainline_moves()):
            board.push(move)
            if index % 6:
                continue
            pieces = chess.popcount(board.occupied)
            if pieces < lo or pieces > hi or board.is_game_over() or board.is_check():
                continue
            key = board.epd()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(board.fen())
    rng.shuffle(candidates)
    return candidates[:count]


def build(arguments: argparse.Namespace) -> None:
    fens = sample_positions(arguments.count, arguments.seed, arguments.min_pieces, arguments.max_pieces)
    print(f"{len(fens)} positions sampled; labelling at depth {arguments.depth}")
    sf = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH.resolve()))
    sf.configure({"Threads": 1, "Hash": 64})
    suite = {"depth": arguments.depth, "positions": []}
    started = time.time()
    for index, fen in enumerate(fens, start=1):
        board = chess.Board(fen)
        info = sf.analyse(board, chess.engine.Limit(depth=arguments.depth))
        best = info["pv"][0].uci() if info.get("pv") else ""
        suite["positions"].append({
            "fen": fen, "best": best, "eval": cp(info["score"], board.turn), "after": {},
        })
        if index % 50 == 0:
            print(f"  {index}/{len(fens)}  {time.time() - started:.0f}s", flush=True)
    sf.quit()
    arguments.suite.parent.mkdir(parents=True, exist_ok=True)
    arguments.suite.write_text(json.dumps(suite, indent=1), encoding="utf-8")
    print(f"wrote {arguments.suite}")


def choose(agent: ModuleType, board: chess.Board, seconds: float) -> str:
    engine = agent.FastEngine()
    if hasattr(engine, "prepare"):
        engine.prepare(board, 0)
    else:
        engine.pos.load(board)
    now = time.monotonic()
    move = engine.choose(now + seconds, now + seconds * 1.5)
    return agent._fb.move_to_uci(move)


def run(arguments: argparse.Namespace) -> None:
    suite = json.loads(arguments.suite.read_text(encoding="utf-8"))
    depth = int(suite["depth"])
    agent = load_agent(arguments.agent.resolve())
    agent._TABLEBASE = None
    sf = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH.resolve()))
    sf.configure({"Threads": 1, "Hash": 64})
    losses: list[int] = []
    by_band: dict[str, list[int]] = {}
    matches = 0
    dirty = False
    started = time.time()
    for index, entry in enumerate(suite["positions"], start=1):
        board = chess.Board(entry["fen"])
        move = choose(agent, board, arguments.seconds)
        if move == entry["best"]:
            loss = 0
            matches += 1
        else:
            if move not in entry["after"]:
                child = board.copy()
                child.push(chess.Move.from_uci(move))
                if child.is_game_over():
                    outcome = child.outcome()
                    value = 0 if outcome is None or outcome.winner is None else (MATE_CP if outcome.winner == board.turn else -MATE_CP)
                else:
                    info = sf.analyse(child, chess.engine.Limit(depth=max(depth - 1, 1)))
                    value = cp(info["score"], board.turn)
                entry["after"][move] = value
                dirty = True
            loss = max(0, int(entry["eval"]) - int(entry["after"][move]))
        losses.append(loss)
        pieces = chess.popcount(board.occupied)
        band = "5-8" if pieces <= 8 else ("9-12" if pieces <= 12 else "13-16")
        by_band.setdefault(band, []).append(loss)
        if index % 50 == 0:
            print(f"  {index}/{len(suite['positions'])}  mean loss so far {sum(losses) / len(losses):.1f}  {time.time() - started:.0f}s", flush=True)
    sf.quit()
    if dirty:
        arguments.suite.write_text(json.dumps(suite, indent=1), encoding="utf-8")
    n = len(losses)
    print(
        f"\n{arguments.agent}: {n} positions at {arguments.seconds}s: mean loss {sum(losses) / n:.1f} cp, "
        f"median {sorted(losses)[n // 2]} cp, best move {matches / n:.1%}, "
        f">=100 cp on {sum(1 for x in losses if x >= 100) / n:.1%}"
    )
    for band in ("5-8", "9-12", "13-16"):
        if band in by_band:
            xs = by_band[band]
            print(f"  {band:>5} pieces: n={len(xs):>3} mean loss {sum(xs) / len(xs):.1f} cp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Endgame move-quality suite.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--count", type=int, default=400)
    b.add_argument("--depth", type=int, default=18)
    b.add_argument("--seed", type=int, default=3)
    b.add_argument("--min-pieces", type=int, default=5)
    b.add_argument("--max-pieces", type=int, default=16)
    b.add_argument("--suite", type=Path, default=SUITE)
    r = sub.add_parser("run")
    r.add_argument("--agent", type=Path, default=Path("."))
    r.add_argument("--seconds", type=float, default=2.5)
    r.add_argument("--suite", type=Path, default=SUITE)
    arguments = parser.parse_args()
    if arguments.command == "build":
        build(arguments)
    else:
        run(arguments)


if __name__ == "__main__":
    main()
