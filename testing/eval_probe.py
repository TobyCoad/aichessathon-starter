"""Per-move static-eval probe: is the net wrong, or is the search wrong?

For each position, evaluate every legal move's child with the agent's static eval (the
number the search actually compares against its margins) and with Stockfish at a fixed
depth, both from the root side's point of view, and print them side by side ranked by
Stockfish. A net that ranks the best move well below a bad one is an evaluation defect;
a net that ranks them right while the engine still plays the bad one is a search defect.

  .venv/Scripts/python.exe -m testing.eval_probe --agent . --fen "<fen>" [--fen ...]
  .venv/Scripts/python.exe -m testing.eval_probe --agent opponents/v95net --fens file.txt

Static evals are the agent's `FastEngine.evaluate()` after `prepare()`, so they include
whatever per-piece-count scaling the tree applies. Stockfish scores are UCI centipawns.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[1]


def load_agent(directory: Path):
    directory = directory.resolve()
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("probe_agent", directory / "agent.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["probe_agent"] = mod
    spec.loader.exec_module(mod)
    return mod


def static_eval(mod, engine, board: chess.Board) -> int:
    """Side-to-move centipawns as the search sees them."""
    engine.prepare(board, 0)
    return int(engine.evaluate())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=Path, default=Path("."))
    ap.add_argument("--fen", action="append", default=[])
    ap.add_argument("--fens", type=Path, default=None, help="file with one FEN per line")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--top", type=int, default=8, help="moves to print per position")
    ap.add_argument("--json", type=Path, default=None, help="also write rows as JSON lines")
    args = ap.parse_args()
    fens = list(args.fen)
    if args.fens:
        fens += [x.strip() for x in args.fens.read_text(encoding="utf-8").splitlines() if x.strip()]

    mod = load_agent(args.agent)
    engine = mod._FAST if getattr(mod, "_FAST", None) is not None else mod.FastEngine()
    sf_path = ROOT / "engines/stockfish/stockfish-windows-x86-64-avx2.exe"
    sf = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    sf.configure({"Threads": 1, "Hash": 128})
    out = open(args.json, "a", encoding="utf-8") if args.json else None
    scale = getattr(mod, "PIECE_SCALE", None)
    for fen in fens:
        board = chess.Board(fen)
        pieces = chess.popcount(board.occupied)
        root_static = static_eval(mod, engine, board)
        info = sf.analyse(board, chess.engine.Limit(depth=args.depth), game=object())
        root_sf = info["score"].pov(board.turn).score(mate_score=3000)
        rows = []
        for move in board.legal_moves:
            board.push(move)
            child_static = -static_eval(mod, engine, board)
            cinfo = sf.analyse(board, chess.engine.Limit(depth=args.depth - 1), game=object())
            child_sf = -cinfo["score"].pov(board.turn).score(mate_score=3000)
            board.pop()
            rows.append((child_sf, child_static, move.uci()))
        rows.sort(reverse=True)
        ps = f" piece_scale {float(scale[pieces]):.2f}" if scale is not None else ""
        print(f"\n{fen}\n  pieces {pieces}  root: static {root_static:+d}  sf{args.depth} {root_sf:+d}{ps}")
        print("  move     sf_child  static_child   (root POV; sorted by Stockfish)")
        best_sf = rows[0][0]
        best_static = max(r[1] for r in rows)
        for child_sf, child_static, uci in rows[: args.top]:
            flag = ""
            if child_static == best_static and child_sf < best_sf - 100:
                flag = "  <-- net's favourite, Stockfish says it loses"
            print(f"  {uci:6s}  {child_sf:+7d}      {child_static:+7d}{flag}")
        net_choice = max(rows, key=lambda r: r[1])
        print(
            f"  net's top child: {net_choice[2]} (static {net_choice[1]:+d}, sf {net_choice[0]:+d});"
            f" best by sf: {rows[0][2]} ({rows[0][0]:+d}); gap {rows[0][0] - net_choice[0]:+d} cp"
        )
        if out:
            out.write(json.dumps({"fen": fen, "agent": str(args.agent), "root_static": root_static, "root_sf": root_sf, "rows": rows}) + "\n")
    sf.quit()


if __name__ == "__main__":
    main()
