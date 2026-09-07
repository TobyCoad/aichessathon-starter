"""Does the net RANK moves right? A corpus-wide static-eval ranking audit, no games.

For every position in a corpus (default: the 60-position blunder corpus), score every
legal child with the agent's static eval and with Stockfish at a fixed depth, then report
per piece-count band:

  bias      mean (static - stockfish) at the root, root POV       -> systematic optimism
  |err|     mean |static - stockfish| at the root                  -> accuracy
  top-loss  Stockfish loss of the child the static ranks FIRST     -> would a 1-ply search blunder?
  top3-hit  how often Stockfish's best move is in the static's top 3
  spearman  rank correlation between static and Stockfish over the children

The search compares static evals against fixed margins, so `bias` and `top-loss` are the
two numbers that predict pruning damage and one-ply blunders respectively.

  .venv/Scripts/python.exe -m testing.eval_rank --agent . --depth 12
  .venv/Scripts/python.exe -m testing.eval_rank --agent opponents/v95net --depth 12
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from testing.eval_probe import load_agent, static_eval

ROOT = Path(__file__).resolve().parents[1]
BANDS = ((2, 8), (9, 12), (13, 16), (17, 20), (21, 24), (25, 32))


def spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return 0.0
    ra = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: a[i]))}
    rb = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: b[i]))}
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=Path, default=Path("."))
    ap.add_argument("--corpus", type=Path, default=ROOT / "overnight/eval/fd60-scored.jsonl")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="JSON lines of per-position rows")
    args = ap.parse_args()
    lines = args.corpus.read_text(encoding="utf-8").splitlines()
    fens = [json.loads(line)["fen"] for line in lines if line.startswith("{")]
    if args.limit:
        fens = fens[: args.limit]
    mod = load_agent(args.agent)
    engine = mod._FAST if getattr(mod, "_FAST", None) is not None else mod.FastEngine()
    sf_exe = ROOT / "engines/stockfish/stockfish-windows-x86-64-avx2.exe"
    sf = chess.engine.SimpleEngine.popen_uci(str(sf_exe))
    sf.configure({"Threads": 1, "Hash": 128})
    rows = []
    for fen in fens:
        board = chess.Board(fen)
        pieces = chess.popcount(board.occupied)
        root_static = static_eval(mod, engine, board)
        info = sf.analyse(board, chess.engine.Limit(depth=args.depth), game=object())
        root_sf = info["score"].pov(board.turn).score(mate_score=3000)
        statics, sfs, moves = [], [], []
        for move in board.legal_moves:
            board.push(move)
            statics.append(-static_eval(mod, engine, board))
            cinfo = sf.analyse(board, chess.engine.Limit(depth=args.depth - 1), game=object())
            sfs.append(-cinfo["score"].pov(board.turn).score(mate_score=3000))
            board.pop()
            moves.append(move.uci())
        if not moves:
            continue
        best_sf = max(sfs)
        top_static = max(range(len(moves)), key=lambda i: statics[i])
        order = sorted(range(len(moves)), key=lambda i: -statics[i])[:3]
        rows.append({
            "fen": fen, "pieces": pieces, "root_static": root_static, "root_sf": root_sf,
            "top_loss": best_sf - sfs[top_static], "top3_hit": int(sfs.index(best_sf) in order),
            "spearman": spearman(statics, sfs), "n": len(moves),
        })
        print(
            f"  {fen[:50]:50s} pcs {pieces:2d} bias {root_static - root_sf:+5d}"
            f" top-loss {best_sf - sfs[top_static]:4d} rho {spearman(statics, sfs):+.2f}",
            flush=True,
        )
    sf.quit()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"\n{args.agent}: {len(rows)} positions at depth {args.depth}")
    print("  band      n   bias   |err|  top-loss  top3-hit  spearman")
    for lo, hi in BANDS + ((2, 32),):
        sel = [r for r in rows if lo <= r["pieces"] <= hi]
        if not sel:
            continue
        bias = statistics.mean(r["root_static"] - r["root_sf"] for r in sel)
        err = statistics.mean(abs(r["root_static"] - r["root_sf"]) for r in sel)
        tl = statistics.mean(r["top_loss"] for r in sel)
        hit = statistics.mean(r["top3_hit"] for r in sel)
        rho = statistics.mean(r["spearman"] for r in sel)
        label = "all" if (lo, hi) == (2, 32) else f"{lo}-{hi}"
        print(f"  {label:7s} {len(sel):3d}  {bias:+6.0f}  {err:6.0f}  {tl:8.0f}  {hit:8.2f}  {rho:+8.2f}")


if __name__ == "__main__":
    main()
