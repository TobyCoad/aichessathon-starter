"""Compare two `testing.bench --json` runs position by position: the over-pruning check.

A pruning switch that merely searches less looks fine on node count and only shows its
damage as SCORE SWINGS at fixed depth -- IMPROVING benched 0.506x nodes and was caught by
"bench pos 39 swings +534 -> +214", not by the count. So read the per-position deltas,
not the total.

  .venv/Scripts/python.exe -m testing.bench_diff overnight/eval/bench-tree-d8.json \
      overnight/eval/bench-s1-all-d8.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    base = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for other_path in sys.argv[2:]:
        other = json.loads(Path(other_path).read_text(encoding="utf-8"))
        by_fen = {r["fen"]: r for r in other}
        rows = [(b, by_fen[b["fen"]]) for b in base if b["fen"] in by_fen]
        deltas = [o["score"] - b["score"] for b, o in rows]
        absd = [abs(d) for d in deltas]
        changed = [(b, o) for b, o in rows if b["best"] != o["best"]]
        nodes_b = sum(b["nodes"] for b, _ in rows)
        nodes_o = sum(o["nodes"] for _, o in rows)
        print(f"\n{Path(other_path).name} vs {Path(sys.argv[1]).name}: {len(rows)} positions")
        print(f"  nodes {nodes_o:,} / {nodes_b:,} = {nodes_o / nodes_b:.3f}x")
        print(
            f"  |score delta| mean {statistics.mean(absd):.1f}  median {statistics.median(absd):.0f}"
            f"  max {max(absd)}   (>100 cp on {sum(d > 100 for d in absd)}, >300 on {sum(d > 300 for d in absd)})"
        )
        print(f"  best move changed on {len(changed)}/{len(rows)}")
        worst = sorted(rows, key=lambda r: -abs(r[1]["score"] - r[0]["score"]))[:4]
        for b, o in worst:
            flag = "" if b["best"] == o["best"] else f"  best {b['best']}->{o['best']}"
            print(f"    {b['score']:+5d} -> {o['score']:+5d}  {b['fen'][:48]}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
