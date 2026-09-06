"""Decode the first complete BINP chunks of a probe head and report stats.

Read-only research probe: verifies that a candidate binpack decodes with our
training/binpack_decode.py reader, and reports the internal-unit score
distribution so the 0.262 cp/unit scale can be sanity-checked before any
full download.

Regenerate the 3 MB heads the 6 Sep survey used (see ../data_sources.md):

    R="Range: bytes=0-3145727"
    B=https://huggingface.co/datasets
    curl -sL -H "$R" -o resc_leela.head \
      $B/vondele/rescored/resolve/main/test80-2023-06-jun-2tb7p.min-v2-orig_Leela.binpack
    curl -sL -H "$R" -o resc_sf20k.head \
      $B/vondele/rescored/resolve/main/test80-2023-06-jun-2tb7p.min-v2-rescore_SF_n20000.binpack
    curl -sL -H "$R" -o resc_sf5k.head \
      $B/vondele/rescored/resolve/main/test80-2023-06-jun-2tb7p.min-v2-rescore_SF_n5000.binpack
    curl -sL -H "$R" -o bt4_feb24.head \
      $B/xushawn/test80-bt4-relabel/resolve/main/test80-2024-02-feb-2tb7p.min-v2.v6.relabel.binpack
    for f in nodes5000pv2_UHO wrongIsRight_nodes5000pv2 dfrc_n5000; do
      curl -sL -H "$R" -o $f.head \
        $B/official-stockfish/master-binpacks/resolve/main/$f.binpack
    done

and the head of our own file (it is .zst, so stream-decompress instead):

    python -c "import zstandard as z; \
      open('local_feb24.head','wb').write(z.ZstdDecompressor().stream_reader( \
      open('data/sf/test80-2024-02-feb-2tb7p.min-v2.v6.binpack.zst','rb')).read(3145728))"
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from training.binpack_decode import VALUE_NONE, iter_chunk


def chunks(path: Path, limit: int = 2):
    data = path.read_bytes()
    at = 0
    n = 0
    while at + 8 <= len(data) and n < limit:
        if data[at : at + 4] != b"BINP":
            raise SystemExit(f"{path.name}: bad magic at {at}: {data[at:at+4]!r}")
        size = int.from_bytes(data[at + 4 : at + 8], "little")
        if at + 8 + size > len(data):
            break
        yield data[at + 8 : at + 8 + size]
        at += 8 + size
        n += 1


def report(path: Path) -> None:
    seen = illegal = bad_board = marked = 0
    scores: list[int] = []
    plies: list[int] = []
    pieces: list[int] = []
    first_fen = None
    for chunk in chunks(path):
        try:
            for board, move, score, ply, result in iter_chunk(chunk):
                seen += 1
                if first_fen is None:
                    first_fen = board.fen()
                if board.status() & ~chess.STATUS_BAD_CASTLING_RIGHTS:
                    bad_board += 1
                if move is not None and not board.is_pseudo_legal(move):
                    illegal += 1
                if abs(score) >= VALUE_NONE:
                    marked += 1
                else:
                    scores.append(score)
                plies.append(ply)
                pieces.append(bin(board.occupied).count("1"))
        except Exception as exc:  # noqa: BLE001
            print(f"  DECODE ERROR after {seen}: {type(exc).__name__}: {exc}")
            break
    absolute = sorted(abs(s) for s in scores)

    def q(p: float) -> int:
        return absolute[int(p * (len(absolute) - 1))] if absolute else -1

    print(f"== {path.name}")
    print(f"  entries={seen} bad_boards={bad_board} illegal_moves={illegal} skip_marked={marked}")
    print(f"  first_fen={first_fen}")
    if absolute:
        print(
            f"  |score| units: median={q(0.5)} p75={q(0.75)} p90={q(0.9)} p99={q(0.99)} "
            f"max={absolute[-1]} mean={statistics.mean(absolute):.1f}"
        )
        big = sum(1 for s in absolute if s > 1145)  # 300 cp at 0.262 cp/unit
        print(f"  frac |score| > 300cp-equivalent (1145 units): {big / len(absolute):.3f}")
    print(f"  ply: median={statistics.median(plies):.0f} min={min(plies)} max={max(plies)}")
    print(
        f"  pieces: median={statistics.median(pieces):.0f} "
        f"frac<=16={sum(1 for p in pieces if p <= 16) / len(pieces):.3f} "
        f"frac<=10={sum(1 for p in pieces if p <= 10) / len(pieces):.3f}"
    )


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        report(Path(arg))
