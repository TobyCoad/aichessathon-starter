"""Pair two binpack heads position-by-position and compare their score labels.

Used to measure how the Stockfish-rescored / BT4-relabelled variants of the SAME
month differ from the original Leela labels, in internal units, without needing
Stockfish locally.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from training.binpack_decode import VALUE_NONE, iter_chunk

LIMIT = 120_000


def chunks(path: Path):
    data = path.read_bytes()
    at = 0
    while at + 8 <= len(data):
        if data[at : at + 4] != b"BINP":
            return
        size = int.from_bytes(data[at + 4 : at + 8], "little")
        if at + 8 + size > len(data):
            return
        yield data[at + 8 : at + 8 + size]
        at += 8 + size


def stream(path: Path):
    n = 0
    for chunk in chunks(path):
        try:
            for board, _move, score, ply, _result in iter_chunk(chunk):
                yield board.epd(), score, ply
                n += 1
                if n >= LIMIT:
                    return
        except Exception:  # noqa: BLE001
            return


def main(a: Path, b: Path) -> None:
    pairs = []
    mismatch = 0
    for (epd_a, sa, _pa), (epd_b, sb, _pb) in zip(stream(a), stream(b), strict=False):
        if epd_a != epd_b:
            mismatch += 1
            if mismatch > 20:
                break
            continue
        if abs(sa) >= VALUE_NONE or abs(sb) >= VALUE_NONE:
            continue
        pairs.append((sa, sb))
    print(f"== {a.name}  vs  {b.name}")
    print(f"  paired={len(pairs)} epd_mismatches={mismatch}")
    if not pairs:
        return
    ax = [p[0] for p in pairs]
    bx = [p[1] for p in pairs]
    print(f"  mean|A|={statistics.mean(map(abs, ax)):.1f}  mean|B|={statistics.mean(map(abs, bx)):.1f}")
    med_a = statistics.median(map(abs, ax))
    med_b = statistics.median(map(abs, bx))
    print(f"  median|A|={med_a:.0f}  median|B|={med_b:.0f}  ratio B/A={med_b / max(med_a, 1):.3f}")
    # least-squares slope through the origin, B = k*A
    num = sum(x * y for x, y in pairs)
    den = sum(x * x for x in ax)
    print(f"  regression slope B = k*A : k={num / den:.4f}")
    ma = statistics.mean(ax)
    mb = statistics.mean(bx)
    va = sum((x - ma) ** 2 for x in ax) ** 0.5
    vb = sum((y - mb) ** 2 for y in bx) ** 0.5
    cov = sum((x - ma) * (y - mb) for x, y in pairs)
    print(f"  pearson r={cov / (va * vb):.4f}")
    agree = sum(1 for x, y in pairs if (x > 0) == (y > 0))
    print(f"  sign agreement={agree / len(pairs):.3f}")
    big = [(x, y) for x, y in pairs if abs(x) > 1145]
    if big:
        print(
            f"  on |A|>1145 units (~300cp): n={len(big)} "
            f"mean|A|={statistics.mean(abs(x) for x, _ in big):.0f} "
            f"mean|B|={statistics.mean(abs(y) for _, y in big):.0f}"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
