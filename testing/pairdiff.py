"""Difference two gauntlet logs pair by pair against one fixed external opponent.

Two builds measured separately against the same Stockfish rung are usually
compared as two percentages, which carries the opening-difficulty term twice.
`openings.pairs` is deterministic -- pair `i` is always `fens[i % 40]` played
from both colours -- so two runs of the same `--games` against the same opponent
play the IDENTICAL schedule, and the pairs can be joined on their index and
differenced. That removes the opening term and shrinks the standard error.

Usage:
    python -m testing.pairdiff a.gauntlet.log b.gauntlet.log

Reads the `[pair N S]` suffix that arena.py writes on every progress line.
Only pairs present in BOTH logs are used (a run stopped early simply
contributes fewer pairs). Prints the paired difference in score and the
equivalent Elo, with the unpaired comparison beside it so the gain from
pairing is visible rather than assumed.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PAIR = re.compile(r"\[pair (\d+) ([0-9.]+)\]")


def read_pairs(path: Path) -> dict[int, float]:
    """pair index -> pair score (0, 0.5 or 1) from one gauntlet log."""
    pairs: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PAIR.search(line)
        if match:
            # A repeated index would mean the schedule was not what we think it
            # is; keep the first and let the count mismatch show up below.
            pairs.setdefault(int(match.group(1)), float(match.group(2)))
    return pairs


def elo(score: float) -> float:
    """Logistic Elo of a score, clamped so a clean sweep is a number not an error."""
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = read_pairs(a_path), read_pairs(b_path)
    shared = sorted(set(a) & set(b))
    if not shared:
        print(f"no shared pair indices ({len(a)} in A, {len(b)} in B) -- "
              "were these run with the same --games against the same opponent?")
        return 1

    diffs = [a[i] - b[i] for i in shared]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)

    a_mean = sum(a[i] for i in shared) / n
    b_mean = sum(b[i] for i in shared) / n
    # Unpaired SE of the difference of two means over the same pairs, i.e. what
    # you would have got by comparing the two percentages independently.
    a_var = sum((a[i] - a_mean) ** 2 for i in shared) / (n - 1) if n > 1 else 0.0
    b_var = sum((b[i] - b_mean) ** 2 for i in shared) / (n - 1) if n > 1 else 0.0
    se_unpaired = math.sqrt((a_var + b_var) / n)

    print(f"A {a_path.name}: {a_mean:.3f} over {n} shared pairs ({len(a)} total)")
    print(f"B {b_path.name}: {b_mean:.3f} over {n} shared pairs ({len(b)} total)")
    print(f"  A - B = {mean:+.4f} +/- {se:.4f} (paired)")
    print(f"          {mean:+.4f} +/- {se_unpaired:.4f} (unpaired, for comparison)")
    if se_unpaired > 0:
        print(f"  pairing changes the SE by {100 * (se / se_unpaired - 1):+.0f}%")
    print(f"  A - B = {elo(0.5 + mean / 2):+.1f} Elo "
          f"+/- {elo(0.5 + (mean + 1.96 * se) / 2) - elo(0.5 + mean / 2):.1f} (95%)")
    won = sum(1 for d in diffs if d > 0)
    lost = sum(1 for d in diffs if d < 0)
    print(f"  pairs A better {won}, B better {lost}, level {n - won - lost}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
