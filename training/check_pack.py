"""Validate a packed position file.

The packer throws away the FEN, so nothing downstream can notice if it wrote
nonsense. These checks decode the stored feature indices back into pieces and
squares and assert the result is a legal-looking chess position.

Run: .venv\\Scripts\\python.exe -m training.check_pack --file data/positions.npy
Exits non-zero on any failure.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from training.features import FEATURES, MAX_PIECES, black_from_white

PIECE_NAMES = ("pawn", "knight", "bishop", "rook", "queen", "king")


def decode(index: int) -> tuple[bool, int, int]:
    """Feature index back to (is_white, piece_type_0based, square), white POV."""
    own = index < 384
    rest = index % 384
    return own, rest // 64, rest % 64


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a packed position file.")
    parser.add_argument("--file", type=Path, default=Path("data/positions.npy"))
    parser.add_argument("--sample", type=int, default=20000)
    arguments = parser.parse_args()

    records = np.load(arguments.file, mmap_mode="r")
    total = len(records)
    rng = np.random.default_rng(0)
    sample = records[rng.choice(total, min(arguments.sample, total), replace=False)]

    failures = Counter[str]()
    piece_histogram = Counter[str]()

    for record in sample:
        count = int(record["count"])
        indices = [int(i) for i in record["idx"][:count]]

        if not 2 <= count <= MAX_PIECES:
            failures["piece count out of range"] += 1
        if any(not 0 <= i < FEATURES for i in indices):
            failures["index out of range"] += 1
            continue
        if len(set(indices)) != count:
            failures["duplicate index"] += 1

        kings = {True: 0, False: 0}
        squares = set()
        for index in indices:
            is_white, piece, square = decode(index)
            piece_histogram[PIECE_NAMES[piece]] += 1
            if square in squares:
                failures["two pieces on one square"] += 1
            squares.add(square)
            if piece == 5:
                kings[is_white] += 1
            # Pawns cannot stand on the first or last rank.
            if piece == 0 and not 8 <= square < 56:
                failures["pawn on rank 1 or 8"] += 1

        if kings[True] != 1 or kings[False] != 1:
            failures["not exactly one king per side"] += 1

        if not -2000 <= int(record["cp"]) <= 2000:
            failures["cp out of range"] += 1
        if int(record["stm"]) not in (0, 1):
            failures["stm not 0 or 1"] += 1

        # Padding beyond `count` is left as zero by the packer. Index 0 means "own
        # pawn on a1", which is unreachable, so padding can never be confused with a
        # real feature -- but everything downstream must still slice by `count`
        # rather than trusting the array to be self-describing.
        if count < MAX_PIECES and any(int(i) != 0 for i in record["idx"][count:]):
            failures["padding is not zero"] += 1

    # The black-perspective derivation must round-trip back to the white one.
    white = sample["idx"][:, 0].astype(np.int64)
    if not np.array_equal(black_from_white(black_from_white(white)), white):
        failures["black_from_white is not an involution"] += 1

    quiet = float((np.abs(sample["cp"]) <= 100).mean())
    stm = float(sample["stm"].mean())
    pieces = float(sample["count"].mean())

    print(f"file            : {arguments.file}  ({total:,} positions)")
    print(f"sampled         : {len(sample):,}")
    print(f"mean pieces     : {pieces:.1f}")
    print(f"white to move   : {stm:.1%}")
    print(f"quiet (|cp|<100): {quiet:.1%}")
    print(f"cp range        : {int(sample['cp'].min())} .. {int(sample['cp'].max())}")
    print("piece mix       : " + ", ".join(
        f"{name} {piece_histogram[name] / max(sum(piece_histogram.values()), 1):.0%}"
        for name in PIECE_NAMES
    ))

    if failures:
        print("\nFAILURES:")
        for name, count in failures.most_common():
            print(f"  {name}: {count}")
        sys.exit(1)

    warnings = []
    if not 0.45 <= stm <= 0.55:
        warnings.append(f"side-to-move is skewed ({stm:.1%} white)")
    if quiet < 0.4:
        warnings.append(f"only {quiet:.0%} quiet positions; the balancer may not have run")
    if warnings:
        print("\nwarnings:")
        for warning in warnings:
            print(f"  {warning}")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
