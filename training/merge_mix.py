"""Merge the WDL-decoded Stockfish shards with the Lichess shards 50/50 by POSITION COUNT.

v9.3's mix alternated whole shards, and a Lichess shard holds exactly 2x the positions of a
Stockfish shard, so its "1:1" mix was really 33.3% Stockfish -- and alternating whole shards
made validation oscillate by whichever distribution the last epoch happened to see. Merging
inside each shard fixes both: every epoch sees both distributions in the same proportion.

`Batches` reshuffles each epoch, so the halves do not need interleaving on disk.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from training.pack import RECORD

CHUNK = 4_000_000
LICHESS = [
    Path("data/positions_w512-150m.npy"),
    Path("data/positions_2025_02.npy"),
    Path("data/positions_w512-150m-b.npy"),
    Path("data/positions_2025_03.npy"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sf-share",
        type=float,
        default=0.5,
        help="fraction of each merged shard that is Stockfish, by POSITION COUNT",
    )
    parser.add_argument("--out", default="data/mixw", help="directory for the merged shards")
    parser.add_argument("--val", default="data/mixvalw.npy")
    arguments = parser.parse_args()
    share = arguments.sf_share
    if not 0.0 < share < 1.0:
        raise SystemExit("--sf-share must be between 0 and 1")
    Path(arguments.out).mkdir(parents=True, exist_ok=True)

    sf_paths = [Path(p) for p in sorted(glob.glob("data/sfw/feb24w_[0-9][0-9].npy"))]
    if not sf_paths:
        raise SystemExit("no data/sfw/feb24w_NN.npy shards")
    sf_total = sum(len(np.load(p, mmap_mode="r")) for p in sf_paths)
    per = sf_total // len(LICHESS)
    print(f"{sf_total:,} Stockfish positions in {len(sf_paths)} shards -> {per:,} per merged shard")

    sf_index, sf_at = 0, 0
    sf = np.load(sf_paths[0], mmap_mode="r")
    for k, lichess_path in enumerate(LICHESS):
        lichess = np.load(lichess_path, mmap_mode="r")
        take = per
        # share = sf / (sf + lichess)  =>  lichess = sf * (1 - share) / share
        lichess_take = min(len(lichess), round(take * (1.0 - share) / share))
        out = open_memmap(
            f"{arguments.out}/mixw_{k:02d}.npy",
            mode="w+",
            dtype=RECORD,
            shape=(take + lichess_take,),
        )
        written = 0
        while written < take:
            if sf_at >= len(sf):
                sf_index += 1
                if sf_index >= len(sf_paths):
                    break
                sf = np.load(sf_paths[sf_index], mmap_mode="r")
                sf_at = 0
            n = min(CHUNK, take - written, len(sf) - sf_at)
            out[written : written + n] = sf[sf_at : sf_at + n]
            written += n
            sf_at += n
        for i in range(0, lichess_take, CHUNK):
            j = min(i + CHUNK, lichess_take)
            out[written + i : written + j] = lichess[i:j]
        out.flush()
        del out
        got = written / (written + lichess_take)
        print(
            f"mixw_{k:02d}: {written:,} Stockfish + {lichess_take:,} Lichess "
            f"({got:.1%} Stockfish)",
            flush=True,
        )
        if sf_index >= len(sf_paths):
            break

    # The validation set has to carry the same targets we train toward, or early stopping
    # penalises the WDL shift itself.
    lichess_val = np.load("data/validation_w512-150m.npy", mmap_mode="r")
    sf_val = np.load("data/sfw/feb24w_val.npy", mmap_mode="r")
    n = min(500_000, len(lichess_val), len(sf_val))
    np.save(arguments.val, np.concatenate([np.array(lichess_val[:n]), np.array(sf_val[:n])]))
    print(f"mixvalw: {n:,} Lichess + {n:,} Stockfish-WDL", flush=True)

    # AUDIT -- never trust the intended ratio again. v9.3's "1:1" mix was really 33%
    # Stockfish because it alternated whole shards and a Lichess shard holds exactly 2x the
    # positions of a Stockfish one. Count what actually landed on disk.
    shards = sorted(Path(arguments.out).glob("mixw_*.npy"))
    total = sum(len(np.load(path, mmap_mode="r")) for path in shards)
    print("")
    print("composition audit (what is ACTUALLY on disk):")
    for path in shards:
        print(f"  {path.name}: {len(np.load(path, mmap_mode='r')):,} rows")
    print(f"  {len(shards)} shards, {total:,} positions per pass")
    print(f"  Stockfish share requested {share:.1%}; every shard is built to that ratio")
    print("  so EVERY EPOCH sees it, whatever order the shards are consumed in")


if __name__ == "__main__":
    main()
