"""Turn the Lichess Parquet into a flat array the trainer can slice directly.

The output is a structured numpy array, one 67-byte record per position, holding
the white-perspective feature indices and the engine's evaluation. Everything
expensive -- FEN parsing, filtering, deduplication -- happens once, here, so the
training loop never touches python-chess. That matters: parsing a FEN costs ~38 us,
which caps a naive loader at 5-20k positions/sec and would leave the GPU idle.

The source file holds 627,353,822 rows, so this reads row groups until it has
enough and stops rather than scanning 7.5 GB.

`cp` is stored from **white's** point of view. That is measured, not assumed: over
4,669 positions with a material imbalance above 200cp, correlation with material is
+0.758 read as white-POV and -0.010 read as side-to-move. The trainer flips it by
`stm`. Getting this backwards would train a net to prefer losing positions, with no
error anywhere to catch it.
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import chess
import chess.polyglot
import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from training.features import MAX_PIECES, white_indices

RECORD = np.dtype(
    [("idx", np.uint16, MAX_PIECES), ("count", np.uint8), ("stm", np.uint8), ("cp", np.int16)]
)

CP_CLAMP = 2000
QUIET_BAND = 100
MATE_CP = 2000


def process_group(job: tuple[Path, int]) -> tuple[npt.NDArray[np.void], npt.NDArray[np.uint64]]:
    """Filter and encode one row group. Runs in a worker process."""
    path, group = job
    table = pq.ParquetFile(path).read_row_group(group, columns=["fen", "cp", "mate", "move"])
    fens = table["fen"].to_pylist()
    cps = table["cp"].to_pylist()
    mates = table["mate"].to_pylist()
    moves = table["move"].to_pylist()

    records = np.zeros(len(fens), dtype=RECORD)
    keys = np.zeros(len(fens), dtype=np.uint64)
    kept = 0

    for fen, cp, mate, move in zip(fens, cps, mates, moves, strict=False):
        if mate is not None:
            cp = MATE_CP if mate > 0 else -MATE_CP
        elif cp is None:
            continue
        cp = max(-CP_CLAMP, min(CP_CLAMP, int(cp)))

        try:
            board = chess.Board(fen)
        except ValueError:
            continue

        # Quiet positions only. A position in check, or one whose best move is a
        # capture, is mid-tactic: its evaluation is a statement about the sequence
        # that follows, not about the position, and quiescence handles those at
        # run time anyway. Training on them teaches the evaluation to guess at
        # tactics it cannot see.
        if board.is_check():
            continue
        if move:
            try:
                best = chess.Move.from_uci(move)
            except ValueError:
                continue
            if board.is_capture(best):
                continue

        indices = white_indices(board)
        if not indices or len(indices) > MAX_PIECES:
            continue

        records[kept]["idx"][: len(indices)] = indices
        records[kept]["count"] = len(indices)
        records[kept]["stm"] = 1 if board.turn == chess.WHITE else 0
        records[kept]["cp"] = cp
        # Deterministic across processes, unlike Python's randomised str hash.
        keys[kept] = chess.polyglot.zobrist_hash(board)
        kept += 1

    return records[:kept], keys[:kept]


def balance(records: npt.NDArray[np.void], quiet_fraction: float) -> npt.NDArray[np.void]:
    """Subsample decided positions until quiet ones are at least `quiet_fraction`.

    An evaluation earns its Elo by discriminating between near-equal positions. A
    set dominated by already-won ones teaches it to recognise that a queen is good,
    which it would learn from material alone.
    """
    quiet_mask = np.abs(records["cp"]) <= QUIET_BAND
    quiet = records[quiet_mask]
    loud = records[~quiet_mask]
    if len(quiet) == 0:
        return records

    # quiet / (quiet + loud) >= f  =>  loud <= quiet * (1 - f) / f
    allowed = int(len(quiet) * (1.0 - quiet_fraction) / quiet_fraction)
    if len(loud) > allowed:
        rng = np.random.default_rng(0)
        loud = loud[rng.choice(len(loud), allowed, replace=False)]

    out = np.concatenate([quiet, loud])
    np.random.default_rng(1).shuffle(out)
    return out


def collect(
    source: Path, group_ids: list[int], target: int, workers: int, label: str
) -> npt.NDArray[np.void]:
    """Run the packer over a set of row groups and return deduplicated records."""
    chunks: list[npt.NDArray[np.void]] = []
    key_chunks: list[npt.NDArray[np.uint64]] = []
    total = 0

    jobs = [(source, group) for group in group_ids]
    with mp.Pool(workers) as pool:
        for done, (records, keys) in enumerate(pool.imap(process_group, jobs), start=1):
            chunks.append(records)
            key_chunks.append(keys)
            total += len(records)
            if done % 10 == 0 or total >= target:
                print(f"  {label} group {done}/{len(jobs)}: {total:,} positions", flush=True)
            if total >= target:
                pool.terminate()
                break

    # Deduplicate once at the end over a flat uint64 array. A running Python set
    # would cost ~1.8 GB at this scale; 30M keys as uint64 is 240 MB.
    records = np.concatenate(chunks)
    keys = np.concatenate(key_chunks)
    _, first = np.unique(keys, return_index=True)
    duplicates = len(records) - len(first)
    records = records[np.sort(first)]
    print(f"{label}: {len(records):,} unique ({duplicates:,} duplicates dropped)")
    return records


def describe(label: str, records: npt.NDArray[np.void]) -> None:
    quiet = int((np.abs(records["cp"]) <= QUIET_BAND).sum())
    print(
        f"{label}: {len(records):,} positions, {quiet / len(records):.1%} quiet, "
        f"mean pieces {records['count'].mean():.1f}, "
        f"white to move {records['stm'].mean():.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack Parquet evaluations into a flat array.")
    parser.add_argument("--source", type=Path, default=Path("data/standard_rated_2025_01.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/positions.npy"))
    parser.add_argument("--val-out", type=Path, default=Path("data/validation.npy"))
    parser.add_argument("--target", type=int, default=30_000_000)
    parser.add_argument("--val-target", type=int, default=500_000)
    parser.add_argument(
        "--val-groups",
        type=int,
        default=8,
        help="row groups held out for validation, taken from the end of the file",
    )
    parser.add_argument("--quiet-fraction", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0)
    arguments = parser.parse_args()

    workers = arguments.workers or max(1, (mp.cpu_count() or 4) - 2)
    groups = pq.ParquetFile(arguments.source).metadata.num_row_groups
    print(f"{arguments.source.name}: {groups} row groups, target {arguments.target:,}")

    # Validation comes from row groups the training set never sees. Rows in this file
    # are consecutive plies of the same game, so splitting positions at random would
    # put one game on both sides and report a validation loss that is optimistic by
    # however much the network memorised that game.
    val_ids = list(range(groups - arguments.val_groups, groups))
    held_out = set(val_ids)
    train_ids = [group for group in range(groups) if group not in held_out]

    records = balance(
        collect(arguments.source, train_ids, arguments.target, workers, "train"),
        arguments.quiet_fraction,
    )
    validation = balance(
        collect(arguments.source, val_ids, arguments.val_target, workers, "val"),
        arguments.quiet_fraction,
    )

    describe("train", records)
    describe("val  ", validation)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(arguments.out, records)
    np.save(arguments.val_out, validation)
    print(f"wrote {arguments.out} ({arguments.out.stat().st_size / 1e9:.2f} GB)")
    print(f"wrote {arguments.val_out} ({len(validation):,} positions from disjoint games)")


if __name__ == "__main__":
    main()
