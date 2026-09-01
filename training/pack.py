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

# numpy.random is a lazy import: it loads a DLL the first time it is touched, which
# in this pipeline was after all the packing work was finished. Windows Application
# Control blocked that DLL once, mid-run, and destroyed 145M packed positions at the
# final shuffle. Touching it here means a block fails in the first second instead of
# the fifteenth minute -- and _shuffle below survives it either way.
try:
    np.random.default_rng(0).random(1)
    _NUMPY_RNG = True
except Exception:  # blocked DLL, restricted environment
    _NUMPY_RNG = False
import numpy.typing as npt
import pyarrow.parquet as pq

from training.features import MAX_PIECES, white_indices

RECORD = np.dtype(
    [("idx", np.uint16, MAX_PIECES), ("count", np.uint8), ("stm", np.uint8), ("cp", np.int16)]
)

CP_CLAMP = 2000
QUIET_BAND = 100
MATE_CP = 2000


def _relabelled(labels: Path | None, group: int, rows: int) -> list[int] | None:
    """Scores for one row group from relabel shards, or None if not fully covered.

    Partial coverage returns None rather than a half-filled array: silently mixing
    two labelling regimes inside one group is exactly the inconsistency relabelling
    exists to remove, and it would be invisible afterwards.
    """
    if labels is None:
        return None
    shard = 100_000
    needed = -(-rows // shard)
    out: list[int] = []
    for index in range(needed):
        path = labels / f"g{group:04d}_s{index:03d}.npy"
        if not path.is_file():
            return None
        out.extend(int(v) for v in np.load(path))
    return out[:rows] if len(out) >= rows else None


def ply_of(fen: str) -> int:
    """Ply from a FEN by string inspection, without building a Board.

    Parsing a FEN costs about 38 us and this decides whether the position is worth
    parsing at all, so it has to be cheaper than the thing it guards.
    """
    parts = fen.split()
    if len(parts) < 2:
        return 10**6
    try:
        fullmove = int(parts[-1]) if parts[-1].isdigit() else 1
    except ValueError:
        fullmove = 1
    return (fullmove - 1) * 2 + (0 if parts[1] == "w" else 1)


def process_group(
    job: tuple[Path, int, Path | None, int],
) -> tuple[npt.NDArray[np.void], npt.NDArray[np.uint64]]:
    """Filter and encode one row group. Runs in a worker process."""
    path, group, labels, min_ply = job
    table = pq.ParquetFile(path).read_row_group(group, columns=["fen", "cp", "mate", "move"])
    fens = table["fen"].to_pylist()
    cps = table["cp"].to_pylist()
    mates = table["mate"].to_pylist()
    moves = table["move"].to_pylist()

    # Our own labels replace both cp and mate: they are one engine at one depth, and
    # the mate encoding is already folded into the score.
    replacement = _relabelled(labels, group, len(fens))
    if replacement is not None:
        # NO_LABEL marks a position the labeller could not score. It becomes None so
        # the existing "cp is None -> skip" path drops it, rather than being trained
        # on as a real evaluation.
        cps = [None if v == -2_147_483_648 else v for v in replacement]
        mates = [None] * len(fens)

    records = np.zeros(len(fens), dtype=RECORD)
    keys = np.zeros(len(fens), dtype=np.uint64)
    kept = 0

    for fen, cp, mate, move in zip(fens, cps, mates, moves, strict=False):
        # Opening positions the book answers at runtime. The network is never asked
        # about them in a real game -- the book plays those plies instantly, and a
        # search rooted after the book only ever descends to deeper plies, never back
        # to shallower ones. They are also the most duplicated part of a human corpus,
        # since every game starts the same way, so they consume budget twice over.
        if min_ply and ply_of(fen) < min_ply:
            continue
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


def _shuffle(out: npt.NDArray[np.void], seed: int) -> None:
    """Shuffle in place, without depending on numpy.random being loadable.

    The fallback permutes by sorting a splitmix64 hash of each index. That is a
    deterministic, well-distributed bijection built from integer arithmetic alone,
    so it needs no RNG at all. Slower than a real shuffle, and worth it: the
    alternative is discarding an hour of packing because a DLL was unavailable for
    a moment.
    """
    if _NUMPY_RNG:
        np.random.default_rng(seed).shuffle(out)
        return
    keys = np.arange(len(out), dtype=np.uint64) + np.uint64(seed * 0x9E3779B97F4A7C15)
    keys ^= keys >> np.uint64(30)
    keys *= np.uint64(0xBF58476D1CE4E5B9)
    keys ^= keys >> np.uint64(27)
    keys *= np.uint64(0x94D049BB133111EB)
    keys ^= keys >> np.uint64(31)
    out[:] = out[np.argsort(keys, kind="stable")]


def balance(records: npt.NDArray[np.void], quiet_fraction: float) -> npt.NDArray[np.void]:
    """Subsample decided positions until quiet ones are at least `quiet_fraction`.

    An evaluation earns its Elo by discriminating between near-equal positions. A
    set dominated by already-won ones teaches it to recognise that a queen is good,
    which it would learn from material alone.
    """
    if quiet_fraction <= 0.0:
        # Balancing disabled. The rule this implements -- "at least 50% of the data
        # should have evaluation values between -100 and 100" -- comes from a single
        # paper that publishes no ablation of it, and the natural fraction here is
        # 34%, so enforcing it costs a third of the corpus, all of it decided
        # positions. That is a large price for an unmeasured convention, and the
        # engine is data-starved: same data, 4x the parameters, measurably worse.
        # So it is now a switch, and the two settings get compared like anything
        # else -- by playing games.
        out = records.copy()
        _shuffle(out, 1)
        return out

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
    _shuffle(out, 1)
    return out


def collect(
    source: Path,
    group_ids: list[int],
    target: int,
    workers: int,
    label: str,
    scratch: Path,
    labels: Path | None = None,
    min_ply: int = 0,
) -> npt.NDArray[np.void]:
    """Run the packer over a set of row groups and return deduplicated records.

    Records are staged through a memory-mapped file rather than a list of chunks.
    The obvious implementation -- collect chunks, `np.concatenate`, then gather the
    deduplicated rows -- holds three copies at once and peaks near three times the
    corpus: measured, that put 150M positions at ~27 GB against 31 GB of RAM, which
    was the only thing capping how much data this project could train on.

    Staged this way the memmap lives on disk (page cache, evictable), so resident
    memory is the key array plus the final result: about 11 GB at 150M rather than
    27, and the ceiling becomes free disk instead of RAM.
    """
    # Generous: dedup and filtering keep well under this, and the file is truncated.
    capacity = int(target / 0.55) + 2_000_000
    scratch.mkdir(parents=True, exist_ok=True)
    staging = scratch / f".{label}_staging.dat"
    staged = np.memmap(staging, dtype=RECORD, mode="w+", shape=(capacity,))
    keys = np.empty(capacity, dtype=np.uint64)
    total = 0

    jobs = [(source, group, labels, min_ply) for group in group_ids]
    try:
        with mp.Pool(workers) as pool:
            for done, (records, chunk_keys) in enumerate(pool.imap(process_group, jobs), start=1):
                take = min(len(records), capacity - total)
                staged[total : total + take] = records[:take]
                keys[total : total + take] = chunk_keys[:take]
                total += take
                if done % 10 == 0 or total >= target or take < len(records):
                    print(f"  {label} group {done}/{len(jobs)}: {total:,} positions", flush=True)
                if total >= target or take < len(records):
                    pool.terminate()
                    break

        _, first = np.unique(keys[:total], return_index=True)
        order = np.sort(first)
        duplicates = total - len(order)

        # Gather in slices so the staged file is read back a piece at a time rather
        # than materialising a second full copy of it.
        result = np.empty(len(order), dtype=RECORD)
        step = 2_000_000
        for start in range(0, len(order), step):
            rows = order[start : start + step]
            result[start : start + len(rows)] = staged[rows]
    finally:
        del staged
        staging.unlink(missing_ok=True)

    print(f"{label}: {len(result):,} unique ({duplicates:,} duplicates dropped)")
    return result


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
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="directory of relabel shards; overrides the corpus cp/mate columns",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=0,
        help="drop positions before this ply; 16 covers what the opening book plays",
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
        collect(
            arguments.source,
            train_ids,
            arguments.target,
            workers,
            "train",
            arguments.out.parent,
            labels=arguments.labels,
            min_ply=arguments.min_ply,
        ),
        arguments.quiet_fraction,
    )
    validation = balance(
        collect(
            arguments.source,
            val_ids,
            arguments.val_target,
            workers,
            "val",
            arguments.out.parent,
            labels=arguments.labels,
            min_ply=arguments.min_ply,
        ),
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
