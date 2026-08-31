"""Relabel positions with our own Stockfish at a fixed depth.

MEASURED AND REJECTED -- do not run this at depth 10. Kept because the negative
result is worth more than the code, and because re-deriving it costs a night.

The premise was that the corpus labels are of unknown, variable depth. That was
wrong, and wrong for an embarrassing reason: fishnet runs a documented fixed
budget of 1.5M nodes, reaching a median depth of about 21. The depth metadata
exists upstream in lichess_db_eval.jsonl.zst; the parquet we fetched simply drops
the column, and "no depth column in our copy" was mistaken for "no known depth".

Adjudicated directly. On 200 positions where a depth-10 label and the corpus label
disagreed by more than 200cp, refereed at depth 18:

    corpus label closer to truth    197/200   (98.5%)   mean error 190.9 cp
    our depth-10 label closer         3/200    (1.5%)   mean error 594.5 cp

and that test favoured depth 10, whose referee was the same binary. Depth 10 is
333x less search than the labels we already own for free.

The cost was also underestimated threefold. A single-threaded benchmark of
8.9 ms/position divided by 14 workers assumes linear scaling; 16 logical cores over
8 physical give about 48% efficiency, so 100M positions is ~54 hours, not 17.7.

The bottleneck was never label quality. 9.1M -> 62.5M positions with labels
untouched measured +151 Elo, and the 1024-wide net lost, so volume is confirmed
and capacity is rejected. More data at existing labels is the axis with slope.

The `cp` column in the corpus comes from Lichess community analysis: volunteers
run Stockfish on donated hardware and stop when they stop. The dataset records
the number but not the depth, so a label might be depth 15 or depth 30 and there
is no way to tell, filter or down-weight. The network learns from a mixture of
careful and careless targets with no signal separating them.

This replaces them with labels of known provenance -- one engine, one depth,
every position. The trade is real and not obviously favourable: consistency is
gained, absolute accuracy is lost, and whether that helps is an empirical
question. Hence `--target 10000000` first, train, SPRT, and only then commit a
full overnight pass.

Permitted explicitly: the rules ban *shipping* Stockfish or third-party weights.
Training on engine-annotated positions is allowed, and is how Stockfish's own
networks are made.

Designed around the fact that a ten-hour job will be interrupted. Work is split
into ~100k-position shards, each written atomically by the worker that finished
it, so a restart re-does at most one shard per worker -- about a minute -- and
`--resume` is the normal way to run it rather than a recovery path.
"""

import argparse
import contextlib
import multiprocessing as mp
import os
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pyarrow.parquet as pq

SHARD = 100_000
# Mate is stored the way engines conventionally do it, as a score near the top of
# the range that decreases with distance, so "mate in 3" stays worse than "mate in
# 1" and the trainer can clamp or drop them without a separate column.
MATE_BASE = 30_000
# A failed label must not be storable as a real evaluation. Zero was the original
# choice and it is the worst available one: it is indistinguishable from a genuine
# 0.00 assessment, and |cp| <= QUIET_BAND means every failure would land in the
# quiet bucket the packer deliberately preserves.
NO_LABEL = -2_147_483_648
ENGINE_PATH = "engines/stockfish/stockfish-windows-x86-64-avx2.exe"


def shard_path(out: Path, group: int, index: int) -> Path:
    return out / f"g{group:04d}_s{index:03d}.npy"


def label_shard(job: tuple[Path, Path, int, int, int, int, str]) -> tuple[int, int, float]:
    """Label one shard and write it atomically. Returns (rows, group, seconds)."""
    source, out, group, index, depth, _total, engine_path = job
    destination = shard_path(out, group, index)
    if destination.exists():
        return 0, group, 0.0

    started = time.perf_counter()
    table = pq.ParquetFile(source).read_row_group(group, columns=["fen"])
    fens = table["fen"].to_pylist()[index * SHARD : (index + 1) * SHARD]
    if not fens:
        return 0, group, 0.0

    scores = np.zeros(len(fens), dtype=np.int32)
    engine = chess.engine.SimpleEngine.popen_uci(os.path.abspath(engine_path))
    try:
        engine.configure({"Threads": 1, "Hash": 16})
        limit = chess.engine.Limit(depth=depth)
        for position, fen in enumerate(fens):
            try:
                board = chess.Board(fen)
            except ValueError:
                scores[position] = NO_LABEL
                continue
            if board.is_game_over(claim_draw=False):
                # Terminal positions have no search value at all.
                scores[position] = NO_LABEL
                continue
            try:
                info = engine.analyse(board, limit)
                score = info["score"].white()
                mate = score.mate()
                if mate is not None:
                    scores[position] = (
                        MATE_BASE - abs(mate) if mate > 0 else -(MATE_BASE - abs(mate))
                    )
                else:
                    scores[position] = int(score.score() or 0)
            except Exception:
                scores[position] = NO_LABEL
    finally:
        with contextlib.suppress(Exception):
            engine.quit()

    # Write to a temporary name and rename, so an interrupted write can never leave
    # a truncated shard that a later --resume would trust and skip.
    temporary = destination.with_suffix(".tmp.npy")
    np.save(temporary, scores)
    temporary.replace(destination)
    return len(fens), group, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel positions with Stockfish.")
    parser.add_argument("--source", type=Path, default=Path("data/standard_rated_2025_01.parquet"))
    parser.add_argument("--out", type=Path, default=None, help="default: data/relabel/d<depth>")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--target", type=int, default=100_000_000, help="positions to label")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--engine", default=ENGINE_PATH)
    arguments = parser.parse_args()

    out = arguments.out or Path("data/relabel") / f"d{arguments.depth}"
    out.mkdir(parents=True, exist_ok=True)

    workers = arguments.workers or max(1, (mp.cpu_count() or 4) - 2)
    metadata = pq.ParquetFile(arguments.source).metadata
    per_group = metadata.num_row_groups and metadata.num_rows // metadata.num_row_groups
    groups_needed = min(metadata.num_row_groups, -(-arguments.target // max(per_group, 1)))
    shards_per_group = -(-per_group // SHARD)

    jobs = [
        (arguments.source, out, group, index, arguments.depth, per_group, arguments.engine)
        for group in range(groups_needed)
        for index in range(shards_per_group)
    ]
    done_already = sum(1 for job in jobs if shard_path(out, job[2], job[3]).exists())

    print(f"relabelling at depth {arguments.depth} on {workers} workers")
    print(f"  target      {arguments.target:,} positions ({groups_needed} row groups)")
    print(f"  shards      {len(jobs):,} of {SHARD:,}, {done_already:,} already done")
    print(f"  output      {out}")
    print("  resume      safe -- rerun this exact command", flush=True)

    labelled = 0
    started = time.perf_counter()
    with mp.Pool(workers) as pool:
        for finished, (rows, group, _seconds) in enumerate(
            pool.imap_unordered(label_shard, jobs), start=1
        ):
            labelled += rows
            if finished % 10 == 0 or finished == len(jobs):
                elapsed = time.perf_counter() - started
                rate = labelled / max(elapsed, 1e-9)
                left = max(0, arguments.target - done_already * SHARD - labelled)
                eta = left / rate / 3600 if rate > 0 else 0.0
                print(
                    f"  {labelled:>12,} labelled  group {group:>3}  "
                    f"{rate:>7,.0f} pos/s  eta {eta:>5.1f} h",
                    flush=True,
                )

    print(f"\ndone: {labelled:,} positions in {(time.perf_counter()-started)/3600:.2f} h")
    print(f"shards in {out}: {len(list(out.glob('g*.npy'))):,}")


if __name__ == "__main__":
    main()
