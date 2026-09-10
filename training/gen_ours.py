r"""Our-distribution training data: positions from OUR engine's games, labelled by
Stockfish, with the game result blended in exactly as the sf80kf corpus does it.

Why this corpus exists (9 Sep). The net trains on Stockfish self-play (sf80kf) and human
games. Stockfish never reaches the positions a 250 knps engine reaches -- the draw audit's
repetition endgames scored +0.8 that are 0.00, the "up a minor, no pawns" endings still
valued -- so the net has never been asked about them. Every strong NNUE pipeline generates
from its own engine and labels with the oracle; this is that step.

Rows are RAW: the trainer never reads them directly. `training.mix_ours` turns them into
sf80kf-convention shards, applying the UCI -> corpus cp mapping (training.uci_scale) and
the lambda=0.75 result blend at that point, so the mapping can change without replaying.
  idx     features.white_indices (white perspective, own pieces first)
  stm     1 white to move, 0 black          (binpack_decode's convention, NOT gen_wdl's)
  cp      white-POV Stockfish UCI cp, mates +-3000, otherwise unscaled and unblended
  result  +1 white won, 0 draw, -1 black won (the ply cap is a draw, as on the platform)

Games: the platform/book openings plus two random plies, then our engine against itself
at --movetime-ms a move (the tree's agent.py, whatever is shipped), to the end or
--max-plies (a draw, as the platform rules it). Positions in check are skipped like the
decoder does. Labels at --label-nodes per position, one Stockfish process per worker.

  .venv\Scripts\python.exe -m training.gen_ours --workers 10 --hours 6
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import features  # noqa: E402
from training.generate import openings, stockfish_path  # noqa: E402

STORE_FROM_PLY = 4  # plies into OUR game (openings already start at ply 10-16)
MATE_CP = 3000
RAW = np.dtype(
    [("idx", "<u2", (32,)), ("count", "u1"), ("stm", "u1"), ("cp", "<i2"), ("result", "i1")]
)


def pack(board: chess.Board, cp: int, result: int) -> tuple[Any, ...]:
    idx = features.white_indices(board)
    return (idx + [0] * (32 - len(idx)), len(idx), 1 if board.turn == chess.WHITE else 0, cp, result)


def label(sf: chess.engine.SimpleEngine, board: chess.Board, nodes: int) -> int:
    info = sf.analyse(board, chess.engine.Limit(nodes=nodes), game=object())
    score = info["score"].white()
    if score.is_mate():
        mate = score.mate() or 0
        return MATE_CP if mate > 0 else -MATE_CP
    return max(-MATE_CP, min(MATE_CP, int(score.score() or 0)))


def play_one(engine: Any, sf: Any, rng: random.Random, starts: list[str], a: argparse.Namespace):
    board = chess.Board(rng.choice(starts))
    for _ in range(2):
        moves = list(board.legal_moves)
        if not moves:
            return []
        board.push(rng.choice(moves))
    if board.is_game_over(claim_draw=True):
        return []
    seen: list[tuple[chess.Board, int]] = []
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < a.max_plies:
        if plies >= STORE_FROM_PLY and not board.is_check():
            seen.append((board.copy(stack=False), label(sf, board, a.label_nodes)))
        now = time.monotonic()
        try:
            move = engine.play(board, now + a.movetime_ms / 1000.0, now + a.movetime_ms / 1000.0 * 1.5)
            if move not in board.legal_moves:
                move = rng.choice(list(board.legal_moves))
        except Exception:
            move = rng.choice(list(board.legal_moves))
        board.push(move)
        plies += 1
    outcome = board.outcome(claim_draw=True)
    winner = None if outcome is None else outcome.winner  # None past the cap = draw
    result = 0 if winner is None else (1 if winner == chess.WHITE else -1)
    return [pack(position, cp, result) for position, cp in seen], result, plies


def worker(index: int, a: argparse.Namespace) -> None:
    import agent  # the tree's engine; ~40 s cold, cached after

    rng = random.Random(a.seed * 1000 + index)
    starts = openings()
    out = Path(a.out) / f"ours_{index:02d}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Any, ...]] = []
    tally = {1: 0, 0: 0, -1: 0}
    started = time.monotonic()
    last_save = started
    games = 0
    with chess.engine.SimpleEngine.popen_uci(str(stockfish_path().resolve())) as sf:
        sf.configure({"Threads": 1, "Hash": 32})
        engine = agent.FastEngine()
        while time.monotonic() - started < a.hours * 3600:
            got = play_one(engine, sf, rng, starts, a)
            if not got:
                continue
            new_rows, result, plies = got
            rows.extend(new_rows)
            tally[result] += 1
            games += 1
            if index == a.first_index and games % 10 == 0:
                rate = len(rows) / (time.monotonic() - started)
                print(
                    f"  worker 0: {games} games (W{tally[1]} D{tally[0]} L{tally[-1]}), last {plies} plies,"
                    f" {len(rows):,} rows, {rate:.1f} rows/s -> ~{rate * a.workers * 3600 / 1e6:.2f}M/h"
                    f" across {a.workers}",
                    flush=True,
                )
            if rows and time.monotonic() - last_save > a.save_every:
                np.save(out, np.array(rows, dtype=RAW))
                last_save = time.monotonic()
    np.save(out, np.array(rows, dtype=RAW))
    print(f"  worker {index}: {games} games, wrote {len(rows):,} rows to {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/ours"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--hours", type=float, default=6.0, help="wall clock per worker")
    parser.add_argument("--movetime-ms", type=int, default=100)
    parser.add_argument("--label-nodes", type=int, default=80_000)
    parser.add_argument("--max-plies", type=int, default=500)
    parser.add_argument("--save-every", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--first-index", type=int, default=0, help="worker/file index offset")
    a = parser.parse_args()
    procs = [mp.Process(target=worker, args=(a.first_index + i, a)) for i in range(a.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    total = sum(len(np.load(f, mmap_mode="r")) for f in Path(a.out).glob("ours_*.npy"))
    print(f"done: {total:,} rows in {a.out}", flush=True)


if __name__ == "__main__":
    main()
