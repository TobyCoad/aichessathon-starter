r"""Generate a WDL-labelled corpus by Stockfish self-play.

The n80000 corpus has fields (idx, count, stm, cp) and NO game outcome. That is the
single biggest gap in it: training on engine eval alone caps the net at the labeller's
judgement, because nothing ever tells it where the labeller was wrong. Every strong NNUE
trainer blends the two -- `lambda * sigmoid(cp/scale) + (1-lambda) * result` -- so the
eval teaches positional judgement and the result corrects it.

This writes the same row plus a `result` field, so a shard can be fed to the existing
loader unchanged and the blend switched on in the loss.

Two things that matter more than they look:

  * OPENING DIVERSITY. Self-play from the start position produces near-identical games:
    a deterministic engine plays the same moves. Each game therefore opens with a random
    walk of `--random-plies` (default 8) before the engine takes over, and the walk is
    rejected if it leaves the position already lost, so the corpus is not full of
    resignable junk. Without this, 50,000 games are worth about 50.

  * THE LABEL AND THE OUTCOME COME FROM DIFFERENT STRENGTHS. Moves are chosen at
    --play-nodes (cheap, so games finish), but each stored position is scored at
    --label-nodes (dearer, so the cp label is worth having). Scoring every position at
    play strength would make the label no better than the move that produced it.

The random opening walk means the first few plies of each game are nonsense, so they are
not stored: `--skip-plies` drops them.

Run (one worker per core, each writing its own shard):
  .venv\Scripts\python.exe -m training.gen_wdl --out data/wdl --workers 10 --games 4000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np

from training import features

STOCKFISH = Path("engines/stockfish/stockfish-windows-x86-64-avx2.exe")

# Matches the n80000 corpus exactly, plus `result`: +1 the side to move went on to win,
# -1 it lost, 0 drawn. Stored from the SIDE TO MOVE's point of view, like cp.
ROW = np.dtype(
    [("idx", "<u2", (32,)), ("count", "u1"), ("stm", "u1"), ("cp", "<i2"), ("result", "i1")]
)

# The n80000 corpus clamps at +-2000 and 40% of its 3-4 piece rows sit on that cap, which
# is why the net saturates in endgames. Keep the same clamp so the two corpora are
# commensurable -- correcting the clamp is a separate experiment, not a silent change.
CP_CLAMP = 2000


def pack(board: chess.Board, cp: int, result: int) -> tuple[Any, ...]:
    """One row. Feature indices are the side-to-move perspective, as the trainer expects."""
    idx = features.indices(board, board.turn, mirrored=False)[:32]
    padded = idx + [0] * (32 - len(idx))
    return (padded, len(board.piece_map()), 0 if board.turn == chess.WHITE else 1,
            max(-CP_CLAMP, min(CP_CLAMP, cp)), result)


def random_opening(rng: random.Random, plies: int) -> chess.Board | None:
    """A random legal walk, rejected if it is already resignable or over."""
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            return None
        board.push(rng.choice(moves))
    if board.is_game_over(claim_draw=True):
        return None
    return board


def play_one(
    sf: chess.engine.SimpleEngine, rng: random.Random, arguments: argparse.Namespace
) -> list[tuple[Any, ...]]:
    board = random_opening(rng, arguments.random_plies)
    if board is None:
        return []
    # Reject an opening the engine already considers decided: those games teach the net
    # about positions it will never be asked to play.
    check = sf.analyse(board, chess.engine.Limit(nodes=arguments.label_nodes), game=object())
    opening_cp = check["score"].pov(chess.WHITE).score(mate_score=20000)
    if opening_cp is None or abs(opening_cp) > arguments.opening_cap:
        return []

    seen: list[tuple[chess.Board, int]] = []
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < arguments.max_plies:
        if len(board.move_stack) >= arguments.random_plies + arguments.skip_plies:
            info = sf.analyse(board, chess.engine.Limit(nodes=arguments.label_nodes), game=object())
            score = info["score"].pov(board.turn).score(mate_score=20000)
            if score is not None:
                seen.append((board.copy(stack=False), int(score)))
        played = sf.play(board, chess.engine.Limit(nodes=arguments.play_nodes))
        if played.move is None:
            break
        board.push(played.move)

    # outcome None means we hit the ply cap: a draw for our purposes, not a result to trust.
    outcome = board.outcome(claim_draw=True)
    winner = None if outcome is None else outcome.winner
    rows = []
    for position, cp in seen:
        result = 0 if winner is None else (1 if winner == position.turn else -1)
        rows.append(pack(position, cp, result))
    return rows


def worker(index: int, arguments: argparse.Namespace) -> None:
    rng = random.Random(arguments.seed * 1000 + index)
    out = Path(arguments.out) / f"wdl_{index:02d}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Any, ...]] = []
    started = time.monotonic()
    last_save = started
    with chess.engine.SimpleEngine.popen_uci(str(STOCKFISH)) as sf:
        sf.configure({"Threads": 1, "Hash": 64})
        for game in range(arguments.games):
            rows.extend(play_one(sf, rng, arguments))
            if index == 0 and game % 25 == 0 and game:
                rate = len(rows) / (time.monotonic() - started)
                print(
                    f"  worker 0: {game} games, {len(rows):,} rows, {rate:.0f} rows/s"
                    f" -> ~{rate * arguments.workers * 3600 / 1e6:.1f}M rows/h across"
                    f" {arguments.workers} workers",
                    flush=True,
                )
            # Checkpoint often. Every save rewrites the WHOLE array, so saving too often
            # is quadratic in rows; saving too rarely risks hours of work on a crash and
            # makes progress unmonitorable. Append-only to a growing list, flushed on a
            # wall-clock interval rather than a game count, which is stable as games get
            # longer in the endgame-heavy tail.
            if rows and time.monotonic() - last_save > arguments.save_every:
                np.save(out, np.array(rows, dtype=ROW))
                last_save = time.monotonic()
    np.save(out, np.array(rows, dtype=ROW))
    print(f"  worker {index}: wrote {len(rows):,} rows to {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/wdl"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--games", type=int, default=4000, help="games PER worker")
    parser.add_argument("--play-nodes", type=int, default=5000, help="nodes per move played")
    parser.add_argument("--label-nodes", type=int, default=20000, help="nodes per stored label")
    parser.add_argument("--random-plies", type=int, default=8, help="random opening walk")
    parser.add_argument("--skip-plies", type=int, default=2, help="plies after the walk to drop")
    parser.add_argument("--opening-cap", type=int, default=400, help="reject openings past this cp")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--save-every", type=float, default=120.0,
                        help="seconds between shard flushes")
    parser.add_argument("--seed", type=int, default=907)
    arguments = parser.parse_args()

    procs = [
        mp.Process(target=worker, args=(i, arguments), daemon=False)
        for i in range(arguments.workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
