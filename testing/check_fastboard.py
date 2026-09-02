"""Correctness of the compiled board against python-chess.

An illegal move loses the game outright, so the board earns trust the way move
generators always have: perft on the standard positions against the published
node counts, then a differential fuzz against python-chess on real positions --
move sets, capture sets, Zobrist keys, check status, make/unmake round trips --
and finally the fused accumulator against the engine's existing one.

Run: .venv\\Scripts\\python.exe -m testing.check_fastboard [--positions N] [--depth D]
"""

import argparse
import importlib.util
import random
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import chess
import chess.polyglot
import numpy as np
from numba import njit

import fastboard as fb

# (fen, depth, nodes) from the Chess Programming Wiki perft page.
PERFT = [
    (chess.STARTING_FEN, 5, 4_865_609),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4_085_603),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11_030_083),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15_833_292),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2_103_487),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3_894_594),
]


@njit(cache=False)
def perft(bb: Any, sqa: Any, meta: Any, undo: Any, keys: Any, bufs: Any, depth: Any) -> Any:
    out = bufs[depth]
    n = fb.gen_legal(bb, sqa, meta, out, False)
    if depth == 1:
        return n
    total = 0
    for i in range(n):
        fb.make_light(bb, sqa, meta, undo, keys, out[i])
        total += perft(bb, sqa, meta, undo, keys, bufs, depth - 1)
        fb.unmake_light(bb, sqa, meta, undo, keys)
    return total


def legal_set(pos: fb.Position, captures: bool) -> set[str]:
    out = np.zeros(fb.MOVE_CAP, dtype=np.int32)
    n = fb.gen_legal(pos.bb, pos.sq, pos.meta, out, captures)
    return {fb.move_to_uci(int(out[i])) for i in range(n)}


def compare(pos: fb.Position, board: chess.Board, where: str) -> list[str]:
    problems = []
    mine = legal_set(pos, False)
    theirs = {m.uci() for m in board.legal_moves}
    if mine != theirs:
        problems.append(
            f"{where}: moves differ  missing {sorted(theirs - mine)}  extra {sorted(mine - theirs)}"
        )
    mine_c = legal_set(pos, True)
    theirs_c = {m.uci() for m in board.generate_legal_captures()}
    if mine_c != theirs_c:
        problems.append(
            f"{where}: captures differ  missing {sorted(theirs_c - mine_c)}"
            f"  extra {sorted(mine_c - theirs_c)}"
        )
    key = int(pos.keys[pos.meta[fb.PLY]])
    if key != chess.polyglot.zobrist_hash(board):
        problems.append(
            f"{where}: key {key:016x} != polyglot {chess.polyglot.zobrist_hash(board):016x}"
        )
    if bool(fb.in_check(pos.bb, pos.meta)) != board.is_check():
        problems.append(f"{where}: in_check disagrees")
    if int(pos.meta[fb.PIECES]) != chess.popcount(board.occupied):
        problems.append(
            f"{where}: piece count {pos.meta[fb.PIECES]} != {chess.popcount(board.occupied)}"
        )
    # All six FEN fields: to_board() feeds the tablebase and the fifty-move rule,
    # so the ep square, halfmove clock and move number must survive the round trip.
    mine_fen = pos.to_board().fen()
    theirs_fen = board.fen()
    if mine_fen != theirs_fen:
        problems.append(f"{where}: fen {mine_fen} != {theirs_fen}")
    return problems


def load_agent(directory: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("fb_check_agent", directory / "agent.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["fb_check_agent"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the compiled board.")
    parser.add_argument("--positions", type=int, default=20_000)
    parser.add_argument("--walk", type=int, default=40, help="random plies from each position")
    parser.add_argument("--parquet", type=Path, default=Path("data/standard_rated_2025_01.parquet"))
    parser.add_argument(
        "--agent", type=Path, default=Path("."), help="agent whose accumulator to match"
    )
    parser.add_argument("--skip-perft", action="store_true")
    arguments = parser.parse_args()
    failures = 0

    started = time.perf_counter()
    fb.warm_up()
    print(f"compile: {time.perf_counter() - started:.1f}s")

    # 1. Perft.
    if not arguments.skip_perft:
        bufs = np.zeros((8, fb.MOVE_CAP), dtype=np.int32)
        for fen, depth, expected in PERFT:
            pos = fb.Position(chess.Board(fen))
            t = time.perf_counter()
            nodes = perft(pos.bb, pos.sq, pos.meta, pos.undo, pos.keys, bufs, depth)
            dt = time.perf_counter() - t
            ok = nodes == expected
            failures += not ok
            print(
                f"perft {depth} {fen[:30]:32s} {nodes:>12,} "
                f"{'ok' if ok else f'EXPECTED {expected:,}'}"
                f"  {nodes / dt / 1e6:5.1f} Mnps"
            )

    # 2. Differential fuzz on real positions and random walks from them.
    rng = random.Random(0)
    fens: list[str] = []
    if arguments.parquet.exists():
        import pyarrow.parquet as pq

        table = pq.ParquetFile(arguments.parquet).read_row_group(400, columns=["fen"])
        fens = table["fen"].to_pylist()
        rng.shuffle(fens)
        fens = fens[: arguments.positions]
    fens += [f for f, _, _ in PERFT]
    problems: list[str] = []
    checked = walked = 0
    t = time.perf_counter()
    for fen in fens:
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        if not board.is_valid():
            continue
        pos = fb.Position(board)
        problems += compare(pos, board, fen)
        checked += 1
        snapshot = (pos.bb.copy(), pos.sq.copy(), pos.meta.copy())
        for _ in range(arguments.walk):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            fb.make_light(pos.bb, pos.sq, pos.meta, pos.undo, pos.keys, fb.move_from_chess(move))
            board.push(move)
            walked += 1
            problems += compare(pos, board, f"{fen} after {board.move_stack}")
            if len(problems) > 20:
                break
        # Unwind and demand the exact starting arrays back.
        while pos.meta[fb.PLY] > 0:
            fb.unmake_light(pos.bb, pos.sq, pos.meta, pos.undo, pos.keys)
        if not (
            np.array_equal(pos.bb, snapshot[0])
            and np.array_equal(pos.sq, snapshot[1])
            and np.array_equal(pos.meta, snapshot[2])
        ):
            problems.append(f"{fen}: unmake did not restore the position")
        if len(problems) > 20:
            break
    for p in problems[:20]:
        print("  " + p)
    failures += len(problems)
    print(
        f"fuzz: {checked:,} positions, {walked:,} random plies, {len(problems)} problems"
        f"  ({time.perf_counter() - t:.0f}s)"
    )

    # 3. Fused accumulator against the agent's own, through zone crossings.
    agent = load_agent(arguments.agent)
    zones_n = getattr(agent, "KING_ZONES", 1)
    width = agent.ACC_SIZE
    white = np.zeros(width, dtype=np.float32)
    black = np.zeros(width, dtype=np.float32)
    astack = np.zeros((fb.MAX_PLY, 2, width), dtype=np.float32)
    zones = np.zeros(2, dtype=np.int64)
    reference = agent.Accumulator()
    drift = plies = 0
    for fen in fens[:300]:
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        if not board.is_valid():
            continue
        pos = fb.Position(board)
        fb.refresh(pos.bb, pos.sq, pos.meta, agent.W1, agent.B1, white, black, zones, zones_n)
        reference.refresh(board)
        for _ in range(arguments.walk):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            fb.make_full(
                pos.bb,
                pos.sq,
                pos.meta,
                pos.undo,
                pos.keys,
                fb.move_from_chess(move),
                agent.W1,
                agent.B1,
                white,
                black,
                astack,
                zones,
                zones_n,
            )
            reference.push(board, move)
            board.push(move)
            plies += 1
            if not (
                np.allclose(white, reference.white, atol=1e-3)
                and np.allclose(black, reference.black, atol=1e-3)
            ):
                drift += 1
                if drift <= 3:
                    print(f"  ACC DRIFT after {move.uci()} in {board.fen()}")
        while pos.meta[fb.PLY] > 0:
            fb.unmake_full(
                pos.bb, pos.sq, pos.meta, pos.undo, pos.keys, white, black, astack, zones
            )
            reference.pop()
        if not (
            np.allclose(white, reference.white, atol=1e-3)
            and np.allclose(black, reference.black, atol=1e-3)
        ):
            drift += 1
    failures += drift
    print(
        f"accumulator: {plies - drift}/{plies} plies match the agent's "
        f"({zones_n} zone(s)), unwind exact"
    )

    if failures:
        print(f"\n{failures} FAILURES")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
