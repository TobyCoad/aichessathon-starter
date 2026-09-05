"""Hold the compiled search to the Python search.

Three checks, in order:
  constants  every constant fastsearch mirrors from agent.py is equal.
  exact      with the transposition table off in both, score and node count are
             identical at depths 1..N on the bench positions and on random
             positions reached by short random playouts.
  table      with the table on, best-move agreement and node-count ratio at
             one depth: the array table replaces entries differently from the
             dict, so this is a similarity check, not an equality.

Run:  .venv\\Scripts\\python.exe -m testing.check_fastsearch
      .venv\\Scripts\\python.exe -m testing.check_fastsearch --depth 4 --random 200
"""

import argparse
import random
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent
import fastboard as fb
import fastsearch as fs
from testing.bench import POSITIONS


class NoTable:
    """A transposition table that remembers nothing."""

    def get(self, key: int) -> None:
        return None

    def __setitem__(self, key: int, value: object) -> None:
        pass

    def __len__(self) -> int:
        return 0

    def clear(self) -> None:
        pass


def check_constants() -> None:
    pairs = {
        "MATE": agent.MATE, "DISTANCE_THRESHOLD": agent.DISTANCE_THRESHOLD,
        "INFINITY": agent.INFINITY, "OUTPUT_SCALE": agent.OUTPUT_SCALE,
        "DELTA_MARGIN": agent.DELTA_MARGIN, "BIG_DELTA": agent.BIG_DELTA,
        "RFP_MAX_DEPTH": agent.RFP_MAX_DEPTH, "RFP_MARGIN": agent.RFP_MARGIN,
        "NMP_MIN_DEPTH": agent.NMP_MIN_DEPTH, "NMP_REDUCTION": agent.NMP_REDUCTION,
        "MAX_PLY": agent.MAX_PLY, "POLL_MASK": agent._POLL_MASK,
    }
    for name, value in pairs.items():
        assert getattr(fs, name) == value, f"{name}: fastsearch {getattr(fs, name)} agent {value}"
    assert tuple(fs.MVV) == agent._MVV
    assert tuple(fs.FUTILITY_MARGIN) == agent.FUTILITY_MARGIN
    print("constants: identical")


def python_engine(board: chess.Board, table_on: bool) -> agent.FastEngine:
    engine = agent.FastEngine()
    engine.pos.load(board)
    fb.refresh(
        engine.pos.bb, engine.pos.sq, engine.pos.meta, agent.W1, agent.B1,
        engine.white, engine.black, engine.zones, agent.KING_ZONES,
    )
    engine.root_side = int(engine.pos.meta[0])
    engine.draw_root = 0
    engine.deadline = time.monotonic() + 3600
    if not table_on:
        engine.table = NoTable()  # type: ignore[assignment]
    return engine


class Kernel:
    """The compiled search on a fresh state."""

    def __init__(self, board: chess.Board, table_on: bool) -> None:
        self.pos = fb.Position(board)
        acc = agent.ACC_SIZE
        self.white = np.zeros(acc, np.float32)
        self.black = np.zeros(acc, np.float32)
        self.astack = np.zeros((fb.MAX_PLY, 2, acc), np.float32)
        self.zones = np.zeros(2, np.int64)
        fb.refresh(
            self.pos.bb, self.pos.sq, self.pos.meta, agent.W1, agent.B1,
            self.white, self.black, self.zones, agent.KING_ZONES,
        )
        self.table = fs.new_table()
        self.killers = np.zeros((fb.MAX_PLY, 2), np.int32)
        self.butterfly = np.zeros(8192, np.int32)
        self.moves = np.zeros((fb.MAX_PLY, fb.MOVE_CAP), np.int32)
        self.scores = np.zeros((fb.MAX_PLY, fb.MOVE_CAP), np.int64)
        self.rep = np.zeros(0, np.uint64)
        self.scratch = np.zeros(2 * acc, np.float32)
        self.counter = np.zeros(4096, np.int32)
        self.ec_key, self.ec_val = fs.new_eval_cache()
        self.exts = np.zeros(4 * fb.MAX_PLY, np.int64)  # 4 lanes, see agent.FastEngine
        self.conthist1 = np.zeros(768 * 768, np.int32)
        self.quiets = np.zeros((fb.MAX_PLY, fb.MOVE_CAP), np.int32)
        self.ctrl = np.zeros(fs.CTRL_SIZE, np.int64)
        self.ctrl[fs.C_TT_OFF] = 0 if table_on else 1
        self.ctrl[fs.C_HYGIENE] = 1 if agent.HYGIENE else 0
        self.ctrl[fs.C_FUTILITY] = 1 if agent.FUTILITY else 0
        self.ctrl[fs.C_ROOT_SIDE] = int(self.pos.meta[0])
        self.ctrl[fs.C_QS_CAP] = 8  # the reference's quiescence cap

    def search(self, depth: int) -> tuple[int, int, int, float]:
        """(score, nodes, best move, seconds) after iterative deepening to `depth`."""
        pos = self.pos
        score = 0
        started = time.perf_counter()
        for d in range(1, depth + 1):
            score = fs.search(  # type: ignore[call-arg]
                pos.bb, pos.sq, pos.meta, pos.undo, pos.keys, agent.W1, agent.B1,
                self.white, self.black, self.astack, self.zones, agent.KING_ZONES,
                agent._W2T, agent.B2, agent.W3, agent.B3, *self.table,
                self.killers, self.butterfly, self.moves, self.scores, self.rep,
                self.ctrl, time.monotonic() + 3600, d, -agent.INFINITY, agent.INFINITY, 0,
                self.scratch, self.counter, self.quiets, self.ec_key, self.ec_val, self.exts,
                self.conthist1, 0,
            )
        seconds = time.perf_counter() - started
        slot = int(pos.keys[0] & fs.TT_MASK)
        best = 0
        if self.table[0][slot] == pos.keys[0]:
            best = int(fs.unpack_move(self.table[1][slot]))
        return int(score), int(self.ctrl[fs.C_NODES]), best, seconds


def python_search(
    board: chess.Board, depth: int, table_on: bool
) -> tuple[int, int, int, float]:
    engine = python_engine(board, table_on)
    score = 0
    started = time.perf_counter()
    for d in range(1, depth + 1):
        score = engine.search(d, -agent.INFINITY, agent.INFINITY, 0)
    seconds = time.perf_counter() - started
    entry = engine.table.get(int(engine.pos.keys[0])) if table_on else None
    best = int(entry[3]) if entry else 0
    return int(score), int(engine.nodes), best, seconds


def random_positions(count: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    out: list[chess.Board] = []
    while len(out) < count:
        board = chess.Board()
        for _ in range(rng.randint(6, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over() and list(board.legal_moves):
            out.append(board)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compiled search vs Python search.")
    parser.add_argument("--depth", type=int, default=4, help="exact check up to this depth")
    parser.add_argument("--random", type=int, default=100)
    parser.add_argument("--table-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()

    check_constants()
    fs.warm_up(agent.W1, agent.B1, agent._W2T, agent.B2, agent.W3, agent.B3, agent.KING_ZONES)
    # The kernel does not probe tablebases inside the tree (the root still does),
    # so the reference runs without them too; the root probe is exercised by play.
    agent._TABLEBASE = None
    boards = [chess.Board(fen) for fen in POSITIONS]
    boards += random_positions(arguments.random, arguments.seed)

    failures = 0
    python_time = 0.0
    kernel_time = 0.0
    for index, board in enumerate(boards):
        ps, pn, _, seconds = python_search(board, arguments.depth, False)
        python_time += seconds
        ks, kn, _, seconds = Kernel(board, False).search(arguments.depth)
        kernel_time += seconds
        if ps != ks or pn != kn:
            failures += 1
            print(
                f"  MISMATCH {index}: python {ps} {pn} nodes, "
                f"kernel {ks} {kn} nodes  {board.fen()}"
            )
    print(
        f"exact (table off, depth {arguments.depth}): "
        f"{len(boards) - failures}/{len(boards)} identical; "
        f"python {python_time:.1f}s kernel {kernel_time:.1f}s "
        f"({python_time / max(kernel_time, 1e-9):.2f}x)"
    )

    agree = 0
    ratios = []
    python_time = kernel_time = 0.0
    for board in boards[: len(POSITIONS)]:
        ps, pn, pb, seconds = python_search(board, arguments.table_depth, True)
        python_time += seconds
        ks, kn, kb, seconds = Kernel(board, True).search(arguments.table_depth)
        kernel_time += seconds
        if pb == kb:
            agree += 1
        if pn:
            ratios.append(kn / pn)
    ratios.sort()
    print(
        f"table on, depth {arguments.table_depth}: best move agreement {agree}/{len(POSITIONS)}, "
        f"node ratio kernel/python median {ratios[len(ratios) // 2]:.2f} "
        f"(min {ratios[0]:.2f}, max {ratios[-1]:.2f}); "
        f"python {python_time:.1f}s kernel {kernel_time:.1f}s "
        f"({python_time / max(kernel_time, 1e-9):.2f}x)"
    )
    if failures:
        raise SystemExit(f"FAIL: {failures} mismatches")
    print("PASS")


if __name__ == "__main__":
    main()
