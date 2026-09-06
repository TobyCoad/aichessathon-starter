"""Clock safety at the real time control, under a charging overhead.

The SPRT cannot see a time-management problem: it runs at 8 s where the reserve
never binds, and it counts a flag as a lost game rather than as the thing being
measured. This plays self-play games at the tournament control and charges the
agent more than it thinks it spent -- `--factor 1.5` bills every move at one and
a half times its wall time -- which stands in for the platform's referee overhead,
a slower core, a noisy neighbour or a garbage-collection pause. An engine that
manages its clock in wall time only flags when it has left itself no margin for
those, so the number to watch is the lowest the clock ever gets.

Both colours are played by the same process, so this measures the budget logic
alone, not the harness. Pass means: no flags and the clock never below
`--min-clock-ms` at any point in any game.

Run:  .venv\\Scripts\\python.exe -m testing.clocktest --agent overnight/challengers/026-time-v2
"""

import argparse
import importlib.util
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import chess

from testing import openings


def load_agent(directory: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("clocktest_agent", directory / "agent.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {directory / 'agent.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["clocktest_agent"] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Job:
    agent: Path
    fen: str
    base_ms: int
    increment_ms: int
    factor: float
    overhead_ms: float
    ply_cap: int


@dataclass(frozen=True)
class Report:
    fen: str
    plies: int
    result: str
    flagged: str
    min_clock_ms: float
    max_move_ms: float
    clock_at_30: float
    used_white_ms: float
    used_black_ms: float
    error: str


def play(job: Job) -> Report:
    try:
        agent = load_agent(job.agent)
    except Exception as exc:  # an import failure is the finding, not an error here
        return Report(job.fen, 0, "*", "", 0.0, 0.0, 0.0, 0.0, 0.0, f"init: {exc!r}")

    board = chess.Board(job.fen)
    clock = {chess.WHITE: float(job.base_ms), chess.BLACK: float(job.base_ms)}
    used = {chess.WHITE: 0.0, chess.BLACK: 0.0}
    lowest = float(job.base_ms)
    longest = 0.0
    at_30 = float(job.base_ms)
    flagged = ""
    error = ""
    plies = 0

    while not board.is_game_over(claim_draw=True) and plies < job.ply_cap:
        side = board.turn
        started = time.perf_counter()
        try:
            uci = agent.get_move(board.fen(), int(clock[side]))
        except Exception as exc:
            error = f"crash: {exc!r}"
            break
        charged = (time.perf_counter() - started) * 1000.0 * job.factor + job.overhead_ms
        clock[side] -= charged
        used[side] += charged
        longest = max(longest, charged)
        lowest = min(lowest, clock[side])
        if clock[side] < 0.0:
            flagged = "white" if side == chess.WHITE else "black"
            break
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            move = chess.Move.null()
        if move not in board.legal_moves:
            error = f"illegal: {uci}"
            break
        board.push(move)
        clock[side] += job.increment_ms
        plies += 1
        if plies == 30:
            at_30 = min(clock.values())

    result = board.result(claim_draw=True) if not flagged and not error else "*"
    return Report(
        job.fen, plies, result, flagged, lowest, longest, at_30,
        used[chess.WHITE], used[chess.BLACK], error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clock safety under a charging overhead.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--base-ms", type=int, default=120_000)
    parser.add_argument("--increment-ms", type=int, default=500)
    parser.add_argument("--factor", type=float, default=1.5, help="charge wall time x this")
    parser.add_argument("--overhead-ms", type=float, default=20.0, help="charged per move")
    parser.add_argument("--min-clock-ms", type=float, default=5_000.0)
    parser.add_argument("--ply-cap", type=int, default=600)
    parser.add_argument("--workers", type=int, default=0)
    arguments = parser.parse_args()

    agent = arguments.agent.resolve()
    if not (agent / "agent.py").is_file():
        raise SystemExit(f"{agent} has no agent.py")

    schedule = openings.pairs(2 * arguments.games)
    fens = [fen for index, (fen, _) in enumerate(schedule) if index % 2 == 0]
    jobs = [
        Job(
            agent, fen, arguments.base_ms, arguments.increment_ms, arguments.factor,
            arguments.overhead_ms, arguments.ply_cap,
        )
        for fen in fens[: arguments.games]
    ]
    workers = arguments.workers or min(len(jobs), 6)
    print(
        f"{agent.name}: {len(jobs)} games at {arguments.base_ms / 1000:g}s"
        f"+{arguments.increment_ms / 1000:g}s, charged x{arguments.factor:g}"
        f" +{arguments.overhead_ms:g} ms/move, {workers} workers",
        flush=True,
    )

    reports: list[Report] = []
    # One game per process: the agent keeps module state for a whole game, and a
    # fresh process is the only way to give every game the clean start it gets on
    # the platform.
    with mp.Pool(workers, maxtasksperchild=1) as pool:
        for report in pool.imap_unordered(play, jobs):
            reports.append(report)
            status = f"FLAG {report.flagged}" if report.flagged else (report.error or report.result)
            low = report.min_clock_ms / 1000
            used = f"w {report.used_white_ms / 1000:5.0f}s b {report.used_black_ms / 1000:5.0f}s"
            print(
                f"  {report.plies:4d} plies  {status:<12}  lowest clock {low:6.1f}s"
                f"  longest move {report.max_move_ms / 1000:5.1f}s"
                f"  at ply 30 {report.clock_at_30 / 1000:6.1f}s  used {used}",
                flush=True,
            )

    flags = sum(1 for r in reports if r.flagged)
    errors = sum(1 for r in reports if r.error)
    lowest = min((r.min_clock_ms for r in reports), default=0.0)
    longest = max((r.max_move_ms for r in reports), default=0.0)
    print()
    print(
        f"flags {flags}/{len(reports)}  errors {errors}  lowest clock {lowest / 1000:.1f}s"
        f"  longest move {longest / 1000:.1f}s"
    )
    ok = flags == 0 and errors == 0 and lowest >= arguments.min_clock_ms
    floor = arguments.min_clock_ms / 1000
    print("PASS" if ok else f"FAIL (need no flags, no errors, clock >= {floor:g}s)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
