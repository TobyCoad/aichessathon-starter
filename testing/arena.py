"""Parallel, paired, SPRT-terminated match runner.

What this adds over `harness/arena.py`, which stays as the reference:

  * games start from 40 balanced openings instead of always the initial position,
    so two deterministic agents do not replay one game N times;
  * every opening is played from both sides, and the pair is scored together;
  * games run across cores, because a 500-game match is otherwise an afternoon;
  * the match stops when the evidence is decisive rather than at a round number.

Agent failures -- crash, illegal move, flag, init -- are counted separately and
reported loudly. They are bugs, not results, and a match containing them should
not be read as a measurement.
"""

import argparse
import sys
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

from harness.rules import PLY_CAP
from harness.sandbox import local
from testing import openings, sprt
from testing.referee import FAILED_TERMINATIONS, play_match

FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100


def _initialise() -> None:
    """Install the Windows transport shim inside each worker process."""
    if sys.platform == "win32":
        from tools.winshim import install

        install()


@dataclass(frozen=True)
class Task:
    pair: int
    agent: Path
    opponent: Path
    fen: str
    agent_white: bool
    base_ms: int
    increment_ms: int
    ply_cap: int


@dataclass(frozen=True)
class GameResult:
    pair: int
    score: float
    termination: str
    plies: int
    agent_failed: bool


def play(task: Task) -> GameResult:
    white, black = (
        (task.agent, task.opponent) if task.agent_white else (task.opponent, task.agent)
    )
    outcome = play_match(
        local(white),
        local(black),
        task.base_ms,
        task.increment_ms,
        ply_cap=task.ply_cap,
        start_fen=task.fen,
    )
    if outcome.result == "draw" or outcome.result == "void":
        score = 0.5
    elif (outcome.result == "white") == task.agent_white:
        score = 1.0
    else:
        score = 0.0

    # A failure termination is attributed to whichever side failed; the referee
    # awards the game to the opponent, so a score of 0 on a failed termination
    # means our agent was the one that broke.
    agent_failed = outcome.termination in FAILED_TERMINATIONS and score == 0.0
    return GameResult(task.pair, score, outcome.termination, outcome.plies, agent_failed)


@dataclass
class Tally:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    failures: int = 0
    plies: int = 0
    games: int = 0
    terminations: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pair_scores: list[float] = field(default_factory=list)

    def record(self, result: GameResult) -> None:
        self.games += 1
        self.plies += result.plies
        self.terminations[result.termination] += 1
        if result.agent_failed:
            self.failures += 1
        if result.score == 1.0:
            self.wins += 1
        elif result.score == 0.5:
            self.draws += 1
        else:
            self.losses += 1


def run(
    agent: Path,
    opponent: Path,
    max_games: int,
    base_ms: int,
    increment_ms: int,
    ply_cap: int,
    workers: int,
    elo0: float,
    elo1: float,
    quiet: bool = False,
) -> tuple[Tally, sprt.Verdict]:
    """Play until the SPRT concludes or `max_games` is reached."""
    schedule = openings.pairs(max_games)
    tasks = [
        Task(index // 2, agent, opponent, fen, white, base_ms, increment_ms, ply_cap)
        for index, (fen, white) in enumerate(schedule)
    ]

    tally = Tally()
    partial: dict[int, list[float]] = defaultdict(list)
    verdict = sprt.Verdict("continue", 0.0, 0, 0.0, float("inf"))
    queued = 0
    pending: set[Future[GameResult]] = set()

    with ProcessPoolExecutor(max_workers=workers, initializer=_initialise) as pool:
        try:
            while queued < len(tasks) or pending:
                while queued < len(tasks) and len(pending) < workers * 2:
                    pending.add(pool.submit(play, tasks[queued]))
                    queued += 1

                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    tally.record(result)
                    partial[result.pair].append(result.score)
                    if len(partial[result.pair]) == 2:
                        tally.pair_scores.append(sum(partial.pop(result.pair)) / 2.0)
                        verdict = sprt.evaluate(tally.pair_scores, elo0, elo1)
                        if not quiet:
                            print(f"  {tally.games:>5} games  {verdict.summary}", flush=True)

                if verdict.decision != "continue":
                    for future in pending:
                        future.cancel()
                    break
        except KeyboardInterrupt:
            for future in pending:
                future.cancel()
            raise

    return tally, verdict


def report(agent: Path, opponent: Path, tally: Tally, verdict: sprt.Verdict) -> None:
    score = (tally.wins + tally.draws / 2.0) / max(tally.games, 1)
    print(f"\n{agent} vs {opponent}")
    print(
        f"  +{tally.wins} ={tally.draws} -{tally.losses}  "
        f"score {score:.1%}  over {tally.games} games"
    )
    print(f"  {verdict.summary}")
    print(f"  mean game length {tally.plies / max(tally.games, 1):.0f} plies")
    print(
        "  terminations: "
        + ", ".join(f"{name} {count}" for name, count in sorted(tally.terminations.items()))
    )
    if tally.failures:
        print(f"\n  !! your agent failed {tally.failures} games. This is a bug, not a result.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired, parallel, SPRT-terminated match.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/greedy"))
    parser.add_argument(
        "--games", type=int, default=1000, help="upper bound; the SPRT usually stops first"
    )
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--workers", type=int, default=0, help="0 picks cores-2")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=20.0)
    arguments = parser.parse_args()

    import os

    workers = arguments.workers or max(1, (os.cpu_count() or 4) - 2)
    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()

    print(
        f"{agent.name} vs {opponent.name} | {arguments.base_ms / 1000:g}s"
        f"+{arguments.increment_ms / 1000:g}s | {workers} workers | "
        f"SPRT[{arguments.elo0:g}, {arguments.elo1:g}]"
    )
    tally, verdict = run(
        agent,
        opponent,
        arguments.games,
        arguments.base_ms,
        arguments.increment_ms,
        arguments.ply_cap,
        workers,
        arguments.elo0,
        arguments.elo1,
    )
    report(agent, opponent, tally, verdict)
    raise SystemExit(1 if tally.failures else 0)


if __name__ == "__main__":
    main()
