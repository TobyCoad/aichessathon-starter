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
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path

from harness.sandbox import local
from testing import openings, sprt
from testing.referee import FAILED_TERMINATIONS, PLATFORM_PLY_CAP, play_match

FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100


def default_workers() -> int:
    """Concurrent games that will not oversubscribe the CPU.

    Each game runs *two* agent processes, so the number of games must be half the
    core count, not the whole of it. Getting this wrong is not merely slow: agents
    measure their deadline in wall time but only poll it every 1024 nodes, so a
    descheduled process blows through its budget and flags. Measured -- at 12
    workers on 16 cores, 1.5x oversubscribed, a 222-game match produced 46 flags;
    the same engines single-threaded overshot on zero moves out of 59.
    """
    import os

    return max(1, ((os.cpu_count() or 4) // 2) - 1)


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
    pgn_dir: str = ""


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
    if task.pgn_dir:
        # Keep the game so it can be replayed. Cheap, and the only way to see what
        # a score actually looked like on the board.
        colour = "w" if task.agent_white else "b"
        name = f"{task.pair:03d}{colour}-{white.name}-vs-{black.name}.pgn"
        try:
            Path(task.pgn_dir).mkdir(parents=True, exist_ok=True)
            (Path(task.pgn_dir) / name).write_text(outcome.pgn + "\n", encoding="utf-8")
        except OSError:
            pass
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


def checkpoint_verdict(
    verdict: sprt.Verdict, games: int, checkpoint: int, promote_at: float, reject_at: float
) -> sprt.Verdict:
    """The every-N-games rule: promote a positive trend, reject a clearly negative one
    from the second checkpoint on, otherwise play on."""
    if verdict.elo >= promote_at:
        print(
            f"  checkpoint {games}: {verdict.elo:+.0f} Elo, trending positive -> PROMOTE early",
            flush=True,
        )
        return replace(verdict, decision="accept")
    if games >= 2 * checkpoint and verdict.elo <= reject_at:
        print(f"  checkpoint {games}: {verdict.elo:+.0f} Elo, negative -> REJECT early",
              flush=True)
        return replace(verdict, decision="reject")
    print(f"  checkpoint {games}: {verdict.elo:+.0f} Elo, undecided -> {checkpoint} more",
              flush=True)
    return verdict


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
    pgn_dir: str = "",
    checkpoint: int = 0,
    promote_at: float = 10.0,
    reject_at: float = -10.0,
) -> tuple[Tally, sprt.Verdict]:
    """Play until the SPRT concludes or `max_games` is reached.

    With `checkpoint` > 0 the run is also judged every `checkpoint` games: an Elo
    estimate at or above `promote_at` accepts early (many small gains shipped fast
    beat one certain verdict), one at or below `reject_at` rejects early from the
    second checkpoint on, and anything in between plays on to the next checkpoint.
    """
    schedule = openings.pairs(max_games)
    tasks = [
        Task(index // 2, agent, opponent, fen, white, base_ms, increment_ms, ply_cap, pgn_dir)
        for index, (fen, white) in enumerate(schedule)
    ]

    tally = Tally()
    partial: dict[int, list[float]] = defaultdict(list)
    verdict = sprt.Verdict("continue", 0.0, 0, 0.0, float("inf"))
    queued = 0
    next_check = checkpoint
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
                        if checkpoint and tally.games >= next_check:
                            next_check += checkpoint
                            if verdict.decision == "continue":
                                verdict = checkpoint_verdict(
                                    verdict, tally.games, checkpoint, promote_at, reject_at
                                )

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
    parser.add_argument("--ply-cap", type=int, default=PLATFORM_PLY_CAP)
    parser.add_argument(
        "--workers", type=int, default=0, help="concurrent games; 0 picks a safe default"
    )
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=20.0)
    parser.add_argument(
        "--pgn-dir",
        default=None,
        help="where to keep every game; default overnight/pgn/<agent>-vs-<opponent>-<time>",
    )
    arguments = parser.parse_args()

    workers = arguments.workers or default_workers()
    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    pgn_dir = arguments.pgn_dir
    if pgn_dir is None:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        pgn_dir = f"overnight/pgn/{agent.name or 'agent'}-vs-{opponent.name}-{stamp}"

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
        pgn_dir=pgn_dir,
    )
    report(agent, opponent, tally, verdict)
    print(f"  games saved under {pgn_dir}")
    raise SystemExit(1 if tally.failures else 0)


if __name__ == "__main__":
    main()
