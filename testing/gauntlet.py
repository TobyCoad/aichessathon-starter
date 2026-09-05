"""Decide whether a challenger should replace the champion.

This is the safety interlock for unattended development. An automated loop that
edits the engine overnight will, without a stopping rule, cheerfully promote a
regression on a lucky 40-game score and report success. So the loop never decides
anything itself: it builds a challenger, runs this, and obeys the exit code.

    0  PROMOTE      passed the crash gate and won the SPRT
    1  REJECT       lost the SPRT, or failed the crash gate
    2  INCONCLUSIVE  ran out of games without a verdict; champion stays

Stage 1 is a crash gate, not a strength test. An agent that crashes, flags or
returns an illegal move loses that game outright, and no amount of playing
strength compensates. A challenger that fails a game for any of those reasons is
rejected without ever reaching the SPRT.
"""

import argparse
from pathlib import Path

from testing import arena, sprt
from testing.referee import FAILED_TERMINATIONS

GATE_GAMES = 24
GATE_BASE_MS = 4_000
GATE_INCREMENT_MS = 50


def gate(challenger: Path, workers: int) -> tuple[bool, str]:
    """Play a short match against the random baseline hunting for failures.

    Random play reaches the odd corners -- promotion, en passant, stalemate,
    positions with a single legal move -- far faster than a real opponent does.
    """
    tally, _ = arena.run(
        challenger,
        Path("baselines/random").resolve(),
        GATE_GAMES,
        GATE_BASE_MS,
        GATE_INCREMENT_MS,
        300,
        workers,
        0.0,
        20.0,
        quiet=True,
    )
    if tally.failures:
        broken = ", ".join(
            f"{name} {count}"
            for name, count in sorted(tally.terminations.items())
            if name in FAILED_TERMINATIONS
        )
        return False, f"failed {tally.failures}/{tally.games} games ({broken})"
    score = (tally.wins + tally.draws / 2.0) / max(tally.games, 1)
    if score < 0.8:
        return False, f"only scored {score:.0%} against the random baseline"
    return True, f"clean over {tally.games} games, {score:.0%} vs random"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate and SPRT a challenger.")
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--champion", type=Path, default=Path("."))
    parser.add_argument("--games", type=int, default=800)
    parser.add_argument("--base-ms", type=int, default=8_000)
    parser.add_argument("--increment-ms", type=int, default=80)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=20.0)
    parser.add_argument("--checkpoint", type=int, default=200, help="judge every N games; 0 = off")
    parser.add_argument("--promote-at", type=float, default=10.0, help="checkpoint Elo to promote")
    parser.add_argument(
        "--reject-at", type=float, default=-10.0, help="2nd+ checkpoint Elo to reject"
    )
    arguments = parser.parse_args()

    workers = arguments.workers or arena.default_workers()
    challenger = arguments.challenger.resolve()
    champion = arguments.champion.resolve()

    if not (challenger / "agent.py").is_file():
        print(f"REJECT  {challenger} has no agent.py")
        raise SystemExit(1)

    print(f"stage 1: crash gate, {challenger.name} vs random")
    ok, detail = gate(challenger, workers)
    print(f"  {detail}")
    if not ok:
        print(f"\nREJECT  {challenger.name} is not safe to ship")
        raise SystemExit(1)

    print(f"\nstage 2: SPRT[{arguments.elo0:g}, {arguments.elo1:g}], challenger vs champion")
    tally, verdict = run_sprt(challenger, champion, arguments, workers)
    arena.report(challenger, champion, tally, verdict)

    if tally.failures:
        print(f"\nREJECT  {challenger.name} failed games during the SPRT")
        raise SystemExit(1)
    if verdict.decision == "accept":
        print(f"\nPROMOTE  {challenger.name}  {verdict.elo:+.0f} Elo over {tally.games} games")
        raise SystemExit(0)
    if verdict.decision == "reject":
        print(f"\nREJECT  {challenger.name}  no better than the champion")
        raise SystemExit(1)
    print(f"\nINCONCLUSIVE  {challenger.name}  ran out of games; champion stays")
    raise SystemExit(2)


def run_sprt(
    challenger: Path, champion: Path, arguments: argparse.Namespace, workers: int
) -> tuple[arena.Tally, sprt.Verdict]:
    return arena.run(
        challenger,
        champion,
        arguments.games,
        arguments.base_ms,
        arguments.increment_ms,
        300,
        workers,
        arguments.elo0,
        arguments.elo1,
        checkpoint=arguments.checkpoint,
        promote_at=arguments.promote_at,
        reject_at=arguments.reject_at,
    )


if __name__ == "__main__":
    main()
