"""Estimate absolute strength against externally-rated opponents.

Every other measurement in this project is relative -- "better than my last one" --
which is the only thing that matters for iterating, and tells you nothing about
where you would land in a field of strangers. This plays a ladder of opponents with
published ratings and reports a bracket.

**Read the output as a bracket, not a number.** Three reasons, all measured:

  * A skill-limited engine does not have a scalar rating. DeepMind measured the
    same unchanged model at 2895 Elo against humans and 2299 against bots -- a 596
    point gap from opponent pool alone -- because an engine whose mistakes are
    unlike human mistakes is easier or harder to beat depending on who is playing.
  * Stockfish's Skill Level ladder is engine-scale (CCRL Blitz), calibrated against
    stash-bot builds, and its own author's caveat is "+/- 100 Elo of CCRL, probably
    because it depends quite a bit on the opponent pool".
  * Skill 0 scored only 31% in its own calibration pool, so the bottom rung is
    extrapolation rather than measurement.

What this is genuinely good for is transfer validation. Self-play SPRT can drift --
you optimise against your own weaknesses and the gains stop generalising -- and
roughly 60% of self-play Elo is reckoned to survive into games against strangers.
A ladder of outsiders is how that gets caught.
"""

import argparse
import math
from pathlib import Path

from testing import arena

# Measured in Stockfish's own recalibration (commit a08b8d4, PR #4341): Ordo over
# matches at 120s+1s against stash-bot versions ranked on CCRL. Not human Elo.
# Skill 0 and 2 were dropped after the first run: we swept them 100% and 97%, so
# they only bounded us from below, and their calibration is the ladder's weakest
# (skill 0 scored just 31% in its own pool, making everything under ~1450
# extrapolation). Skill 8 and 10 replace them, to find the rung where the score is
# near 50% -- that is the only rung that actually locates strength rather than
# bounding it.
LADDER: dict[str, tuple[float, int]] = {
    "sf-skill6": (2363.2, 4379),
    "sf-skill8": (2596.2, 4975),
    "sf-skill10": (2788.3, 5511),
}

# A second engine family, at fixed depth. These have no published rating, so they
# are rated here by playing a rung of known strength against them -- `--anchor`.
#
# The point is not another number but a check on the first one. The Stockfish rungs
# disagreed with each other by 500 Elo, climbing monotonically with opponent
# strength, and four rungs of one engine share failure modes so they cannot say
# whether that is Stockfish's calibration or our matchup. A different engine can.
CROSSCHECK: tuple[str, ...] = ("weiss-d4", "weiss-d6", "weiss-d8")


def elo_difference(score: float, games: int) -> tuple[float, float]:
    """Elo difference implied by a score, and its 95% half-width.

    A clean sweep implies an infinite difference, so the score is clamped: with a
    lopsided result the honest statement is a bound, not a point estimate.
    """
    clamped = min(max(score, 0.5 / (games + 1)), 1.0 - 0.5 / (games + 1))
    difference = -400.0 * math.log10(1.0 / clamped - 1.0)
    error = math.sqrt(max(clamped * (1.0 - clamped), 1e-9) / games)
    high = min(clamped + 1.96 * error, 1.0 - 1e-9)
    low = max(clamped - 1.96 * error, 1e-9)
    span = (-400.0 * math.log10(1.0 / high - 1.0)) - (-400.0 * math.log10(1.0 / low - 1.0))
    return difference, abs(span) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate absolute strength.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponents", type=Path, default=Path("opponents"))
    parser.add_argument("--games", type=int, default=60, help="games per rung")
    parser.add_argument("--base-ms", type=int, default=20_000)
    parser.add_argument("--increment-ms", type=int, default=200)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--anchor",
        default="",
        help="rate the CROSSCHECK opponents by playing this rung against them, then stop",
    )
    arguments = parser.parse_args()

    workers = arguments.workers or arena.default_workers()

    if arguments.anchor:
        reference = (arguments.opponents / arguments.anchor).resolve()
        rating = LADDER[arguments.anchor][0]
        print(f"anchoring against {arguments.anchor} (nominal {rating:.0f})\n")
        for name in CROSSCHECK:
            directory = (arguments.opponents / name).resolve()
            if not (directory / "agent.py").is_file():
                continue
            tally, _ = arena.run(
                reference, directory, arguments.games, arguments.base_ms,
                arguments.increment_ms, 300, workers, 0.0, 20.0, quiet=True,
            )
            if not tally.games:
                continue
            score = (tally.wins + tally.draws / 2.0) / tally.games
            difference, margin = elo_difference(score, tally.games)
            # The reference scored `score` against this opponent, so the opponent
            # sits that far below it.
            print(
                f"  {name:<12} reference scored {score:>5.1%} over {tally.games:>3} games"
                f"  -> {rating - difference:.0f} +/- {margin:.0f}"
            )
        return

    estimates: list[tuple[str, float, float, float, float]] = []

    for name, (rating, calibration_games) in LADDER.items():
        directory = (arguments.opponents / name).resolve()
        if not (directory / "agent.py").is_file():
            print(f"  {name}: not built, skipping")
            continue
        print(f"\n=== {name} (nominal {rating:.0f}, from {calibration_games:,} games) ===")
        tally, _ = arena.run(
            arguments.agent.resolve(),
            directory,
            arguments.games,
            arguments.base_ms,
            arguments.increment_ms,
            300,
            workers,
            0.0,
            20.0,
            quiet=True,
        )
        if not tally.games:
            continue
        score = (tally.wins + tally.draws / 2.0) / tally.games
        difference, margin = elo_difference(score, tally.games)
        estimate = rating + difference
        print(
            f"  +{tally.wins} ={tally.draws} -{tally.losses}  score {score:.1%}  "
            f"-> {difference:+.0f} Elo  => {estimate:.0f} +/- {margin:.0f}"
        )
        if tally.failures:
            print(f"  !! {tally.failures} failures -- this rung is a bug report, not a result")
        estimates.append((name, rating, score, estimate, margin))

    if not estimates:
        raise SystemExit("no rungs played")

    print("\n" + "=" * 62)
    print("  opponent      nominal   score        implied strength")
    for name, rating, score, estimate, margin in estimates:
        print(f"  {name:<12} {rating:>7.0f}   {score:>5.1%}   {estimate:>7.0f} +/- {margin:.0f}")

    # Weight each rung by how informative it is: a 50% score localises strength,
    # a 100% score only bounds it from below.
    weights = [1.0 / max(margin, 1.0) ** 2 for _, _, _, _, margin in estimates]
    combined = sum(w * e for w, (_, _, _, e, _) in zip(weights, estimates, strict=True)) / sum(
        weights
    )
    spread = [e for _, _, _, e, _ in estimates]
    print(
        f"\n  bracket: {min(spread):.0f} to {max(spread):.0f}, "
        f"precision-weighted centre {combined:.0f}"
    )
    print("  engine-scale (CCRL Blitz), not human Elo; treat +/-100 as the floor on error")


if __name__ == "__main__":
    main()
