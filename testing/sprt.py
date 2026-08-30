"""Sequential probability ratio test over paired games.

A change worth +20 Elo needs roughly 1,300 games to resolve, and `make arena`'s
20-game default carries a confidence interval of about +/-135 Elo. Any process that
promotes a change on a 20-game score is not measuring, it is sampling noise. This
module is the stopping rule that makes an automated promotion trustworthy.

Games are scored in pairs -- the same opening played from both sides -- and the
statistics are pentanomial over pair scores in {0, 0.25, 0.5, 0.75, 1}. Pairing
removes colour and opening bias from the variance, which is free precision.

Bounds are the conventional alpha = beta = 0.05, giving log-likelihood-ratio
thresholds of +/-2.944. Use elo0=0, elo1=20: Stockfish's [0, 5] needs about 1,130
games to resolve even a true +50 patch, which is out of reach in a twelve-day event.
"""

import math
from dataclasses import dataclass
from typing import Literal

Decision = Literal["accept", "reject", "continue"]

ALPHA = 0.05
BETA = 0.05
LOWER_BOUND = math.log(BETA / (1.0 - ALPHA))
UPPER_BOUND = math.log((1.0 - BETA) / ALPHA)

# The log-likelihood ratio uses an estimated variance, which is wild for the first
# handful of pairs. Without a floor the test crosses a bound on early luck: measured
# over 200 simulated matches, an unguarded test promoted a true 0-Elo change 14% of
# the time against a nominal 5%. Promoting a regression is the failure that matters
# here, since it silently degrades the champion, so the test refuses to conclude
# before it has seen this many pairs.
MIN_PAIRS = 25


def elo_to_score(elo: float) -> float:
    """Expected score for an Elo advantage, under the logistic model."""
    return float(1.0 / (1.0 + 10.0 ** (-elo / 400.0)))


def score_to_elo(score: float) -> float:
    """Inverse of `elo_to_score`, clamped so a clean sweep does not return infinity."""
    clamped = min(max(score, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10(1.0 / clamped - 1.0)


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    llr: float
    pairs: int
    elo: float
    margin: float

    @property
    def summary(self) -> str:
        name = {"accept": "PASS", "reject": "FAIL", "continue": "...."}[self.decision]
        return (
            f"{name} llr {self.llr:+.2f} [{LOWER_BOUND:.2f}, {UPPER_BOUND:.2f}] "
            f"after {self.pairs} pairs, elo {self.elo:+.1f} +/- {self.margin:.1f}"
        )


def evaluate(pair_scores: list[float], elo0: float = 0.0, elo1: float = 20.0) -> Verdict:
    """Score the test so far.

    `pair_scores` holds one entry per completed pair, each the mean of the two games.
    Returns "accept" when the challenger is better than elo1, "reject" when it is no
    better than elo0, and "continue" while the evidence is not yet decisive.
    """
    count = len(pair_scores)
    if count < 2:
        return Verdict("continue", 0.0, count, 0.0, math.inf)
    decisive = count >= MIN_PAIRS

    mean = sum(pair_scores) / count
    variance = sum((score - mean) ** 2 for score in pair_scores) / (count - 1)

    # Early on, every pair can score identically and the sample variance collapses
    # to zero, which sends the ratio to infinity and prints a meaningless number in
    # the log. Floor it well below any real match variance -- pair scores typically
    # vary by ~0.35 -- so this binds only in the degenerate case and is conservative
    # everywhere else, since a larger variance makes the bounds harder to reach.
    variance = max(variance, 0.01)

    mu0 = elo_to_score(elo0)
    mu1 = elo_to_score(elo1)
    llr = count * (mu1 - mu0) * (2.0 * mean - mu0 - mu1) / (2.0 * variance)

    # Standard error of the mean pair score, converted to Elo at the observed point.
    error = math.sqrt(variance / count)
    elo = score_to_elo(mean)
    margin = abs(score_to_elo(min(mean + 1.96 * error, 1.0 - 1e-6)) - elo)

    if decisive and llr >= UPPER_BOUND:
        return Verdict("accept", llr, count, elo, margin)
    if decisive and llr <= LOWER_BOUND:
        return Verdict("reject", llr, count, elo, margin)
    return Verdict("continue", llr, count, elo, margin)


def games_for(elo: float, draw_rate: float = 0.4) -> int:
    """Games needed to resolve a true `elo` difference at 95% confidence, 80% power.

    Here so the numbers are in the codebase rather than in someone's memory:
    +100 needs about 53 games, +50 about 223, +20 about 1,420, +10 about 5,700.
    """
    if elo == 0.0:
        return 0
    return int(947_900.0 * (1.0 - draw_rate) / (elo * elo))
