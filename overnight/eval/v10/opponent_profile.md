# Profile of the leader, "Emile Andrieu" (measured 6 Sep 2026)

Everything below is computed from `overnight/pgn/leader/*.pgn` (29 games) and
`overnight/pgn/platform/round-*.pgn` (37 games), with Stockfish 1 thread / 64 MB hash at
**depth 16** as the reference (analysis only). Every position of all 66 games was scanned:
4,609 own-move records. Scripts and raw outputs are in this directory (`_struct.py`,
`_sfscan.py`, `_acpl.py`, `_depth.py`, `_convert*.py`, `_ourmoves.py`, `_scoreours.py`,
`_same.py`, `_elo.py`; outputs `sf_*.json`, `acpl_full.txt`, `bands.txt`, `depth.txt`,
`convert*.txt`, `same_positions*.txt`, `pooled.txt`, `style.txt`).

---

## 1. Ten-line summary

1. The gap is **+501 Elo** (bootstrap 95% CI **[+334, +741]**), from a logistic fit over the
   61-game union of both archives; the naive common-opponent figure is +540.
2. Widening the common-opponent set from 9 to **15 shared opponents** (filename slugs, not
   the `[White]/[Black]` tags -- 21 of our PGNs carry `?` for both players): they score
   **16.0/17 = 94.1%**, we score **7.5/18 = 41.7%**.
3. Corpus-wide ACPL: **15.1 cp (n=2166) them vs 21.2 cp (n=2443) us**; restricted to
   contested positions (|ref eval| <= 300 cp), **6.4 vs 9.3**.
4. On **521 identical positions** replayed by our own engine with the leader's own clock,
   the paired deficit is **+6.5 cp/move [+3.0, +10.7]** -- but it is *entirely* concentrated
   in positions that are already won.
5. In dead-level positions (|eval| <= 100 cp, n=255) the paired difference is **+0.0 cp
   [-1.6, +1.5]**: move for move, **our engine is their equal when the game is balanced**.
6. In won positions the deficit explodes: **+14.6 cp/move [+8.0, +21.7]** at +300..+800
   (n=107) and **+49.1 [+13.5, +91.9]** above +800 (n=38). Corpus-wide it is the same shape:
   9.0 vs 27.1 cp at +300..+800.
7. In *lost* positions we are **better** than they are: -16.3 cp/move [-29.5, -5.5] at
   eval < -300 (10.8 vs 27.1). We defend; we do not finish.
8. Consequence: they spend **32.4% of their moves in won positions (>+300) vs our 16.3%**,
   and reach +500 in **22 of 24 decided games (92%)** against our **16 of 37 (43%)**; they
   drop below -150 in **2/24 (8%)**, we in **13/37 (35%)**, scoring 0.077 from there.
9. **No search advantage is detectable.** Agreement with Stockfish peaks at depth 14 for
   both (53.8% them, 52.2% us); moves Stockfish only prefers from depth >= 12 are 2.46% of
   theirs and 2.82% of ours; on the same 280 contested positions both match SF-d16 at
   **exactly 52.9%**.
10. Their clock is a near-constant fraction: spend/clock median **4.06%, IQR [3.60%,
    4.87%]**, and their **longest move in 2,137 measured moves is 4.28 s** (ours 11.14 s).
    In identical won positions they spend **1.55 s median to our 0.90 s** -- we throttle
    thinking exactly where we are weakest.

---

## 2. ACPL comparison, with sample sizes and error bars

Reference: Stockfish depth 16, evaluation clamped to +-2000 cp, CPL = eval(before, mover POV)
- eval(after, mover POV), floored at 0. `+-` is a 95% normal interval on the mean.

### 2a. Each side in its own games (different positions, different opponents)

| set | n moves | ACPL | median | matches SF-d16 | >=200 cp | <=10 cp |
|---|---|---|---|---|---|---|
| leader, all positions | 2166 | **15.1 +- 4.4** | 0 | 55.4% | 1.1% | 82.1% |
| us, all positions | 2443 | **21.2 +- 3.7** | 0 | 53.2% | 1.8% | 75.2% |
| leader's opponents | 2156 | 26.4 +- 4.1 | 1 | 49.7% | 2.4% | 69.0% |
| our opponents | 2443 | 24.1 +- 4.4 | 0 | 52.6% | 1.8% | 73.5% |
| leader, \|eval\| <= 300 | 1421 | **6.4 +- 0.9** | 0 | 52.6% | 0.1% | 83.7% |
| us, \|eval\| <= 300 | 1773 | **9.3 +- 1.1** | 0 | 52.1% | 0.2% | 78.1% |

Our opponent pool is if anything *weaker* than theirs (contested ACPL 13.4 vs 11.5), so
opponent quality does not explain the result gap.

### 2b. The head-to-head game, both colours, identical positions

`round-29-loss-white-vs-emile-andrieu-f1aabee5.pgn` = `round-21-win-black-vs-make-no-mistakes-f1aabee5.pgn`

| | n | ACPL | contested ACPL | matches SF-d16 | >=200 cp errors |
|---|---|---|---|---|---|
| us (White) | 63 | 25.6 +- 9.2 | 17.7 +- 8.9 (n=35) | 50.8% | 0 |
| leader (Black) | 64 | 16.3 +- 13.0 | 7.5 +- 5.5 (n=33) | 60.9% | 1 |

One game, so this is indicative only. It is a pure grind: no single move of ours lost 200 cp;
the reference eval slid from +11 at ply 0 to -23 (ply 10), -58 (50), -327 (70), -766 (100),
mate (120). We were never blundered out, we were out-played 20 cp at a time.

### 2c. The controlled test -- our engine in their seat

280 + 241 = **521 leader-to-move positions** from their 29 games, replayed by our current
`agent.py` through `get_move(fen, their_actual_remaining_clock_ms)`, scored against the same
depth-16 reference. Paired (same position, same reference, same machine, same background
load), bootstrap CIs from 4,000-5,000 resamples.

| band | n | leader ACPL | our ACPL | paired diff (ours - leader) | 95% CI |
|---|---|---|---|---|---|
| \|eval\| <= 100 | 255 | 5.9 | 5.9 | **+0.0** | [-1.6, +1.5] |
| +100..+300 | 79 | 12.9 | 20.8 | +7.9 | [-0.8, +17.9] |
| +300..+800 | 107 | 8.4 | 23.0 | **+14.6** | [+8.0, +21.7] |
| > +800 | 38 | 65.4 | 114.5 | **+49.1** | [+13.5, +91.9] |
| < -300 | 36 | 27.1 | 10.8 | **-16.3** | [-29.5, -5.5] |
| all pooled | 521 | 13.4 | 20.0 | +6.5 | [+3.0, +10.7] |

Load control: a gauntlet (8 `harness/runner.py` processes, `v94-120s` vs `153-mixnet2`) was
confirmed running when the scans started and had finished by the time they ended; its exact
end time is unknown, so both replay runs saw some background load. That cannot manufacture
this result -- the *same* engine on the *same* machine under the *same* conditions shows
**zero** deficit in level positions and a large one only when winning. Both replays were
interleaved across the whole sample, so any load change is spread across all bands.

Sample-size honesty: the >+800 band is 38 paired positions and its interval is very wide; the
+100..+300 band (n=79) straddles zero. The two solid claims are the level band (n=255, tight
zero) and the +300..+800 band (n=107, clearly non-zero). Corpus-wide band ACPL (`bands.txt`)
reproduces both: 4.8 vs 7.3 at |eval|<=100 (position-selection differences), 9.0 vs 27.1 at
+300..+800, 55.1 vs 78.2 above +800.

---

## 3. Phase breakdown

### By piece count (contested positions only, |eval| <= 300)

| pieces | leader ACPL | our ACPL | ratio | leader match | our match |
|---|---|---|---|---|---|
| >= 27 | 6.9 +- 1.2 (n=323) | 11.4 +- 2.9 (n=356) | 1.65x | 63.5% | 52.2% |
| 17-26 | 7.2 +- 1.4 (n=479) | 12.0 +- 1.9 (n=558) | 1.67x | 61.2% | 62.0% |
| 11-16 | 10.1 +- 3.2 (n=212) | 15.2 +- 5.0 (n=243) | 1.50x | 61.8% | 52.7% |
| <= 10 | 3.2 +- 1.5 (n=407) | 3.4 +- 0.9 (n=616) | 1.06x | 29.2% | 42.7% |

### By piece count (all positions)

| pieces | leader ACPL | our ACPL | leader >=200 cp | our >=200 cp |
|---|---|---|---|---|
| >= 27 | 7.0 (n=333) | 11.8 (n=364) | 0.0% | 0.3% |
| 17-26 | 10.5 (n=610) | 16.8 (n=689) | 0.7% | 0.7% |
| 11-16 | 19.8 (n=417) | 28.2 (n=437) | 1.9% | **3.4%** |
| <= 10 | 19.6 (n=806) | 24.9 (n=953) | 1.5% | **2.5%** |

### The paired test, by piece count *within won positions* (eval > +300)

| pieces | n | leader | ours | paired diff | 95% CI |
|---|---|---|---|---|---|
| 17-26 | 45 | 29.8 | 48.2 | +18.4 | [+4.6, +37.0] |
| 11-16 | 45 | 11.2 | 25.4 | +14.3 | [+5.2, +26.1] |
| <= 10 | 45 | 32.0 | 73.5 | **+41.6** | [+14.9, +78.4] |

**Read against our known static-eval error (70 / 155 / 475 cp at 27-32 / 17-26 / 11-16, net
error 331.9 / 262.6 / 184.4):** the piece-count axis is *not* where the leader beats us. In
contested positions the ratio is a flat ~1.5-1.7x from 32 pieces down to 17, and vanishes at
<= 10 pieces. The axis that separates us is the **evaluation** axis, and inside that the
worst single cell is **won endgames at <= 10 pieces (+41.6 cp/move)** -- exactly where a
search cannot help and a technique/eval term must. Our two largest single departures are both
there: `round-23-win-black-vs-ransom` ply 142, 7 pieces, +1273 -- they lost 0 cp, we lost
**581**; `round-22-win-white-vs-lightning-tree` ply 211, 6 pieces, +1311 -- they 0, we **382**.

### Error concentration

| | cp lost per game (median) | top-3 moves' share | worst 1% of moves carry | worst 5% carry |
|---|---|---|---|---|
| leader | 757 | 59% | 50% | 75% |
| us | **1157** | 51% | 38% | 67% |

Their losses are a few discrete errors; ours are a broader drip.

---

## 4. Search-depth / node-rate estimate

**Estimate: their effective search is not measurably deeper or faster than ours -- both sit
around a depth-10-14 Stockfish-agreement plateau. I find no evidence of a search advantage,
and three independent tests agree.**

**Method 1 -- agreement-versus-depth.** One Stockfish search per position to depth 16, keeping
the PV move at every depth; then for each depth d, the share of a player's moves equal to
Stockfish's own d-ply choice. The depth that maximises agreement is a crude effective-depth
estimate.
- Leader: d1 49.8% -> d7 52.3% -> **d14 53.8%** -> d16 52.6%.
- Us: d1 48.3% -> d7 51.4% -> **d14 52.2%** -> d16 52.1%.
Same peak depth, 1.6 points apart. *Uncertainty:* the curve is nearly flat (4 points across
16 plies), so this test has very little power -- it can exclude a huge depth gap, not a small
one. With n=1421/1773 the binomial SE is ~1.3/1.2 points, so the 1.6-point gap is itself
barely 1 sigma.

**Method 2 -- depth-sensitive positions.** Restrict to contested positions where Stockfish's
choice at depth 4 differs from its choice at depth 16 (30% of theirs, 26% of ours). A deeper
searcher should play the d16 move more and the refuted d4 move less.
- Leader: plays the d16 move 30.6%, the refuted d4 move 23.2% (ratio 1.32), ACPL 6.1 +- 1.5.
- Us: plays the d16 move 29.9%, the refuted d4 move 28.1% (ratio 1.06), ACPL 9.4 +- 2.0.
A mild signal in their favour. But decisively: **their ACPL in depth-sensitive positions
(6.1) equals their ACPL in depth-insensitive ones (6.5), and so does ours (9.4 vs 9.3)**. If
depth were the differentiator, the gap would widen where depth matters. It does not.

**Method 3 -- moves only a deep search finds.** Share of a player's contested moves that
Stockfish first prefers at depth >= 12: **leader 2.46%, us 2.82%**. They play *fewer* of them.

**Method 4 -- the paired replay.** On 280 identical contested positions our engine, given
their own clock (median 1.78 s to their 1.74 s), matched Stockfish-d16 on **52.9%** of moves.
The leader matched on **52.9%**. Identical to the decimal.

**Method 5 -- time and complexity.** Their spend correlates +0.872 with piece count and -0.590
with |eval|, versus our +0.405 and -0.126; their spend/clock IQR is [3.60%, 4.87%] against our
[2.52%, 5.52%]; their longest move in 2,137 measured moves is 4.28 s (ours 11.14 s). Their
manager is close to a fixed 1/24 of the remaining clock with almost no per-position variance.
That is the signature of a manager that does *not* extend on complexity -- so they are not
buying depth with time either. Their clock decay (86.0 / 64.3 / 48.4 / 36.8 / 28.1 s left
after 10 / 20 / 30 / 40 / 50 own moves) is slightly *flatter* than ours (91.8 / 66.3 / 48.4 /
30.4 / 21.4).

**Node rate:** not estimable. Nothing in a PGN constrains nodes per second, and every
proxy above is confounded by their evaluation function. All I can bound is the *product*
(search x eval), and that product equals ours in level positions.

---

## 5. Structurally distinctive things about how they play

- **Openings are not a factor.** Every game starts from a platform-supplied `[FEN]`/`[SetUp]`
  position around move 6-9. Their 29 starts average **-0.3 cp** from their own POV (median -1,
  range -58..+41), ours **-9.5 cp** across 37 starts (median -10, range -76..+48). Colours are
  balanced (they 15W/14B, we 19W/18B). Instant (<0.15 s) moves: 5.4% them, 6.3% us.
- **They finish by mate.** 22 of their 24 result-bearing games ended in checkmate; the other
  two are insufficient-material draws. **Zero threefold, zero fifty-move.** We have 4 threefold
  and 2 fifty-move draws.
- **Our draws are mostly honest, not thrown.** Of our 11 draws, only **one** was a genuinely
  squandered win: `round-11-draw-black-vs-saucybeans` peaked at **+2000** and ended in threefold
  at 17 pieces. One more is marginal (`round-31` peak +326, 323 plies, insufficient material).
  The other 9 peaked at +6..+174. **The previously recorded "8 extra wins / +130 Elo from the
  draw rate" is not supported by the eval data.**
- **They are almost never worse.** Trough <= -150 in **2/24** games (8%) versus our **13/37**
  (35%); trough <= -300 in 2/24 versus 10/37. From <= -300 we have scored **0.000 in 10 games**.
- **They compound; we plateau.** Holding +100..+800, mean reference-eval change over the next
  20 plies -- 27-32 pieces: **+661 them / +265 us** (median +451 / +30); 17-26: +321 / +337
  (median **+197 / +116**, share that fell more than 100 cp **2% / 12%**); 11-16: +415 / +361
  (4% / 6%).
- **Conversion from a decisive edge is NOT our problem.** Once at +500 both sides finish:
  they won 21 of 22 such games, we won 15 of 16. We are in fact *faster* (median 32 plies from
  first +500 to mate versus their 41). The difference is **getting there**: 92% of their games
  versus 43% of ours.
- **They attack the king.** Non-pawn moves landing within 2 squares of the enemy king: **23.5%
  them vs 17.9% us** (mean distance 4.11 vs 4.25). Checks given per game 10.4 vs 6.7 received;
  ours 6.6 given vs 7.5 received.
- **They push pawns when ahead.** 23.3% of their moves when >= +100 are pawn moves, against our
  17.0%; **23 promotions to their opponents' 3**, while we have 9 to our opponents' 15.
- **Neither side avoids simplification.** Capture rate when ahead >= +100: 20.5% them, 20.1% us;
  level 13.0% / 14.8%. When behind they capture less (8.6% vs our 15.4%, n=70 -- small).
- **Their games are longer wins.** Median win 127 plies ending at 6 pieces (15 of their 21 wins
  end at <= 10 pieces); our median win is 88 plies ending at 8 pieces.
- **Clock correction.** The earlier note in `leader_benchmark.md` has the two sides swapped.
  Measured: median clock left at the end **25.1 s them vs 18.8 s us**; median total spent
  **124.4 s them vs 132.4 s us**; longest single move median 4.24 s / 6.60 s, max **4.28 s vs
  11.14 s**. In the head-to-head specifically they finished with **25.07 s** and we with
  **14.59 s**, having spent 122.8 s to our 136.9 s. **We spend more clock than they do, not
  less** -- and we still under-spend in the one place it matters (won positions).

---

## 6. What to change, ranked by expected Elo

The ranking is by the size of the measured deficit times the confidence that closing it is
reachable. Elo figures are estimates with the stated basis; only item 0 is a hard bound.

**0. (bound) The whole conversion-side prize is about +70 Elo, not +130.** In our 22 games
that reached +150 we scored 0.795; they score 0.955 in the same situation. Closing that
exactly is +3.5 points in 37 games: 0.554 -> 0.649, i.e. **+69 Elo**. Anything below that is a
share of this number. The rest of the 500-Elo gap is not visible as a conversion failure.

**1. Stop throttling the search in won positions. (est. +30 to +60 Elo; highest confidence)**
Measured: on identical positions with eval > +300 they spend **1.55 s median, we spend 0.90 s**
-- a 42% cut -- and we lose **23.0 vs their 8.4 cp/move** there. Two mechanisms in `agent.py`
produce this and both are already written: `_STABILITY_SCALE = (1.2, 1.1, 1.0, 0.9, 0.8)`
shrinks the soft budget once the best move repeats (which is exactly what happens in a won
position), and **`CONVERT_BUDGET` is `False`** -- the switch built to extend both deadlines
while `_CONV_LO <= score <= _CONV_HI` (120..900 cp) is off. Turning `CONVERT_BUDGET` on, and/or
suppressing the stability shrink when |root score| > ~300, is a one-flag experiment that
targets the largest measured cell. It also spends clock we are already banking: we finish with
18.8 s median, and their `_CONV_MAX_FRACTION`-style cap of 16% is nowhere near binding.

**2. Evaluation/technique in won endgames at <= 10 pieces. (est. +20 to +50 Elo)**
Measured: paired deficit **+41.6 cp/move [+14.9, +78.4]** on won positions at <= 10 pieces, and
our two worst single moves in the whole study are here (-581 cp at 7 pieces from +1273; -382 cp
at 6 pieces from +1311). Corpus-wide our <= 10-piece blunder rate (>= 200 cp) is 2.5% against
their 1.5%. Note the shape: in *contested* <= 10-piece positions we are their equal (3.4 vs 3.2)
and our SF match rate is *higher* (42.7% vs 29.2%). The failure is specifically converting a
large material edge -- shuffling instead of making progress. `_tablebase_move` already exists;
widening its coverage, or adding a progress/king-opposition/passed-pawn term that only fires
when |eval| is large, is the targeted fix. This is where their 23-versus-3 promotion count
comes from.

**3. Growth of a small edge, +100..+300. (est. +10 to +40 Elo; weakest evidence)**
Measured: paired +7.9 cp/move [-0.8, +17.9] (n=79, straddles zero); corpus-wide 15.2 vs 12.3;
and the outcome-level tell -- holding +100..+800 at 17-26 pieces, our advantage falls more than
100 cp within 20 plies **12% of the time against their 2%**. This is the band that decides
whether a game becomes a +500 win (their 92%) or drifts back to level (our 43%). I cannot
separate "eval flatness in slightly-better positions" from "search settling too early" with the
data I have; the honest next step is to reuse `_ourmoves.py` with a much larger sample of the
+100..+300 band.

**4. Repetition and progress guards. (est. +10 Elo)**
Measured: exactly one thrown win (`round-11`, peak +2000 -> threefold at 17 pieces) plus 4
threefold and 2 fifty-move draws total, against their zero of each. Worth 0.5-1.0 points in 37
games. Small, cheap, and already partly addressed by the twofold/contempt work.

**5. Aggression / king-attack weighting. (est. 0 to +20 Elo; correlational only)**
Measured: their non-pawn moves land within 2 squares of the enemy king 23.5% of the time to our
17.9%, and they give 10.4 checks per game to our 6.6. This is a *description* of an engine that
is usually attacking, not proof that a king-safety weight causes it. The king-zone work already
in the tree (`kz8/kz16/kz32`) is the existing lever; I would not spend time here before items
1-3.

**Explicitly NOT worth pursuing (measured negatives, so nobody re-runs them):**
- **More search depth or nodes.** Four independent tests (section 4) find no depth advantage;
  our engine already matches SF-d16 at the same 52.9% rate they do on their own positions.
- **More clock, or a more aggressive time manager in general.** We already spend more total
  clock than they do (132.4 s vs 124.4 s median) and think longer on our longest move (11.14 s
  vs 4.28 s max). The fix is *redistribution* toward won positions (item 1), not more time.
- **Openings or an opening book.** Platform-supplied start FENs, balanced (-0.3 vs -9.5 cp),
  instant-move rates 5.4% vs 6.3%.
- **Level-position middlegame play.** Paired difference **+0.0 cp/move [-1.6, +1.5]** over 255
  positions. This is the single most useful negative result in the study.
- **Defence.** In lost positions we are measurably *better* than they are (10.8 vs 27.1 cp/move).

---

## 7. What I could NOT determine

- **Their engine's actual depth, node rate, or evaluation.** Nothing in a PGN identifies these
  separately; every proxy I used measures the product of search and evaluation. All I can state
  is that the product equals ours in level positions and beats ours in won ones.
- **Whether their won-position edge is search, evaluation, or a tablebase.** A 7-piece endgame
  played at 0 cp loss is equally consistent with Syzygy access, a good endgame term, or a
  search that is simply not throttled. Their zero fifty-move draws and 23 promotions are
  suggestive of real endgame knowledge, but suggestive is all.
- **Whether they use an opening book at all.** Games begin at move 6-9 from a supplied FEN;
  5.4% of their moves are instant, which is not distinguishable from a fast search.
- **The +100..+300 band.** 79 paired positions, CI [-0.8, +17.9]. The one band where the
  mechanism most likely lives is the one I have the least power on.
- **Their behaviour in real time trouble.** Their minimum clock is high and their spend is a
  flat fraction, so there are almost no sub-10-second samples to characterise.
- **Whether the 501-Elo gap is stable.** 24 result-bearing games of theirs, 37 of ours, one
  head-to-head. Five of their 29 PGNs carry `Result "*"` (their `[Round]` tags 15, 22, 26, 28,
  32) and were excluded from all record and Elo arithmetic but kept for move-quality scanning.
- **Whether fixing item 1 actually wins games.** That needs a gauntlet, which I was instructed
  not to run. Everything here is a measurement of move quality, not of results.
- **How much of our measured won-position deficit is the background gauntlet.** Some load was
  present for part of both replays; the level-position null result argues it is not the cause,
  but a repeat on an idle machine would settle it.
- **Anything about the other 11 competitors.** Only the leader and ourselves were profiled.
