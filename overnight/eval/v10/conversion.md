# Failed conversion: what actually takes the won games away

Sources: `overnight/pgn/platform/round-*.pgn` (37 rated platform games with `[%clk]`),
`overnight/postmortem/round-*.json` (full per-ply Stockfish trajectories, our POV, for 22 of
them) and `overnight/postmortem/aichessathon-round-{1..5}-*.json` (rounds 1-5). Engine facts
read from `agent.py` / `fastsearch.py`; nothing in the tree was modified. Scripts:
`overnight/eval/v10/_conv.py`, `_conv2.py`, `_conv3.py`, `_conv4.py`, `_conv5.py`, `_conv6.py`
(read-only, ~2 s each). Companions: `games.md` (rounds 1-24), `rounds25-29.md`, `round31.md`.

**Corpus correction up front: there are 37 rated platform games in the tree, not 41**
(`round-01` .. `round-37`; the five `aichessathon-round-N-*.pgn` files are duplicates of
rounds 1-5). Record 15W-11L-11D. **Reference evaluations exist for the 11 losses, the 11
draws, and 3 of the 15 wins** -- post-mortems were only ever built for non-wins plus rounds
1-3. Every rate below is therefore over the 22 lost/drawn games unless stated.

## Summary -- ten claims, each with the number behind it

1. **We have never converted a middling advantage.** Of the 22 games with reference
   trajectories, **11 reached a peak of >= +100** and **7 reached >= +150**; they returned
   **0 wins, 4 draws, 3 losses** at >= +150 and **0W/6D/5L** at >= +100. The 3 wins that have
   trajectories peaked at **+500 / +561 / +1012**. We convert crushing; we have never
   converted +100..+300.
2. **It is not the fifty-move rule.** Across the 22 games, **0 of our 103 moves played at
   reference >= +100 had a halfmove clock >= 20**, and the highest halfmove clock ever seen
   while ahead is **14**. Of the 184 our-moves played at halfmove clock >= 40, the maximum
   reference eval is **+18** and **0.0%** are >= +50. We shuffle only in positions that are
   already dead.
3. **It is not the clock.** At the 11 conversion peaks the mean clock in hand was **70.7 s**
   of 120 s; at the 23 flagged moves played with the reference at >= +100 the mean was
   **60.1 s in hand and 2.64 s spent -- 4.8% of the bank**. Three of the seven >= +150 games
   ended with 26.6 s, 35.9 s and 41.6 s unspent.
4. **It is not contempt or draw-acceptance, any more.** Exactly one game was drawn while
   winning: **round 11, threefold at reference +1004 with a mate on the board**. That is
   already fixed -- `REPETITION_TWOFOLD` was added for it and is `True` in the tree. In the
   other draws the eval was already 0 when the draw arrived (round 27: reference exactly 0 for
   the 61 moves before the fifty-move claim; round 31: 0 from ply 111 to ply 323).
5. **The loss is a cliff, not a slide.** From the peak to the first ply below half the peak,
   the median share of the decay carried by **one single move of ours is 75%** (n=10; 99%,
   100%, 100%, 83%, 79%, 71%, 60%, 57%, 56%, 52%). The opponent's moves contribute
   near-nothing: our losses in those windows total 1,695 cp against 54 cp of opponent gains.
6. **Half of those cliffs are pure budget.** In the 10 conversion games (round 11 excluded),
   **5/10 have `horizon` as the cause of the single biggest loss** and **7/10 contain a
   `horizon` loss played with more than 20 s on the clock**, worth **1,060 cp in total**.
   `horizon` in `testing/postmortem.py` means our own engine, handed a fresh 120 s clock
   (~3-10 s of search), plays the reference move. Those seven moves were played in 1.4-9.0 s.
7. **Our net is not systematically optimistic -- it is systematically noisy in exactly the
   band that matters.** On flagged moves with the reference at >= +100, mean **|static - ref|
   is 125 cp** in the +150..+400 band and 88 cp at +50..+150, with the sign mixed (optimistic
   on 14 of 23). The extremes on decisive moves: round 31 ply 78 static **+363** vs reference
   +158; round 22 ply 62 **+387** vs +164; round 37 ply 28 **+22** vs **+200**.
8. **Round 37 is not a failed conversion at all.** Its "peak +200" was never visible to us:
   our static at that ply reads **+22**, and +22 again at the next. The same is true of round
   18 (+15 vs +121) and round 5 (-10 vs +102). Three of the eleven "failed conversion" rows in
   REPORT.md are an advantage we never saw, not one we squandered.
9. **The clock is spent where it cannot buy anything.** Over the 22 games we spent **1,290 s
   (43.5% of 2,968 s) on 891 moves at |reference| < 30** and **122 s (4.1%) on the 49 moves at
   reference >= +150**. Adding the +100..+150 band still only reaches 9.7%.
10. **The time manager has no idea any of this is happening.** `_budget_v6` is
    `remaining / max(30, 56 - 0.4 * fullmove) + 0.7 * inc` -- a function of clock and move
    number only. Measured spend relative to that ideal is flat across every eval band
    (1.35x level, 1.41x at +150..+400, 1.48x at +50..+100, 1.78x when losing): the engine
    treats the move that decides a won game exactly like the 891 dead shuffle moves.

## The distribution asked for

Peak = the highest reference evaluation, our POV, on a position handed to us to move, mate
scores excluded. "half@" = first ply at or after the peak below half the peak.

| rd | result | termination | peak | ply | pcs at peak | clock at peak | half@ | plies peak->end | our moves peak->half | biggest single our-move loss (cause) |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | draw | threefold | **+1108** | 58 | 17 | 42.2 s | - | 8 | 0 | -- (repeated into a threefold with mate on) |
| 31 | draw | insuff. material | +290 | 76 | 14 | 31.0 s | 79 | 248 | 2 | 138 (**horizon**, 31.0 s in hand) |
| 22 | loss | checkmate | +248 | 58 | 20 | 44.9 s | 63 | 76 | 3 | 126 (search); also 76 horizon at 44.2 s |
| 37 | loss | checkmate | +200 | 28 | 25 | 75.2 s | 29 | 106 | 1 | 148 (search, our static +22) |
| 24 | loss | checkmate | +181 | 36 | 26 | 73.9 s | 44 | 96 | 4 | 50 (search) |
| 35 | draw | insuff. material | +167 | 54 | 20 | 48.4 s | 59 | 72 | 3 | 106 (**horizon**, 44.3 s in hand) |
| 9 | draw | threefold | +166 | 46 | 25 | 57.1 s | 49 | 12 | 2 | 52 (evaluation) |
| 27 | draw | fifty moves | +141 | 23 | 26 | 97.9 s | 50 | 149 | 14 | 85 (**horizon**, 78.5 s in hand) |
| 18 | loss | adjudication | +121 | 13 | 30 | 101.6 s | 14 | 288 | 1 | 138 (search, our static +15) |
| 17 | draw | threefold | +113 | 39 | 23 | 78.9 s | 51 | 6 | 6 | 60 (**horizon**, 57.5 s in hand) |
| 5 | loss | checkmate | +111 | 14 | 30 | 98.4 s | 17 | 60 | 2 | 471 (**horizon**, 89.2 s in hand) |

Two structural readings. **Piece count at the peak is 20-30 in 8 of the 11 games** -- this is a
middlegame failure, not the known 11-16-piece endgame weakness (round 31 at 14 pieces is the
one clean endgame case, and `round31.md` already showed the v9.3 net fixes its decisive ply).
And **the decay is fast**: median 4 plies from peak to half-peak, with a 248- or 288-ply tail
of dead-drawn play afterwards.

For scale: those 11 games returned **3.0 points of 11**. Turning the seven >= +150 games from
2.0/7 into 4.5/7 is +2.5 points over a 37-game season -- 6.8 percentage points of score.

## The five hypotheses, quantified

**(a) Evaluation drift -- PARTIALLY CONFIRMED, but it is noise, not bias.** Mean |static - ref|
on flagged moves is 125 cp at +150..+400 and 88 cp at +50..+150, against 75 cp when level.
The sign is mixed: 14 of 23 optimistic. In 7 of the 11 games we over-read the position by
37-223 cp on the decisive move (so a 100 cp giveback leaves the root score still "winning" and
nothing looks wrong); in 3 (rounds 5, 18, 37) we under-read it by 86-178 cp and never knew we
were better. Both directions destroy conversion, for opposite reasons.
**Caveat: the `static` column is the *current tree's* net re-scoring an old game, not the net
that played it** (`testing/postmortem.py` loads `--agent .`). It is the right forward-looking
measurement and the wrong historical one.

**(b) Fifty-move / repetition -- REFUTED as a live mode.** See claim 2: zero our-moves at
>= +100 with halfmove clock >= 20 in 22 games; max halfmove clock while ahead is 14. Rounds 16
and 27 both hit the fifty-move rule with the reference at exactly 0 for the whole counter run.
The one real instance, round 11, is fixed by `REPETITION_TWOFOLD`.

**(c) Contempt / draw scoring -- REFUTED for current builds.** Round 11 (pre-fix) is the only
game drawn while winning. The remaining threefolds are round 9 (reference +56, us 200 cp down
on material -- taking the perpetual is defensible), round 17 (-4) and round 21 (0). `CONTEMPT`,
`ADJUDICATION` and `ADJ_V2` are all inert on this failure: by the time a draw score is consulted
the position is already 0.

**(d) Time exhaustion -- REFUTED.** Mean 70.7 s in hand at the peak; mean 60.1 s at the
critical flagged moves; **zero `time`-caused flags in any of the 11 conversion windows**. The
low-clock floor (`games.md` failure mode 2) is real and costs games, but it costs them
*elsewhere*: 27 `time` flags in the corpus, none of them in a won position.

**(e) Genuine technique -- CONFIRMED as the residue, and it is the majority.** Of 221 flagged
moves corpus-wide the causes are search 113, evaluation 46, horizon 32, time 27, book 3. Inside
the conversion windows our 2,116 cp of losses split **horizon 1,060 / search 720 /
unflagged 236 / evaluation 100** -- so **~39% is `search` or `evaluation` and not fixable by
more time.** The archetype is round 31 moves
46-52: three consecutive liquidations (`Qxe5` trading into a drawn B+N vs 2B structure, then
`f6`, then `f5` giving up the passed e-pawn), each priced by our net as still +244..+363.
`round31.md` measured 34.9 s of search on move 52 still choosing `f5`. **More time does not fix
that one.**

**Synthesis.** There is not one mechanism, there are two, and they compose:
a **±125 cp evaluation error in the +100..+400 band** means the engine cannot tell that this
particular move is the one that decides the game, and a **budget that is a function of clock
and move number only** means it never spends extra there even though it has 60 s in hand and
43.5% of its clock is going on provably dead positions. The evaluation half is the standing
net workstream. The budget half is unaddressed, cheap, and is the single largest thing in the
data that is not already someone's open item: **1,060 cp of `horizon` losses inside the
conversion windows, every one played with >20 s in hand.**

## Primary recommendation -- `CONVERT_BUDGET`

**File:** `agent.py` only. Kernel untouched, so `testing.check_fastsearch` stays exact.

**Switch:** `CONVERT_BUDGET: Final = False` (default **OFF**, as the fixed rules require).

**Constants**, next to the `DRAW_BUDGET` block at line ~1241:

```python
CONVERT_BUDGET: Final = False
_CONV_LO: Final = 120          # cp; root score at which a conversion is live
_CONV_HI: Final = 900          # cp; above this the game wins itself
_CONV_MOVES: Final = 2         # consecutive own searches inside the band
_CONV_MULT: Final = 2.0        # multiplies BOTH soft and hard
_CONV_MIN_CLOCK: Final = 20.0  # seconds left below which we never extend
_CONV_MAX_FRACTION: Final = 0.16  # never more than this share of the clock on one move
```

**New function**, immediately after `_draw_budget_soft` (line ~2758), mirroring it:

```python
def _convert_budget(time_left_ms: int, soft: float, hard: float) -> tuple[float, float]:
    """CONVERT_BUDGET: spend the bank on the two or three moves a game that decide a
    won position. `_DRAW_SCORES` already holds our own last root scores."""
    remaining = max(time_left_ms - 400.0, 50.0) / 1000.0
    if (
        len(_DRAW_SCORES) >= _CONV_MOVES
        and all(_CONV_LO <= s <= _CONV_HI for s in _DRAW_SCORES[-_CONV_MOVES:])
        and remaining > _CONV_MIN_CLOCK
    ):
        now = time.monotonic()
        cap = now + remaining * _CONV_MAX_FRACTION
        soft = min(now + (soft - now) * _CONV_MULT, cap)
        hard = min(max(soft, now + (hard - now) * _CONV_MULT), cap)
    return soft, hard
```

**Wiring**, at the single existing call site (line ~2974), beside `DRAW_BUDGET`:

```python
soft, hard = _budget(board, time_left_ms)
if DRAW_BUDGET:
    soft = _draw_budget_soft(board, time_left_ms, soft)
if CONVERT_BUDGET:
    soft, hard = _convert_budget(time_left_ms, soft, hard)
...
if DRAW_BUDGET or CONVERT_BUDGET:          # the score feed must run for either
    _note_draw_score(board, int(_FAST.root_score))
```

**Why it must scale `hard`, not the stop-rule factor.** `round31.md` measured the decisive
move 45 spending **2.63 s against a hard cap of 3.03 s -- 87% of everything the manager could
legally give**. The `choose` stop rule's stability / score-drop / node-effort product multiplies
`soft` and is clamped to [0.4, 1.5]; it never reaches `hard`. **A stop-rule change would be
inert on the exact case it is meant to fix.** With `_CONV_MULT = 2.0` at round 31 move 45 the
hard cap becomes `min(3.03 x 2, 0.16 x 32.8) = 5.25 s` and soft becomes 2.42 s -- landing in
the 3-10 s range at which the post-mortem's replay finds `45...g5`.

**Why `_DRAW_SCORES` and not a new list.** It already exists, is already reset per game
(`_DRAW_LAST_PLY`), already holds our own last six completed root scores, and is fed from the
same place. The two switches are the same mechanism with opposite signs: `DRAW_BUDGET` banks
clock when the root score has hugged zero, `CONVERT_BUDGET` spends it when the root score has
held a win. **They should ship in the same bundle** -- `DRAW_BUDGET` was measured to bank
30-35 s a game (round 31 item 4a) and `CONVERT_BUDGET` costs, on this corpus, 78 our-moves at
reference >= +120 over 22 games = 3.5 a game, roughly +5-9 s a game at `_CONV_MULT = 2.0`.

**Honest ceiling.** This buys the `horizon` subset and nothing else: **1,060 cp across 7 of the
10 conversion games**, ~28% of the in-window decay is `search`-labelled and untouched. Call it
**+0..+15 Elo at 120 s**, most of it in a small number of games, and note that a 8 s SPRT's
+/-12 Elo resolution probably cannot see it -- it is a bundle rider, not a solo slot.

**Known risk, stated plainly.** The trigger is *our* root score, which claim 7 shows is often
150-200 cp optimistic. In a game we have misread as won (rounds 25, 29 recorded statics of
+775 and +187 against true -709 and -1042) this burns clock in a lost position. That is
bounded by `_CONV_MAX_FRACTION 0.16` and `_CONV_MIN_CLOCK 20.0`, and it costs a game that was
already lost -- but it is the reason the clocktest below is mandatory rather than inherited.

## How to gate it

**Pre-gate, before writing any code (~20 min, no gauntlet).** The whole switch rests on "more
time changes the move in won positions". Test it directly and cheaply:

```
python -m testing.endgame_suite run --agent . --seconds 2.5
python -m testing.endgame_suite run --agent . --seconds 5.0
```

`overnight/eval/endgame_suite.json` has **97 of its 400 positions with `eval` in [120, 900]**
(piece counts 5-16, 15 of them at 13 pieces, 19 at 15). The runner only bands by piece count,
so read `positions[i]["eval"]` in a throwaway script and report mean loss over that subset at
both budgets. **Proceed only if mean loss on the 97-position conversion subset falls by >= 15%
from 2.5 s to 5.0 s.** If doubling the budget does not improve move choice in won positions,
`CONVERT_BUDGET` cannot work and should be dropped -- and that is a finding worth writing down
either way.
*Limitation, stated: the suite is 5-16 pieces, and 8 of the 11 conversion peaks are at 20-30
pieces. The suite can falsify the mechanism but cannot fully confirm it.*

**Ship gate, in order.**
1. `ruff check`, `mypy agent.py fastsearch.py`, `python -m testing.check_fastsearch --depth 4
   --random 30` -- must be byte-identical with the switch off (it is `agent.py`-only).
2. **`testing.clocktest` at the 1.5x charge, mandatory and NOT inheritable** from
   `drawcap2-clocktest-l` -- this switch raises the hard cap, which is precisely what
   clocktest exists to catch. **PASS = 0 flags in 6 games and lowest clock >= 5.0 s.** This is
   the switch's real risk and its real gate.
3. Ride in the next time-management bundle; the bundle's 8 s SPRT decides, and the mandatory
   **40 games at 120 s on `platform_openings.txt` must be non-negative**. Do not give it a solo
   gauntlet slot.
4. **Instrument before shipping**: log per game how many moves fired the extension and the
   seconds they consumed. Nothing in any artefact records root scores, so **the firing rate is
   currently unmeasurable** (see below) and one clocktest game's log settles it.

## What I could NOT determine from the data

- **The true conversion rate.** No reference trajectories exist for 12 of the 15 wins, so
  "0 of 7 games at >= +150 were won" is a statement about the 22 lost/drawn games only. Some of
  the 12 unanalysed wins certainly passed through +150 on the way to mate. **The correct fix is
  to run `testing.postmortem` over the 12 missing wins** -- it is the cheapest way to turn this
  report's central rate into a real one, and I did not run it (it needs Stockfish over ~1,200
  plies, well past the 2-minute CPU limit I was given).
- **How often `CONVERT_BUDGET` would fire, and what it would cost.** The trigger reads our own
  root score; root scores appear in no PGN, JSON or log. My 3.5 moves/game estimate substitutes
  the *reference* eval >= +120 (78 of 1,699 our-moves) and is a **lower bound**, because our
  score runs optimistic. Instrument it; do not trust the estimate.
- **Whether the eval error causes the short search or vice versa.** I hypothesised that a high,
  rising, stable root score makes TIME_V6's `2 ** (drop/100)` and `_STABILITY_SCALE` factors
  shrink the budget in exactly the won positions that need it. **The data does not support
  this** and I am retracting it: spend relative to the ideal `_budget_v6` soft is flat across
  every band (1.41x at +150..+400 vs 1.35x when level, n=44 vs 509). The engine is
  *indifferent* to winning, not biased against it. The recommendation stands on indifference.
- **Which build played rounds 32-37.** `JOURNAL.md` pins rounds 25/27/29 to v8/v8.5/v9 and
  `round31.md` pins round 31 to v9.1, but I found no entry mapping rounds 32-37 to builds.
  Round 37 was played on 6 Sep, after v9.4 shipped at 12:13, but I could not confirm the slot.
- **Whether the round 31 / 35 / 37 statics reflect the net that played.** `postmortem.py`
  re-scores with `--agent .`, i.e. the tree at analysis time. Claim 7's numbers are the current
  net's view of old positions.
- **Opponent strength as a confound.** Rounds 22, 24, 29 and 37 are against opponents that
  never blundered back; I cannot separate "we failed to convert" from "they defended well"
  without playing the positions, which the CPU limit forbids.
