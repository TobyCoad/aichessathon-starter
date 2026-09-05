# v10 evidence: what the 22 platform games actually say

Sources: `overnight/pgn/platform/*.pgn` (all 22 rated games, clocks included),
`overnight/postmortem/*.json` (Stockfish-referenced per-move deltas for 15 of them --
every loss and draw plus rounds 1-3), `~/Downloads/aichessathon-round-{17,18}.log` and
the v1/v2/v7/v8/v10 build logs (platform-side init and per-move times),
`harness/referee.py` (the actual rules), `agent.py` (TIME_V4b, contempt, book, TB).
Analysis scripts were run read-only; nothing in the tree was modified.

## Summary (10 lines)

1. Record 10-6-6 over 22 rated games (59.1%). Every loss and every draw reached <= 16 pieces.
2. **The endgame is the whole problem.** In losses our ACPL is 34.9 below 17 pieces vs 21.3 in the middlegame; 13 of 17 blunders happen at <= 16 pieces.
3. **It is an evaluation failure, not a depth failure.** Mean |our static - reference| on flagged moves is 70 cp at 27-32 pieces, 155 cp at 17-26, and **475 cp at 11-16**. A 5 s re-search recovers the reference move on only 17% of the 11-16 flags.
4. **The clock has an absorbing floor at ~13 s.** 10 of 22 games hit < 15 s and then play the remaining 8-77 moves at 0.2-1.4 s. `RESERVE_FRACTION 0.10` locks 12.05 s away permanently and below `LOW_CLOCK 15` the budget `rem/30` equals the 0.5 s increment exactly, so we can never climb out.
5. 5 of 17 blunders are cause `time`, **all** at clock < 17.2 s; they were decisive in rounds 4, 8 and 22. Round 22 was thrown away in 5 moves at 0.4-1.0 s each with 13 s sitting unusable.
6. Yet we have never flagged: minimum clock across 22 games is 12.7 s, and clocktest at a 1.5x charge bottoms at 9.7-10.5 s. We are paying ~12 s of insurance against a risk that has never fired.
7. **Round 18 was lost to a rule, from a dead-equal position.** The referee adjudicates at 300 match plies on raw material; we held K+R+N vs K+Q (-100) at reference eval 0. The fifty-move draw was 23 plies away and the cap arrived first.
8. **The book is a non-factor.** Mean 1.09 in-book plies per game, 12 of 22 games get zero book moves, and the eval on leaving book was never outside [-46, +45]. No game was ever "left in a bad line".
9. Syzygy is 3-4 men only (70 files, 4.4 MB, `TB_MEN = 4`), probed at the root and in search. 203 of our moves were played at <= 5 men but every 5-man position we reached was already correctly drawn -- a 5-man subset would have changed no result so far.
10. Init is safe: 34.8-38.6 s of a 90 s budget clean, 50.1 s worst case (the ponder build). Clock accounting on the platform is exact and honest; the only oddity is the confirmed process suspension between moves.

## Evidence table -- all 22 platform games

`ACPL` columns are mean centipawn loss against the reference, split by piece count at the
position before our move. `book` is how many of our moves came out of `weights/book.bin`.
`exit eval` is the reference evaluation, our point of view, at the first move we had to search.
`tt<15` is the number of our moves played with under 15 s on the clock.
Blank analysis cells = no post-mortem JSON (rounds 6, 7, 12, 14, 15, 19, 20 -- all wins).

| rd | col | result | opponent | termination | plies | our mv | ACPL >=27 | ACPL 17-26 | ACPL <=16 | blunders (our move / cp / pieces / cause) | low clk | tt<15 | opp end clk | book | exit eval | peak eval | final men |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | W | win | sunfish | checkmate | 22 | 11 | 6.9 | - | - | - | 74.9 | 0 | 81.4 | 0 | +26 | mate | 27 |
| 2 | B | win | imperial knights | checkmate | 55 | 28 | 6.6 | 17.2 | 0.0 | 16/-289/21p/horizon | 44.9 | 0 | 57.0 | 1 | +21 | mate | 15 |
| 3 | B | win | danya's disciple | checkmate | 83 | 42 | 16.2 | 5.1 | 20.7 | 27/-224/15p/horizon, 34/-164/15p/search | 23.4 | 0 | 37.3 | 3 | +28 | mate | 11 |
| 4 | B | **loss** | checkers | checkmate | 136 | 68 | 11.8 | 35.2 | **45.4** | 27/-163/17p/search, 56/-166/7p/**time**, 62/-910/7p/horizon | 12.9 | 24 | 28.2 | 0 | -3 | +15 | 6 |
| 5 | W | **loss** | blunder-buss | checkmate | 73 | 36 | **80.4** | 22.1 | **125.1** | 8/-471/29p/horizon, 31/-979/11p/**eval**, 32/-751/10p/search | 17.4 | 0 | 36.3 | 0 | +35 | +111 | 9 |
| 6 | W | win | therook | checkmate | 80 | 40 | | | | | 22.6 | 0 | 36.8 | 0 | | | 17 |
| 7 | B | win | chimera | checkmate | 123 | 62 | | | | | 12.9 | 11 | 24.7 | 0 | | | 3 |
| 8 | W | **loss** | no more ammo | checkmate | 107 | 53 | 21.3 | 10.1 | **49.5** | 30/-215/14p/search, 33/-252/12p/search, 47/-165/9p/**time** | 13.5 | 5 | 24.9 | 0 | -46 | -9 | 8 |
| 9 | W | draw | fuzzybot | threefold | 57 | 28 | 14.6 | 20.7 | - | - | 41.6 | 0 | 54.5 | 2 | +12 | +166 | 24 |
| 10 | W | **loss** | keep kann and caro on | checkmate | 84 | 42 | 26.4 | 19.1 | **51.5** | 35/-197/14p/**eval** | 21.6 | 0 | 11.4 | 3 | -30 | -13 | 9 |
| 11 | B | draw | saucybeans | threefold | 65 | 32 | 14.8 | 36.2 | - | 32/-818/17p/search | 35.9 | 0 | 51.3 | 0 | +3 | mate | 17 |
| 12 | B | win | alphabeta | checkmate | 134 | 67 | | | | | 15.4 | 0 | 18.1 | 0 | | | 7 |
| 13 | W | draw | neural gambit | insuff. material | 108 | 54 | 14.8 | 2.3 | 2.8 | - | 19.7 | 0 | 48.9 | 0 | -43 | +36 | 2 |
| 14 | W | win | bitsentangled | checkmate | 34 | 17 | | | | | 70.0 | 0 | 64.4 | 1 | | | 21 |
| 15 | B | win | omega3 fish | checkmate | 167 | 84 | | | | | 13.4 | 15 | 3.7 | 2 | | | 5 |
| 16 | B | draw | lubina | fifty moves | 210 | 105 | 10.2 | 30.9 | 5.4 | - | 13.8 | 40 | 12.3 | 3 | -21 | +71 | 5 |
| 17 | W | draw | waterside research | threefold | 151 | 76 | 10.9 | 7.5 | 0.8 | - | 13.9 | 11 | 43.2 | 4 | -45 | +113 | 7 |
| 18 | B | **loss** | pheanup | **adjudication** | 300 | 150 | 19.9 | 25.8 | 4.3 | - | 12.7 | 78 | 2.1 | 0 | -23 | +121 | 5 |
| 19 | W | win | adashima | checkmate | 64 | 32 | | | | | 59.6 | 0 | 46.6 | 3 | | | 14 |
| 20 | B | win | lightning tree | checkmate | 159 | 80 | | | | | 14.8 | 1 | 27.6 | 0 | | | 3 |
| 21 | B | draw | mate in one | threefold | 277 | 139 | 16.2 | 10.2 | 2.3 | - | 13.0 | 31 | 5.0 | 2 | -18 | +18 | 4 |
| 22 | W | **loss** | sobriety | checkmate | 133 | 66 | 10.4 | 14.8 | **85.3** | 59/-263/14p/**time**, 62/-933/13p/**time**, 63/-935/13p/**time** | 13.4 | 7 | 21.0 | 0 | +45 | +248 | 13 |

Aggregates over the 15 post-mortemed games (position-weighted ACPL):

| | >= 27 pieces | 17-26 pieces | <= 16 pieces |
|---|---|---|---|
| all | 18.0 (148 mv) | 19.1 (249 mv) | 18.2 (533 mv) |
| **losses** | **26.8** (58) | 21.3 (116) | **34.9** (241) |
| draws | 13.3 (69) | 19.4 (95) | 3.1 (270) |
| wins | 9.0 (21) | 12.1 (38) | 19.8 (22) |

(The draws' 3.1 is not skill: 270 of those moves are dead-drawn 4-7 man shuffles where
no move loses anything. The honest comparison is losses vs middlegame, 34.9 vs 21.3.)

Flag anatomy by piece band -- 121 flagged moves, `|static - ref|` is our net's own error:

| band | flagged | mean \|static - ref\| | 5 s search finds the reference move | short search repeats our move | causes |
|---|---|---|---|---|---|
| 27-32 | 28 | 70 cp | 5/28 = 18% | 46% | search 19, book 4, horizon 4, eval 1 |
| 17-26 | 44 | 155 cp | 14/44 = 32% | 61% | search 28, horizon 9, eval 7 |
| **11-16** | 29 | **475 cp** | **5/29 = 17%** | 62% | search 12, **eval 11**, horizon 2, time 4 |
| <= 10 | 20 | 141 cp | 8/20 = 40% | 65% | **time 9**, eval 6, search 4, horizon 1 |

## Failure mode 1 -- the 11-16 piece band, where the net hallucinates

Numbers: mean absolute static-vs-reference error **475 cp** at 11-16 pieces, 3x the
17-26 band and 7x the opening. 11 of the 29 flags there are classed `evaluation`
(our engine still prefers its move at 5 s *and* its static disagrees with the reference
by a lot) against 1 of 28 in the opening. A 5 s re-search rescues only 17% of them, so
this is not a depth problem and no amount of search speed fixes it.

The clearest single instance is **round 22, our move 59 (PGN 118, b7)**: reference -23,
**our static +512**, we played it, reference after -286; three moves later at move 62 the
reference is -557 while our static reads +86, and at move 63 the reference is -1065 while
our static reads **+565**. A queen-and-passer position of 13-14 pieces read five pawns
wrong in our favour. Round 5's collapse is the same shape: move 31 Bd1 (-979, `evaluation`,
11 pieces) and move 32 Bc2 (-751, `search`, 10 pieces).

Round 8 is the pure form of the mode: at 16 pieces the reference had us at -12, and over
the next 24 moves (9-15 pieces) essentially every move is flagged, walking -12 -> -165 ->
-280 -> -477 -> -670 -> -1085 -> mate. No single blunder above -252; an accuracy slide.

Structural cause: the net has 8 output buckets keyed on piece count
(`bucket = (pieces-1)*8//32`, so 9-12 and 13-16 are buckets 2 and 3), but positions in that
range are ~5% of Lichess training data, and the earlier review already measured 10x worse
mse on near-equal <= 16-piece positions. The loss-reweighting attempt (`kz8w`,
`--weight-endgame`) selected epoch 0, i.e. it never beat its own starting checkpoint --
reweighting the loss did not fix it. What has *not* been tried is resampling the data so the
low-piece buckets are actually represented, or bounding the head's output below 17 pieces.

## Failure mode 2 -- the clock's absorbing floor at ~13 s

`RESERVE_FRACTION = 0.10` of a 120.5 s peak = **12.05 s that can never be spent**, and
`LOW_CLOCK = 15.0` switches the soft budget to `remaining / 30`, which at 13-15 s yields
0.43-0.50 s -- precisely the 0.5 s increment. The state is absorbing by construction: once
touched, we play out the game at increment rate.

Measured spend profile over all 22 games (our moves only):

| our move | 1-10 | 11-20 | 21-30 | 31-40 | 41-50 | 51-60 | 61-70 | 71-80 | 81-90 | 91-100 | 101+ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean s | 3.17 | 3.21 | 2.69 | 2.22 | 1.37 | 0.96 | 0.68 | 0.51 | 0.50 | 0.35 | 0.24-0.47 |
| n | 220 | 208 | 196 | 160 | 124 | 107 | 83 | 56 | 34 | 30 | 114 |

10 of 22 games cross below 15 s, and they cross at 46-92% of the way through:
round 4 at our move 45 of 68, round 8 at 49 of 53, round 15 at 64 of 84, round 16 at 62 of
105, round 17 at 66 of 76, round 18 at 73 of 150, round 21 at 64 of 139, round 22 at 60 of
66. Round 18 played **78 moves** in the low-clock regime; round 16, 40; round 21, 31.

Meanwhile the front is over-banked: we spend 51-70% of the clock in the first 20 searched
moves in the games that go long, and 11 of 22 games *end* with more than 20 s unspent.

The damage is concentrated and identifiable. All 5 `time`-caused blunders occur at
clock < 17.2 s:

| game | our move | move | cp | pieces | clock | spent |
|---|---|---|---|---|---|---|
| r4 | 56 | Ke4 | -166 | 7 | 14.4 s | 0.46 s |
| r8 | 47 | Rc5+ | -165 | 9 | 17.2 s | 0.95 s |
| r22 | 59 | b7 | -263 | 14 | 16.1 s | 0.99 s |
| r22 | 62 | Nf3 | -933 | 13 | 14.2 s | 0.72 s |
| r22 | 63 | Kh3 | -935 | 13 | 14.3 s | 0.39 s |

Round 22 is the cleanest loss of a whole point to the clock: the reference had the game
level (+0) at our move 58; five moves at 0.4-1.0 s each ended it, with 13 s of reserve
untouched and the opponent finishing on 21.0 s.

Against that risk: **we have never flagged.** Minimum clock across 22 rated games is 12.7 s;
`testing.clocktest` at a 1.5x charge bottoms at 9.7-10.5 s over 6 games with 0 flags. The
insurance is priced for a hazard that has not occurred in ~22 rated + several hundred
gauntlet games.

Node-count evidence for what the tail costs: bench is 1,605,437 nodes to depth 8 over 40
positions (40 k/position) and 2,085,202 to depth 10 over 12 positions (174 k/position),
so the effective branching factor is ~2.08 -- **one ply per doubling of time**. Playing at
0.4 s instead of 1.5 s is ~1.9 plies thrown away; 0.5 s instead of 1.2 s is ~1.3 plies.
The measured worth of a ply here is +50-65 Elo at 8 s (compiled search +67 for 1.7x speed,
LMR +47) and the journal's own rule that 120 s gains are about half of 8 s gains puts it at
**~25-35 Elo per ply at 120 s**, in the phase where our accuracy is already 1.6x worse.

### TIME_V5 (floor 18 + stable-score refund), evaluated

Replaying the schedule offline over the real game lengths:

| model | 80-move game: m60 spend | m80 spend | end clock | 150-move game: m100 | end clock |
|---|---|---|---|---|---|
| current (floor 26, LOW 15, reserve .10) | 1.10 s | 0.55 s | 13.6 s | 0.50 s | 13.0 s |
| TIME_V5 (floor 18, refund) | 1.30 s | 0.53 s | 14.6 s | 0.52 s | 13.8 s |
| proposed (floor 26, LOW 9, reserve .04, drain tail) | ~1.5 s | ~0.9 s | ~6 s | ~0.8 s | ~6 s |

Verdict: **TIME_V5 is directionally right and far too small.** It moves the move-60 budget
by +0.2 s (+0.25 ply) and leaves the 13 s reserve and the absorbing floor exactly where they
are -- it cannot touch any of the five `time` blunders above, all of which happened below
17 s where the floor never binds and the refund is irrelevant. Worse, the refund only fires
after two stable iterations, i.e. it hands back time precisely in the positions where extra
depth matters least. Estimate +3..+8 Elo. It is a strict subset of the change that is
actually needed; fold it into a proper TIME_V6 rather than shipping and testing it alone.

## Failure mode 3 -- long games and the ply-300 material adjudication

`harness/referee.py` checks `board.outcome(claim_draw=True)` first (so threefold and
fifty-move are claimed *for* us automatically), then at `len(move_stack) >= 300` calls
`_adjudicate`, which awards the game on **raw material sum** -- draw only on an exact tie.

Round 18 is a whole point lost to this and nothing else. Final position
`2K5/r7/8/n2k4/8/6Q1/8/8 b - - 77 157`: K+R+N against K+Q, no pawns, reference evaluation
**0** for the last 85 of our moves, and we lost. Material balance -100 (queen 900 vs rook
500 + knight 300), so the adjudication went to White. The halfmove clock stood at **77**:
the automatic fifty-move draw was 23 plies away and the ply cap arrived first.

Our engine is partly aware of this -- `_contempt` ramps `CONTEMPT_AHEAD` toward
`ADJUDICATION_PLY` -- but two things are wrong:

* When behind, the draw score is a flat `-CONTEMPT_BEHIND = +20 cp`, with no ramp. At ply
  280 with a losing adjudication a draw is worth a full half point, not 20 cp; the search
  saw a repetition as barely better than playing on and did not force one.
* `game_ply = 2*(fullmove_number - 1) + turn` counts **chess** plies from the real initial
  position, but the referee counts **match** plies from the curated start FEN. Round 18
  started at chess ply 13, so our ply counter runs 13 ahead of the referee's -- conservative
  here, but the offset is a bug and should be pinned from the first request.
* Nothing anywhere knows that when `game_ply + (100 - halfmove_clock) <= 300` a fifty-move
  draw is *reachable before* the cap, which turns "avoid zeroing moves" into a concrete
  winning-a-half-point plan for the side losing the adjudication.

3 of 22 games ran past 200 plies (rounds 16, 18, 21) and one of the three was lost to the
rule from a drawn position -- 4.5% of games, one full point.

## Opening book

Measured now: `weights/book.bin` is 9.76 MB, 610,028 entries, covers **28 of the 80**
curated platform start positions, 2.6 moves per covered position, mean 1.38 in-book plies
from a pool start.

In the games actually played: **mean 1.09 in-book plies, 12 of 22 games get zero book
moves**, best case 4 (round 17). The eval on the first move we had to search was never
outside [-46, +45] across all 15 post-mortemed games -- the hypothesis "the book left us
in a bad line" is **false on this data**. Round 5's -471 at move 8 was `horizon`, 29 pieces,
9.66 s spent at 89 s on the clock, and came 8 moves after leaving book.

Ceiling on a better book: the platform's start positions are already at move 6-10, so even
a perfect book adds at most 2-4 plies before it runs out of theory. At ~3.2 s/move that is
6-13 s banked, deposited in the front of the game where our spend is already too high --
i.e. it feeds the over-banking in Failure mode 2 rather than helping.

Against that: `065-nobook` measured 51.4% (inconclusive) on the platform pool,
`094-bookverify` -94 +/- 53, the earlier max-drop-30 prune tested exactly 50.0%, and
`105-bookprune` (mc20/md10) was closed on coverage before a gauntlet -- 60 of 599 row groups
of 2025_01 with min-count 20 kept only 31,200 moves and collapsed pool coverage to **7/80**
with 0.26 mean in-book plies, meaning ~91% of platform games would open book-less and no
600-game SPRT could resolve anything. That verdict is right and the diagnosis is right:
the prune starved the counts, it did not prune badly.

**Verdict: the book is worth ~0 and every experiment so far agrees. A full-month rescan is
a 4.5 h CPU job for a plausible upside of +0..+8 Elo. Do not spend the CPU before the
freeze.** A small learned book from our own engine-vs-engine results is worse still: it
would be trained on our own 11-16-piece blind spot and cannot cover 80 curated positions
from a few thousand games.

## Endgame and tablebases

Current: `TB_MEN = 4`, 70 files, 4.4 MB -- **3-4 men, not 3-4-5**. Probed at the root
(`_tablebase_move`, with the zeroing-before-DTZ ordering that fixed the KPvK shuffle) and
inside the search on both the python and compiled paths (`agent.py:1255`,
`fastsearch.py`/`agent.py:1662`, gated on `pieces <= TB_MEN`). So the probe placement is
already right; the only question is coverage.

Residency across all 22 games, counting our moves:

| men | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12+ |
|---|---|---|---|---|---|---|---|---|---|---|
| our moves | 13 | 55 | 135 | 73 | 66 | 45 | 69 | 35 | 22 | 799 |
| games reaching | 2 | 4 | 7 | 7 | 9 | 10 | 11 | 12 | 10 | 22 |

68 of our moves are already exact (<= 4 men). A 5-man set would add **135 moves in 7 games**.

Would it have changed a result? **No, not one.** Every 5-man position we reached was
correctly handled already: round 16 (fifty-move draw at 5 men, reference -19..-46, we were
the worse side and drew), round 21 (threefold at 5 then 4 men, reference +0 throughout),
round 18 (5 men, reference 0, drawn on the board and lost only to the ply cap), round 13
(insufficient material, reference +0). The 5-man value is (a) instant moves banking ~15 s
of clock in the games that go there and (b) insurance against botching a *won* 5-man ending
we have not yet been handed.

Size: a full 5-man WDL+DTZ set is ~939 MB, WDL alone ~380 MB -- both far over the 50 MB cap.
Current unpacked bundle is 27.9 MB, so a curated ~20 MB subset (KRPvKR, KQvKR, KRvKB,
KRvKN, KPPvKP, KRPvKP) is the only feasible form, and it competes directly with a larger
net for the remaining ~22 MB. 6-man is arithmetically impossible (~150 GB).

A dedicated endgame *evaluation* is the better use of the same effort: the losses' damage
is at 9-16 pieces, well above any shippable tablebase, and that is exactly where the net's
static error is 475 cp.

## Platform-specific

**Init.** Observed on their box: 34.8, 35.4, 35.7, 36.9, 38.2, 38.6 s clean; **50.1 s** worst
(round 18, the PONDER + 16-zone-net build) of a 90 s budget -- 56% used. Local import is
31 s under load, so their box is ~1.15x ours for import, not the 1.8x that applies to search.
Headroom is 40-53 s. Every `njit` in `fastsearch.py` (19) and `fastboard.py` (36) is
`cache=False`, so all 55 kernels recompile every game.

Shipping a numba cache directory would need `cache=True` plus `NUMBA_CPU_NAME=generic` and
`NUMBA_CPU_FEATURES=""` set identically at build and run time, otherwise the cache keys on
the build CPU and silently misses on an Azure worker. Upside: init drops to roughly 5-10 s.
**Elo value of that on its own is zero** -- init is outside the 120 s game clock, and we are
at 41-56% of budget. Its real value is insurance (an init timeout is a lost game) plus
~28 s of headroom, which is what would let a bigger net or a 5-man subset ship at all.
Cost of the generic-CPU flags is a small loss of vectorisation in the kernels, which
directly costs knps -- it must be benched before it is shipped, and it may be a net negative.
Given the freeze on 10 Sep, this is a "only if a bigger net needs the room" item.

**Memory.** TT is `1 << 22` = 4.19 M slots x 2 uint64 = 67 MB; net 13.6 MB (float16 W1);
book 9.8 MB memory-mapped; syzygy 4.4 MB. Nowhere near the 2 GB cap. `MAX_TABLE = 400_000`
caps the python-side fallback table at ~0.3 GB, which is also fine.

**Platform behaviour.** No timeouts, no illegal moves, no errors in any of 22 games.
Clock accounting is exact: round 17, 120 + 0.5x76 - 144.1 used = 13.9 s left, matches;
round 18, 120 + 0.5x150 - 181.8 = 13.2 s, matches. The increment is applied *after* the move
and the flag is checked *before* it is added (`referee.py`), which our budget already
assumes. The one real oddity is confirmed and already acted on: **the process is suspended
between moves.** The round-18 ponder-diag lines show 6-11 k pondered nodes for gaps up to
9.6 s -- pure thread-start artefact -- and the validator states it outright. Pondering is
dead and correctly closed.

Also worth noting for the Swiss: in 4 of 6 losses the opponent finished with more clock than
we did (r4 28.2 vs 13.4, r5 36.3 vs 17.9, r8 24.9 vs 14.9, r22 21.0 vs 14.4). Our mean end
clock is 28.9 s against the field's 33.5 s, but the median tells the real story -- ours is
19.9 s and in the long games it is 13 s.

## Ranked ideas for v10

E = expected Elo at 120 s on the platform's own openings.

| # | idea | E | cost | risk | state |
|---|---|---|---|---|---|
| 1 | **TIME_V6** -- reserve 0.10 -> 0.04, LOW_CLOCK 15 -> 9, drain the bank in the tail instead of holding it | **+10..+25** (pt +15) | ~25 lines, one switch | medium: flagging. Gated by clocktest at 1.5x + 200-game crash hunt | not built |
| 2 | **Endgame eval below 17 pieces** -- shrink the net's output toward a material/PSQT baseline as pieces fall, and/or retrain with the data *resampled* by piece bucket rather than the loss reweighted | **+10..+30** | high (a retrain is hours of GPU; the shrink is ~15 lines) | medium: the shrink can cost real evaluation in sharp endings | shrink not built; `104-kz16r` retrain running (unstratified) |
| 3 | **Adjudication awareness** -- ramp the behind-contempt toward a full half point as match ply -> 300, pin the ply counter to match plies, and treat a reachable fifty-move draw (`ply + 100 - halfmove <= 300`) as a draw in hand | **+5..+15** | ~30 lines, one switch | low; must not make us draw-happy before ply ~200 | not built |
| 4 | v8.5 bundle: LMR_AGGRESSIVE + PRUNE_V2 + SINGULAR + LAZY_ACC | +10..+30 combined | built | low | `110-v85all` in its crash gate |
| 5 | `104-kz16r` five-month retrain | +5..+15 | running | low | GPU step pending |
| 6 | TIME_V5 as built (floor 18 + refund) | +3..+8 | built | low | **fold into #1, do not ship or gauntlet alone** |
| 7 | Counter-move heuristic / history malus / side-indexed history | +8..+25 | medium | low | not built (V7_PLAN item 11) |
| 8 | Curated ~20 MB 5-man syzygy subset | +3..+10 | high; competes with the net for the 50 MB | medium | not built; **would have changed no result in 22 games** |
| 9 | numba `cache=True` + generic-CPU flags to cut init to ~5-10 s | 0 direct; insurance + ~28 MB/28 s headroom | medium | medium: cache misses silently; generic CPU costs knps | not built |
| 10 | Full-month book rescan (min-count 20 over all 599 row groups) | +0..+8 | 4.5 h CPU | low | **do not spend the CPU before the freeze** |
| 11 | 6-man tablebases | n/a | ~150 GB | impossible | closed |

### Scoping the top three

**1. TIME_V6.** One switch in `agent.py` touching `_budget_v2` only; the kernel is untouched
so `check_fastsearch` stays exact and 8 s play is *not* byte-identical this time (LOW_CLOCK
binds at 8 s too), so it needs the 8 s SPRT as well as the long-TC gates.
Concretely: `RESERVE_FRACTION 0.10 -> 0.04` (4.8 s locked instead of 12.05 s);
`LOW_CLOCK 15.0 -> 9.0`; below `LOW_CLOCK` replace `remaining/30` with
`0.42 + max(0.0, remaining - 5.0) / 50.0`, which spends the increment *plus* a slow drain
toward a genuine 5 s floor instead of sitting at an absorbing 13 s; keep TIME_V5's
`expected` floor of 18 and drop its stable-score refund (it fires in the wrong places).
Gate: `testing.clocktest` at the 1.5x charge must show **0 flags and lowest clock >= 3.0 s**
over 6 games (today's baseline is 9.7-10.5 s, so the acceptance band is explicit), then the
200-game crash hunt, then 40 games at 120 s on `GAUNTLET_OPENINGS=platform`, then the 8 s
SPRT. If the clocktest lowest clock lands under 3.0 s, back the reserve off to 0.06 and
re-run rather than abandoning. This is the single highest-confidence item: it is a schedule
change validated by an offline replay of 22 real games, and the three losses it targets
(rounds 4, 8, 22) are documented above with clock and spend at every blunder.

**2. Endgame evaluation.** Two independent attempts, in cost order.
(a) *Shrink* (cheap, ship-or-kill in a day): below 17 pieces blend
`eval = w*net + (1-w)*material`, `w` falling from 1.0 at 16 pieces to ~0.55 at 6, applied in
both `agent._eval_bucket_kernel`'s caller and the compiled path so the two stay identical
under `check_fastsearch`. This directly bounds the +512-in-a-level-position failure.
(b) *Resampled retrain*: rebuild the packed shards so buckets 0-3 (1-16 pieces) are sampled
to ~4x their natural rate instead of reweighting the loss (which `kz8w` already showed does
not work), and select on a bucket-balanced validation loss.
**Build the instrument first**, before either: the existing `testing/endgame_suite.py`
measures the *chosen move's* cp loss at 2.5 s (baseline 7.0 cp) and is blind to a
hallucinating evaluation that still picks a sane move -- which is why RFP_PHASE looked bad
there and this problem has never shown up. Add a mode that reports mean `|static - SF d18|`
per piece band over the same 400 positions; the platform games predict roughly 70 / 155 /
475 / 141 cp for 27-32 / 17-26 / 11-16 / <= 10, and any fix must move the 11-16 number.
That measurement is minutes of CPU and settles (a) without a single gauntlet game.

**3. Adjudication awareness.** Smallest and most certain of the three. In `_contempt`:
replace the flat `-CONTEMPT_BEHIND` with a ramp `20 + 230 * late` (so a draw is worth ~250 cp
at the cap when we are losing the material adjudication), and pin `game_ply` to match plies
by recording `board.ply()` on the first request and offsetting from it rather than deriving
it from `fullmove_number`. Add a `FIFTY_REACHABLE` term: when
`match_ply + (100 - halfmove_clock) <= ADJUDICATION_PLY` and we are behind on adjudication
material, add a bonus for a high halfmove clock so the search stops volunteering captures
that reset it. Gate on the 8 s SPRT (repetition scoring is exercised there) plus the 40-game
120 s match; watch specifically that the draw rate in games under 150 plies does not rise.
Round 18 is the whole case for it: one point, from a position the reference scored 0,
missed by 23 plies.
