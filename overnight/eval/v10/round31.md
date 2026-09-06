# Round 31 -- draw, Black vs Abhi's chess demon (6 Sep 08:00 slot)

Source: `overnight/postmortem/round-31-draw-black-vs-abhi-s-chess-demon-409a4645.json`,
`overnight/pgn/platform/round-31-...pgn` (clocks). Piece counts replayed from the PGN.
Read-only; the only file written is this one. Companions: `rounds25-29.md` (the previous
pass), `games.md` (rounds 1-24), `V10_PLAN.md` (the closed list).

**Headline: no new failure mode.** The one decisive ply is already fixed by the net that
shipped 45 minutes before the game started, measured directly below. Nothing in this game
justifies a gauntlet slot.

## 0. Which build played it

Curated start FEN at move 8 (`1rbqk1nr/... w KQk - 2 8`), 323 plies, 161 of our moves,
`insufficient_material`, 0 blunders, 3 mistakes, 5 inaccuracies, causes search 7 / horizon 1,
**zero `time` flags**, min clock 10.27 s.

JOURNAL 6 Sep 07:45 records v9.2 and v9.3 as "emailed and awaiting the human's uploads", so
the live build at the 08:00 slot was almost certainly **v9.1** (= v9 + TIME_V6, old kz16
net). Two independent confirmations:

- **Clock signature.** TIME_V6's live values in the tree are `RESERVE_FRACTION_V6 = 0.06`,
  `LOW_CLOCK_V6 = 12.0` -- not the 0.04 / 9 in the brief; those were the untamed values that
  failed clocktest twice. Below 12 s `_budget_v6` sets `soft = hard = remaining/18`, whose
  equilibrium against a 0.5 s increment is a ~10 s clock. Observed: the clock sits in
  10.27-13.31 s for our last 107 moves, spending 0.55-0.66 s a move. TIME_V5's
  `remaining/30` below 15 s plus a 12 s reserve floors at 13 s -- exactly what rounds 25 /
  27 / 29 showed (13.0, 13.0, 14.8 s). **TIME_V6 is live and is why the floor moved
  13.0 -> 10.27 s.**
- **Move choice.** The tree (v9.3, mixnet2 net) plays the reference move at the decisive
  ply at the same clock; the game did not (section 1).

Caveat that governs every static number below: the postmortem's `static` column and my
probes use the **tree = v9.3**, not the net that played. That is the useful comparison going
forward, but it is not the playing engine's own error.

## 1. (a) Where the half point went

Reference trajectory (our POV, Stockfish): +137 (mv 19, 28 pcs) -> +188 (mv 22) -> +95
(mv 33, 17 pcs) -> +258 (mv 43) -> **+290 peak (mv 45, 14 pcs)** -> +158 (46) -> +105 (47,
12 pcs) -> +39 (48) -> +4 (after 52...f5) -> **0 from move 55 to the end (114 more of our
moves)**. The JSON's `conversion` field: "peak +290 at ply 76, below half by ply 80".

All eight flagged moves, with the TIME_V6 budget that governed each:

| mv | move | pcs | delta | ref best | cause | clock | spent | soft/hard | our static | ref | err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | Na5 | 28 | -99 | Ba6 | search | 83.3 s | 3.77 s | 2.08 / 5.19 | +104 | +143 | 39 |
| 30 | e4 | 19 | -35 | f5 | search | 56.3 s | 1.69 s | 1.52 / 3.79 | +135 | +161 | 26 |
| 38 | Kg8 | 15 | -30 | Kg8(!) | search | 41.0 s | 1.30 s | 1.29 / 3.23 | +9 | +146 | 137 |
| 44 | Qf6 | 14 | -35 | Qc2 | search | 34.9 s | 2.23 s | 1.26 / 3.16 | +385 | +272 | 113 |
| **45** | **Bh6** | **14** | **-138** | **g5** | **horizon** | **33.2 s** | **2.63 s** | **1.21 / 3.03** | **+429** | **+290** | **139** |
| 46 | Qxe5 | 14 | -56 | Kg7 | search | 31.0 s | 1.87 s | 1.14 / 2.85 | +363 | +158 | 205 |
| 47 | Bg7 | 12 | -59 | Kg8 | search | 29.7 s | 2.39 s | 1.09 / 2.72 | +244 | +105 | 139 |
| 52 | f5 | 12 | -73 | e2 | search | 21.2 s | 1.38 s | 0.94 / 2.08 | +157 | +77 | 80 |

The half point goes in **one nine-move stretch, moves 44-52, at 14 down to 12 pieces**:
-138 -56 -59 -73 = **-326 cp of the -391 cp lost below 17 pieces**, and the whole slide from
+290 to 0 happens there. The mechanism is a chain of simplifications our eval prices as
still-winning: 45...Bh6 instead of 45...g5; then 46...Qxe5, trading into B+N+4P vs 2B+2P
with every pawn on the f-h files (a drawn structure); then 52...f5 instead of 52...e2,
which wastes the passed e-pawn (given up outright at 65...e2 66.Kxe2).

The 28-piece mistake (20...Na5, -99) cost nothing durable -- the reference was back to +188
two moves later.

**Engine probes** (three positions, six searches, ~52 s of engine time in one process, plus
one 34.9 s search in a second; tree = v9.3, all v9.4 switches off):

| position | clock handed in | tree plays | played in game | ref best |
|---|---|---|---|---|
| mv 45 `8/5pbk/5qp1/2Q2n1p/2B1pB2/6PK/7P/8 b` | 33.2 s (real) | **g6g5 in 1.38 s** | Bh6 | g5 |
| mv 45 | 120 s | g6g5 in 3.48 s | | |
| mv 46 `8/5p1k/5qpb/4Qn1p/2B1pB2/6PK/7P/8 b` | 29.7 s (real) | **f6c6 (Qc6, declines the trade)** | Qxe5 | Kg7 |
| mv 46 | 120 s | f6e5 (Qxe5) in 4.73 s | | |
| mv 52 `8/2B4k/5ppb/7p/2Bn4/4p1P1/6KP/8 b` | 20.3 s (real) | f6f5 in 1.16 s | f5 | e2 |
| mv 52 | 120 s / 600 s (34.9 s of search) | f6f5 in both | | |

Readings:

1. **The decisive ply is already fixed.** At the identical clock the current champion finds
   45...g5 in 1.38 s -- inside the budget the game actually had. The v9.1 build that played
   needed the postmortem's 10 s replay to find it (`horizon`). The only tree-vs-v9.1
   differences are NMP_V2 and the mixnet2 net; a change of this size is almost certainly the
   net.
2. **Move 46 is improved but not solved** -- at the real clock the tree declines the queen
   trade (Qc6); given four times the time it goes back to Qxe5.
3. **Move 52 is not a horizon problem at all.** 34.9 s of search -- an order of magnitude
   more than the game had -- still plays f5 over e2 (+77 -> +4). At 12 pieces, with our
   static already 80 cp too high, this is the standing evaluation weakness, not depth.

## 2. (b) Clock profile

Available: 120 s + 161 x 0.5 s = 200.5 s. Used 187.2 s (93.4%). **Ended holding 13.31 s.**

| phase | our moves | total | mean | max |
|---|---|---|---|---|
| mv 8-27 (32-24 pcs, ref +140..+190) | 20 | 67.5 s | 3.38 s | 5.61 s |
| mv 28-44 (24-14 pcs) | 17 | 37.8 s | 2.23 s | 4.08 s |
| **mv 45-55 (the decay, 14-12 pcs)** | **11** | **21.8 s** | **1.98 s** | **2.73 s** |
| mv 56-62 (12 pcs, ref 0) | 7 | 7.4 s | 1.06 s | 1.49 s |
| mv 63-169 (12-4 pcs, ref 0 throughout) | 106 | 52.7 s | 0.50 s | 1.27 s |

- **Lowest clock 10.267 s (move 146)**, final 13.31 s; never below 10.2 s; 0 flags. The
  `time_trouble_moves = 107` count is the sub-1 s shuffle, not real trouble.
- **Did TIME_V6 spend the bank? Partly, and on nothing.** It moved the absorbing floor from
  TIME_V5's 13.0 s to 10.27 s -- **2.7 s of the ~12 s bank games.md predicted, 23% of the
  headline**. Every second of it went on moves 63-169, where the reference evaluation is
  exactly 0. We still finished with 13.31 s (11% of the base clock) unspent, and 52.7 s
  (28% of all time spent) went on 106 provably drawn moves.
- **No error correlates with a short think, and none is fixable by more time.** The three
  mistakes were played with 83.3 s, 33.2 s and 21.2 s in hand. At the decisive move 45 we
  spent 2.63 s against a **hard cap of 3.03 s** -- 87% of everything the manager could
  legally give. The cap is `min(10% of remaining, 2.5 x soft)`, and the stop rule's
  stability / score-drop / node-effort product is clamped to [0.4, 1.5] and multiplies
  `soft`, never `hard`, so no stop-rule tuning reaches it. At move 52 we spent 1.38 s =
  1.47x soft, i.e. the 1.5 clamp -- and 34.9 s does not change the move.
- **TIME_V6 gave the decisive move less than TIME_V5 would have.** At move 45 TIME_V5's hard
  cap was `min(0.12 x 32.76, 3 x 1.42) = 3.93 s` against TIME_V6's 3.03 s -- **23% less**.
  Third game running (round 27 was -17%). TIME_V6 remains insurance against flagging, not
  Elo.
- **The move model is 3x wrong on game length.** `expected = max(30, 56 - 0.4 x fullmove)`
  said 38 moves remained at move 45; 123 did. Harmless only because the low-clock regime is
  an asymptote rather than a countdown.

## 3. (c) Piece count at every error -- CONFIRMED in kind, collapsed in magnitude

| pieces | our moves | cp lost | ACPL | flagged | mean abs(static - ref) |
|---|---|---|---|---|---|
| 27-32 | 16 | 192 | 12.0 | 1 | 39 (n=1) |
| 17-26 | 13 | 81 | 6.2 | 1 | 26 (n=1) |
| **11-16** | **39** | **480** | **12.3** | **6** | **136 (n=6)** |
| <= 10 | 93 | 28 | 0.3 | 0 | -- |

**61% of the cp lost, and 6 of 8 flagged moves, sit at 11-16 pieces on 24% of our moves.**
The standing finding holds in kind. It does **not** hold in magnitude: mean static error at
<= 16 pieces is **136 cp** here, against games.md's 475 and rounds25-29's 674 at 11-16.
The direction is still optimistic (5 of 6 over-state our advantage; move 38 is 137 cp
pessimistic). This is the v9.3 net measuring a v9.1 game, and it agrees with the eg_calib
numbers in JOURNAL 07:45 (331.9 cp at 5-8 pieces vs the old net's 673.7).

The <= 10 band -- rounds25-29's P1, 387 cp in round 25 -- is **28 cp over 93 moves** here.
Nothing left to instrument in that band in this game; the position was dead before we
entered it.

## 4. (d) Failure modes

**The game shows no new failure mode.** Its one decisive ply (45...Bh6, -138, `horizon`) is
already fixed by the shipped v9.3 net, measured at the same clock. Its residual error
(52...f5) is the known 11-16-piece evaluation weakness, now 3.5x smaller than games.md
recorded. Its clock never fell below 10.2 s and no error correlates with a short think.
**No item below deserves a gauntlet slot**; with the queue ~10 h deep and uploads closing
11 Sep, the right action on this game is to record it and keep shipping net work.

Ranked, genuinely new content first.

**1. DRAW_BUDGET's guards are mis-calibrated -- it would have been inert here.**
Not a new idea (the switch exists, off; `drawcap-clocktest-l` is queued) but a new
measurement on it. The guards are `pieces <= 10` **and** `time_left > LOW_CLOCK_V6 (12.0 s)`.
In round 31 the reference eval is 0 from move 55, but we were at 12 pieces until move 76 and
the clock fell below 12 s at move 66. The two guards overlap on roughly **3 of the 106 drawn
shuffle moves**, so DRAW_BUDGET as written banks ~2 s of the 60.1 s spent from move 55 on.
Fix: `_DRAW_PIECES` 10 -> 14 and the clock guard 12.0 -> ~8.0, which arms it at move 55 and
banks ~30-35 s. *Elo at 120 s: +0..+5* -- it banks time it cannot spend backwards, so it is
worth nothing in this game; its value is only in games that go dead and come alive (the
rounds 16 / 21 / 27 / 31 shape, 4 of 27 post-mortemed games). *Gate:* two constants in
`agent.py`, kernel untouched, `check_fastsearch` stays exact; `drawcap-clocktest-l`'s lowest
clock must **rise**, plus a 40-game 120 s non-negative. Ship in the v9.5 filler bundle. No
slot.

**2. Drawn-structure scale factor at <= 14 pieces (single-flank pawns / opposite bishops).**
The only new *mechanism* the game exposes. Evidence: static +429 / +363 / +244 against
reference +290 / +158 / +105 at 14 / 14 / 12 pieces, in a position that is objectively drawn
-- B+N+4P vs 2B+2P with every pawn on the f-h files, entered voluntarily by 46...Qxe5. Round
25's turning point was the same class (opposite-coloured bishops read as level). Distinct
from ENDGAME_SHRINK (closed 6 Sep 07:45): that blended the net toward *material*, which the
mixnet2 net now beats in every band; this is a multiplicative scale on the eval magnitude
keyed by pawn-file span and the bishop-colour flag, which a material blend cannot express.
Not on the closed list. *Elo at 120 s: +3..+10.* *Gate:* `testing.eg_calib` per-band static
error at 9-16 pieces must fall below the v9.3 net's 262.6 / 184.4 cp with 5-8 no worse,
**then** an 8 s SPRT. *Cost:* 3-4 h, and it touches the kernel's evaluate call site (hot
path). *Honest verdict:* not worth a slot before the freeze -- two games of evidence against
a hot-path risk.

**3. The ply-300 adjudication model is contradicted by this game -- verify, do not build.**
Round 18 was adjudicated a loss at exactly 300 plies counted from its curated FEN (move 7).
Round 31 counts plies the same way and **reached 323 un-adjudicated**; at ply 300 White was
ahead K+B+P vs K+B, so a raw-material adjudication would have been a loss for us. Round 21
(277) and round 16 (210) never tested it. `ADJUDICATION` is live (v9) and `ADJ_BEHIND_LATE`
adds up to +300 cp to the behind-side draw score as `_match_ply` -> 300 -- a bias whose
premise this game denies. It cost nothing here (drawn either way), but it is a live bias on
a model with one confirming and one contradicting observation. *Elo: 0; it removes a risk.*
*Gate:* none -- re-read the platform rules and check whether any other game passed ply 300
un-adjudicated. Minutes, no machine time, no slot.

**4. Passed-pawn-on-the-6th blindness at <= 14 pieces -- record only.**
52...e2 (+77) against the played 52...f5 (+4) survives 34.9 s of search, so it is static,
not horizon. One data point; the buildable form is a targeted training resample (passed
pawns on ranks 6-7 at <= 14 pieces), a GPU job rather than a search switch. *Elo: unknown,
+0..+8.* *Gate:* eg_calib. Too speculative for a slot with five days left. Log it; if a
second game shows the same shape it becomes a net-data item.

**Not proposed:** no time-management change (the decisive move used 87% of its hard cap and
34.9 s does not fix the other one); no book item (zero book flags in the game); no tablebase
item (the game was drawn long before 4 men); nothing from the closed list.
