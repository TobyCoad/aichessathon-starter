# v7 plan -- from four fresh-context reviews (4 Sep night)

Ladder after round 14: make_no_mistakes 1854, rank 22 of 238, 7-3-4. Top five 2147,
2114, 2110, 2082, 2060; then a cliff to 1911 at #10. The target is a top-5 seed
(~+150), not the +300 to #1. Seats fill in Swiss order; tiebreaks points, Buchholz,
head-to-head, earlier upload. Reliability in rounds 1-4 is worth more than any single
engine change. Freeze Wed 10 Sep evening; Thu morning is emergency-only.

## What the evidence says we are (weak-spot review, 14 platform games)

* Move quality equals the field's in the middlegame (ACPL 16-20 vs 14-19) and is
  2-2.4x worse below 16 pieces (29-36 vs 15-16). All four losses were decided there;
  in three of them our peak eval never exceeded +15. Accuracy losses, not blunders.
* 78% of the flagged moves are repeated by our own engine at 5 s: most errors are
  not depth-limited. The "depth-class" reading of rounds 4/5/8/10 was too generous;
  re-run the post-mortems on v6 to measure how much depth actually recovers.
* The top bots spend LESS time per move (p50 1.3-1.8 s vs our 2.3 s), play games
  25-40 plies longer, and convert long low-material endings (fifty-move and ply-300
  terminations we never reach).
* The book has negative expected value on the curated pool: 35% coverage, 0.7
  firings per game, ~20 cp loss per firing, one singleton line loses to a Greek gift.
* Draws given away: ~1.1 points of 14, almost all round 11 (referee claim rule; fixed).

## Ranked v7 candidates (E = expected Elo at 8 s; the 120 s gate is what counts)

| # | Change | E | Cost | State |
|---|---|---|---|---|
| 1 | Pondering (allowed by the rules; runner keeps the process alive) | +30..60 | built, switch PONDER | queue5 061 |
| 2 | Piece-count-scaled RFP/futility margins (static err 2-6x worse <17 pieces) | +0..40 | built, RFP_PHASE | queue5 064 + endgame suite |
| 3 | Book off, judged on the platform's own start positions | +5..12 | built, BOOK_ENABLED | queue5 065 (GAUNTLET_OPENINGS=platform) |
| 4 | 8-zone net retrained on 3 months (control) / with endgame loss weighting | +5..25 | trainer built | queue6 kz8c, kz8w |
| 5 | 16 king zones from the 8-zone net (Berserk 8->16: +5; Viridithas +24) | +5..15 | maps built | queue6 kz16 |
| 6 | PVS on top of LMR (rejected alone; the pairing is the standard one) | +0..30 | switch PVS | queue5 063 |
| 7 | Internal iterative reduction | +5..20 | built, IIR | queue5 066 |
| 8 | No consecutive null moves (fakes a repetition; rare, 2% nodes) | +0..10 | built, NMP_GUARD | queue5 062 |
| 9 | Exact speed: eval scratch + blocked head (+28% knps), one-sided zone rebuild | 0 (speed) | committed, exact | in v6 |
| 10 | Time: expected-moves floor 26 -> 18; the field finishes with more clock | +0..15 | not built | 120 s only |
| 11 | Counter-move heuristic / history malus and side indexing | +8..25 | not built | after 1-8 |
| 12 | Curated 5-man Syzygy subset (~20 MB) for the 2-8 piece band | +5..10 | not built | competes for the 50 MB with a 16/32-zone net |

Closed, do not reopen: correction history (x2), endgame fine-tune of the current
net on a reweighted shard, wider accumulator (speed tax exceeds the gain at this
node rate; SF/Berserk/Ethereal all measured it), int8/int16 heads in numba (slower,
measured twice), sparse-head tricks, HalfKP/HalfKA (over the size cap, data-starved),
LMP (-40 on top of the compiled search), 5-man full tablebases (378 MB).

## Instruments

* 8 s SPRT vs the champion, 119 openings: search changes.
* GAUNTLET_OPENINGS=platform: the 80 curated start positions: book / opening changes.
* testing/endgame_suite.py: 400 positions with 5-16 pieces from our own games,
  Stockfish d18 labels, mean cp loss at 2.5 s: evaluation and pruning changes.
* Clock replay x1.5 and 40 games at 120 s: anything touching speed or time.

## Process

One switch per challenger, judged against 060-v6 (the assembled v6). Nets are judged
on the stratified validation loss, the endgame suite and the gauntlet, and at 120 s
before shipping. The v7 bundle repeats the full gate (crash hunt, clock replay,
120 s match, unpacked size < 50 MB). Pondering is gauntleted with 6 workers and must
show zero flags in the 120 s games.
