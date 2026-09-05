# V10 plan -- consolidated from the 5 Sep research pass

Sources: overnight/eval/v10/search.md, network.md, games.md, timeman.md (speed.md pending).
State when written (5 Sep 17:40): v8 live on the platform (ladder record 10-6-6 before it);
v8.5 = v8 + LMR_AGGRESSIVE + LAZY_ACC + TIME_V5 + PRUNE_V2 + SINGULAR PROMOTED at 8 s
(+36 Elo over 477 games, 55.2%), clocktest PASS, the 40-game 120 s gate re-queued on the
desktop after a load-induced init-timeout run. Freeze target: 10 Sep evening; uploads close
11 Sep 11:00. Each 8 s SPRT costs ~3 h of the laptop or ~5 h of the desktop, so about
6-8 more verdicts fit. Elo figures are at 120 s (the 8 s SPRT reads ~2x).

## What the evidence says (three lines)
- Every platform loss and draw reached <= 16 pieces; the net's static error there is 475 cp
  vs 70 cp in the middlegame. That is an evaluation problem, and only a net or an endgame
  eval fix touches it. Search changes buy depth, which recovers only 17% of those errors.
- Time is the second cause: a 13 s absorbing floor (RESERVE 0.10 + rem/30 below LOW_CLOCK)
  banks ~12 s per game that is never spent, and 5 of 17 blunders were time-caused at < 17 s.
- The search is already modern; what it lacks are the accuracy terms (continuation
  history, improving/cutnode) that let reference engines prune harder, plus a shallow null
  move (R = 2 + d/6 vs 3 + d/4 everywhere else).

## Ranked list (expected value per gauntlet slot)

| # | Idea | Elo @120 s | Cost | Gate | Owner / when |
|---|---|---|---|---|---|
| 1 | **TIME_V6**: reserve 0.10 -> 0.04, LOW_CLOCK 15 -> 9, credit the increment, node-effort factor `max(0.5, (1.5 - bestFrac)*1.7)`, best-move stability table {2.5,1.2,0.9,0.8,0.75}, score-drop `2^(-drop/100)` clamped [0.5,2], never pre-estimate the next iteration; absorbs TIME_V5 | +15..30 | 4-6 h | clocktest at 1.5x charge + 40 games at 120 s (8 s cannot see it) | laptop build tonight; desktop 120 s gate |
| 2 | **CONT_HIST bundle**: 1-ply continuation history (piece-to) in ordering + LMR adjust + quiet pruning; fix the stale `quiets[]` malus (HISTORY2_FIX); clear killers[ply+2] on entry and the table between moves | +12..25 | 5-7 h | 8 s SPRT vs v8.5 | laptop, after #1 is built |
| 3 | **ADJUDICATION**: the referee adjudicates on raw material at match ply 300 (round 18 lost a dead-equal K+R+N vs K+Q that way); pin the ply counter to match plies, ramp behind-contempt toward a full half point as ply -> 300, prefer lines that reach a fifty-move draw before the cap | +3..8 plus avoided losses | 2 h | 8 s SPRT (the harness enforces the same cap: 7 adjudications in 477 games) | laptop, small |
| 4 | **NET_V10**: mirrored king buckets (own-king file mirror, 16 zones over the a-d half board, 0 MB) + rebalanced 12 endgame-dense output buckets, warm-started from b8-kz16 | +5..12 | 2 GPU h + export/inference change (9 files touch the hot path) | val loss, endgame suite (needs a v8 baseline first, ~18 min), 8 s SPRT, 120 s | GPU after 104-kz16r; one gauntlet slot |
| 5 | **IMPROVING + CUTNODE** flags gating RFP / futility / LMR / NMP | +8..18 | 3-5 h | 8 s SPRT (bundle with #6) | laptop |
| 6 | **NMP_V2**: R = 3 + d/4 + eval margin, skip when TT says fail-low, verification search at depth >= 10 | +6..15 | 2-3 h | 8 s SPRT (bundle with #5) | laptop |
| 7 | **CAPTURE_ORDER**: SEE-ordered captures, losing captures below quiets, capture history | +6..14 | 2-4 h | 8 s SPRT | if a slot frees |
| 8 | **QS_EVAL_CACHE on** (exact, +4.2% knps under v8.5 per speed.md; `see` allocation and evaluate blocking measured at the noise floor) | +2..4 | 0 h | bench only; in the v9 bundle | done in speed.md |
| 9 | **QS transposition table** (probe + store in quiescence) | +4..10 | 2-3 h | 8 s SPRT | later |
| 10 | **INIT_FOLD**: constant-fold the settled switches into the kernel (fs.warm_up -18%) + eager signatures in fastboard (61 redundant specialisations, 7 -> ~4 s); numba cache / AOT REJECTED (segfaults, native binaries forbidden) | 0 direct; init 31.7 -> ~24 s idle | 2-3 h | import time in a clean dir + exactness | v9.1 |
| 11 | ENDGAME_SHRINK: below 17 pieces blend the net toward a material/PSQT baseline, or resample training data by piece bucket | +5..15 | 3 h | endgame suite + 8 s SPRT | risky; only with a v8 suite baseline |
| 12 | Aspiration widening 1.5x per fail (no jump to +/-INF), root improvements, killer decay | +0..5 each | 0.5 h each | fold into a bundle | filler |
| 13 | 104-kz16r five-month net (in flight) | +0..5 | running | suite + SPRT | fold when it lands |
| 14 | Endgame shard in rotation (1 of 6 shards) | -5..+8 | 1.5 GPU h | suite must not regress | only if #4 passes |
| 15 | Curated 5-man syzygy subset (~15-20 MB) | +3..6 | 0 build | none | would have changed no platform result; skip |

Closed (measured or dominated; do not reopen before 11 Sep): staged movegen, multi-cut,
IID, more TT replacement, QS checks/evasions, correction history, width 768/1024,
distillation, int8/int16 inference (1.9x slower than float32, measured three times), sparse head,
shipped numba cache / AOT (segfaults on rebuild; native binaries forbidden), self-play labelling at scale (costs
the verdict machine), 6-man tablebases, full-month book rescan, HalfKA features.

## Sequencing
1. Tonight: #1 TIME_V6 and #3 ADJUDICATION built as switches; TIME_V6 to the desktop 120 s
   gate, ADJUDICATION into the next laptop SPRT with #2.
2. #2 CONT_HIST bundle built next; SPRT vs v8.5 on the laptop (~3 h).
3. #5 + #6 as one "pruning accuracy" bundle after that.
4. GPU: v8 endgame-suite baseline, then #4 NET_V10 once 104-kz16r finishes; its SPRT on
   whichever machine is free.
5. v9 candidate = v8.5 + every pass; full gate (SPRT, crash hunt, clocktest, 40 x 120 s),
   zip, human upload. Repeat as v9.5 if #4 lands in time. Freeze 10 Sep evening.
