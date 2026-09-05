# v9 -- champion + QS_EVAL_CACHE + ADJUDICATION + HISTORY2_FIX + KILLER_CLEAR

Ready to upload: **C:/Users/tobyc/Downloads/aichessathon-v9.zip** (21.6 MB zip,
27.9 MB unpacked; also `submission-v9.zip` in the repo root). Built from the
TESTED challenger dir `overnight/challengers/132-v9core`.

## What changed (vs v8.5, the current upload)
- **QS_EVAL_CACHE** -- memoise quiescence static evals by position key. Exact
  (identical nodes), +4.2% knps.
- **ADJUDICATION** -- fifty-move-rule awareness near the 300-ply platform cap:
  when ahead, avoid drifting into a material-adjudication draw; when behind and
  a fifty-move draw is reachable, steer the search toward it (a horizon of
  non-zeroing plies scores as the draw we want).
- **HISTORY2_FIX** -- zero the quiet-history slot for non-quiet moves so the
  cutoff malus stops punishing stale entries.
- **KILLER_CLEAR** -- clear killers[ply+2] on node entry and the whole table
  between root moves; cross-subtree killers were ordering noise.

## Measured
- 8 s SPRT vs v8.5 (132-v9core, laptop): **PROMOTE, +23 Elo at checkpoint 200**
  (+70 =73 -57, 53.2%).
- Clocktest (v9core-clocktest-l): **PASS** -- 0/6 flags, 0 errors, lowest clock
  11.3 s, longest move 13.2 s.
- Bench depth 8 vs champion: 1.00x / 1.02x / 0.97x nodes for the three
  non-exact switches; QS_EVAL_CACHE bit-identical nodes.
- Cold import of the clean unzip: **33.6 s** (measured under gauntlet load;
  local budget 45 s, platform ~60 s vs its 90 s budget).
- Exactness check after promoting the switches in the tree: 70/70 identical, PASS.
- 40 games at 120 s (v9core-120s, desktop) still queued -- informational only
  for this bundle (no time-management change in it).

## Next bundle (v9.1, already in test)
- **CONT_HIST** (1-ply continuation history): built, 0.89x nodes at depth 8,
  gauntlet 133-conthist queued on the laptop.
- **INIT_FOLD**: constant-folds settled switches at kernel compile; exact
  (bit-identical bench), -4.8 s import. Ships with v9.1, no gauntlet needed.
- **TIME_V6** (smarter clock use): laptop clocktest PASS (lowest 5.7 s);
  judged by the 120 s runs now in flight -- joins v9.1 only if they pass.
