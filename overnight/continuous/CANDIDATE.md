# v9.2 -- v9.1 + NMP_V2 (null-move rework)

Ready to upload: **C:/Users/tobyc/Downloads/aichessathon-v9.2.zip** (21.7 MB zip,
28.0 MB unpacked; also `submission-v92.zip` in the repo root). Built from the TESTED
challenger dir `overnight/challengers/143-nmp` (v9.1's kernel + the NMP_V2 switch), no
untested extras.

## What changed (vs v9.1)
- **NMP_V2** -- the null move as the reference engines do it: dynamic reduction
  R = 3 + depth/4 + min((static eval - beta)/200, 3) instead of the old 2 + depth/6,
  tried only when the static eval is at or above beta, and skipped when the table
  already holds an upper bound below beta. One to three plies more reduction on the
  null search, which the research pass flagged as our most under-tuned pruning term.

## Measured
- 8 s SPRT vs v9.1: **PROMOTE at the 200-game checkpoint, +26 Elo** (+76 =63 -62, 53.5%).
- Crash gate: clean, 24/24 vs random.
- Cold import of the clean unzip: **33.7 s** here (platform ~55-60 s of its 90 s budget).
- Clock test: not re-run (no time-management change; v9.1's replay stands).

## Also decided tonight
- Stockfish-data net (152-sfnet): REJECT, -76 +/- 48 at 8 s. Cause found: my score-scale
  calibration made its evaluations 1.7x too loud for the search's pruning margins, and it
  had forgotten human positions. Scale corrected, shards rescaled, a mixed
  Stockfish + Lichess net is retraining (153-mixnet2) and will be gauntleted at ~05:30.
  (An earlier "150-sfnet PROMOTE +19" was void: the worker had tested the champion
  against itself; that worker bug is fixed.)
- IMPROVING + CUTNODE: REJECT (40% over 140). CONT_HIST: REJECT.
- INIT_FOLD and eager fastboard signatures (exact, ~8 s off the import) are built and
  verified but not in this zip; they ship with the next version after a clean-unzip check.

## Next
- Queue: capture ordering (144-caporder), the QS-table + aspiration filler bundle
  (145-v93fill), SEE-of-quiets (147-seequiet), then the mixed net (153-mixnet2).
