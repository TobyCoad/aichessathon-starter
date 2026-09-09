# v17 research -- EVALUATION side, ours vs Alexandria 9.0 (9 Sep 13:50)

Reference: engines/alexandria/src/{nnue.cpp,nnue.h,eval.h} at tag v9.0.0 (gitignored; ideas
only -- rules.md bans shipping any port/translation or any net we did not train). Ours:
agent.py (net load, kernels, FastEngine.evaluate), fastboard.py (accumulator push/rebuild),
fastsearch.py (evaluate, ENDGAME_SHRINK blend). Everything architectural is ALREADY DONE as
of v16f: pairwise, dual activation, endgame-dense buckets, factoriser, 16 mirrored king
buckets, same 768 feature encoding (verified bit-exact against their binary). What is left
is inference machinery and post-processing.

## Verdict in three lines
- The network is no longer the gap (v16f beat v12 +116 at 8 s; Alexandria's own net ranked no
  better than v12 at 1 ply). Eval-side items below are worth +5..+10 combined, not +50.
- The one real mechanism we lack is an ACCUMULATOR CACHE for king-zone crossings (their
  "Finny table"). Everything else is either equal, closed by measurement, or tiny.
- Spend gauntlet slots on search (S2, RAZOR/CUTNODE retest, ordering/history/time reports);
  the eval items ride along as bundle fillers.

## Ranked

| # | Item | Theirs | Ours | Elo @120 s | Cost | Gate |
|---|---|---|---|---|---|---|
| E1 | **ACC_CACHE** (Finny table) | nnue.cpp povActivateAffine: per (side, bucket, flip) cached accumulator + 12 occupancy bitboards; on every eval diff the cached entry against the board (add/remove only the pieces that differ) | fastboard.make_full: a king move that changes zone calls `rebuild` for that perspective = sum of ALL piece rows (~30 x 1024 floats); normal moves push 2-4 rows | +3..8 (pure speed; bigger on the 1024 net than it was on 512) | 3-4 h | bench knps at d8; check_nnue's "accumulator matches full rebuild" is the exactness proof |
| E2 | **HMC_DAMP** 50-move damping | eval.h adjustEval: `eval * (200 - halfmove) / 200` | none (ADJUDICATION/DRAW_BUDGET act at the root only) | +2..5 (fewer shuffles into 50-move draws when ahead; makes progress) | 0.5 h | draws audit (testing/draws.py) + 120 s; invisible at 8 s |
| E3 | MATERIAL_SCALE | eval.h ScaleMaterial: `eval * (22400 + material) / 32 / 1024` -- ~0.68x at bare kings, ~0.93x at full material, linear in material | EVAL_SCALE / EVAL_SCALE_PHASE / EVAL_SCALE_SMOOTH (all OFF; fitted per piece count, v11.1 at scale 300 lost 40%/40) and ENDGAME_SHRINK (OFF) | -10..+10; same direction as our own fit (net too loud in the endgame) but every scaling test here lost games | 0.5 h | 8 s gauntlet; do NOT bundle with anything else, it changes every margin |
| E4 | TT static eval reuse | stores rawEval in the TT, skips evaluate on a hit | TT_EVAL REJECTED; QS_EVAL_CACHE + eval cache ON | closed | -- | -- |
| E5 | Correction history (pawn / non-pawn / cont) | tune.h corrhistory*Weight, applied in adjustEval | CORRECTION / CORRECTION_QS built, REJECTED x2 | closed unless the agent report finds our version differed materially | -- | -- |
| E6 | Eval scale for pruning margins (their NET_SCALE 362 vs our 400 logit) | one constant, margins tuned to it | PIECE_SCALE flat 400; margins tuned to it | equal by construction | -- | -- |
| E7 | int16/int8 quantised inference | yes | measured 1.9x SLOWER in Python, closed 3x | closed | -- | -- |

## E1 in detail (the one to build)
Mechanism: keep `acc_cache[side, block, 1024]` (block = zone + KING_ZONES*flip, 32 blocks per
side) and `occ_cache[side, block, 12]` bitboards. On a zone crossing for `side`, instead of
`rebuild`: take the cached entry for the NEW block, XOR its 12 occupancy boards against the
board's, add rows for pieces present-but-not-cached and subtract rows for cached-but-absent,
then copy the result into the live accumulator AND back into the cache with the current
occupancy. First visit to a block in a search = a full rebuild into the cache (same cost as
today, once). Entries never go stale: they always describe exactly the position their
occupancy records, so a diff is exact by construction. Memory 2 x 32 x 1024 x 4 B = 256 KB +
6 KB of bitboards. Numba: two nested loops over set bits, reusing `_acc_row_one`.
Where the win is: kings cross zones on most king moves in a 16-zone mirrored map (every
file change across d/e flips, every rank change on ranks 1-2 crosses); each crossing today
costs ~30 row adds against ~3 for an ordinary move. If ~7% of made moves are crossings that
is ~2 extra rows per move on average, i.e. roughly 40% of all accumulator work -- and the
accumulator is 15.4% of node time (speed.md), so ~6% of node time, ~+6% knps ceiling, more on
v16f's 1024-wide rows than the 512 profile above measured.
Verification: `training.check_nnue` already compares the incremental accumulator against a
full rebuild over 5,972 plies including castling/promotion/en passant -- that is the exactness
gate; then bench d8 knps (nodes must be IDENTICAL, only time moves).
Switch: ACC_CACHE in agent.py; cache arrays owned by FastEngine, passed like `astack`; the
crossing branch in fastboard.make_full picks cache-diff vs rebuild on a flag.

## E2 in detail
`fastsearch.evaluate` already reads `meta[fb.HALFMOVE]`-adjacent fields for draw scoring;
apply `score * (200 - hmc) // 200` there (and in FastEngine.evaluate for root contempt) when
hmc > some floor (Alexandria applies always). Judge on testing/draws.py (ahead-draws count)
at 120 s; 8 s games rarely reach hmc > 40.

## Not gaps (equal or ours is fine)
- Incremental push on make vs their rebuild-by-diff on every eval: different designs, same
  cost class once E1 exists; LAZY_ACC already defers ours to the first evaluate on a line.
- Output bucket map, dual activation, pairwise, factoriser, mirroring: identical ideas now.
- Eval clamping to the non-mate range: DISTANCE_THRESHOLD guards do the same job.
