# v11.1 -- v11 + EVAL_SCALE (the search's evaluation scaled x0.75)

Ready to upload: **C:/Users/tobyc/Downloads/aichessathon-v11.1.zip** (21.6 MB zip; also
`submission-v111.zip` in the repo root). Built from the tree by harness.package after the
switch was flipped; testing.check_bundle cold-imported it in a fresh subprocess: compiled
board on, compiled search on, mirroring agrees, kernels from the zip, legal moves.

## What changed (vs v11)
- **EVAL_SCALE on at 300** (was 400): every evaluation the search compares against its
  pruning margins is 0.75x what v11 used. Same net (md5 1298b2c0), same search. The net
  over-states positions on the lines our search actually reaches, so with the old scale
  reverse futility, futility, the null move and the aspiration window all pruned against a
  number that was too rosy.

## Measured (59-position mistake replay at the real 120 s clock, testing.mistakes)
| build | fixed | better | same | worse | cp given away |
|---|---|---|---|---|---|
| pre-v11 baseline | 33 | 6 | 17 | 2 | 2,901 |
| v11 as shipped | 37 | 2 | 11 | 5 | 4,032 |
| v11.1 (scale 300) | 42 | 1 | 15 | 2 | 2,424 |
| v11.1 repeat | 42 | 1 | 15 | 2 | 2,375 |
| per-bucket table 220..125 | 37 | 3 | 18 | 2 | 3,660 |
| scale UP 526 (kernel-consistent) | 37 | 3 | 16 | 4 | 3,894 |
- Clock replay (6 games at 120 s + 0.5 s, x1.5 charge): PASS, 0 flags, lowest clock 5.7 s,
  longest move 10.5 s.
- Cold import of the bundle: 44.9 s in-process here, measured while a GPU fine-tune loaded
  the CPU (v11 measured 33.2 s idle); platform budget 90 s.
- Exactness check: 69/70 identical, the one divergence a 10-node rounding difference at an
  identical score (int truncation of out*300 differs between the two evaluate paths).
- A 40-game 8 s match vs v11 as shipped was started and skipped at the human's request;
  its result is recorded in the journal when it finishes.

## Why not the bucketed table
Both per-bucket tables (the fitted 220..125 and a slope-derived up-table) fixed fewer
positions and gave away 500-1,500 cp more than the flat 0.75 on the same replay.

## Next
- v12 candidate net: v11's mirrored architecture fine-tuned on n80000 x Lichess mixed data,
  epoch ~26/32, validated on a mixed set; gated by the ranking audit (testing.eval_rank),
  then its own scale chosen by the mistake replay before any gauntlet.
