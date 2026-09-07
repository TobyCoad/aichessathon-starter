# v12 -- the mirrored net fine-tuned on mixed Stockfish + human data, unscaled

Ready to upload: **C:/Users/tobyc/Downloads/aichessathon-v12.zip** (21.6 MB; also
`submission-v12.zip` in the repo root). Built by harness.package from the tree;
testing.check_bundle cold-imported it in a fresh subprocess: compiled board on, compiled
search on, mirroring agrees, kernels from the zip, legal moves. Net md5 6170827d.

## What changed (vs v11)
- **The net.** v11's mirrored 16-zone architecture, warm-started from v11 and fine-tuned
  32 epochs on Stockfish n80000 shards interleaved with the four Lichess human months
  (validation on a mixed 800k set). v11 had been trained on engine positions only and had
  lost the human-position judgement; this puts it back while keeping most of the n80000 gain.
- **EVAL_SCALE off** (400, unscaled). v12's own scale is ~1.0 on the mixed set, and the
  mistake replay prefers it unscaled (see below). v11.1's x0.75 does NOT apply to v12.

## Measured
- Held-out loss, same positions for both: Lichess (human) 0.005227 vs v11 0.007007 (-25%);
  Stockfish n80000 0.007522 vs 0.006910 (+9%); mixed 0.006372 vs 0.006968 (-8.5%).
- Ranking audit (testing.eval_rank, 60 blunder positions, every legal move vs Stockfish d12):
  root error 114 cp (v11 133), loss of the net's top move 201 (v11 239), Stockfish's best
  in the net's top 3: 35% (v11 27%), rank correlation 0.40 (v11 0.30). Best of every net
  measured; the shared 13-16-piece weakness is unchanged.
- Mistake replay (59 positions, real 120 s clock): **41 fixed / 4 better / 14 same / 1 worse,
  2,094 cp given away** -- v11: 37 / 2 / 11 / 5 / 4,032; the best row of any build so far.
  At scale 300 it is worse (39 / 2 / 17 / 2 / 3,174), hence unscaled.
- Exactness check 70/70 identical, 40/40 best move. Cold import 63 s measured under a GPU
  training run plus a 4-worker match; v11's idle figure with the same net size was 33 s.
- NOT yet: the 40-game 8 s match vs v11 and the clock replay are running now; their results
  are recorded in the journal and sent as a follow-up if either disagrees.

## Overnight
v13 = v12 continued on all 38 Stockfish shards x five human months, 150 epochs; then v13b =
v13 + an endgame-heavy shard; each audited, replayed at both scales, and v13 plays 400 games
vs v12 at 8 s. Results by morning.
