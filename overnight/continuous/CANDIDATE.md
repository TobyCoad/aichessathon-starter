# v15 -- the mirrored net trained FROM SCRATCH on mixed Stockfish + human data, unscaled

Ready to upload: **C:/Users/tobyc/Downloads/aichessathon-v15.zip** (21.7 MB; also
`submission-v15.zip` in the repo root). Built by harness.package from the tree;
testing.check_bundle cold-imported it in a fresh subprocess: compiled board on, compiled
search on, mirroring agrees, kernels from the zip, legal moves. Net md5 cca8f2d6.
Cold import 38.2 s in-process with the audit chain loading the CPU.

## What changed (vs v12)
- **The net, trained from scratch** on the wide mixed rotation: all 38 Stockfish n80000
  shards alternating with five human months (four Lichess + 2024_11, 40M positions each),
  lr 1e-3 with a two-epoch warm-up, 300 epochs (~7 h), validated on the mixed 800k set.
  v12 was a 32-epoch warm start from v11 on 8 SF shards + 4 human; two attempts to
  continue v12 (lr 1e-4 and 3e-5) never improved on it, so a fresh optimum was the lever.
- Search unchanged. EVAL_SCALE off (unscaled): v15's own scale is 1.01 on the mixed set.

## Measured
- Held-out loss, same positions for both, v12 -> v15: Lichess (human) 0.005227 -> 0.004852
  (-7%); Stockfish n80000 0.007522 -> 0.007485 (-0.5%); mixed 0.006372 -> 0.006171 (-3%).
  Best mixed-val of any net; still improving in the last epochs.
- Exactness check 70/70 identical, 40/40 best move.
- RUNNING at ship time (the human asked to ship without waiting): the 60-position ranking
  audit, the mistake replays at scale 400 / 300 / per-bucket, and the clock replay. Results
  follow in the journal and a note if any disagrees. No self-play match by request.

## Versions this week
v11 (mirrored net, pure Stockfish data) -> v11.1 (on hold: scale 300 helped the blunder
replay but read 40% in 40 games) -> v12 (mixed-data warm start, +56% in 40 games vs v11)
-> v15 (mixed data from scratch).
