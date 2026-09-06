# v9.3 -- v9.2 + the mixed Stockfish/Lichess net

Ready to upload: **C:/Users/tobyc/Downloads/aichessathon-v9.3.zip** (21.7 MB zip, 28.0 MB
unpacked; also `submission-v93.zip` in the repo root). Built from the TESTED challenger dir
`overnight/challengers/153-mixnet2` (v9.2's search + the new net, md5 45f73c3f).

## What changed (vs v9.2)
- **The evaluation net is retrained on engine-distribution data.** 581M positions from
  Stockfish's own self-play training data (Feb 2024 binpack, decoded by our new
  training/binpack_decode.py, scale-corrected) were interleaved with the four Lichess
  months, and the current net was fine-tuned on that mix (24 epochs, best at 21).
  It halves the error on engine-like positions (SF-val 0.002702 vs 0.005318) while
  staying close on human positions; the search is unchanged.

## Measured
- 8 s SPRT vs v9.2: **PROMOTE at the 200-game checkpoint, +19 Elo** (this time verified:
  the challenger's net differs from the champion's by md5, and the gauntlet log shows it).
- Endgame suite at 2.5 s: 13.8 cp vs the champion's 10.8 -- better below 9 pieces
  (8.8 vs 17.0), worse at 9-12 (23.8 vs 12.0). Mixed on the suite, positive in games.
- Cold import of the clean unzip: 35.7 s here (platform ~60 s of its 90 s budget).
- Clock test / 120 s games: not re-run (no time-management change).

## Tonight's Stockfish-data experiment, in order
- Decoded 1.03B positions from the binpack in 46 min (581M kept after quiet filters).
- The champion net's loss on engine positions was 48% worse than on human ones: the
  distribution gap was real.
- Pure Stockfish retrain: REJECT (-76). Cause: my score scale was 1.7x too loud and the net
  forgot human positions. Scale corrected, shards rescaled.
- Mixed retrain with the corrected scale: this net, +19. A variant with its output head
  rescaled to unit slope (155-mixnet2s) is queued next and may replace it if it scores higher.

## Also overnight
- v9 (QS cache, adjudication, history fix, killer clearing, +23), v9.1 (TIME_V6, 55% at
  120 s), v9.2 (NMP_V2, +26) shipped by email. Capture ordering: flat (closed). Improving /
  cut-node flags: REJECT. Continuation history: REJECT.
