# v9.1 -- v9 + TIME_V6 (time management rebuilt)

Ready to upload: **C:/Users/tobyc/Downloads/aichessathon-v9.1.zip** (21.7 MB zip,
28.0 MB unpacked; also `submission-v91.zip` in the repo root). Built from the TESTED
challenger dir `overnight/challengers/v9-120s-l` (v9's kernel + the TIME_V6 switch).

## What changed (vs v9, the current upload)
- **TIME_V6** -- the clock is used instead of banked. The old scheme kept a 10% reserve
  and dropped to 0.5 s moves below 15 s, which the post-mortems showed left ~12 s per game
  unspent and caused time-trouble blunders. The new budget credits the increment it
  measures between calls, keeps a 6% reserve, uses the literature's stop rule at each
  iteration end (best-move stability x score-drop x node-effort, from Ethereal / Stash /
  Weiss / Alexandria) instead of predicting the next iteration, and below 12 s spends an
  exact eighteenth of the clock per move so it settles near 6-8 s instead of 13 s.

## Measured
- 40 games at 120 s + 0.5 s on the platform openings vs v9: **+11 =22 -7, 55.0%**.
- Clock replay (1.5x charge, 6 games): **PASS**, 0 flags, lowest clock 5.8 s, longest
  move 11.9 s (v9's replay: lowest 11.3 s -- the new floor is by design).
- 8 s SPRT: not applicable (below 12 s the low-clock rule dominates; TIME_V6 is a 120 s
  change, so the 120 s games are its gate).
- Cold import of the clean unzip: 41 s here under heavy load (decode + tests running);
  v9 measured 33.6 s; platform budget 90 s (v8.5 took 63 s there).

## Not included
- CONT_HIST (continuation history): REJECT at 8 s vs v9 -- closed.
- IMPROVING / CUTNODE: built (off), not yet tested; next bundle.

## Next
- v9.2 bundle: IMPROVING + CUTNODE + NMP_V2 (+ CAPTURE_ORDER if built), 8 s SPRT + clocktest.
- Stockfish-data net: the February 2024 Stockfish self-play binpack is decoding into
  ~580M positions now; a warm-started retrain on it runs tonight and is compared with
  the Lichess-trained net on the endgame suite and a gauntlet. Ships as its own version
  if it wins.
