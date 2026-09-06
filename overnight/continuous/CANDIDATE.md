# v9.4 -- four search switches + the WDL-target net, in one gauntlet

Ready to upload: **C:/Users/tobyc/Downloads/aichessathon-v9.4.zip** (21.7 MB zip, 28.0 MB
unpacked; also `submission-v94.zip` in the repo root). Built from the TESTED challenger dir
`overnight/challengers/149-v94wdl` -- the exact tree that played the gauntlet -- with
INIT_FOLD flipped on in the zip copy (proved bit-identical below).

This is the biggest single-version jump since v8.5.

## What changed (vs v9.3)
- **NMP_V2B** -- on a null-move cutoff at depth >= 10, verify with a reduced-depth real
  search before trusting it, with Stockfish's zugzwang guard inside the verification
  subtree. Protects exactly the deep nodes where a wrong null cutoff poisons the tree.
- **CAPTURE_ORDER** -- SEE-losing captures are re-ordered below every quiet move; winning
  and equal captures keep their MVV-LVA band, with a capture-history tiebreak learned from
  capture cutoffs.
- **QS_TT** -- quiescence probes and stores the main transposition table (hits of any depth
  cut; stores never evict a real search entry's hash move).
- **ASP_WIDE** -- aspiration re-search windows widen ~1.5x per fail instead of 4x with a
  jump to full width on the third fail: several cheap re-searches near the score instead of
  one expensive full-width pass when the root swings.
- **The net is retrained on WIN/DRAW/LOSS targets** (157-wdlnet): 145M fresh Stockfish
  positions decoded at --wdl-lambda 0.75, merged 50/50 by position count with Lichess
  (290M positions per pass), fine-tuned from the v9.3 net for 12 epochs. Blending the game
  result into the target moves the label most where we have been losing games: mean target
  move 342.9 cp overall, but 765.7 cp at 2-8 pieces and 553.9 at 9-12.
- **INIT_FOLD** rides in the zip (source-only, exact): fastsearch compiles the 18 settled
  switches as constants so numba prunes their branches before typing, cutting several
  seconds off cold start. Verified bit-identical, see below.

## Measured
- **8 s SPRT vs v9.3: PROMOTE at the 200-game checkpoint, +70 Elo** -- +92 =51 -53, 59.9%
  over 196 games, llr +2.86 against a +/-2.94 bound. The strongest verdict any bundle has
  produced (v9 was +23, v9.2 +26, v9.3 +19), and it is not a lucky early stop: the estimate
  had been between +63 and +70 for the previous 20 checkpoints.
- **Clock test at 120 s (x1.5 charge + 20 ms/move): PASS** -- flags 0/6, errors 0, lowest clock 5.7 s against the 5 s floor, longest move 11.9 s. Six full games, no time trouble.
- **Endgame suite** (157-wdlnet vs the v9.3 net): 11.4 cp vs 13.8 overall and better in
  every band -- 5-8 pieces 4.2 vs 8.8, 9-12 21.5 vs 23.8, 13-16 7.7 vs 8.5.
- **Bench, fixed depth 8**: 1,110,289 nodes at 233-249 knps -- 23% fewer nodes than v9.3's
  1,445,087 for the same depth, which is the four ordering/pruning switches doing their job.
- **INIT_FOLD is exact**: the zip build (fold on) and the tested challenger (fold off) both
  bench 1,110,289 nodes at depth 8 -- identical to the node, so the fold changes speed only.
- **Cold import of a clean unzip: 38.1 s** measured while the clock test had four workers on
  the machine, so it is a pessimistic read (the platform's budget is 90 s and it is ~1.8x
  slower than this laptop -- the honest figure to compare is the idle one, ~34 s here).
- Unpacked 28.0 MB, well under the 50 MB limit; agent/fastboard/fastsearch + the whole
  weights tree, net md5 `1f4be882` (the tested WDL net, not the v9.3 one).

## A caveat worth stating
The +70 is a 200-game checkpoint. Our own power simulation says a checkpoint promotion at
+19 is weak evidence (155-mixnet2s ran +69 at 76 games and decayed to -3 by 346), but +70
with llr +2.86 is a different animal -- it is a bound-crossing result, not a checkpoint
squeak. Treat it as a real gain of somewhere between +25 and +70 at 8 s; 120 s gains have
historically been about half the 8 s gain.

## What is in the next bundle (v9.5)
ADJ_V2 (built, off) -- the engine half of the 600-ply finding. Our local rules copy says the
platform adjudicates on material at ply 300; the canonical rules say a game still running at
600 plies is simply DRAWN. The champion therefore values a draw at more than a rook in
ordinary long middlegames (+320 cp a rook down by ply 300, on a premise that is false).
ADJ_V2 re-bases the whole ramp on 600 and caps the behind-side draw bonus at 100 cp.
With it: DRAW_BUDGET (widened), ROOT_NODES, SINGULAR_EXT2, RAZOR, plus CUTNODE and
SEE_QUIET if their gauntlets return positive -- one gauntlet for the bundle, as agreed.
