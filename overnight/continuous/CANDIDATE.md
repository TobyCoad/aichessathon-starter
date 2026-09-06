# v9.5 -- v9.4 + a net retrained on engine data with Stockfish labels

Ready to upload: **C:/Users/tobyc/Downloads/aichessathon-v9.5.zip** (21.7 MB zip, 28.0 MB
unpacked; also `submission-v95.zip` in the repo root). Built from the TESTED challenger dir
`overnight/challengers/180-sf100`, with INIT_FOLD flipped on in the zip copy (proved exact:
bench depth 8 is 1,014,119 nodes with the fold on AND off).

Search is UNCHANGED from v9.4. This is a net-only change.

## The defect this fixes, traced end to end
Our evaluation overvalued positions where we are attacking. Against a Stockfish d16 reference
on 258 positions from our own games, the signed error in attacking positions was -452 cp for
the original Lichess-trained net and -209 for v9.4's, while quiet positions sat near zero.

The cause is selection bias in HUMAN games. Humans enter attacking positions when the attack
works, so the corpus contains a filtered sample of the successful ones. Measured on 2,000
positions from real games (ours and the leader's), the TRUE median value of having an attack
is **+5 cp**; the Lichess corpus teaches **+440**; engine self-play teaches **+6 to +10**.
The corpus lies by 435 cp and the net inherited 452 cp of it -- a near one-for-one match that
closes the chain from corpus to blunder.

That is the mechanism behind our blown wins: we build an attack, believe we are ~450 cp better
than we are, commit, it does not break through, and a won game becomes a draw or a loss.

## What changed
The net is retrained on **321 M positions of engine self-play with Stockfish n20000 labels**
(vondele/rescored), replacing the 50/50 human/engine mix. No human data at all. The labels
correlate **0.978** with our Stockfish reference against **0.861** for the labels v9.4 learned
from. Scale re-derived locally as 0.2584 cp/unit before training (the step that cost a night
when 0.45 was used instead of 0.262).

## Measured
- **Endgame suite, 400 positions at 2.5 s: 7.5 cp mean loss, the best any net has recorded**
  (v9.4 11.4, v9.3 13.8, the old champion 10.8). The 9-12 piece band **halved, 21.5 -> 9.8**.
  This is the instrument that scores MOVE CHOICE on positions from our own games below 16
  pieces, which is where every platform loss and draw we have analysed ended up.
- **Attack bias eliminated**: signed error in attacking positions -158 -> **+11** on a neutral
  2,000-position probe; quiet +27 -> -1; weighted evaluation error **-24%**.
- check_nnue: all checks passed. check_fastsearch: 70/70 + 40/40 PASS.
- The zip was UNZIPPED AND RUN, not just built: cold import 36.0 s, plays e2e4, a sane K+P
  endgame move and a sane middlegame move. Net md5 9e2b0006 (the tested net).

## The honest caveat
**There is no game evidence yet.** The 8 s SPRT against the v9.4 champion is running now and
its 200-game checkpoint lands around 21:00. Our own notes warn that the endgame suite is "a
veto on catastrophes, not the ranking signal" -- v9.3's net scored WORSE on it (13.8) and still
won +19 Elo in games. Three instruments agree here and two of them are not circular, but none
of them is Elo. If the gauntlet comes back negative I will say so immediately.

## Update 6 Sep 20:35 -- one gate PASSED, the other aborted before it played a game

**`v95net-clocktest-l`: PASS** (20:07). flags 0/6, errors 0, lowest clock 5.5 s, longest move
12.8 s -- measured on THIS net, not on the earlier v9.5 that shared the number. This is the
mandatory gate, and the zip is safe to run: it does not flag and it does not error.

**The 8 s SPRT still has no result, and this time it is our test harness, not the engine.**
`181-v95-vs-v94` printed `REJECT ... failed 7/24 games (init 7)` at 20:12. That verdict is a
crash-gate abort, NOT a strength verdict: it never reached stage 2. All seven failures are
`init` -- the 24-game gate ran 7 games in parallel, i.e. 14 engine processes each compiling the
numba kernel from cold at the same moment, and seven of them took longer than the 90 s init
budget. Nothing was played and nothing about the net was measured. Re-queued as
`182-v95-vs-v94` with the concurrency halved (4 workers); its 200-game checkpoint lands about
an hour after it starts.

**What that abort does say, and it is not nothing.** Our cold init is 36.0 s single-process
here; the platform's box is ~1.8x slower against a 90 s budget, so v9.5 lands around 65 s of
90 with no margin for a loaded machine. Tonight our own gate showed init blowing straight
through 90 s the moment the machine is busy. That is the case for v9.6 (INIT_FOLD +
INIT_ASYNC, both already built and measured), whose clocktest is on the machine now.

**Recommendation, unchanged and explicit: v9.5 is safe to upload but its strength is still
unproven.** Three non-game instruments agree it is better and none of them is Elo. If you would
rather wait, the SPRT verdict will be in your inbox tonight; if you upload now, nothing in the
clocktest suggests it will misbehave.
