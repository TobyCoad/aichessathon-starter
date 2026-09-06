# v9.6 -- v9.5 + INIT_ASYNC: the init timeout can no longer lose a game

Ready to upload: **C:/Users/tobyc/Downloads/aichessathon-v9.6.zip** (21.7 MB zip, 28.0 MB
unpacked; also `submission-v96.zip` in the repo root). Built from the TESTED challenger dir
`overnight/challengers/v96-clocktest-l`.

**The engine is byte-for-byte v9.5** -- same search, same net (md5 9e2b0006), depth-8 bench
identical at 1,014,119 nodes. The only change is *when* the numba compile happens. This is a
robustness fix, not a strength fix, and I would rather it went up than v9.5 did.

## The defect
The platform starts a fresh process for every ladder game and gives `import agent` a fixed 90 s
budget. A game whose import overruns is lost outright ("game ended white by init") before a move
is played. Our four platform samples: **74.1 s, >90 s (LOST), 88.1 s, 64.1 s** -- one loss in
four, and the two survivors at 74 and 88 were inside the budget by seconds. 89% of that import is
numba compiling the search kernel.

## What INIT_ASYNC does
The kernel compile moves to a daemon thread. Import waits for it only until `INIT_READY_S = 72.0`
seconds from the top of `agent.py`; past that, import returns, the runner prints its ready line
inside the 90 s budget, and the **first `get_move` joins the thread and charges the wait to its
own move budget** (`_join_warmup` subtracts it from `time_left_ms`, floor 200 ms). A slow first
move on a 120 s + 0.5 s clock is survivable; a failed init is a certain loss. All four samples
above become safe.

## Evidence
- **`v96-clocktest-l` PASS** (21:07): flags 0/6, errors 0, lowest clock 6.0 s, longest move 13.7 s,
  at 120 s + 0.5 s charged x1.5. Measured on this exact build.
- **The deadline path tested directly.** The clocktest cannot reach it: the compile takes ~30 s
  here, well inside 72 s, so locally the switch is a no-op by construction. I ran a copy with
  `INIT_READY_S = 3.0` to reproduce the platform's slow box: import returned at **5.6 s** printing
  `init-async: ready at 5.5s with the search kernel still compiling`, the first `get_move` joined
  the compile (**37.2 s**, charged to itself) and returned a legal `e2e4`, the second move was
  instant. That is the whole mechanism, end to end.
- Clean-unzip cold import **43.0 s**, first move 0.0 s -- measured while an 8 s SPRT was loading
  the machine, so this is a stress number, and it is under the 45 s bar.
  **It is NOT lower than v9.5's 36.0 s and it was never going to be:** below the deadline the
  switch is byte-identical behaviour. Anyone reading this as a speedup is reading it wrong. The
  gain is entirely in the tail we have already lost a game to.
- ruff PASS, mypy PASS, `check_fastsearch --depth 4 --random 30` PASS (70/70 exact, 40/40 best
  move, node ratio 1.00), bench depth 8 = 1,014,119 nodes = v9.5's exactly (the INIT_FOLD gate).

## What is still unproven, unchanged from v9.5
The **net has still never won a rated self-play game against v9.4.** `181-v95-vs-v94` was killed
by our own crash gate (init timeouts from 14 simultaneous cold compiles -- fixed, and every task
now carries `workers: 4`); the re-run **`182-v95-vs-v94`** started at 21:08 and its 200-game
checkpoint lands around 22:20. I will email the verdict either way.

The one piece of game evidence the new net does have is external, not self-play:
**`p8-sf10` finished 62.5% (+21 =8 -11, +88.7 +/- 124.8 Elo) over 40 games at 8 s vs
Stockfish skill 10**, where the same probe on v9.4's net finished at -17.4 Elo. Forty games is
far too few to call it -- the interval spans zero comfortably -- but the sign matches what the
endgame suite and the static-error probe both predicted. It also settles a testing question: at
62.5% that rung is inside the 40-70% band, so **Stockfish skill 10 at 8 s is now our screening
opponent** and we can stop grading ourselves against ourselves.

## Next
`182-v95-vs-v94` (600 games, 8 s, the net's strength verdict) -> `p8-sf12` -> `v95-vs-sf14-120s`.
v9.7's bundle: SEE_QUIET (its rejection was an init-crash abort, not a strength result -- it is
open again), ROOT_NODES, KILLER_SHIFT, ENDGAME_SHRINK, RAZOR.
