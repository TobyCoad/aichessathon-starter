# The leaderboard leader as a control group (measured 6 Sep 14:50)

The current leader is **Emile Andrieu**. Their 29 public games are fetched to
`overnight/pgn/leader/*.pgn` (filenames are indexed by OUR fetch order, not by the PGN's
own `[Round]` tag -- read the tags). They carry `[%clk]` annotations.

## The headline: the gap is conversion, not survival

| | leader | us |
|---|---|---|
| record | 21-1-2 (~89%) | 15-11-11 (~55%) |
| **draw rate** | **2/29 = 7%** | **11/42 = 26%** |
| decided games ending in checkmate | 22 of 24 | -- |
| threefold draws | **0** | 4 |
| fifty-move draws | **0** | 2 |
| adjudication losses | 0 | 1 |
| median game length (plies) | 128 | 124 |
| longest game | 525 plies | 323 |

If we converted at their draw rate that is roughly **8 extra wins in 42 games**, worth on the
order of **+130 Elo** -- more than everything this project has shipped since v7.1. The gap
is concentrated in exactly one phase.

## Two theories I tested and REJECTED, so nobody re-runs them

- **Clock usage is not the differentiator.** Median clock left at the end: 18.6 s them,
  22.3 s us. Total clock spent per game: 134.4 s vs 126.4 s. They spend about 8 s more --
  real, consistent with our known habit of banking time we never use, but nowhere near
  worth 300 Elo.
- **No opening-book edge.** Instant (sub-0.15 s) opening moves average 0.2 for them and 0.3
  for us. Neither side has a meaningful book. Our own book measured a non-factor already.

## Head to head

We have played them once: `overnight/pgn/platform/round-29-loss-white-vs-emile-andrieu-f1aabee5.pgn`
(we were White, lost by checkmate, 127 plies). Their copy is
`overnight/pgn/leader/round-21-win-black-vs-make-no-mistakes-f1aabee5.pgn`. That game is
already post-mortemed and is our worst on record for evaluation error: 674 cp mean at 11-16
pieces. In it we spent 94.9 s of clock to their 105.4 s, our longest single move was 4.15 s
to their 6.25 s, and we finished with 25.1 s unspent to their 14.6 s.

## Still open, needs CPU

Whether they are simply stronger per move or specifically better at converting is not
answerable from PGNs alone. The test is a Stockfish-referenced ACPL comparison of both
sides of the head-to-head game (`testing.postmortem` takes `--colour`), plus the same over
their wins. Run it when no clocktest or gauntlet is live.
