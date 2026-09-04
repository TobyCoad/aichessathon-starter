# Search depth plan (4 Sep 2026)

Goal: two to three more plies at the tournament control. Profile of the current
search (8 s, 963k nodes): evaluation kernel 41%, Python search loop 25%,
make/unmake dispatch 25%, table dict 9%. Node ~4 us idle, ~240 knps.

Every stage is one change, tested before the next starts. The champion on the
platform (v4) is untouched until the whole chain passes the 120 s gate.

## Stage 0 -- measurement (testing/bench.py)
Fixed set of 40 positions (openings, middlegames, endings, tactics).
- fixed depth: nodes, time, best move, score -- the determinism reference
- fixed time: depth reached -- the number we are trying to move
Gate: runs green on the current engine; numbers recorded in overnight/eval/bench-*.log.

## Stage 1 -- compiled search (fastsearch.py)
Negamax with quiescence inside one recursive numba kernel, same semantics as
FastEngine.search: TT probe/store, RFP, null move, futility d1-2, check extension,
killers, butterfly history, MVV-LVA ordering, delta pruning, contempt draw score,
repetition against the game history and the stack.
- TT: fixed arrays (keys uint64[N], packed data int64[N]), replace if deeper or older.
- Abort: kernel checks the clock through objmode every 4096 nodes and sets a flag;
  every frame unwinds normally, so the position arrays are always consistent.
- Root loop stays in Python: iterative deepening, per-root-move calls, TIME_V2/V3/V4
  unchanged. Tablebase probing stays at the root; in-tree probing is dropped
  (4-man positions in the tree are scored by the net + search).
Gates, in order:
  a. differential: with the TT disabled in both, score and best move identical to
     FastEngine at depths 1-5 on the 40 bench positions and 200 random positions.
  b. with the TT: best move agreement >= 95% at depth 6, node counts within 20%.
  c. speed: >= 3x knps on the bench.
  d. crash hunt: 500 games vs random, zero failures; fallback path still works.
  e. clock replay (testing/clocktest) floor >= 10 s under 1.5x charge.
  f. gauntlet SPRT[0, 20] vs champion at 8 s -- expected to pass easily.

## Stage 2 -- cheaper evaluation inside the kernel
- static-eval cache in the TT entry, reused after null move.
- int16 head (quantized W2/W3) if evaluate is still > 30% of node time.
Gate: NNUE check error < 2 cp vs float; bench speed; gauntlet no regression.

## Stage 3 -- fewer nodes per ply, one switch each, SPRT[0, 20] at 8 s
1. PVS + aspiration windows
2. LMR (log-based reductions, re-search on fail high)
3. late move pruning at depth <= 3
4. SEE pruning of losing captures in quiescence
Each promoted only on PASS. The bundle then plays 40 games at 120 s vs the champion.

## Stage 4 -- ship
clocktest + crash hunt + 120 s match on the final build; package with
`python -m harness.package --include fastboard.py --include fastsearch.py`;
the user uploads. Hard go/no-go: Tuesday 8 Sep evening. If stage 1 has not passed
gate (f) by then, the port is dropped and v4 stays.
