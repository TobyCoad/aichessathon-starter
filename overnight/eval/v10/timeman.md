# v10 research: time management (literature, measured SPRT results)

Compiled 5 Sep 2026 from OpenBench commit messages (each an SPRT with Elo +/- 95% CI).
Reference time controls: STC 8+0.08, LTC 40+0.4 / 60+0.6; our 120+0.5 sits between.

## Summary

1. Total worth of a modern TM layer over a naive `remaining/20 + inc` allocation:
   ~25-40 Elo measured directly (Berserk #320: +30.6 LTC, +27.7 STC), up to ~60 across
   a full development chain (Alexandria/Weiss sums). Gains shrink at long TC without
   increment and are largest with increments -- our 0.5 s/move increment is the
   favourable case.
2. Biggest single idea: **node-effort scaling** -- spend less when the best root move
   took most of the nodes, more when it did not: Alexandria +19.8 STC / +15.9 LTC,
   Weiss +11.4 / +11.8, Ethereal +9.9 / +9.7 (+20.9 at 40/10). Formula family
   `factor = max(floor, (K - bestMoveNodeFraction) * M)`: Ethereal `max(0.5, 2*nonBest+0.4)`,
   Alexandria `(1.53 - frac)*1.74`, Stormphrax `max(2.59 - 1.6*frac, 0.188)`,
   Viridithas `(1.62 - frac)*1.40`, Weiss `0.52 + 3.73*nonBest`. Needs per-root-move
   node counts (Weiss's follow-up fix of that accounting was neutral: accounting matters,
   retune after).
3. **Best-move stability**: Stash/Viridithas table `{2.50, 1.20, 0.90, 0.80, 0.75}`
   (clamped at 4 stable iterations), Ethereal `1.20 - 0.04*stability` (cap 10),
   Berserk `1.311 - 0.0533*s`, Stormphrax power law. Measured +2 to +7 (Stormphrax
   +6.7 STC / +4.9 LTC on the 4th attempt; Alexandria +3.3/+3.3; Ethereal +2.3/+2.8).
   Gate on depth >= 5-6. Not universal: Weiss has none.
4. **Score-drop factor** (spend more when the score falls over the last 3 iterations):
   Stash `2^(-delta/100)`; Ethereal `clamp(0.05*drop, 0.75, 1.25)`; Alexandria
   `clamp(0.86 + 0.010*idDrop + 0.025*searchDrop, 0.81, 1.50)`; Viridithas +34% per
   aspiration fail-low (max 2). Worth +2 to +4, more at long TC (Ethereal: failed at
   10+0.1, passed at 60+0.6 and 40/40).
5. **Two bounds only**: a soft bound checked at iteration end, a hard bound checked
   every ~1024 nodes. Adding the soft bound was +57/+36/+28 for Weiss (#254, off a weak
   baseline); Alexandria's simplification to exactly two bounds was +4.3/+5.8.
6. **Never predict whether the next iteration fits**: Ethereal removed its
   `estimatedUsage` (branching-factor extrapolation) for +11.9/+6.5/+6.8/+5.7 at four
   TCs. Start the iteration; let the hard bound abort it.
7. Partial iteration results: Ethereal commits a root fail-high immediately and rolls
   back on fail-low; Berserk discards partial iterations entirely. Either is safe;
   naive "use the partial best move" is not (fail-high then fail-low hazard).
8. "Uncertain" flag (Weiss): when the iteration's best move differs from the previous
   one, keep starting iterations past the soft bound: +7.4 STC / +2.9 LTC, +3.5 LTC.
9. All factors multiply against one `ideal_usage`, product clamped (Stormphrax min 0.09).
   Retune constants after any change (Alexandria SPSA retune of TM alone: +4.9/+6.8).
10. Berserk's base allocation `(remaining + 50*inc - overhead)/20`, max
    `min(0.75*remaining, 5.5*alloc)` is the shape to copy; ours (TIME_V4b) already has
    soft/hard bounds but banks ~12 s that is never spent (see games.md) and has none of
    the node-effort / stability / score-drop factors.

## Mapping to our engine (agent.py `choose`, TIME_V4b/TIME_V5)

- We have: soft/hard budgets, aspiration, a floor on expected moves (26; TIME_V5 lowers
  it to 18 with a "two stable iterations -> 1.0 soft budget" refund). We lack:
  node-effort (needs per-root-move node counts: the root loop already searches moves one
  at a time in `root_search`, so `ctrl[C_NODES]` before/after each root move gives the
  fraction), best-move stability scaling, score-drop scaling, the uncertain flag.
- Proposed TIME_V6 switch (120 s only, judged by clocktest + 40 games at 120 s, plus the
  8 s SPRT is byte-identical below LOW_CLOCK so it cannot see it):
  `ideal = (remaining + 40*inc - overhead) / 20` capped by hard = `min(0.5*remaining,
  4*ideal)`; at each completed iteration (depth >= 5) compute
  `factor = stability[min(s,4)] * clamp(2^(-drop/100), 0.5, 2.0) * max(0.5, (1.5 - bestFrac) * 1.7)`
  and stop when elapsed > ideal * factor; never pre-estimate the next iteration; keep the
  hard bound in the kernel poll. Expected +15-30 at 120 s given games.md's finding that
  time is a top-2 loss cause and ~12 s per game is banked unused.
- Also fix the 13 s absorbing floor: RESERVE_FRACTION 0.10 and `rem/30` below LOW_CLOCK
  (games.md items 4-6): allow spending down to ~4 s + increment, i.e. reserve = 3 s flat.

Sources: Ethereal timeman.c and commits 4e894306, 083c7286, 60f4d5c5; Berserk search.c,
commit b7d22980, PR #324, #412; Weiss #254 (e707cf2f), #752 (14012110), #297, #507;
Alexandria #167 (07395162), #281 (a6fedcab), #438, #524, #618; Stormphrax PR #197;
Stash timeman.c; Viridithas timemgmt.rs; Koivisto timemanager.cpp; CPW Time Management.
