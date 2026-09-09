# v17 research -- TIME / ID / TT / ROOT / EXTENSIONS / DRAW, ours vs Alexandria 9.0 (9 Sep)

Static reading only. Reference: `engines/alexandria/src/{time_manager.cpp, search.cpp, ttable.cpp/h,
tune.h, history.cpp, init.cpp}` (eval.h is not in the checkout; its adjustEval is covered by
`v17/eval.md` E2). Ours: `agent.py` (root loop `FastEngine.choose` L2817-3070, budgets L3395-3480),
`fastsearch.py` (kernel `search` L931-1636, TT L423-486). Ideas only, no code text. Nothing here
re-proposes the closed list (PVS alone, IIR, TT_KEEP -32, TT_BUCKETS, TT_EVAL, ponder, NMP guard,
check-ext cap, ASP_WIDE).

## Verdict in four lines
- **Time is the one structural gap.** Alexandria's optimum is close to ours, but its CEILING per move
  is 5-8x optimum (stability x2.38, node-effort x2.66, eval-instability x1.25, hard = 0.76 of the
  clock). Ours clamps the product at 1.5x and hard at 10% of the clock. HORIZON.md measured the
  median horizon move needed 1.57x. That is the only item here worth a 120 s slot.
- Aspiration, root handling, TT layout: equal or better in idea; the differences are fillers
  (+0..5 each, root-loop ones invisible to the bench by construction).
- Extensions: two cheap mechanisms we lack that need NO improving flag -- singular multi-cut and
  LMR do-deeper/do-shallower re-search -- plus the fact that Alexandria's `improving` is used in
  exactly ONE pruning term (LMR +1) and one margin; ours fed three. Split it.
- Draw handling: equivalent except the missing 50-move eval damping (eval.md E2) and one
  five-line TT guard. And the tree carries an unresolved contradiction about the referee cap
  (ADJUDICATION assumes ply-300 material; ADJ_V2's comment says "600 and a draw, verified twice").
  Settle that before touching contempt.

## Slot budget (5-6 left before 11 Sep 11:00)
The final bundle gate (SPRT + crash hunt + clocktest + 40 x 120 s) eats ~2 slots itself, so there
are TWO experiments: (A) one 8 s SPRT on a depth bundle, (B) one 120 s run on TIME_V7. Everything
else rides as filler on bench evidence or does not ship.

| Rank | Item | Elo @120 s (honest) | Cost | Gate |
|---|---|---|---|---|
| 1 | **TIME_V7** upside multiplier (section 1) | +5..15, CI +/-20 at 40 games | 2-3 h | clocktest 1.5x + HORIZON postmortem replay (the only instrument with power) + 40 x 120 s |
| 2 | **S3 depth bundle**: SINGULAR_MULTICUT + LMR_DEEPER + IMPROVING_LMR + HINDSIGHT(built) + ASP_V2 + TT_HMC90 | +5..15 at 8 s | 3-4 h | bench each arm at d8 first (free), then one 8 s SPRT |
| 3 | TT_PV bit (ttPv into singular margin + LMR) | +2..6 | 2-3 h | bench + rides on S3 if ready |
| 4 | TT_INTERLEAVE (one array, key/data adjacent; exact) | +1..3 (speed) | 1-2 h | bench knps only, no slot |
| 5 | OPP_HIST_EVALDIFF (opponent-quiet history from eval swing) | +1..4 | 1 h | bench, filler |
| 6 | TT_LENIENT replacement (depth + 4 >= old) | +0..3 | 0.5 h | bench, filler |
| -- | root history, root LMR, cuckoo cycle detection, eval-stability alone, IIR variants, check-ext removal | +0..3 each or negative | -- | not before the freeze |

---

## 1. TIME MANAGEMENT

### Alexandria (time_manager.cpp, search.cpp)
- Base (`Optimum`, L8-38): overhead = min(25 ms, t/2); movesToGo = 50; `timeLeft = t + inc*49 -
  overhead*52`; `optScale = min(0.025, 0.20 * t / timeLeft)`; `opt = optScale * timeLeft`;
  `max = 0.76 * t - overhead`. (tune.h L110-111: optScaleFixed 25, optScaleTimeLeft 200.)
- Scaling (`ScaleTm`, L44-56), applied after every completed iteration once RootDepth > 7
  (search.cpp L294-297): `opt_scaled = opt * node * bm * eval`, capped by max only:
  - node: `(1.53 - bestMoveNodeFraction) * 1.74` (tune.h L126-127); no floor, no ceiling.
    frac 0.3 -> 2.14; 0.6 -> 1.62; 0.9 -> 1.10; 0.95 -> 1.01.
  - bm (best-move stability 0..4): {2.38, 1.29, 1.07, 0.91, 0.71} (tune.h L114-118).
  - eval stability 0..4: {1.25, 1.15, 1.03, 0.92, 0.87} (L121-125); counter increments when the
    score is within +/-10 of a RUNNING AVERAGE `avg = (avg + score)/2` (search.cpp L272,
    L285-291), resets otherwise -- symmetric, not direction-only.
- Stops: soft checked only at iteration end (`StopEarly`, L40-42; search.cpp L300); hard every
  1024 nodes (`TimeOver`, L64-69). No prediction of the next iteration. On abort Negamax returns 0
  (search.cpp L423-430) and the best move is whatever the PV table holds -- a partial iteration's
  move IS taken if it raised alpha at the root (L880-890). Same policy as our TIME_V4.

### Ours (agent.py)
- `_budget_v6` L3395-3423: `remaining = clock - 0.4 s`; `expected = max(30, 56 - 0.4*fullmove)`;
  `soft = remaining/expected + 0.7*inc_observed`; `hard = min(0.10*remaining, 2.5*soft)`; reserve
  6% of the peak clock; below 12 s `soft = remaining/18, hard = soft`.
- Stop rule in `choose` L3022-3042 (depth >= 5): `factor = STABILITY[min(s,4)] (1.2,1.1,1.0,0.9,0.8)
  x 2^(drop/100) (drop = score[-4] - score[-1], clamped +/-100) x max(0.5, 2.0 - 1.6*frac)`,
  **clamped [0.4, 1.5]**; stop when `elapsed > factor * soft`. Kernel hard poll every 256 nodes
  (fastsearch.py L71, L942).

### Numbers at 120 s + 0.5 s (their opt vs our soft, then their ceiling vs ours)
| clock left | Alex opt | ours soft | Alex ceiling (unstable) | ours ceiling |
|---|---|---|---|---|
| 120 s (move 1) | 3.6 s | 2.5 s | ~91 s hard; realistically opt x 2.38 x 2.14 x 1.25 = 23 s | 1.5 x 2.5 = 3.75 s (hard 6.25) |
| 90 s (move ~20) | 2.8 s | 2.2 s | 68 s hard, ~18 s realistic | 3.3 s (hard 6.2) |
| 60 s (move ~40) | 2.1 s | 1.85 s | 45 s / ~13 s | 2.8 s (hard 4.6) |
| 30 s | 1.33 s | 1.35 s | 22 s / ~8 s | 2.0 s (hard 3.0) |
| 12 s | 0.88 s | 0.67 s (hard = soft) | 9 s / ~5 s | 0.67 s |
| 3 s | 0.6 s | 0.17 s | 2.3 s | 0.17 s |

Optimum: within 30% everywhere -- NOT the gap. Alexandria credits ~1.2 increments per move
(0.025 x 49 x 0.5) against our 0.7; its "low clock" regime only starts when `0.2*t/timeLeft <
0.025`, i.e. ~3.3 s, against our 12 s. Both are second-order.

Ceiling: the gap is 3-6x on the moves that matter. HORIZON.md (33 horizon moves, 13 games): median
clock 62 s, median move 45, |eval| 122 cp, the engine already spends 2x its median there, and the
fix needed 1.57x the soft budget -- just past our 1.5 clamp. NOTES also records TIME_V6 as "tamed",
recovering ~23% of the banked clock, and the first untamed cut (4 soft budgets, 9 s low clock, 4%
reserve) drained to 1.6 s under the 1.5x clocktest charge. So the ceiling must be raised ONLY where
the clock is healthy, and the low-clock regime left exactly as is.

### Proposal: TIME_V7 (switch in agent.py; `_budget_v6` + the stop rule; no kernel change)
1. Above 30 s remaining: `hard = min(0.20*remaining, 4.0*soft)` (still 5x tighter than
   Alexandria's 0.76); 12-30 s: unchanged `min(0.10, 2.5 soft)`; < 12 s: unchanged.
2. Stability table Alexandria-shaped but damped: {2.0, 1.25, 1.05, 0.9, 0.75}; node factor
   `(1.53 - frac) * 1.74` floored at 0.5; keep our score-drop term; add their symmetric
   eval-instability counter as a x1.25 -> x0.87 table (needs the running average, 5 lines).
3. Product clamp [0.4, 3.0] above 30 s remaining, [0.4, 1.5] below (today's behaviour).
4. Keep never-predict (already so). Keep TIME_V4's partial-iteration rule.
Expected: +5..15 at 120 s -- the HORIZON bucket is 35% of lost value; funding every horizon move
costs 2.5 s/game against 17 s median unused. Risk: flags (the reason TIME_V6 was tamed). Mitigation
is the 30 s gate plus the 20%-of-clock hard cap: worst case one 24 s move at 120 s, and the
existing 6% reserve still applies to hard.
What can be measured: NOT 40 games at 120 s (+/-70 Elo). Use HORIZON.md's pre-registered
falsifier: re-run `testing/postmortem.py` over the 29 failed games with the switch on and require
the horizon bucket (13,231 cp in pm2-v95) to shrink by more than the 12% noise floor while the
`time` bucket (8,059 cp) does not grow; then clocktest at 1.5x (0 flags, lowest clock > 3 s); then
40 x 120 s as a non-negative sanity gate only. Note the 8 s gauntlet plays its ENTIRE game below
LOW_CLOCK_V6 (12 s), so it cannot see any of this -- and a cheaper 30 s + 0.3 s SPRT
(precedent: `036-lmr.30s`) would at least exercise the factor path, at ~4x the cost of 8 s games.

## 2. ITERATIVE DEEPENING + ASPIRATION

Alexandria `AspirationWindowSearch` (search.cpp L326-388): delta 12 from depth >= 3, window
around `averageScore` (running average, L272), fail-low: `beta = (alpha+beta)/2`, `alpha = score -
delta`, depth reset to the root depth; fail-high: `beta = score + delta`, **`depth = max(depth-1,
1)`**; `delta *= 1.44` per fail; never jumps to infinity.
Ours (`choose` L2858-2999): window 15 from depth 4 around the LAST score; ASP_WIDE widens
x1.5^fails, full width after 10 fails; fail-high keeps depth; no beta shrink on fail-low; TIME_V4
takes the fail-high move immediately (L2971-2973, correct).
Gaps, all filler, all root-loop (bench-invisible by construction, NOTES standing caveat 6):
- **ASP_V2** (one switch): (a) fail-high re-search at depth-1 (their L380) -- halves the cost of a
  root fail-high pass at our branching ~2.1x/ply; (b) fail-low pulls beta to the midpoint (L372);
  (c) centre the window on the running average. +2..5 combined; 0.5-1 h; risk: none the 8 s
  gauntlet would not show. Interaction: the depth-1 pass still counts as `depth` for the stop
  rule and stability -- keep it that way (Alexandria does).
- Their `depth >= 3` vs our 4 and 12 vs 15: noise.

## 3. TRANSPOSITION TABLE

| | Alexandria (ttable.h L7-26, ttable.cpp L77-124) | Ours (fastsearch.py L423-486, probe L992-1018, store L1600-1635) |
|---|---|---|
| entry | 10 B: move16, score16, eval16, key16, depth8, age5+pv1+bound2 | 16 B over two arrays: full key64 + data64 (score16, eval16, move16, flag2, depth8, age6) |
| bucket | 3 entries / 32 B line, mul-high index | 2 slots (even = deep, odd = always-replace), mask index, 2^22 entries = 64 MB |
| slot choice | victim = lowest `depth - 4*age_distance` among 3 (L106-108) | same key, else deep slot if aged or `depth >= old`, else odd slot |
| replace rule | exact bound, OR new key, OR `depth + 5 + 2*pv > old`, OR old age (L114-118) | deep slot only if `depth >= old depth` |
| move | keep the old move when the new is null (L111) | store best_move even on upper bounds (better) |
| cutoff | non-PV only, `cutNode == (ttScore >= beta)`, `50mr < 90` (L470-482) | any node incl. PV, `stored_depth >= depth`, no hmc guard |
| static eval | stored raw; on hit skips evaluate AND uses ttScore as eval when the bound agrees | stored; on hit skips evaluate (cached_eval L1006). The ttScore-as-eval half is TT_EVAL, rejected |
| ttPv | bit stored, propagates into singular margin and LMR | none |
| QS | probe + store depth 0 | QS_TT on, same idea |

Material differences and what they are worth:
- **Replacement leniency.** Alexandria overwrites a deeper entry when the new one is at most 4
  shallower (6 if PV). Ours requires >= old depth; TT_KEEP (stricter still) lost -32. The
  measured direction says lenient wins. **TT_LENIENT**: deep slot if `depth + 4 >= old depth`.
  +0..3, 0.5 h, bench nodes at d8 (should drop slightly; may show nothing at fixed depth), filler.
  Do not bundle with anything that changes node counts you want to attribute.
- **ttPv bit (TT_PV).** Steal one age bit (5 bits, MAX_AGE 32 as theirs) for a "was PV" flag set
  when `beta - alpha > 1` at store or inherited from the hit. Uses: singular margin
  `tt_score - depth*5/8 - depth*(ttPv and not pv)` (their L742), LMR -1 for quiets at ttPv nodes
  (L808-809), and no LMR at all for captures at ttPv nodes (L785). +2..6 at our depth (their
  LMR is tuned around it; ours is not, so the first cut may be negative -- bench it). 2-3 h incl.
  check_fastsearch with the flag off; risk: the age wrap at 32 searches is harmless (age only
  breaks ties).
- **TT_HMC90**: skip the cutoff when `meta[HALFMOVE] >= 90` (their L474). Five lines. Matters
  more for us than for them because ADJUDICATION lowers C_HMC_DRAW to `hmc + 16` near the cap, so
  a stored score from an hmc-20 transposition can mask a draw-scored line. +0..1, zero risk, filler.
- **TT_INTERLEAVE** (speed, exact): one uint64 array of 2*N with key and data adjacent, so a probe
  touches one cache line instead of two arrays (the docstring at L424 already knows it is two).
  At 64 MB the table is miss-bound; expect +2..4% knps -> +1..3 Elo. 1-2 h; every TT site
  (search probe/store, quiesce, qs_tt_store, predicted_reply, probcut store) changes index
  arithmetic; nodes must be IDENTICAL on the bench, which is the exactness proof. No slot.
- TT cutoff at PV nodes: ours cuts, theirs does not. Cutting truncates the PV and can return a
  stale exact score at the root's PV; at 250 knps the node saving is worth more. Leave it.
- The `cutNode == (ttScore >= beta)` gate needs CUTNODE (0.995x nodes, in `200-v97-prune`). Not
  worth adding until CUTNODE has a verdict.
- Not gaps: age per search (ours `self.age += 1` in `play` L3079), mate-distance to/from table,
  prefetch (n/a), table size (64 MB ours, theirs default 16 MB).

## 4. ROOT HANDLING

Alexandria: the root is an ordinary Negamax node. Ordering = movepicker with history + 4 x
rootHistory (history.cpp L191-194); rootHistory gets bonus/malus at root cutoffs (L96-120,
tune.h L173-178), zeroed per search (search.cpp L267). Root moves receive full LMR (L785 has no
root guard; only the pruning block at L693 is `!rootNode`). Per-root-move node counts
(`nodeSpentTable`, L868-869) feed the TM only, never ordering. Best move on abort: PV table.
Ours: ROOT_ORDER by previous scores (L2870-2896), ROOT_NODES off (node-count ordering -- a
Stockfish idea, not Alexandria's; bundle filler as recorded), ROOT_LMR off (r = 1 from the 4th
move, 2 from the 10th, agent.py L171-173), no root history, TIME_V4 partial-iteration rule.
- Root history: the root is ONE node searched ~12 times a move; its ordering is already by exact
  scores from the previous iteration, which is strictly more information than a history bonus.
  +0..2. Skip.
- Root LMR: theirs works because the reduction is the fully-featured LMR (history, ttPv,
  cutnode, do-deeper). Our ROOT_LMR is a fixed 1/2-ply scheme with no history term; it was built
  and left off. Not before the freeze; if anything, let the kernel's own LMR see ply 0 (one guard
  change) rather than the root-loop version. +0..4, risky at our depth (root moves 4-10 reduced
  by 2 at depth 10 = the depth-8 blunder class). Skip.
- Best move on abort: equivalent. Nothing to do.

## 5. EXTENSIONS / DEPTH

Singular (their L732-769 vs ours L1291-1325):
| | Alexandria | ours |
|---|---|---|
| entry | depth >= 6, ttMove, lower bound, ttDepth >= depth-3, extension budget `ply*2 < rootDepth*5` | depth >= 7, exact-or-lower, ttDepth >= depth-3, `exts[ply] < 6` |
| margin | `ttScore - depth*5/8 - depth*(ttPv && !pv)` | `tt_score - 2*depth` (much wider = stricter = fewer extensions) |
| verify depth | (depth-1)/2 | same |
| results | ext 1; non-PV: ext 2 if < sBeta-10, ext 3 if < sBeta-75, AND `depth += 1` if depth < 10; **multi-cut**: `singularScore >= beta -> return it`; ext -2 if ttScore >= beta; ext -2 if cutNode | ext 1; SINGULAR_EXT2 (off): ext 2 if < sBeta-25 at non-PV, -1 if ttScore >= beta |
- **SINGULAR_MULTICUT** (missing, and the cheapest real mechanism in this report): the excluded
  search just proved a SECOND move holds `sbeta - 1 >= beta` when tt_score is well above beta --
  return `value` without searching the node. Two lines inside the existing block, bench-visible
  (nodes drop at d8), +2..6 at 8 s. Guard `abs(value) < DISTANCE_THRESHOLD`. Risk: the verify
  search is half depth, so it is a speculative cut -- exactly what every reference engine accepts.
- Margin: ours -2d is 3x theirs; at depth 8 ours needs alternatives 16 cp below, theirs 5. Fewer
  extensions than theirs; whether that is right for a 250 knps engine is a bench question
  (nodes at d8 rise with a narrower margin). Try `-depth` as SINGULAR_MARGIN filler, +0..4.
- Their `depth += 1` on a strong double extension (L755) has no analogue; skip (depth explosion
  risk at our EXT_CAP).
- Extension budget: theirs by ply (`ply < 2.5 * rootDepth`), ours by count per line (6).
  Equivalent in effect. Nothing to do.

Check extension: Alexandria has NONE (only "reduce less if the move gives check", L804). Ours:
uncapped +1 (L1038-1044); the CAP was rejected. Removal is a different experiment from a cap and
untested, but every 3500 engine dropped it because LMR+singular cover it; at our depth 10-12
tactics dominate and the risk is real. Not before the freeze.

IIR: theirs (L485, L664) triggers on `ttBound == HFNONE` at depth >= 4 (no entry, or eval-only
entry) and ALSO lowers the RFP margin by 76 and the NMP depth by 1 on such nodes. Ours (rejected)
triggered on `hash_move == 0` at depth >= 4 -- functionally the same trigger. Closed; the RFP/NMP
side-terms are +0..2 and not worth reopening.

Hindsight (their L552, ours L1077-1083): identical formula, HINDSIGHT_EVAL 155. Bench
`bench-s2-hindsight-d8`: 1,348,345 nodes vs tree 1,400,201 = 0.963x (knps 226 vs 254, but the
S2 benches were run under load: s2-lean 276, s2-all 205, so read nodes only). It needs the static
eval at every non-check node; RFP (depth <= 6) and NMP_V2 (depth >= 3) already evaluate almost all
of them, so the true cost is small. +1..3, already built, rides in S3.

Improving without over-pruning: Alexandria's `improving` (L537-545) is the SAME signal as ours
(ss-2 / ss-4 static compare, default true when both in check). It is used in exactly two places:
LMR +1 when not improving (L796/L823) and RFP margin -61 when improving (L391); LMP (L703) does
not exist here. Ours fed three terms -- RFP depth-1, prune2 futility table index depth-1, LMR +1
-- and prune2's table steps by 100 cp per index, so `depth - improving` at depth 2 halves the
margin. That is the 0.506x. **IMPROVING_LMR**: keep only the LMR arm (Alexandria's exact use);
bench it; expect 0.85-0.95x nodes with stable scores. 0.5 h (one extra ctrl check on the two
prune arms). +2..6.
Other depth signals that do not need `improving`:
- **LMR_DEEPER** (their L843-850): after a reduced search beats alpha, the full-depth re-search
  runs at newDepth+1 if `score > bestScore + 77 + 2*newDepth`, at newDepth-1 if `score < bestScore
  + newDepth`. No new state; sits in our reduced-search branch (L1484-1500). +2..5 at 8 s, 1 h,
  bench-visible; risk: tiny (the -1 arm is the one that saves nodes; the +1 arm costs).
- `badNode` (no TT bound) as a "we know nothing here" flag: used only in the IIR family. Closed.
- `complexity` (L527-532) needs correction history (rejected twice). n/a.
- **OPP_HIST_EVALDIFF** (their L519-525, "~6 Elo"): at a non-check node whose parent had a static
  eval and played a quiet, give the PARENT's move a history bonus `clamp(-10*(parent_static +
  my_static), -1830, 1427) + 624` in the opponent band. Reads the IMPROVING lane's per-ply eval.
  Improves ordering only; +1..4, 1 h; bench shows it as fewer nodes at d8 if it works.

## 6. DRAW / ENDGAME IN SEARCH

- Repetition (their `IsRepetition` L21-43): in-tree twofold = draw; game-history positions need a
  true threefold. Ours (REPETITION_TWOFOLD, `rep_keys` + `fb.repeats` L1124-1137): any single
  earlier game occurrence = draw. Ours is one repetition stricter, deliberately (the referee's
  automatic claim), measured +66. Closed.
- Upcoming-repetition detection (`hasGameCycle`, L439-444, L967-972): if the side to move can
  FORCE a repetition and alpha < 0, alpha = 0. Stockfish's cuckoo tables; +4 at their scale. With
  twofold scoring already making these lines draws one ply later, and 2-3 h of numba for a cuckoo
  table, +0..3. Skip.
- 50-move (their L46-68 vs ours L950-957): identical incl. the checkmate-on-move-100 case. Their
  extra: no TT cutoff at hmc >= 90 (section 3, TT_HMC90).
- Draw score: theirs `(nodes & 2) - 1` (+/-1 jitter, no contempt); ours root-side contempt with
  material/late ramps (`_contempt` L3266-3296). Ours is richer and measured. BUT: ADJUDICATION
  (on) ramps toward a ply-300 material award while ADJ_V2 (off, agent.py L1535 comment) states the
  verified rule is 600 plies and a draw. If ADJ_V2's reading is right, the champion is paying up to
  +300 cp to reach a draw it would get anyway, from ply 225. Whoever owns the referee question
  should settle it from the platform's canonical rules text before any contempt work; it is a
  0-hour change with a possibly large sign either way, and 8 s games never reach the ramp.
- Eval damping `eval * (200 - hmc) / 200` and material scaling: eval.md E2/E3. E2 is the one
  search-visible piece (it is what stops a winning side shuffling into the fifty-move rule when
  the root-only DRAW_BUDGET/ADJUDICATION logic is silent); judged on testing/draws.py at 120 s.
- Insufficient material: neither engine scores it in search (their comment says so; the code
  does not). Root TB at 4 men covers ours. Equal.
- Tablebases in search: none in either (their search.cpp has no probe; ours root-only,
  fastsearch.py L16). Equal.
- Mate-range guards: DISTANCE_THRESHOLD vs isDecisive. Equal.

## What to do with the two slots
Slot A (8 s, ~1 h to the 200-game checkpoint): **S3** = HINDSIGHT + SINGULAR_MULTICUT + LMR_DEEPER
+ IMPROVING_LMR + ASP_V2 + TT_HMC90, after a d8 bench of each kernel arm alone (drop any arm above
1.0x nodes or with a score swing > 200 cp on the 40 positions; the S1/S2 benches show the rig can
do this in minutes). Add TT_PV only if it benches <= 1.0x with the same best moves.
Slot B (120 s, ~1 h incl. clocktest): **TIME_V7**, judged by the HORIZON postmortem replay first;
the 40 games are a sanity gate, not the verdict.
Fillers that need no slot: TT_INTERLEAVE (bench knps, exact), TT_LENIENT, OPP_HIST_EVALDIFF.
