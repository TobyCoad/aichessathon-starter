# v10 search research — what is left on the search side

*Written 5 Sep 2026 against the v8 tree (v8.5 bundle under test). Read-only study:
nothing was run, nothing was changed. Numbers marked **(ours)** are from this
project's own logs; numbers marked **(lit)** are from other engines and are
adjusted before use.*

## Summary (10 lines)

1. The search is already a competent modern search. What is missing is not a
   *list of features* — it is the **accuracy terms** that let a modern engine
   reduce and prune much harder than we dare to: continuation history, an
   improving flag, and a cut-node flag. Everything else is small.
2. Ranked best six: **continuation history**, **time-budget shape**, **improving
   + cutnode**, **null-move rework**, **SEE-ordered captures + capture history**,
   **kernel speed (see_ge + evaluate)**. Expected total at 120 s: **+45 to +95**,
   not the +150 the ladder gap needs.
3. Our own calibration: 1 ply ≈ 2.1x nodes; 1 doubling of speed ≈ **+65 Elo at
   8 s**, **+32 at 120 s**. Exact speed-ups realise that in full; lossy pruning
   realises **~27%** of it (LMR: predicted +175, measured +47).
4. Therefore: a change that only saves 8% of nodes is worth about **+2 Elo**.
   Several of the classic candidates (staged movegen, multi-cut, IID, mate
   tricks, more TT replacement work) are in that band. **They will not reach +20.
   Do not spend a day on them.**
5. Our pruning constants are *not* badly mis-scaled — the net's output really is
   in Stockfish centipawns (trained against `sigmoid(cp/400)`). The one clear
   mis-tune is the **null-move reduction, R = 2 + d/6, one to three plies
   shallower than the whole SF family**.
6. LMR_AGGRESSIVE only reached 0.92x nodes because a steeper *table* without
   accuracy terms is self-limiting. Our base table is already steeper than
   Stockfish's base table. The missing plies live in the ±2-ply adjustments.
7. Two genuine defects found by reading: `quiets[ply, searched]` is **stale**
   whenever a non-quiet move precedes a cutoff, so HISTORY2 hands malus to moves
   from a sibling subtree; and `fastboard.see` heap-allocates `np.zeros(32)` on
   every call in the hottest loop in the engine.
8. Post-mortem evidence says **time is a top-2 cause of our losses** (round 22:
   4 of 5 mistakes were time-caused, 10 time-trouble moves, 12 s of reserve never
   spent). Time-budget shape is the highest-value non-Elo-table item here.
9. The rig cannot see anything below about **+12 Elo at 8 s**. Small exact
   changes must be judged by bench nodes/knps + theory and shipped in a bundle,
   exactly as v8 was; single-switch SPRTs on them waste desktop days.
10. Hard constraint on all of it: **import time**. 31 s locally, ~56 s on their
    1.8x box against a 90 s budget. Every new branch in the `search` kernel costs
    numba compile time. Re-measure import on every bundle, not at the end.

---

## 0. How to read the Elo estimates

All estimates below are derived from this project's own measurements, not from
fishtest transplants. The chain:

* **Branching factor.** Bench: 1,605,437 nodes to depth 8 over 40 positions =
  ~40k per position. Mean depth ~12 at 4 s and 195 knps = ~780k nodes.
  780k / 40k = 19.5x over four plies → **EBF ≈ 2.10**, i.e. one ply ≈ **1.07
  doublings** of time. (ours)
* **Elo per doubling.** COMPILED_SEARCH was measured at 1.7–2.3x knps (call it
  one doubling) and scored **+67 ± 40** at 8 s. So **≈ 65 Elo per doubling at
  8 s**. NOTES.md: 120 s gains run about half of 8 s gains → **≈ 32 Elo per
  doubling at 120 s**. (ours)
* **Exact changes** (same tree, faster nodes): realise the full model.
  `Elo(8 s) = 65·log2(speed-up)`, `Elo(120 s) = 32·log2(speed-up)`.
* **Lossy changes** (pruning, reductions): the node model over-predicts badly,
  because each saved node also costs accuracy. LMR cut nodes to 0.14x at fixed
  depth → model says +175; measured **+47**. QS SEE pruning: measured **+25**.
  So use **realised ≈ 27% of the node model** for a reduction/pruning change.
* **Ordering changes** (same semantics, better move order): between the two,
  because a better order does not lose accuracy. Use **~55%**.
* **Rig resolution.** From our own logs, 200–500 games gives ±25–40 Elo.
  SPRT[0,20] resolves a true +20 in roughly 600–1300 games. **Anything under
  ~+12 Elo at 8 s comes back INCONCLUSIVE**, as 090-history2 (+12 ± 31),
  099-ttbuckets and 103-lazyacc all will.

A useful sanity figure: to move from ~1854 to a top-5 seed (~2060–2147) needs
**+150 to +250**. The entire search list below is worth **+45 to +95** at 120 s.
The rest has to come from the net, from the clock, and from not losing games to
avoidable causes. Say this out loud rather than hoping the search list closes it.

---

## 1. Ranked table

Elo columns are **at 120 s**, the control that actually pays. Multiply by ~2 for
the 8 s SPRT figure the gauntlet will report.

| # | Idea | Elo @120 s | Cost (h) | Risk | Confidence | Why |
|---|---|---|---|---|---|---|
| 1 | **Continuation history** (1-ply piece-to), used in ordering + LMR + quiet pruning | **+12 … +25** | 4–6 | med | med-high | Replaces a single counter-*move* with a full table; it is the accuracy term that makes deep LMR safe |
| 2 | **Time-budget shape** (credit the real increment, cut the 10% reserve, scale soft budget by stability/effort) | **+10 … +25** | 3–5 | med (clock) | med | Post-mortem: 4 of 5 mistakes in round 22 were time-caused; 12 s of reserve is banked and never used |
| 3 | **Improving flag + cutnode flag**, gating RFP / futility / LMR / NMP | **+8 … +18** | 3–5 | low-med | med | The two cheapest accuracy bits in the SF family; both are prerequisites for #4 and for harder LMR |
| 4 | **Null-move rework**: R = 3 + d/4 + eval-margin, skip on a TT fail-low, verify at high depth | **+6 … +15** | 2–3 | med (zugzwang) | med | Our R = 2 + d/6 is 1–3 plies shallower than every reference engine; this is the clearest mis-tune we have |
| 5 | **SEE-ordered captures + capture history** (losing captures ranked *below* quiets) | **+6 … +14** | 2–4 | low-med | med | At depth > 5 a queen-takes-defended-pawn is currently searched before both killers |
| 6 | **Kernel speed**: allocation-free `see_ge`, `evaluate` blocking | **+4 … +10** | 3–6 | low (exact) | med-high | `see` calls `np.zeros(32)` per capture; target 195 → 215–230 knps |
| 7 | **TT probe (and store) in quiescence** | **+4 … +10** | 2–3 | med | low-med | QS is most of our nodes and currently has no table at all |
| 8 | **ProbCut** at depth ≥ 5 | **+3 … +8** | 4–6 | med | low-med | Works best on a true-cp eval, which ours is; but our EBF is already low |
| 9 | **Fix `quiets[]` staleness** in HISTORY2 | **+0 … +6** | 0.5 | low | med | Real defect, near-free fix (§3.7) |
| 10 | **Singular double / negative extensions** (after SINGULAR lands) | **+3 … +8** | 1–2 | low | low-med | Only meaningful if 111-singular passes |
| 11 | **Razoring** at depth ≤ 3 | **+0 … +5** | 1 | low | low | SF has trimmed it repeatedly; our RFP already covers most of it |
| 12 | **Aspiration widening** 4^n → 1.5x, no full window before 4 fails | **+0 … +5** | 0.5 | low | low-med | Cheap; the current schedule jumps straight to ±INFINITY |
| 13 | **LMP redone** with improving + a history exception | **+0 … +6** | 1 | med | low | Plain LMP measured **−40** (ours, 053). Only worth retrying *after* #3 |
| 14 | **Null-move verification** alone | **+0 … +4** | 1 | low | low | Targets our endgame weak spot, but the effect is small everywhere it is measured |
| 15 | **History-based LMR made continuous** (r −= hist/6000, ±2) instead of ±1 at ±8000 | **+0 … +4** | 0.5 | low | med | Folds into #1; do not test alone |
| 16 | **Root PVS / root LMR** | **+0 … +3** | 1 | low | med | Aspiration already narrows the root window to ±15 |
| 17 | **Killer clearing** between game moves | **+0 … +3** | 0.3 | low | low | Killers currently persist for the whole game |
| 18 | **Staged move generation** | **+0 … +2** | 3 | low | **high** | `gen_legal` is 0.19 µs of a 5.13 µs node = **3.7%**. Ceiling ≈ +2 Elo. Already REJECTED twice (029, 033). **Closed.** |
| 19 | **Multi-cut** | **+0 … +2** | 3 | med | med | Abandoned by the whole modern field in favour of NMP + ProbCut. **Do not build.** |
| 20 | **IID** (as opposed to IIR) | **+0 … +2** | 2 | low | med | IIR was already INCONCLUSIVE (066). IID is the older, worse form. **Closed.** |
| 21 | **More TT replacement work** (deeper ageing, 4-slot buckets) | **+0 … +2** | 2 | low | med | TT_KEEP −32, TT_BUCKETS node-neutral at d8. This vein is mined out. **Closed.** |
| 22 | **Checks in quiescence / QS evasions** | **−15 … +2** | 2 | high | med | QS_EVASIONS measured **−18 ± 27** (028). Generating checks costs more. **Closed.** |
| 23 | Mate-distance / cutoff tricks | **0** | — | — | high | Already shipped inside SAFE_BITS |
| 24 | Correction history | **0** | — | — | high | Rejected twice. Closed by standing order. |

**Honest lines to keep saying:** items 18–24 will not reach +20 Elo and several
have already been measured at or below zero here. Items 9, 11, 12, 15, 16, 17 are
each worth a few Elo, are individually invisible to an 8 s SPRT, and should be
folded into a bundle rather than gauntleted one at a time.

---

## 2. What we already have, so nobody rebuilds it

Present and on in v8: TT (2^22, packed, 2-slot buckets, ageing, static eval
cached in the entry), iterative deepening, aspiration (±15, from depth 4), LMR
(log-log, from the 3rd quiet), null move (R = 2 + d/6), RFP (80/ply to depth 6),
futility (d ≤ 2), quiescence (cap 14, delta + big-delta + SEE), check extensions,
MVV-LVA + killers + counter-move + side-indexed history with gravity and malus,
SEE pruning of bad captures at d ≤ 5, twofold repetition, mate-distance pruning,
root ordering by previous scores.

Built and off, under test as v8.5: LMR_AGGRESSIVE (+PVS), LAZY_ACC, TIME_V5,
PRUNE_V2 (quiet futility to d4 + history pruning), SINGULAR.

**Genuinely absent:** continuation history, capture history, improving, cutnode,
ProbCut, razoring, TT in quiescence, null-move verification, eval-scaled null
reduction, SEE-based capture *ordering* (we only SEE-*prune*).

---

## 3. Scoping the top six

Ground rules that apply to all of them, taken from `overnight/continuous/NOTES.md`:

* Every change is a **switch in `agent.py`, `False`/`0` by default in the tree**,
  mirrored into a `ctrl` slot in `FastEngine.prepare` (agent.py ~1890–1921).
* `ctrl` is `int64[CTRL_SIZE=40]`; slots **0–34 are taken**, so **35, 36, 37, 38,
  39 are free**. More than five new slots means raising `CTRL_SIZE` (a one-line
  change in `fastsearch.py`, plus the `np.zeros(fs.CTRL_SIZE)` in
  `testing/check_fastsearch.py:105` and `fastsearch.warm_up:937`).
* `testing/check_fastsearch.Kernel.__init__` builds `ctrl` as **all zeros** and
  then sets only `C_TT_OFF`, `C_HYGIENE`, `C_FUTILITY`, `C_ROOT_SIDE`,
  `C_QS_CAP=8`. So **any switch whose "off" value is 0 automatically keeps the
  flags-off kernel bit-identical to the Python reference** — which is the gate
  (`--depth 4 --random 30`, 70/70 exact). A switch with a non-zero off value
  (like `C_QS_CAP`) does not, so do not create one.
* Before commit: `ruff check`, `mypy agent.py fastsearch.py`,
  `python -m testing.check_fastsearch --depth 4 --random 30`.
* One switch per challenger, judged by **SPRT[0,20] at 8 s vs the champion**
  (`testing/gauntlet.py`, defaults 800 games / 8000 ms / 80 ms increment).
  Anything touching the clock also needs `testing.clocktest` (120 s + 0.5 s,
  ×1.5 charge, floor 5 s) and 40 games at 120 s with `openings: platform`.
  Anything touching evaluation accuracy in the endgame also needs
  `testing/endgame_suite.py` (400 positions, 2.5 s/move, ~18 min per run;
  champion baseline **7.4 cp** mean loss: 6.0 at 5–8 pieces, 9.9 at 9–12, 6.1 at
  13–16).

### 3.1 CONT_HIST — continuation history (rank 1)

**What.** Today the only "what followed what" knowledge is `counter`, an
`int32[4096]` holding **one move** per previous `(from, to)` pair
(fastsearch.py:701, scored at `COUNTER_MOVE = (1<<20)-3` in
`fastboard.score_moves`). Replace it with a real table:

```
conthist[(prev_piece * 64 + prev_to) * 768 + (piece * 64 + to)]   int32, 589,824 entries = 2.4 MB
```

Used in three places, which is where the Elo is — not in ordering alone:

1. **Ordering.** In `fb.score_moves`, a quiet's score becomes
   `butterfly[...] + conthist1[...]` (and `+ conthist2[...]` for the 2-ply
   table). Keep the killer/counter bonuses above it.
2. **LMR.** In `fastsearch.search` at the reduction block (lines 746–765),
   replace the coarse `hist > 8000 → −1 / hist < −8000 → +1` with a continuous
   `reduction -= (butterfly + conthist1 + conthist2) // 6000`, clamped to ±2.
3. **Quiet pruning.** In the `prune2` block (lines 728–733), test
   `butterfly + conthist1` against `-HIST_PRUNE_SLOPE * depth` instead of
   `butterfly` alone.

Updated on a cutoff alongside `butterfly` (lines 847–862), same gravity formula,
same bonus `min(depth*depth, 1200)`, same malus loop over `quiets[ply, :searched]`.

**Where the indices come from — no board change needed.** Board ply and search
ply coincide (`Position.load` sets `meta[PLY] = 0`, the root loop makes exactly
one move before `root_search(..., ply=1)`, and `make_null` advances both), so:

* 1-ply: `prev = undo[meta[PLY]-1, U_MOVE]`; `prev_to = (prev>>6)&63`; the piece
  that landed there is **`sq[prev_to]`**, still on the board at node entry —
  free, no undo field, no new argument.
* 2-ply: `sq[prev2_to]` is *not* reliable (the 1-ply move may have captured or
  moved it). Keep a per-ply stack instead (below), written just before each
  recursive call.

**Where to put the per-ply stack.** Do **not** add a new array argument (that
changes the signature in `fastsearch.search`, `fastsearch.quiesce`, `warm_up`,
`agent.FastEngine.root_search` and `testing/check_fastsearch.Kernel.search` —
five sites, and the recursion passes 30+ arguments already). Instead **widen the
existing `exts` array**: allocate `np.zeros(4 * fb.MAX_PLY, np.int64)` at its
five allocation sites (`agent.py:1521`, `check_fastsearch.py:103`,
`fastsearch.py:936`) and index

```
exts[ply]                  # unchanged: the extension count (bit-identical when flags are off)
exts[MAX_PLY   + ply]      # static eval at this ply   (for IMPROVING, §3.3)
exts[2*MAX_PLY + ply]      # piece*64 + to of the move made at this ply (conthist)
exts[3*MAX_PLY + ply]      # spare
```

Nothing reads the new region with the switch off, so `check_fastsearch` stays
exact. This is deliberately the lowest-churn route.

**Switch.** `CONT_HIST: Final = False` in agent.py near `HISTORY2` (~line 936),
`ctrl[_fs.C_CONT_HIST] = 1 if CONT_HIST else 0` in `prepare`, `C_CONT_HIST = 35`.
Arrays: `self.conthist1`, `self.conthist2` allocated in `FastEngine.__init__`
(~line 1524) and added to `__slots__` (~line 1500). They **do** need to be new
kernel arguments; that is unavoidable, and is the one signature change.
Do 1-ply first as its own switch; add `CONT_HIST2` (2-ply) as a second switch on
top only if 1-ply passes.

**Parameters.** Bound ±16384 as now; gravity identical; divisor 6000 for the LMR
term (a quarter-bound gives 2 plies at saturation — tune 4000/6000/8000 by bench
nodes at depth 8/10 before spending a gauntlet).

**Pitfalls.**
* `fb.score_moves` is shared with `agent.FastEngine.search` via `fb.order_moves`
  — do **not** change `order_moves`; add the conthist arguments to `score_moves`
  with default `0` (it already takes `counter=0, base=0` defaults) so the Python
  reference path is untouched and `check_fastsearch` stays exact.
* Under **LAZY_ACC**, `undo[ply, U_MOVER]` is written only on the light path;
  do not read it. Use `sq[prev_to]` / the `exts` stack as above.
* The **null move** writes `undo[ply, U_MOVE] = 0`; treat `prev == 0` as "no
  continuation" and skip both the read and the update, as SF does.
* **HYGIENE** halves `butterfly` every move (`self.butterfly >>= 1`,
  agent.py in `choose`). Halve the conthist tables the same way or they will
  saturate and outrank the killers within a few moves.
* 2.4 MB per table is fine against 2 GB, but it is **cache-hostile**: expect
  −2…−4% knps. That is priced into the estimate.
* `SINGULAR`'s excluded-move search re-enters the same node at the same `ply`;
  make sure the conthist *update* is skipped when `excluded != 0` (the store
  guard at line 875 already has the pattern).

**Test plan.** Bench `--depth 8` and `--depth 10` first: target ≤ 0.90x nodes
(if it is not below 0.95x, the ordering change is not working — debug before
spending games). Then `112-conthist` on the laptop, SPRT[0,20] at 8 s, 800 games.
Expect **+20 … +45 at 8 s**; a result below +12 will read INCONCLUSIVE, so treat
"positive point estimate + node win" as a pass for bundling, as LAZY_ACC is being
treated. Endgame suite is worth one run (ordering changes move endgame accuracy).

**Honest risk.** This is the single most likely item on the list to be worth
+20 at 120 s, and also the one most likely to come back INCONCLUSIVE because the
rig cannot resolve it. Budget the bench evidence accordingly.

### 3.2 TIME_V6 — the shape of the budget (rank 2)

**Why it is this high.** `overnight/postmortem/REPORT.md`, round 22 (a 133-ply
loss): 5 flagged mistakes, **4 of them cause = "time"**, 10 time-trouble moves,
lowest clock 13.4 s. The engine at 5 s finds the right move at move 126 and plays
the losing one because it had 0.4 s. V7_PLAN adds: the top bots spend **less**
time per move (p50 1.3–1.8 s vs our 2.3 s) and play games **25–40 plies longer**.
We are front-loading a clock into a phase of the game where our eval is at its
strongest and starving the phase where it is weakest.

**Where.** `agent._budget_v2` (agent.py:2291) and the stop rule in
`FastEngine.choose` (agent.py ~2030–2055). The kernel is untouched, so
`check_fastsearch` is not at risk at all — this is a pure-Python change.

**Three separable defects, in order of size:**

1. **The increment is only half-credited, as a constant.**
   `soft = remaining / expected + 0.25` with a 0.5 s increment. Over a 66-move
   game the increment is worth 33 s — a fifth of the total clock — and the
   formula treats it as 0.25 s of pocket money. Credit `0.8 * inc` (i.e. 0.40)
   or, better, `soft = (remaining - reserve) / expected + 0.85 * inc`.
2. **`RESERVE_FRACTION = 0.10` banks 12 s that is never spent.** Once the clock
   is inside the reserve the budget collapses to `remaining / 30` ≈ 0.5 s and
   parks there in a stable equilibrium (spend ≈ income) for the rest of the game
   — exactly the round-22 tail. 0.05–0.06 releases ~6 s into moves 60–110 while
   still leaving a full second per move of headroom at the 1.5x charge.
3. **The soft budget itself never reacts to the position.** TIME_V3/V5 only
   adjust the *iteration allowance* (1.0 / 1.5 / 2.5 soft budgets). The
   literature's gain is in scaling the **budget**: a factor in roughly
   [0.6, 1.6] driven by (a) best-move stability across the last 3 iterations and
   (b) "effort", the fraction of this move's nodes spent under the best root
   move. (a) is already computed (`stable_streak`, `unstable`). (b) needs
   per-root-move node counts, which are free: `ctrl[C_NODES]` is read after every
   `root_search` call, so `nodes_after - nodes_before` per root move is a
   subtraction in the existing loop.

**Switch.** `TIME_V6: Final = False` beside `TIME_V5` (~line 862), with
`RESERVE_FRACTION_V6`, `INC_CREDIT` and the scale bounds as named constants. Keep
it strictly disjoint from TIME_V5 so the two can be attributed; if TIME_V5 passes
in the v8.5 bundle, build TIME_V6 on top of it and say so.

**Pitfalls.**
* **Clock safety is the number-one risk in this project** (REVIEW_2026-09-01:
  the pre-TIME_V2 champion bottomed at 1.2 s under a 1.5x charge). Every variant
  must clear `testing.clocktest --factor 1.5` with lowest clock ≥ 5 s **before**
  any gauntlet.
* **8 s games cannot see this.** Below `LOW_CLOCK = 15.0` the budget is
  `remaining / 30` and the new terms never bind, so an 8 s SPRT is byte-identical
  play. Do not queue an 8 s task for it — that is what happened to 102-timev5.
* The **endgame suite is fixed-movetime** and also cannot see a budget change.
  Waive it, as 102-timev5 did.
* `_MAX_CLOCK_MS` is inferred from the largest `time_left_ms` seen. On the
  platform the first request already carries 120 000, so the reserve is right
  from move 1; do not "fix" that.

**Test plan.** (1) An **offline schedule model** first, as loop iteration 4 did
for TIME_V5 — replay the clock arithmetic over a 130-ply game at 120 s + 0.5 s
and print seconds-per-move at moves 20/40/60/80/100/120 for each variant. This
costs nothing and kills bad variants before they touch the desktop. (2)
`clocktest` ×1.5, 6 games, floor ≥ 5 s. (3) 40 games at 120 s,
`openings: platform`, `elo0 -50 / elo1 50` — the same non-closing but strong
evidence format that carried v8 (67.5% over 40).

**Honest risk.** 40 games at 120 s has a ±90 Elo interval. This item will *never*
get a clean verdict in the time left. Decide it on the schedule model plus the
clocktest floor plus the post-mortem, and accept that.

### 3.3 IMPROVING + CUTNODE (rank 3)

**What.** Two one-bit facts every strong engine carries down the tree and we do
not.

*Improving*: `static_eval(ply) > static_eval(ply-2)`. When the position is
improving, prune less; when it is not, prune more. Standard uses: RFP margin
`80*(depth - improving)`, futility margin scaled, LMP count `(3+d*d)/(2-improving)`,
LMR `reduction += 1` when not improving.

*Cutnode*: whether this node is *expected* to fail high. It is not derivable
inside the node — it has to be passed down. Rule (SF's): the child of a
null-window search is a cut node iff its parent was not; the null-move child is
always a cut node; the first child of a PV node is a PV node. Standard use:
`reduction += 1..2` at cut nodes, and IIR at cut nodes.

**Where.**
* Static eval stack: `exts[MAX_PLY + ply]` from §3.1. Populate it right after the
  TT probe, at the point where `standing` is computed (fastsearch.py:593–629).
  Today `standing` is only computed when `depth <= RFP_MAX_DEPTH` (6). Extend to
  every non-check node that reaches the move loop — nodes with remaining depth
  ≥ 7 are **well under 1%** of the tree at EBF 2.1, so this costs ~1% more
  evaluations, and it is **after** the TT cutoff so LAZY_ACC's win is untouched.
  In check, write a sentinel (`-INFINITY`) and treat improving as False, as SF
  does.
* Cutnode: add **one `int64` positional parameter** to `fastsearch.search`.
  This changes the signature — update the recursive calls (10 sites), `warm_up`
  (line 941), `agent.FastEngine.root_search` (line 1852) and
  `testing/check_fastsearch.Kernel.search` (line 118). Pass `0` everywhere when
  the switch is off so behaviour is bit-identical.

**Switch.** `IMPROVING: Final = False`, `C_IMPROVING = 36`; `CUTNODE: Final =
False`, `C_CUTNODE = 37`. Test them **separately** — they are independent and
CUTNODE interacts with LMR_AGGRESSIVE.

**Parameters.** RFP: `standing - RFP_MARGIN * (depth - improving) >= beta`.
LMR: `reduction += 1` if not improving, `+= 1` at a cut node (`+= 2` at a cut
node with depth ≥ 8 is SF's, but that is aggressive for us — start at 1).
PRUNE_V2 futility: use `FUTILITY_MARGIN2[depth - improving]`.

**Pitfalls.**
* `standing` currently doubles as the "every move was pruned" return value at
  line 873 (`return standing`). If you compute `standing` at more nodes you have
  also changed what that path can return at depth > 6 — but that path is only
  reachable when `futile` or `prune2` fired, both of which require
  `standing != -INFINITY` already. Verify by inspection, and by the exact check.
* `ply < 2` has no grandparent: improving must default to **True** at ply 0–1
  (SF's convention) or you will over-prune the whole first two plies.
* Under **SINGULAR**, the excluded re-search re-enters the same `ply`. It must
  not overwrite `exts[MAX_PLY + ply]`, or the parent's improving flag flips
  mid-node. Guard the write with `excluded == 0`.
* After a **null move** the child's static eval is the negation of the parent's;
  writing it naively makes "improving" oscillate. SF simply does not use the
  null child's eval for improving. Skip the write when the move was null.

**Test plan.** Bench depth 8/10 (expect 0.90–0.96x nodes for the pair), then
`113-improving` and `114-cutnode` as separate SPRT[0,20] tasks at 8 s. Endgame
suite for IMPROVING (it changes RFP behaviour, and RFP_PHASE was closed *on* that
suite — the same instrument applies).

**Honest risk.** Individually each is probably +5…+10 at 8 s, i.e. **below the
rig's resolution**. Their value is that they unlock harder LMR and a deeper null
move; judge them by nodes-at-fixed-depth and bundle them with #1 and #4.

### 3.4 NMP_V2 — the null move (rank 4)

**The mis-tune.** `null_depth = depth - 1 - 2 - depth//6` (fastsearch.py:639–641)
gives R = 2 at depth < 6, 3 at 6–11, 4 at 12+. The SF family runs
R ≈ 3 + depth/3 (or /4) **plus** an eval-margin term of up to +3, i.e. R = 6–8 at
depth 12. We are **one to three plies shallower than every reference engine**,
and the null move is the cheapest node-saver in the search. This is the clearest
single mis-tune in the codebase.

**What to change** (all inside the NMP block, fastsearch.py:631–659):
1. `null_depth = depth - 1 - (NMP_BASE + depth // NMP_DIV + min((standing - beta) // 200, 3))`
   with `NMP_BASE = 3`, `NMP_DIV = 4`. This needs `standing` at every node —
   i.e. it depends on §3.3. Without §3.3, use the constant part only.
2. **Skip the null move when the TT says fail-low**: we already have `tt_flag`
   and `tt_score` in locals (lines 538–556) for SINGULAR. If
   `tt_depth >= depth - 3 and tt_flag == 2 and tt_score < beta`, the null move
   is almost certainly wasted; skip it.
3. **Verification** at `depth >= 12`: on `score >= beta`, re-search
   `search(depth - R, beta - 1, beta, ply, ...)` with the null move disabled
   (a `ctrl` slot, or a plain parameter) and only return `beta` if that also
   fails high. This is the zugzwang guard and it points straight at our
   documented endgame weakness.
4. Return `score` rather than `beta` when `abs(score) < DISTANCE_THRESHOLD`
   (fail-soft), which costs nothing and gives the TT a better bound.

**Switch.** `NMP_V2: Final = False`, `C_NMP_V2 = 38`, with `NMP_BASE`,
`NMP_DIV`, `NMP_VERIFY_DEPTH` as module constants mirrored in
`check_fastsearch.check_constants` (add them to the `pairs` dict there, since
that function asserts every mirrored constant).

**Pitfalls.**
* `NMP_GUARD` (no null after a null) was measured **flat** here, but the reason
  it existed is real: two nulls restore the Zobrist key and the repetition scan
  fires, scoring the grandchild as a draw. A **deeper** R makes null-after-null
  more likely, so re-check that interaction; consider folding NMP_GUARD in.
* Under **LAZY_ACC** there is already a careful `C_ACC_PLY` relabel after
  `unmake_null` (lines 650–655). A verification re-search adds a second null-free
  search at the same ply — it does not make a null, so nothing to relabel, but
  read that block before touching anything nearby.
* `non_pawn_material` is already the zugzwang guard for the side to move; keep it.
* Deeper R makes the search *less* reliable at low material — which is exactly
  the band the endgame suite measures. **Run the suite.**

**Test plan.** Bench depth 8/10 first — expect a real node cut, target
**0.80–0.88x** (much larger than LMR_AGGRESSIVE's 0.92x; if you do not see it,
the change did not take). Then `115-nmpv2` SPRT[0,20] at 8 s, 800 games. Then the
endgame suite (champion 7.4 cp) — a rise above ~8.5 cp is a reject even if the
gauntlet likes it, by the same rule that closed RFP_PHASE.

### 3.5 CAPTURE_ORDER — SEE-ordered captures and capture history (rank 5)

**The gap.** `fb.score_moves` scores every capture at
`CAPTURE_BONUS + MVV[victim]*16 - MVV[attacker]`, i.e. **all captures rank above
both killers and every quiet, regardless of whether they lose material**. We
SEE-*prune* bad captures, but only at `depth <= 5` and only from the second move
(`C_SEE_MAIN`, lines 734–743). At depth ≥ 6 a queen capturing a defended pawn is
still searched before the killers.

**What to change** (in `fb.score_moves`, and only under the switch so
`order_moves` and the Python reference stay untouched):
1. Split captures into **good** (`see >= 0`) and **bad** (`see < 0`). Good keep
   `CAPTURE_BONUS + ...`. Bad get a score **below the quiet history band**, e.g.
   `-CAPTURE_BONUS + MVV[victim]*16 - MVV[attacker]`.
2. Add **capture history**: `caphist[(piece*64 + to)*6 + captured]`, `int32`,
   4608 entries — tiny. Added to a good capture's score, and updated with the
   same gravity on a capture cutoff (with a malus loop over the captures already
   tried, which needs a second small array beside `quiets`, or a reuse of a spare
   `quiets` column band).

**Cost model.** Calling `see` for every capture at every node is the expensive
part — which is exactly why §3.6 (`see_ge`) should land first or alongside. With
`see_ge` the ordering call is a boolean early-exit and roughly 2–3x cheaper than
today's full SEE.

**Switch.** `CAPTURE_ORDER: Final = False`, `C_CAP_ORDER = 39`. That is the last
free `ctrl` slot; the next switch after this one must raise `CTRL_SIZE`.

**Pitfalls.**
* `score_moves` does not currently receive `bb`/`meta`, which `see` needs. It
  already receives `sqa`; add `bb` and `meta` as **defaulted** parameters so the
  Python reference call site in `agent.FastEngine.search` (which uses
  `fb.order_moves`, a different function) is unaffected. Verify with
  `check_fastsearch --depth 4 --random 30`.
* Do not let a bad capture sort below a *futility-pruned* quiet — the pruning
  tests run on `plain` (quiet, non-promotion) moves only, so a demoted capture is
  never pruned by `futile` / `prune2`. Good.
* Promotions get `PROMOTION_BONUS` added on top; keep that above the bad-capture
  band or under-promotions will vanish.
* SEE at *every* capture at *every* node is a real cost. Measure knps on the
  bench before the gauntlet; if it is worse than −6% the ordering gain will not
  pay for it.

**Test plan.** Bench depth 8 for both nodes **and** knps (this one can lose on
speed even while winning on nodes). Then `116-caporder` SPRT[0,20] at 8 s.

### 3.6 KERNEL SPEED — `see_ge` and `evaluate` (rank 6)

Two concrete items, both exact, plus an honest ceiling.

**(a) `fastboard.see` allocates on every call.** Line 1019:
`gain = np.zeros(32, dtype=np.int64)`. Under numba this is an NRT heap
allocation, and `see` is called once per capture in quiescence (line 464, with
`C_SEE` on) and again in the main search at depth ≤ 5 (line 741). Both call sites
only ever compare the result against a threshold (`< 0`, `< -20*depth*depth`), so
neither needs the exchange *value* — they need `see_ge(threshold) -> bool`.
The standard formulation uses **scalars only**: no array, no second unwinding
loop, and an early exit as soon as the running balance settles the question.
Expect **2–3x** on SEE. If SEE is ~8–12% of node time (it is not separately
profiled here; `evaluate` 1.58 µs, `make+unmake` 0.41 µs, `gen_legal` 0.19 µs of
a 5.13 µs node leaves 2.95 µs unattributed), that is **+4…+6% knps**.
Keep the existing `see` as well — `see_ge` must be introduced under the same
switch as its call sites, or it changes behaviour if the two ever disagree on a
boundary case. Golden-test `see_ge(m, t) == (see(m) >= t)` over a few thousand
random capture positions before wiring it in.

**(b) `evaluate` is 31% of a node.** 1.58 µs for a 1024→32→1 head = 32,768 FMAs
plus 1024 clipped squares. The head's weights for one bucket are
32 × 1024 × 4 B = **128 KB** — over L1, inside L2 — so this is L2-bandwidth
bound, not FLOP bound. The 4-wide blocking already bought +28% (journal, 4 Sep).
Remaining exact levers, in order of expected return:
* Widen the blocking from 4 to 8 accumulators (one full AVX2 register file of
  running sums) — halves the passes over `hidden`.
* Hoist the clamp: `hidden` is rebuilt from scratch every call; with LAZY_ACC the
  accumulator is already known-current, so the clamp-and-square loop (1024
  elements) can be fused into the first blocking pass instead of a separate pass.
* `fastmath=True` is already set on `evaluate` (line 263) — **do not** add it to
  `search`/`quiesce`; they are integer code and it buys nothing while risking
  reassociation in the score arithmetic.
* `boundscheck` is **off by default** in numba njit, so there is no "disable
  bounds checks" win available. Do not go looking for one.
* int8/int16 heads have been **measured slower twice** here (journal, 4 Sep) and
  are on the closed list. Do not reopen. The real byte-count win would be a
  narrower L2 (32 → 16), which is a *net architecture* change, not a search one.

**Honest ceiling.** A realistic combined target is **195 → 215–230 knps
(+10…+18%)**, which by the calibration in §0 is `65·log2(1.14) ≈ +9 Elo at 8 s`
and **+4…+5 at 120 s**. Kernel speed is *not* where the remaining Elo is. Do (a)
because it is half a day and exact; do (b) only if a day is spare.

**Test plan.** `testing/bench.py --depth 8 --json` against
`overnight/eval/bench-champion-d6.json`-style baselines: node counts must be
**byte-identical** (this is an exact change) and knps must rise.
`check_fastsearch --depth 4 --random 30` must stay 70/70. Then fold into the next
bundle without its own gauntlet — an exact +5% change is invisible to SPRT, and
the precedent is set (QS_EVAL_CACHE closed at +2%, LAZY_ACC promoted on bench).

### 3.7 Two defects found by reading (do these regardless)

**(a) `quiets[]` is stale.** fastsearch.py:769–770:

```python
if history2 and plain:
    quiets[ply, searched] = move
```

`searched` is incremented for **every** move, captures included, but
`quiets[ply, searched]` is only *written* for plain quiets. So when a capture is
searched at index 0 (the usual case — captures sort first), `quiets[ply, 0]`
still holds whatever the *previous node at this ply* wrote. The malus loop at
lines 855–859 then iterates `range(searched)` and applies a history penalty to
that stale move, which belongs to a sibling subtree. Because most cutoffs happen
on move 0 or 1, this fires constantly. **Fix:** write `quiets[ply, searched] = 0`
in the `else` branch. It is three lines, it must be a switch
(`HISTORY2_FIX`) because HISTORY2 is currently **on** in the tree, and it may
well be worth a few Elo on its own. Bench it at depth 8/10; fold into a bundle.

**(b) Killers never expire.** `self.killers2` is allocated once per `FastEngine`
and never cleared — killers from move 10 are still ranked at
`KILLER_FIRST = (1<<20)-1` at move 60, above every history move. Every reference
engine clears `killers[ply+2]` on node entry, and clears the whole table between
game moves. The `>>= 1` decay that HYGIENE applies to `butterfly` has no
equivalent here. Cheap; fold in with (a).

---

## 4. Are our pruning constants mis-tuned for the eval scale?

**The eval really is in centipawns.** `training/train.py` minimises
`(sigmoid(prediction) - sigmoid(target/400))²` with `SCALE = 400.0`, and
`fastsearch.evaluate` returns `int(out * OUTPUT_SCALE)` with the same 400. So the
network's output is a logit fitted to `cp/400`, and the engine's score is the
Stockfish centipawn it was trained on. **The margins are not on a foreign scale**
and the "our NNUE is scaled differently so all the SF constants are wrong" worry
is unfounded. The one real caveat is *resolution*: because the loss lives in
win-probability space, the net is precise near 0 and compresses badly beyond
about ±600 cp. Margins that operate in the ±100–500 band (all of ours) are fine;
anything that would trust an eval of +900 is not.

Constant by constant, against the SF/Ethereal/Berserk/Weiss band:

| Constant | Ours | Reference band | Verdict |
|---|---|---|---|
| `RFP_MARGIN` | 80/ply | 70–90/ply | **Fine.** Do not touch. |
| `RFP_MAX_DEPTH` | 6 | 7–9, with an `improving` reduction | **One to two plies short.** Cheap test: 8, but only *with* IMPROVING (§3.3) — raising the cap without the improving term is how you prune a rising position. |
| `FUTILITY_MARGIN` | (0,150,300) d≤2 | ~100–150/ply | Fine, but **inconsistent with PRUNE_V2's** (0,100,200,300,400). Two overlapping mechanisms with different d=1 margins. Unify to one ladder when PRUNE_V2 lands. |
| `NMP_REDUCTION` | 2 + d/6 | 3 + d/3 or /4, **plus** an eval-margin term to +3 | **Clearly mis-tuned — the biggest one.** See §3.4. |
| `NMP_MIN_DEPTH` | 3 | 3 | Fine. |
| LMR table | log(d)log(m)/2.25 + 0.75 | ≈ /2.4 base, **then ±1–3 plies** of adjustment | **Base is already steeper than SF's base.** The gap is the adjustments, not the table. This is why LMR_AGGRESSIVE (÷1.8) only reached 0.92x nodes: a steeper table with no accuracy terms is self-limiting. |
| LMR history term | ±1 ply at ±8000 | continuous, ±2–3 plies | **Too coarse.** Make it `hist // 6000`, clamped ±2. Folds into §3.1. |
| `HIST_PRUNE_SLOPE` | 1500/ply, bound 16384 | ~2000–3500/ply at similar bounds | Slightly conservative; try 2500 once PRUNE_V2's own verdict lands. |
| `ASPIRATION_WINDOW` | 15 | 10–20 | Fine (already tuned 30 → 15). |
| Aspiration widening | `window * 4**fails`, full window at 3 fails | ×1.3–1.5 per fail, rarely fully open | **Too coarse**, and going fully open throws away the narrow-window saving. Half a day, a couple of Elo. |
| `QS_CAP` / `BIG_DELTA` / `DELTA_MARGIN` | 14 / 975 / 200 | comparable | Fine. |
| `SINGULAR` margin | `2*depth`, depth `(d-1)//2` | 1–3×depth, similar | Fine as a first cut. |

**What a tuning pass could yield, honestly.** In the SF world a full SPSA retune
of the pruning constants is worth roughly +5…+15 Elo, and it costs tens of
thousands of games. Our rig is one laptop and one 16-core desktop, a 600-game
SPRT takes hours, and **uploads close on 11 Sep**. SPSA over even four parameters
would consume the entire remaining desktop budget and would still be
under-powered. **Recommendation: do not run SPSA.** Instead:

* Use **bench nodes at fixed depth 8 and 10** as the cheap objective for
  parameter *shape* choices (it is deterministic, takes minutes, and is what
  correctly predicted that LMR_AGGRESSIVE would disappoint).
* Use the **endgame suite** (400 positions, 2.5 s, ~18 min) as the objective for
  anything that changes what the search *trusts* — it is the instrument that
  correctly closed RFP_PHASE.
* Spend the gauntlet budget on the **structural** items in §3, and on the
  **bundle** gate, not on constants.

The single exception worth a gauntlet: **`NMP_REDUCTION`**, because the gap to
the reference band is 1–3 plies, not 10%. That is a structural change wearing a
constant's clothes.

---

## 5. Sequencing, and what to do with the desktop

Given the freeze on 10 Sep evening, there is room for roughly **three
single-switch SPRTs plus one bundle gate**. Proposed order:

1. **Now, cheap and off the critical path:** §3.7 (a) and (b), §3.6 (a)
   `see_ge`, §4's aspiration widening. All exact or near-exact, all bench-judged,
   all folded into the next bundle without their own gauntlets.
2. **First SPRT: `112-conthist`** (§3.1). Highest expected value, and everything
   after it is worth more once it exists.
3. **Second SPRT: `115-nmpv2`** (§3.4) — plus the endgame suite, mandatory.
4. **Third SPRT: `113-improving`** (§3.3), which is also what lets RFP go to
   depth 8 and LMR reduce harder.
5. **In parallel on the other machine, all 120 s:** TIME_V6 (§3.2) — schedule
   model → clocktest → 40 games at 120 s.
6. **Bundle v10** = v8.5 (if it passes) + whatever of the above is positive,
   through the full gate: SPRT vs champion, 200-game crash hunt
   (`--elo0 900 --elo1 950`), clocktest, 40 games at 120 s, unpacked zip < 50 MB,
   **and an import-time measurement**.

Items 5–24 of the ranked table are explicitly **not** scheduled. ProbCut (#8) and
TT-in-quiescence (#7) are the two worth building if a machine goes idle; both are
half-day builds with real but modest upside.

## 6. The constraint everyone forgets

`submission-v8.zip`: 21.6 MB packed, 27.9 MB unpacked, **import 31 s**. Their box
is ~1.8x slower → **~56 s against a 90 s init budget**. Every switch in §3 adds
branches to one enormous `njit` function, and numba's compile time grows
faster than linearly in a function's branch count. `warm_up` compiles both kernels
inside the init budget by design — so *every* new branch is charged to that 56 s.

**Measure `python -c "import agent"` wall time on every bundle, early**, not at
packaging time. If it passes 45 s locally, the bundle is not shippable however
much Elo it has, and the right response is to cut the least-valuable switch — not
to hope their box is faster than 1.8x.
