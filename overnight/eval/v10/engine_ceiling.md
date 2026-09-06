# The structural ceiling of this engine, measured

*Written 6 Sep 2026, 15:00–16:50, iter 34. Read-only on the tree: every build below is a copy
under `overnight/challengers/ceiling/`. No tracked file was modified. Champion snapshot
`agent.py` md5 `bde60a60`, `fastsearch.py` `662f4697`, `fastboard.py` `7bcb0aa1`,
`weights/net.npz` 6 Sep 12:10 (W1 12288×512 f16 = 16 king zones × 512 accumulator,
W2 8×1024×32). Laptop: Intel Core Ultra 9 285H, 16 cores, python 3.12.10, numba 0.67.0.*

## Summary — ten lines, each with the number behind it

1. **EBF is 2.06, which is healthy.** One matched 40-position set, iterative deepening 1..d,
   fresh engine per position: d6 261,890 → d8 1,110,289 → d10 4,750,385 → d12 15,014,299 nodes.
   Per ply: **2.059** (6→8), **2.068** (8→10), **1.778** (10→12). Move ordering is not the problem.
2. **First-move cutoff is 85.4%,** against the 90%+ of strong engines: 324,267 of 379,527
   fail-high nodes at d10, mean move index at cutoff **0.551**. Perfect ordering would remove
   about a third of the work at fail-high nodes only — worth **+3.5 to +8 Elo at 120 s**, not 300.
3. **One ply costs 2.06× the nodes = 1.043 doublings = +33 Elo at 120 s** on this project's own
   calibration (32 Elo/doubling). The 300–500 Elo gap to the leader is therefore **9.0 to 15.0
   extra plies, i.e. 675× to 49,000× the node rate.** No numba engine on one core is 675× ours.
   **The gap is not search speed and cannot be closed on this axis.**
4. **Evaluation is 0.97 µs of a 3.26 µs node (30%)** — measured by a node-identical double-eval
   build (+4.63 s on a 15.5 s depth-10 bench, node count bit-identical at 4,750,385), and
   independently by micro-benchmark (1382 ns/call × 0.558 calls/node = 0.77 µs). It is the only
   stage in the engine whose cost is unambiguously resolvable.
5. **Nobody profiled the other 70% because there is nothing there to find.** Doubling movegen,
   the TT probe, the accumulator snapshot, move scoring or the pick scan each moves the bench by
   **less than the ±8% process-to-process spread of the bench itself**. Re-run with a 9–10× lever
   arm the figures are: pick scan **≤0.029 µs/node (0.9%)**, TT probe **≤9.8 ns each = 0.006
   µs/node (0.2%)**, make+unmake **0.31 µs/node (10%)**, accumulator **0.21 µs/node (6%)**. The
   only stage above the floor is make/unmake.
6. **The largest single unattributed cost is numba's calling convention.** Adding 10 unused array
   parameters to `search` costs **+0.175 µs/node**; adding 20 costs **+0.267 µs/node** (17.5 and
   13.4 ns per array argument per node). The kernel already passes **30 arrays and 7 scalars**, so
   argument passing is an estimated **0.24–0.45 µs/node = 7–14% of node time.**
7. **82.8% of `search()` entries never generate a move.** At d10, of 3,066,398 entries: **45.1%**
   drop straight into quiescence, **24.4%** return on reverse futility, **12.5%** on a TT cutoff,
   **17.2%** reach the move loop. The non-eval time is per-node *control flow*, not any data
   structure — which is why no single hot spot exists to fix.
8. **The reported node count is inflated 1.41×.** 1,383,106 of 4,750,385 "nodes" at d10 are the
   same position counted twice (a `search()` entry that immediately calls `quiesce()`). Distinct
   positions: 3,367,279. Our 218–332 knps is **155–236 knps** of real nodes — relevant only when
   comparing against another engine's published figure.
9. **The realistic node-rate ceiling is 1.15–1.25×, worth +6 to +10 Elo at 120 s.** Making the
   *entire* non-eval 70% free — physically impossible — would be 3.3× = **+55 Elo**. That is the
   hard upper bound on everything a profile can ever buy here.
10. **The biggest lever on the board is not Elo, it is init.** Init measured **33.8–41.7 s idle**
    and **55.0–55.2 s with an 8-worker gauntlet running** (1.63× load inflation) on a box the
    platform runs ~1.8–2.1× slower than. `AGENTS.md:24` says the budget is **60 s**; four platform
    samples are 74.1 / >90 (**game lost**) / 88.1 / 64.1 s. SEARCH_SPLIT
    (`overnight/eval/v10/initsplit.md`, −13 s local / −27 s platform) is worth more than every
    speed item in this document put together.

---

## 0. Method, and where it is weak

Everything is measured on **one fixed set of 40 positions** — `testing/bench.POSITIONS` in full —
with a **fresh `FastEngine` per position per depth** (fresh TT, killers, history), iterative
deepening 1..d, no time limit. This is the fix for the comparability problem in the note that
commissioned this work: the published bench uses different position counts at different depths, so
its node totals cannot be divided across depths. Every node count below is over the same 40.

Driver: `%TEMP%/claude/.../scratchpad/ebf.py` (a stripped `testing/bench` that also dumps counter
slots). Analysis: `analyse.py`, `timing.py`. Variants: `mkvariants*.py`, `mkargs.py`, `patch_count*.py`.

### Three instruments, three different weaknesses

**(a) Counters.** Extra `ctrl[]` slots incremented inside the kernels. *Exact*: the counter build
reproduces the baseline node count bit-for-bit (91,912 nodes on positions 0–9 at d6 in both
builds; 4,750,385 at d10 in all counter builds). Counters cost ~6% of runtime and **nothing** in
accuracy — they are load-invariant, so every count in this document is exact, not an estimate.

**(b) Duplication A/B.** Run one stage a second (or 6th, or 10th) time per node, keeping the tree
identical, and read the extra time. Node identity is the correctness proof and every timing
variant below reports exactly 4,750,385 nodes. Two things go wrong with it:

* **LLVM deletes the duplicate.** A second `gen_legal` into the same buffer, or nine extra
  `score_moves` passes into the same row, are dead stores and are removed. This is not a
  hypothesis: `micro.py`'s isolated loop measures `gen_legal` at **−0 ns/call**, which is
  impossible, and `gen10x` (nine extra full generations at every movegen node, 8.6M extra
  generations over the bench) moved the bench by **−0.02 s**. Both results are *invalid*, not
  evidence that movegen is free. I could not find a formulation LLVM would not eliminate.
* **Codegen perturbation.** Adding code inside a 650-line kernel changes register allocation for
  the whole function. This showed up decisively: `score2x` measured **+16%** in one run and
  **−1%** in another, and `score3x` (two duplicates) came out **faster than base**. A duplication
  result is only believable if it scales with the number of duplicates. **Every number I quote
  below either passed a linearity check at 5–10× or is labelled as unmeasured.**

**(c) Isolated micro-benchmarks** (`micro.py`, loop-doubling, best of 3). These are *warm-cache*
costs and therefore **lower bounds** on the in-search cost of the same call. The gap is large:
`make_light`+`unmake_light` measures **92 ns** isolated and **313 ns** in situ (`mkun6x`), a 3.4×
cold-cache penalty.

### Error bars

* **Within one process**, three passes over the same workload spread **3–5%**.
* **Between processes**, the same baseline build measured **14.31 / 14.90 / 15.80 / 15.90 / 16.20 s**
  across five independent runs (and 15.23 / 16.10 / 16.20 / 16.70 s as the trailing control in the
  same runs) — a **±8% spread** on identical code, from thermal drift and process placement.
  **±8% = ±0.26 µs/node is the resolution floor.** Anything smaller is not measurable on this rig.
* One whole round of measurements (`vars.log`) had to be **thrown away**: an 8-worker gauntlet
  finished part-way through it and the machine got 24% faster mid-run. All numbers below are from
  runs on an idle machine with a baseline measured before *and* after.

---

## 1. The profile: where non-eval node time goes

Baseline at depth 10 over the 40 matched positions: **4,750,385 nodes in ~15.5 s = 306 knps =
3.26 µs/node** (mean of five independent baseline processes; ±8%).

### 1.1 The duplication table

Δ is against a baseline linearly interpolated between the control runs that bracket each variant
in the same process sequence. Every row is node-identical at 4,750,385 unless marked.

| variant | what it repeats | lever | Δ µs/node | per unit | verdict |
|---|---|---:|---:|---:|---|
| `eval2x` | `evaluate` body | 1 extra | **+0.974** | 0.97 µs/node | **real**: +32%, four times the ±8% floor |
| `mkun6x` | `make_move`+`unmake_move` | 5 extra | **+1.564** | **0.313 µs/node per pair** | **real**: +47% |
| `pick10x` | `pick_move`'s scan | 9 extra | +0.263 | ≤0.029 µs/node per pass | **upper bound**: +8.0%, at the floor |
| `tt10x` | TT probe at a random slot | 9 extra | +0.088 | ≤9.8 ns per cold probe | **upper bound**: +2.6%, below the floor |
| `args10` | 10 unused array parameters | 10 | +0.175 | 17.5 ns/arg/node | estimate: +5.2%, at the floor |
| `args20` | 20 unused array parameters | 20 | +0.267 | 13.4 ns/arg/node | estimate: +7.9%, at the floor |
| `acc2x` | the `astack` accumulator snapshot | 1 extra | +0.00 | — | ≤ noise (≤0.26) |
| `lazyoff` | `LAZY_ACC = False` | — | +0.12 | — | ≤ noise; laziness saves ~4% |
| `ttinterleave` | key+data as one 16-byte entry | — | −0.00 | — | **no measurable win** |
| `ttsmall` | `TT_BITS` 22→18 (64 MB→4 MB) | — | −0.14 | — | ~4%/node faster, **not** node-identical |
| `gen2x` / `gen10x` | `gen_legal` | 1 / 9 extra | −0.03 / −0.01 | — | **INVALID** — LLVM removes it |
| `score2x` / `score10x` | `score_moves` | 1 / 9 extra | +0.49 then −0.01 / +0.02 | — | **INVALID** — dead stores, and irreproducible |
| `pick2x` / `pick3x` | `pick_move` scan | 1 / 2 extra | +0.36 / +0.02 | — | **superseded** — failed linearity, use `pick10x` |
| `evalstub` | `evaluate` → `simple_eval` | — | n/a | — | **not comparable**: 1,080,269 nodes, a 4.4× *smaller* tree |

Only two rows clear the ±8% floor outright: `eval2x` and `mkun6x`. `pick10x`, `tt10x`,
`args10` and `args20` sit **at or under** it, which is why they are quoted as bounds and
estimates — but each is a 9–20× lever, so even as a ceiling they say the per-unit cost is tiny.
`args10`/`args20` carry one extra caveat: adding the dummy parameters broke `fs.warm_up`'s own
call (a `NameError` on the dummy names), so `_FAST_OK` came back False and the kernel compiled
lazily inside the first pass. Passes 2 and 3 ran the compiled kernel and are node-identical at
4,750,385; only those two are used.

The last row of the table matters for the brief that commissioned this work. The "218 knps → 362 knps with
`simple_eval`" measurement, from which "eval is ~40% of node time" was derived, **changes the tree
by 4.4×**: a material-only static score makes reverse futility and futility prune vastly harder,
so the node *mix* is different and the knps ratio is not a cost ratio. The node-identical
`eval2x` figure is **30%**, and the isolated micro-benchmark independently gives **24%**.

### 1.2 The budget

Measured items first, then estimates, against 3.26 µs/node. Call counts are exact (counters, d10).

| stage | calls/node | µs/node | % | source |
|---|---:|---:|---:|---|
| `evaluate` (NNUE head + clipped ReLU) | 0.558 | **0.97** | **30%** | `eval2x`; micro cross-check 0.77 |
| `make_move`+`unmake_move` (board only) | 0.696 | **0.22** | **7%** | `mkun6x` |
| accumulator update (`sync_acc` + `_acc_row`) | 0.540 plies | **0.21** | **6%** | micro `make_full` − `make_light` = 389 ns |
| passing 30 arrays + 7 scalars per call | 0.634 calls | 0.24–0.45 (est) | 7–14% | `args10`/`args20` extrapolated |
| `gen_legal` | 0.201 | 0.04–0.16 (est) | 1–5% | **not measurable**; count × 200–800 ns |
| `score_moves` | 0.201 (3.23 moves) | 0.05–0.15 (est) | 2–5% | **not measurable**; scaled from `pick10x` |
| `pick_move` scan | 1.09 | **0.029** | **0.9%** | `pick10x` |
| TT probe | 0.645 | **0.006** | **0.2%** | `tt10x` |
| `see` | 0.369 | 0.04 (est) | 1% | micro 117 ns/call |
| `in_check` | ~0.65 | 0.03 (est) | 1% | micro 41 ns/call |
| **residual**: search + quiescence prologue on every node, TT store, repetition scan, killer clear, mate-distance, improving/RFP/NMP/singular guards, history and killer updates, clock poll, recursion | — | **1.1–1.5** | **34–46%** | by difference |

The residual is not a measurement failure — it is the finding. **Section 3 shows what it is
made of, and it is not a data structure.**

### 1.3 Two structural facts about memory that are *not* levers

* **The transposition table is two separate 32 MB arrays** (`tt_key`, `tt_data`), so every probe
  touches two cache lines instead of one. I built the interleaved version (one array, key at
  `2i`, data at `2i+1`; buckets then land inside a single 64-byte line) and it is node-identical
  and **measurably worth nothing**. `tt10x` explains why: a cold random bucket probe costs
  **9.8 ns**, so at 0.645 probes/node the whole TT probe path is 0.006 µs/node. Halving a 0.2%
  cost is not a project. Same argument retires the split `ec_key`/`ec_val` eval cache.
* **The accumulator snapshot writes 4 KB per synced ply** (2×512 float32 into `astack`), 2.5 GB
  of copying over the depth-10 bench. Duplicating it (`acc2x`) cost **nothing measurable** — the
  live window of `astack` for a 12-ply line is 48 KB and stays in L2. The accumulator's real cost
  is the **W1 row gathers**, and the micro split puts the whole accumulator at 389 ns/ply =
  0.21 µs/node.

---

## 2. EBF and move ordering, on the matched set

### 2.1 Effective branching factor

40 positions, iterative deepening 1..d, fresh engine per position — the same set at every depth.

| depth | nodes | s | knps | ratio | EBF/ply |
|---|---:|---:|---:|---:|---:|
| 6 | 261,890 | 1.3 | 206 | — | — |
| 8 | 1,110,289 | 5.1 | 217 | ×4.240 / 2 ply | **2.059** |
| 10 | 4,750,385 | 21.1 | 225 | ×4.279 / 2 ply | **2.068** |
| 12 | 15,014,299 | 67.2 | 224 | ×3.161 / 2 ply | **1.778** |

(Those knps are from the counter build under residual load; the clean baseline is 306 knps.)

Per-position EBF, median / min / max: d6→d8 **2.095** / 1.049 / 3.599; d8→d10 **1.905** / 1.021 /
4.025; d10→d12 **1.620** / 1.019 / 3.732. The falling EBF with depth is TT reuse and LMR biting
harder, and it is a *good* sign.

**2.06 sits in the middle of the healthy 1.8–2.5 band.** The brief's hypothesis — "if ours is
materially higher, move ordering is the biggest lever" — is **refuted**. It is not.

This also corrects the figure in `search.md` §0, which derived EBF ≈ 2.10 by comparing 40
positions at depth 8 against a mean depth ~12 at 4 s. That estimate was right by luck; the matched
measurement is 2.06.

### 2.2 Move ordering quality

Depth 10, exact counts. Fail-high nodes = 379,527 = **72.1%** of the 526,313 nodes that generate
moves.

| | count | share |
|---|---:|---:|
| cutoff on the **first move searched** | 324,267 | **85.44%** |
| cutoff on the **first move in the ordering** (`i == 0`) | 323,890 | 85.34% |
| mean move index at cutoff | — | **0.551** |
| quiescence: cutoff on the first capture tried | 240,149 / 271,943 | **88.31%** |

Stable across depth: 83.61% (d6), 84.83% (d8), 85.44% (d10), 84.92% (d12).

**What actually causes the cutoff** (d10, of 379,527 fail-highs):

| cutting move | count | share |
|---|---:|---:|
| capture or promotion (MVV-LVA / SEE-ordered) | 152,043 | **40.06%** |
| killer | 91,050 | 23.99% |
| hash move | 86,012 | 22.66% |
| counter move | 14,719 | 3.88% |
| other quiet (history) | 35,703 | 9.41% |

Two things follow.

* **A hash move exists at only 25.7% of movegen nodes** (135,111 of 526,313) — low by reference-
  engine standards. I tested the obvious culprit and **refuted it**: the hypothesis was that
  `QS_TT` floods the table with depth-0, move-0 quiescence bounds (measured: **1,639,616 quiescence
  store attempts, 1,621,585 kept, against 515,182 main-search stores — 76% of table traffic is
  moveless**). Turning `QS_TT` off moves hash-move availability from 25.7% to **23.7%**, i.e. the
  wrong way. The low rate is the shape of a heavily pruned tree, not TT pollution.
  *(Incidentally: `QS_TT` triples the TT hit rate, 30.7% vs 10.3%, and cuts the tree 8.8%,
  4,750,385 vs 5,208,365 nodes. It is doing a lot of work.)*
* **Every hash-move cutoff is a first-move cutoff** (86,012 of 86,012). The hash move is always
  tried first and always cuts immediately when it cuts. There is no ordering bug here.

### 2.3 How much is the missing 5% of ordering worth?

Mean move index at cutoff is 0.551, so a perfect orderer would remove 0.551 extra searched moves
per fail-high node out of 1.551 = 36% of the move-loop work at fail-high nodes. Those extra moves
are searched with a null window and usually reduced, so their true cost is well under a full
subtree; call it a 10–15% node reduction. `search.md`'s own realisation factor for ordering
changes is 55%, so: 0.55 × 32 × log2(1/0.87) ≈ **+3.5 Elo at 120 s**, ceiling **+8** if the
reduction were 25%. And 85.4% → 100% is not attainable. **Ordering is not the lever.**

---

## 3. Where the node budget really goes: 82.8% of nodes do nothing

Depth 10, 3,066,398 `search()` entries, exact counts, mutually exclusive and exhaustive:

| what the entry does | count | share |
|---|---:|---:|
| drops straight into quiescence (`depth <= 0`) | 1,383,106 | **45.11%** |
| returns on **reverse futility** | 747,888 | **24.39%** |
| **reaches the move loop** | 526,313 | **17.16%** |
| returns on a **TT cutoff** | 383,809 | 12.52% |
| returns on a **null-move cutoff** | 22,323 | 0.73% |
| repetition / fifty-move | 1,895 | 0.06% |
| mate-distance pruning | 1,064 | 0.03% |
| razor / MAX_PLY | 0 | 0.00% |

This is the shape of a correctly-pruned modern search, and it is also the explanation for the
diffuse profile: **the dominant per-node cost is the prologue that every one of those 3.07M entries
executes before deciding to return** — key load, repetition scan, TT probe and unpack, IIR test,
`in_check`, check extension, mate-distance, improving, the RFP arithmetic, the NMP guard chain and
the singular guard — plus the ~37-argument recursive call that got it there. There is no data
structure to optimise because the time is not in a data structure.

Three consequences worth writing down:

* **45.1% of `search()` entries immediately re-enter the same position as a `quiesce()` entry**,
  which probes the *same TT slot again* (`QS_TT` is on) and does its own prologue. That is the
  1.41× node inflation of line 8, and it is also ~0.3–0.5 µs of duplicated prologue on 29% of
  counted nodes.
* **Reverse futility prunes more nodes than the move loop expands** (24.4% vs 17.2%), on a static
  score, with a margin of 80 × depth ≤ 480 cp. **22.1% of those prunes happen at ≤16 pieces**,
  where `games.md` measures the net's mean absolute static error at **475 cp**. The margin is at
  or below the eval's own error in the phase where 13 of 17 platform blunders happened. *This is
  already closed*: `JOURNAL.md` 5 Sep 00:06 records RFP_PHASE tested on the 400-position endgame
  suite — every wider margin made it worse (7.0 cp baseline vs 11.6 / 15.8 / 7.6 / 7.5). Wider
  margins cost depth faster than they buy accuracy. **Do not reopen it**; the entry is here so
  the next reader does not rediscover the same idea.
* **78.5% of generated moves are never made** (15,348,407 generated, 3,303,024 made at d10;
  27.4 moves per main generation, 8.35 `pick_move` calls per movegen node). That is the classic
  case for staged generation — and `gen_legal` is called on only **0.201 of nodes**, so the entire
  generator cannot exceed ~5% of node time whatever its per-call cost. `033-staged-movegen` was
  already REJECTed by gauntlet. It is now also bounded arithmetically.

---

## 4. Is numba being used well?

Introspected live (`numbainfo.py`) on the champion after a full import.

* **No object-mode fallback anywhere.** The only non-nopython artefact in the tree is one lifted
  loop in `fastsearch.timed_out`, which is the deliberate `objmode` `time.monotonic()` call.
* **No recompilation.** `search` and `quiesce` have **exactly one signature each**. Nothing in the
  hot path is being re-typed at runtime.
* **Redundant specialisations are 6, not 61.** fastboard: 41 signatures over 35 functions —
  `score_moves` ×3, `gen_legal` ×3, `rebuild` ×2, `make_light` ×2, `make_full` ×2, everything else
  ×1. fastsearch: 23 over 20 — `pack` ×3, `qs_tt_store` ×2. **This contradicts
  `speed.md` §7 ("61 redundant specialisations: `attackers_to` 9, `is_attacked` 9, `rook_attacks`
  9, `bishop_attacks` 9")**, which was read out of numba's on-disk cache index — an artefact that
  accumulates across builds — rather than from the live dispatchers. The live count agrees with
  `initsplit.md` §4. The eager-signature work in `speed.md`'s ranked table #3 is therefore worth
  ~1.5 s of init, not 2–4 s, and no node rate at all.
* **`FastEngine()` construction after warm-up: 0.001 s.** Nothing is being compiled lazily.
* **The TT layout is not cache-hostile enough to matter** (§1.3), and `boundscheck` is already off.
* **The one real numba cost is the calling convention.** `search` takes 30 array arguments and 7
  scalars. A numba array argument is a multi-word struct (meminfo, parent, nitems, itemsize, data
  pointer, shape, strides), not a pointer; on Windows x64 only four arguments go in registers, so
  the rest are stored and reloaded on every recursive call. Measured marginal cost: **17.5 ns per
  array argument per node** for the first ten added, **9.2 ns** for the next ten. Extrapolating the
  marginal rate to the ~26 stack-passed arrays gives **0.24 µs/node (7%)**; extrapolating the
  first-ten rate gives **0.45 µs/node (14%)**. This is the largest identified item in the residual
  and, as far as I can find, has never been looked at — but both data points sit at the ±8%
  resolution floor, so it is an estimate awaiting a bundled pilot build, not a measurement.

---

## 5. The ceiling, in Elo

Using this project's own calibration (`search.md` §0): exact speed-ups realise
`Elo(120 s) = 32·log2(speed-up)` and `Elo(8 s) = 65·log2(speed-up)`.

| scenario | speed-up | Elo @120 s | Elo @8 s |
|---|---:|---:|---:|
| **everything except `evaluate` becomes free** (impossible) | 3.36× | **+56** | +114 |
| all 0.43 µs of make/unmake + accumulator becomes free | 1.15× | +6.5 | +13 |
| all of the argument-passing overhead removed (0.45 µs) | 1.16× | **+6.9** | +14 |
| half of it removed (0.24 µs) | 1.08× | +3.5 | +7 |
| the `ttsmall` memory effect, if it were free | 1.05× | +2.2 | +4.5 |
| a whole extra ply | 2.06× | **+33** | +68 |
| **the gap to the leader** | **675× – 49,000×** | **+300 – +500** | — |

**Read the last two rows together.** Closing 300 Elo on the search-speed axis needs 9.4 doublings,
which at EBF 2.06 is **9.0 extra plies**; 500 Elo needs **15.0**. We reach depth 10–11 at the real
control; the leader would have to be reaching 19–26 on one core in Python. They are not.
Whatever the leader is doing, it is not out-searching us by nine plies.

**The realistic ceiling of the entire profiling programme is 1.15–1.25×, i.e. +6 to +10 Elo at
120 s.** The physically-impossible ceiling is +56. Neither turns the tide, and this is the honest
answer to "stop patching and find something that turns the tide": *there is nothing on this axis
to find, and the measurements above are what closes it rather than a further opinion.*

Where the 300–500 Elo actually lives is already documented and measured in this repo, and my
counters corroborate it rather than contradict it:
`leader_benchmark.md` (their draw rate 7% vs our 26%; matching it is worth ~+130 Elo),
`games.md` (mean |static − reference| of **475 cp at 11–16 pieces** against 70 cp at 27–32; 13 of
17 blunders at ≤16 pieces; a 5 s re-search recovers only **17%** of them — *i.e. depth does not fix
them*), and `initsplit.md` (four platform init samples, one of them a lost game).
**24.1% of our quiescence evaluations at depth 10 happen at ≤16 pieces**, so a quarter of the
engine's static judgements are made in the band where its error is 6.8× worse.

---

## 6. Ranked structural options

Elo at 120 s. "Init" is the change in local import time; multiply by ~1.8–2.1 for the platform.

| # | option | expected Elo | effort | init cost | risk | basis |
|---|---|---:|---|---:|---|---|
| 1 | **SEARCH_SPLIT** — move the four non-recursive blocks of `search` into njit helpers | 0 direct; **prevents a lost game per ~4 platform starts** | 1–2 d | **−13 s local / −27 s platform** | low (pure code motion, node-identical or wrong) | `initsplit.md`; my init 33.8–41.7 s idle, 55.2 s under load, budget 60 s |
| 2 | **NET_V10 / anything that fixes the ≤16-piece eval** | the only axis with **>+50** headroom | GPU night + gauntlet slot | ~0 | high (needs a verdict) | 475 cp vs 70 cp; 24.1% of our evals are in that band |
| 3 | **Bundle the kernel's 30 array parameters** (numba `structref`, or hoist the 7 read-only weight arrays to module globals) | **+3.5 to +7, but confirm first** | **4 h** for the 7 weight arrays alone, as the pilot; 2–4 d for the full bundle | ~0 to −1 s (smaller signature; the body still dominates inference) | medium — the underlying measurement is at the resolution floor, so build the 7-array pilot and re-measure before spending days | `args10` +0.175, `args20` +0.267 µs/node |
| 4 | **Skip the duplicated prologue on the drop into quiescence** (45.1% of `search()` entries; also removes the 1.41× node-count inflation) | **+1.5 to +2.5** | 4 h | small − | medium — the check extension needs `in_check`, so only the TT/repetition/guard half can move; changes the node counter that TIME_V6 reads | 1,383,106 of 4,750,385 nodes |
| 5 | **Staged ordering** — try hash move, then captures, then killers, before scoring the quiets | **+1 to +2** | 1 d | − (smaller kernel) | medium | 85.4% first-move cutoffs; 78.5% of generated moves never made |
| 6 | **`scores` as int32 instead of int64** (all bands fit: max is `1<<30` + `1<<19`) | 0 to **+1** | 1 h | ~0 | low, exact | halves the ordering array's cache footprint; `pick10x` says the scan is only 0.9% |
| 7 | Accumulator width 512→384, or head 32→16 neurons | +8 to +13 *before* the eval-quality loss | 1 night retrain | −1 to −2 s | **high** | **bad trade**: eval quality is the binding constraint (row 2) |
| 8 | Interleaved TT entry / smaller TT / eval-cache layout | **0** | 1 d | 0 | low | **do not do** — built and measured: `ttinterleave` no win, `tt10x` 9.8 ns/probe |
| 9 | Staged or cheaper move generator | **≤+2** | 2 d | + | medium | **do not do** — `033-staged-movegen` REJECTed; `gen_legal` is 0.201 calls/node so ≤5% by arithmetic |
| 10 | Eager signatures on fastboard leaves | 0 | 4 h | **−1.5 s** (not −2 to −4) | low | only 6 redundant specialisations exist, not 61 |
| 11 | RFP_PHASE (piece-count-scaled pruning margins) | **negative** | — | — | — | **closed 5 Sep**: endgame suite 7.0 → 11.6 / 15.8 / 7.6 / 7.5 |
| 12 | int8/int16 NNUE, sparse head, shipped numba cache, AOT | negative or forbidden | — | — | — | **closed**, `speed.md` |

**Ordered by Elo per unit of risk-weighted effort: 1, 2, 3, 4, 5, 6.** Items 1 and 2 are not
speed work, and that is the point of this document.

---

## 7. What I could not determine

1. **The cost of `gen_legal`.** Every duplication formulation I tried — a second call into a
   different buffer, nine calls in a loop, an isolated rotating-output micro-benchmark — was
   eliminated by LLVM. Bounded only arithmetically at 0.04–0.16 µs/node (1–5%) from the exact call
   count. To measure it you need a build where the generator's output is genuinely consumed
   differently each time, or a hardware profiler (VTune / `perf`) on the compiled kernel.
2. **The cost of `score_moves`.** Same failure: duplicates into the same row are dead stores.
   `score2x` gave +16% in one run and −1% in the next, which is the signature of codegen
   perturbation, not of measurement. Estimated 2–5% by analogy with the `pick10x` scan.
3. **The residual, 34–46% of node time**, is attributed only by difference. I know from the
   counters *which nodes* pay it (the 82.8% that never generate a move) but not how it splits
   between the TT unpack, the repetition scan, `in_check`, the guard chain, and the recursive
   call itself. Splitting that needs an instruction-level profiler, not a duplication A/B.
4. **Whether argument bundling would realise its predicted 7–14%.** The `args10`/`args20`
   extrapolation is linear in the marginal region; the first four arguments are register-passed
   and the real 30 may behave differently. The only honest test is to build one bundled variant.
5. **Anything about the real time control.** Every number here is fixed-depth bench. Depth reached
   at 120 s + 0.5 s in real games, TT reuse across moves, and the interaction with TIME_V6 are all
   outside this measurement. The bench set from a cold TT reaches ~375k nodes/position at depth 12,
   which at 306 knps is 1.2 s — consistent with, but not a measurement of, the reported game depth
   of 10–11.
6. **The platform box's actual numbers.** The 1.8–2.1× slowdown factor is inherited from earlier
   notes, not measured by me. My contribution is the load sensitivity: **init is 1.63× slower with
   eight workers running** (55.2 s vs 33.8 s), which is the mechanism behind the 74.1 / >90 / 88.1 /
   64.1 s platform spread.
7. **Whether the leader's 300–500 Elo is real strength or a ladder artefact.** 89% vs 55% against a
   shared field implies ~310 Elo, but pairing effects are not controlled. I have shown only that
   *whatever it is*, search speed cannot account for it.
8. **`AGENTS.md:24` says the import budget is 60 s; `initsplit.md` and the task brief both use
   90 s.** I did not resolve which is binding. If it is 60 s, item 1 is not a priority, it is an
   emergency: we measure 33.8 s idle here and the platform is ~1.8× slower.

---

## Reproduction

All builds under `overnight/challengers/ceiling/` (copies; the tree was not touched). Only `base/`
keeps a `weights/` copy — `cp -r base/weights <variant>/` to run any of the others; see
`overnight/challengers/ceiling/README.md`. **Note:** the continuous pipeline rewrote `agent.py`
(16:41) and `fastsearch.py` (16:09) in the tree during this session, so `base/` is the frozen
15:01 snapshot, not the current champion — the same hazard `speed.md` hit on 5 Sep. Every A/B here
is against that one snapshot and they are comparable to each other.

```
base          frozen champion copy, the control for every A/B
count         + 33 exact counters (nodes, movegen, ordering, cutoffs, TT, make/unmake)
count2        + the decomposition of search() returns (RFP / TT / quiescence / repetition)
count3        + which move causes each cutoff, pick_move call counts, hash-move availability
count4        + quiescence TT store counters       count4off  same with QS_TT = False
eval2x  gen2x  gen10x  score2x  score3x  score10x  pick2x  pick3x  pick10x
mkun2x  mkun6x  tt2x  tt10x  acc2x  lazyoff  ttinterleave  ttsmall  args10  args20  evalstub
```

Scripts in `%TEMP%/claude/C--dev/.../scratchpad/`: `ebf.py` (matched-set runner + counter dump),
`analyse.py`, `timing.py`, `micro.py`, `numbainfo.py`, `mkvariants{,2,5}.py`, `mkargs.py`,
`patch_count{,2,3}.py`, and the run logs `ebf40.log`, `vars2.log`, `vars3.log`, `vars4.log`,
`vars5.log`, `vars6.log`. `vars.log` is the discarded first round (load changed mid-run).

Canonical command:

```
.venv/Scripts/python.exe scratchpad/ebf.py overnight/challengers/ceiling/<v> out.json 10,10,10
.venv/Scripts/python.exe scratchpad/ebf.py overnight/challengers/ceiling/count3 c.json 10 --counters
```

**Read the node count first.** Every timing variant here must report exactly **4,750,385** nodes at
depth 10 over the 40 matched positions; if it does not, it is measuring a different tree and the
timing means nothing.
