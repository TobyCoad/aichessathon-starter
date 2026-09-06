# v10 — is the feature set the wrong design?

Written 6 Sep ~15:30 by a read-only research pass. Nothing was trained, queued or gauntleted;
no engine file was touched. Every number is either quoted from an existing artefact in this
repo or derived arithmetically from one. Derivations live in `_costs2.py`, `_kingmoves.py` and
`_kingmoves2.py` in this directory and are re-runnable in under two minutes. Estimates are
labelled ESTIMATE.

## Summary (10 lines, each with its number)

1. **`network.md`'s reason for closing HalfKP is wrong.** It said 40,960 inputs are "over the
   size cap". At float16, king-mirrored HalfKP (32 x 641 = 20,512 rows) at accumulator 256 is
   **11.03 MB of net and 25.37 MB unpacked — smaller than the 13.64 MB / 27.98 MB we ship
   today**, with 22 MB of headroom to spare. The size objection is refuted.
2. **And it is cheaper per evaluation.** The head is `2 x A x 32` multiply-adds: 16,384 at
   A=256 against **32,768** today, a 0.50x head and, since the accumulator update is also
   linear in A, an ESTIMATED **1.248x node rate = +10.2 Elo at 120 s** before the net proves
   anything. So yes — mirrored HalfKP at acc 256 is **both smaller on disk and cheaper per
   evaluation than what we ship**. Stated plainly, as asked.
3. **But disk and speed were never the binding constraint. Data per king bucket is.** Our two
   measurements: kz8 -> kz16 was **+31 Elo**; kz16 -> kz32 was worse (val 0.004690 vs
   0.004659, suite 8.1 vs 7.4, gauntlet INCONCLUSIVE 52.2%/600). The winning kz16 run had
   **436 M unique positions / 16 zones = 27.2 M per zone**; the losing kz32 run had the same
   436 M / 32 = **13.6 M per zone**. That is the turnover point, and it is a *density*, not a
   count.
4. **At 1.16 B, 32 buckets clears that bar and 64 does not.** 1.16 B / 32 = **36.2 M per
   zone** (above the 27.2 M that won); 1.16 B / 64 = **18.1 M per zone** (between the winner
   and the loser, nearer the loser). True HalfKP/HalfKA is 64 buckets. To give it the density
   the winning configuration had we would need 64 x 27.2 M = **1.74 B** positions. We have
   1.16 B. **We are still 1.5x short — the data objection survives re-examination even though
   the size objection does not.**
5. **The "king move forces a refresh" objection is already 80% paid.** Measured over 4,000
   corpus positions: 8.63% of legal moves are king moves, and **79.8% of those already cross a
   16-zone boundary and already trigger a full one-perspective rebuild today**. Full
   king-square resolution takes the refresh rate from 6.89% to 8.63% of moves — **+1.74
   percentage points**, an ESTIMATED **+0.8% node time**. HalfKP's famous cost is, for us,
   noise.
6. **The 331.9 cp "static error" is not evidence against the feature set.** Those labels are
   Stockfish **depth-18 search** scores on deliberately hard 5-16 piece positions with mean
   |label| = **860.5 cp** at 5-8 pieces (p90 = 2000). Correlation with the label is
   **0.937 / 0.901 / 0.895** by band, and clipping both sides to +/-1000 cp drops the mean
   error to **130.1 cp**. The "strong NNUE is 30-60 cp" figure is against quiet static
   references, not depth-18 scores on won/drawn endgames. Comparing them is a category error.
7. **32 king zones at accumulator 256 needs zero engine edits.** `KING_ZONES` is
   `W1.shape[0] // 768` and `ACC_SIZE` is `W1.shape[1]`; the 32-zone map already exists in
   `training/features.py`, `agent._zone` **and** `fastboard.zone_of`; every kernel loop reads
   `white.shape[0]`. The .npz shape alone selects the architecture. **It is a retrain and an
   export, nothing else.**
8. **Mirroring — the thing that would make 32 buckets into real HalfKA_hm — is the expensive
   half, and it has already been piloted and lost.** NOTES 6 Sep 10:40: symmetrising the
   champion costs **58%** of its loss, and the pilot was still **38% above control's best**
   when killed at epoch 2 of 6 on 40 M positions. Its engine side is the 9-file hot-path
   surgery. It cannot warm-start.
9. **Width is not free to give back.** The only 256 -> 512 datum we own is `021-w512-150m`,
   **PROMOTE +89 +/- 49 over 96 games** — but it confounded width with a 6.7x data increase
   (145 M vs 21.6 M), min-ply 16 and balancing off. It is not a clean width measurement, and
   it is the reason halving A is a bet, not a free +10 Elo.
10. **Verdict: no feature-set change.** Ship one net task — **32 king zones at accumulator
    256, current 768 feature set, from scratch on the 581 M Stockfish corpus** — which is a
    same-parameter-count (6.42 M vs 6.55 M) reallocation from width to king resolution, is
    **0.53 MB smaller** than today's net, banks the 1.248x node rate, and touches **no engine
    file**. Falsifier in the last section.

---

## Task 1 — the cost table

### Method, so every column can be checked

* **W1 rows** = king buckets x features per bucket. Our scheme is `zone * 768 + index`, so
  *our net with 64 buckets and an identity zone map **is** HalfKA* (64 x 768 = 49,152). HalfKP
  is the same construction with the two king planes dropped: 64 x 641 (641 = 10 x 64 + 1; the
  honest count is 640, the +1 is the Shogi-inherited padding slot — the arithmetic below uses
  641 as asked, which is 0.16% pessimistic).
* **W1 fp16 MB** = rows x A x 2 / 1e6. `export.py --half` already stores W1 as float16 and
  `agent.py` casts it back at import; the 50 MB cap is on the unpacked file, so fp16 is the
  number that counts against it.
* **resident f32 MB** = rows x A x 4 / 1e6 — what actually sits in the engine's RAM after the
  cast, and the number that governs cache behaviour. This column is not in the brief and it is
  the one I would watch.
* **net MB** = W1 fp16 + head (W2 `(8, 2A, 32)` f32 + b2 + W3 + b3 + b1).
* **unpacked MB** = net + source 0.238 + book 9.760 + syzygy 4.346, measured from
  `submission-v94.zip` (27.98 MB total today). Limit 50.00.
* **head MACs** = `2 * A * 32`, exactly the inner loop of `agent._eval_bucket_kernel` and
  `fastsearch.evaluate`. Today: 32,768.
* **knps x** — measured `218 knps` with the net and `362 knps` with `evaluate()` stubbed to
  material, so the net is `1/218 - 1/362 = 1.825 us` of a `4.587 us` node = **39.8%**. Because
  `LAZY_ACC` is `True`, stubbing `evaluate()` also skips the accumulator update, so that 39.8%
  covers **head + accumulator**, and both are linear in A. Model:
  `t_new = 2.762 + 1.825 * (A/512)` us. ESTIMATE — it assumes perfect linearity and no cache
  cliff.
* **Elo speed** = `32 * log2(knps ratio)`, from `search.md`'s ~32 Elo per node doubling at
  120 s (65 at 8 s, halved for the long control). ESTIMATE.
* **Mpos/zone** = 1.16 B / king buckets. Reference points: **27.2 M/zone won** (kz16 at
  436 M), **13.6 M/zone lost** (kz32 at 436 M).

### The table

| scheme | A | W1 rows | params M | W1 fp16 MB | resident f32 MB | net MB | unpacked MB | headroom MB | head MACs | x current | knps x | Elo speed @120s | Mpos/zone @1.16B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **CURRENT 16z x 768** | **512** | **12,288** | **6.55** | **12.58** | **25.2** | **13.64** | **27.98** | **+22.02** | **32,768** | **1.00x** | **1.000x** | **+0.0** | **72.5** |
| CURRENT scheme, A=1024 | 1024 | 12,288 | 13.11 | 25.17 | 50.3 | 27.27 | 41.61 | +8.39 | 65,536 | 2.00x | 0.715x | -15.5 | 72.5 |
| CURRENT scheme, A=768 | 768 | 12,288 | 9.83 | 18.87 | 37.7 | 20.45 | 34.80 | +15.20 | 49,152 | 1.50x | 0.834x | -8.4 | 72.5 |
| CURRENT scheme, A=384 | 384 | 12,288 | 4.92 | 9.44 | 18.9 | 10.23 | 24.57 | +25.43 | 24,576 | 0.75x | 1.110x | +4.8 | 72.5 |
| CURRENT scheme, A=256 | 256 | 12,288 | 3.28 | 6.29 | 12.6 | 6.82 | 21.16 | +28.84 | 16,384 | 0.50x | 1.248x | +10.2 | 72.5 |
| 16z x 768 **mirrored** | 512 | 12,288 | 6.55 | 12.58 | 25.2 | 13.64 | 27.98 | +22.02 | 32,768 | 1.00x | 1.000x | +0.0 | 72.5 |
| **32z x 768 unmirrored** (map already in all three files) | 512 | 24,576 | 12.85 | 25.17 | 50.3 | 26.22 | 40.56 | +9.44 | 32,768 | 1.00x | 1.000x | +0.0 | 36.2 |
| **32z x 768 unmirrored** | 384 | 24,576 | 9.63 | 18.87 | 37.7 | 19.66 | 34.01 | +15.99 | 24,576 | 0.75x | 1.110x | +4.8 | 36.2 |
| **32z x 768 unmirrored — RECOMMENDED** | **256** | **24,576** | **6.42** | **12.58** | **25.2** | **13.11** | **27.45** | **+22.55** | **16,384** | **0.50x** | **1.248x** | **+10.2** | **36.2** |
| HalfKA-hm 32 x 768 mirrored | 512 | 24,576 | 12.85 | 25.17 | 50.3 | 26.22 | 40.56 | +9.44 | 32,768 | 1.00x | 1.000x | +0.0 | 36.2 |
| HalfKA-hm 32 x 768 mirrored | 384 | 24,576 | 9.63 | 18.87 | 37.7 | 19.66 | 34.01 | +15.99 | 24,576 | 0.75x | 1.110x | +4.8 | 36.2 |
| HalfKA-hm 32 x 768 mirrored | 256 | 24,576 | 6.42 | 12.58 | 25.2 | 13.11 | 27.45 | +22.55 | 16,384 | 0.50x | 1.248x | +10.2 | 36.2 |
| HalfKA-hm 32 x 768 mirrored | 128 | 24,576 | 3.21 | 6.29 | 12.6 | 6.56 | 20.90 | +29.10 | 8,192 | 0.25x | 1.425x | +16.4 | 36.2 |
| **HalfKP-hm 32 x 641** | 512 | 20,512 | 10.76 | 21.00 | 42.0 | 22.06 | 36.40 | +13.60 | 32,768 | 1.00x | 1.000x | +0.0 | 36.2 |
| **HalfKP-hm 32 x 641** | 384 | 20,512 | 8.07 | 15.75 | 31.5 | 16.54 | 30.89 | +19.11 | 24,576 | 0.75x | 1.110x | +4.8 | 36.2 |
| **HalfKP-hm 32 x 641** | **256** | **20,512** | **5.38** | **10.50** | **21.0** | **11.03** | **25.37** | **+24.63** | **16,384** | **0.50x** | **1.248x** | **+10.2** | **36.2** |
| **HalfKP-hm 32 x 641** | 128 | 20,512 | 2.69 | 5.25 | 10.5 | 5.52 | 19.86 | +30.14 | 8,192 | 0.25x | 1.425x | +16.4 | 36.2 |
| HalfKP 64 x 641 | 512 | 41,024 | 21.27 | 42.01 | 84.0 | 43.06 | 57.41 | **-7.41** | 32,768 | 1.00x | 1.000x | +0.0 | 18.1 |
| HalfKP 64 x 641 | 384 | 41,024 | 15.95 | 31.51 | 63.0 | 32.30 | 46.64 | +3.36 | 24,576 | 0.75x | 1.110x | +4.8 | 18.1 |
| HalfKP 64 x 641 | 256 | 41,024 | 10.63 | 21.00 | 42.0 | 21.53 | 35.88 | +14.12 | 16,384 | 0.50x | 1.248x | +10.2 | 18.1 |
| HalfKP 64 x 641 | 128 | 41,024 | 5.32 | 10.50 | 21.0 | 10.77 | 25.11 | +24.89 | 8,192 | 0.25x | 1.425x | +16.4 | 18.1 |
| HalfKA 64 x 768 | 512 | 49,152 | 25.43 | 50.33 | 100.7 | 51.38 | 65.73 | **-15.73** | 32,768 | 1.00x | 1.000x | +0.0 | 18.1 |
| HalfKA 64 x 768 | 384 | 49,152 | 19.07 | 37.75 | 75.5 | 38.54 | 52.88 | **-2.88** | 24,576 | 0.75x | 1.110x | +4.8 | 18.1 |
| HalfKA 64 x 768 | 256 | 49,152 | 12.71 | 25.17 | 50.3 | 25.69 | 40.04 | +9.96 | 16,384 | 0.50x | 1.248x | +10.2 | 18.1 |
| HalfKAv2 64 x 704 | 256 | 45,056 | 11.67 | 23.07 | 46.1 | 23.60 | 37.94 | +12.06 | 16,384 | 0.50x | 1.248x | +10.2 | 18.1 |

Headroom is against the 50 MB unpacked limit **with the book and the full syzygy set kept**.
Dropping the book (measured a non-factor: 1.09 in-book plies per game, 12 of 22 games got zero
book moves) adds **9.76 MB**; dropping syzygy adds **4.35 MB**. Budgets: net may be
**35.66 MB** as things stand, **45.42 MB** without the book, **49.76 MB** without either. Only
three rows in the whole table are over the cap as shipped, and all three are cured by dropping
the book. **Disk is not a constraint on any of these designs.** That is the single largest
correction to `network.md`.

### The claim in the brief, checked

> "mirrored HalfKP at acc 256 is BOTH SMALLER ON DISK AND CHEAPER PER EVALUATION than what we
> ship today"

**True.** 11.03 MB of net against 13.64 MB (**-19%**), 25.37 MB unpacked against 27.98 MB, and
16,384 head MACs against 32,768 (**0.50x**), for an ESTIMATED 1.248x node rate. It is also
smaller in resident memory: 21.0 MB against 25.2 MB.

Two things the brief did not ask that follow from the same arithmetic and matter more:

* **`32z x 768` unmirrored at A=256 is nearly the same object and costs nothing to build.**
  13.11 MB net (still 0.53 MB smaller than today), the same 16,384 MACs, the same 1.248x, the
  same resident 25.2 MB as today, **and it requires not one line of engine code** (see task 3).
  HalfKP-hm's extra 2.08 MB saving buys a mirror that costs a 9-file hot-path change.
* **The A=256 rows are a parameter-neutral reallocation.** `32z x 768 @ 256` is 6.42 M
  parameters against today's 6.55 M. This is not "a smaller net". It is the *same* net budget
  spent on 2x the king resolution instead of 2x the accumulator width, at half the inference
  cost. That framing is the whole question.

### The refresh cost, measured not assumed

`overnight/eval/v10/_kingmoves.py` and `_kingmoves2.py` reconstruct boards from the packed
corpus (the record stores white-perspective 768-space indices, which invert exactly to
piece/colour/square) and count king moves and zone crossings. Castling rights are not stored so
castling is undercounted; castling is itself a king move, so this biases the king fraction
**down**.

Over 4,000 positions from `data/positions_2025_03.npy` (125,210 legal moves):

| | rate |
|---|---|
| king moves as a share of legal moves | **8.63%** |
| king moves that already cross a 16-zone boundary today | **79.82%** |
| refresh rate today (16 zones) | **6.89%** of moves |
| refresh rate with full king-square resolution | **8.63%** of moves |
| **marginal increase** | **+1.74 pp (a 1.25x increase in refreshes)** |

By piece band (7,000 positions across `data/sf/feb24_08.npy` and `data/positions_2025_03.npy`):

| band | boards | mean pieces | king moves | kz8 refresh | **kz16 refresh (today)** | **full king-square refresh** |
|---|---|---|---|---|---|---|
| 2-12 | 1,492 | 7.6 | 24.44% | 10.36% | **15.86%** | **24.44%** |
| 13-20 | 1,656 | 16.7 | 10.50% | 4.58% | **8.57%** | **10.50%** |
| 21-32 | 2,852 | 26.0 | 5.85% | 2.09% | **5.05%** | **5.85%** |

The two effects run in opposite directions and cancel: king-move frequency is highest exactly
where a refresh is cheapest. `fastboard.rebuild` costs `pieces x A` adds; an incremental push
costs `2A` (snapshot) + `(n_add + n_rem) x 2 x A` ~ `6A`. So in the 2-12 band a refresh is
`7.6A / 6A` ~ **1.3x** an incremental update while the marginal rate rises 8.6 pp; in the 21-32
band a refresh is `26A / 6A` ~ **4.3x** but the marginal rate rises only 0.8 pp.

Weighted through the measured 39.8% net share and the ~34% accumulator share of it (from
`network.md`'s 29.4% evaluate / 15.4% accumulator profile), the extra refresh work is an
ESTIMATED **+0.8% of node time** — i.e. `1.248x` becomes `~1.238x`. **The refresh objection to
HalfKP is quantitatively dead for this engine**, because our 16-zone scheme already pays 80% of
it. Anyone still repeating it should be shown this table.

### The cost nobody has costed: resident memory

`W1` is cast to float32 at import and randomly accessed 4-6 rows at a time. Today it is
**25.2 MB** — already at the edge of a laptop's last-level cache. The table's `resident f32`
column is therefore a real speed term that the linear model above does **not** capture:

* `HalfKA 64 x 768 @ 512` = **100.7 MB** resident. Four times today's. ESTIMATE: this is where
  the discarded synthetic width bench's pathological numbers probably came from.
* `32z x 768 @ 256` and `HalfKA-hm @ 256` = **25.2 MB** — byte-identical to today. **This is
  the reason to prefer A=256 when raising bucket count**: it holds the memory footprint
  constant while doubling the buckets.
* `HalfKP-hm @ 256` = **21.0 MB**, better still.

Storing W1 as float32 in the npz instead of fp16 would double the disk cost and change nothing
resident; keep `--half`.

### Import time — not a constraint

Measured here: loading and casting the current 12.58 MB fp16 W1 takes **20.5 ms** (best of 3).
That is ~1.6 ms per MB, so even a 50 MB fp16 W1 adds **<100 ms** to a 31.7 s idle import. The
numba kernels are unchanged in shape by any row above — `ACC_SIZE` is a compile-time constant
inside the closure and the loop bodies are identical — so **no row in this table costs
measurable init time.** The brief's worry that "feature-set changes cost import time" does not
apply to bucket-count or width changes. It *would* apply to a mirror (one extra XOR per feature
index, no new specialisations, so still ~0) and to a new activation (a real recompile).

---

## Task 2 — the quality argument

### Measured here (this repo, our games, our gauntlets)

| observation | value | source |
|---|---|---|
| kz8 -> kz16 (2x king buckets) | **+31 Elo**, PROMOTE over 524 games | `072-kz16.gauntlet.log` |
| kz16 -> kz32 (2x again) | val 0.004690 vs 0.004659, suite 8.1 vs 7.4, **INCONCLUSIVE 52.2% / 600** | `train-kz32b.log`, `059-kz32b.gauntlet.log` |
| kz16 training scale | 4 shards x ~145 M = **436 M unique**, 18 epochs | `train-kz16.log` header |
| kz32 training scale | 3 shards x ~145 M = **436 M unique**, 24 epochs, epoch-1 train/val gap **+0.000541** | `train-kz32b.log` header |
| 1024 wide vs 256 wide | ~**-35 Elo**, two runs, at 21.6 M positions | JOURNAL 31 Aug |
| 256 -> 512 wide | **+89 +/- 49 over 96 games**, PROMOTE — *confounded* with 21.6 M -> 145 M, min-ply 16, balancing off | JOURNAL, `021-w512-150m` |
| 9.1 M -> 62.5 M positions | **+151 Elo** | `relabel.py` docstring |
| mirroring pilot | symmetrising costs **58%**; still **38%** above control at epoch 2 of 6 on 40 M | NOTES 6 Sep 10:40 |
| 12 endgame-dense heads pilot | val 0.004458 vs control 0.004481 — **0.5%**, inside noise | NOTES 6 Sep 10:40 |
| champion's broken mirror symmetry | a position and its file-mirrored twin score **56.2 cp apart** (median 37.3, p90 130.4) against mean \|eval\| 388 cp | NOTES |
| val loss as an Elo predictor | kz16 beat kz8 on val by 0.9% and won +31; a 15%-better-loss net measured ~0 | NOTES, JOURNAL |
| SF corpus size | 8 x ~72.4 M + 1.3 M = **580.7 M** | `train-sf.log` header |
| training throughput | **0.29-0.30 M pos/s**, ~250 s per 72.5 M shard | `train-sf.log` |

### The one argument that decides it

The kz8/kz16/kz32 sequence is a *scaling curve on king-bucket count* and it **turned over
between 16 and 32 at 436 M positions**. Convert it to density:

| configuration | positions per king bucket | outcome |
|---|---|---|
| kz8 @ 436 M | 54.5 M | beaten by kz16 |
| **kz16 @ 436 M** | **27.2 M** | **won, +31 Elo** |
| **kz32 @ 436 M** | **13.6 M** | **lost / inconclusive, train-val gap +0.00054 at epoch 1** |

Now apply the corpus we actually have:

| design | buckets | Mpos/bucket @ 1.16 B | Mpos/bucket @ 581 M (SF only) | verdict against the 27.2 M line |
|---|---|---|---|---|
| current kz16 | 16 | 72.5 | 36.3 | comfortably fed |
| kz16 mirrored | 16 | 72.5 | 36.3 | comfortably fed |
| **32 buckets (kz32 / HalfKA-hm / HalfKP-hm)** | **32** | **36.2** | **18.2** | **clears it on the full corpus, misses it on SF alone** |
| **HalfKP / HalfKA (64 buckets)** | **64** | **18.1** | **9.1** | **misses it; nearer the losing config than the winning one** |

To give 64 buckets the density that the *winning* configuration had, we would need
`64 x 27.2 M = 1.74 B` positions. We have 1.16 B. **The re-examination the brief asked for
therefore returns: the size objection was wrong, the data objection was right, and it is still
right — but only for 64 buckets. For 32 it has flipped.**

Two honest caveats on that arithmetic, both against the change:

* The 1.16 B is not 1.16 B of one thing. 581 M is Stockfish self-play (`data/sf/`), the rest is
  Lichess, and the Lichess half is the *same* four or five 145 M shards the kz32 run already
  used. The genuinely *new* data since the kz32 verdict is the 581 M SF corpus, which on its
  own gives 32 buckets 18.2 M/bucket — **below** the 27.2 M line.
* Positions are not independent. Consecutive positions in a game share almost everything;
  effective sample size per bucket is well under the nominal count, and equally so at every row
  of the table, so the *ranking* survives but the absolute threshold may be optimistic.

### Feature set vs width, at ~1 B — what the literature says (IMPORTED, not measured here)

Stated separately because none of it was measured in this repo and none of it was run under our
constraints (Python/numba, batch 1, no SIMD int8):

* Every strong modern NNUE uses king-indexed features (HalfKP, then HalfKAv2, now HalfKAv2_hm)
  rather than plain 768, and horizontal mirroring of the king index is standard. Small engines
  mirror their king buckets as a matter of course; **our unmirrored 16-zone map is the unusual
  choice, not the mirrored one.**
* Those results were obtained at data scales far above 1.16 B (Stockfish's nets are trained on
  many billions of positions, often multiple passes) and with int8/SIMD inference where a wider
  first layer is nearly free per node. Neither condition holds here: our first layer is float32
  numpy/numba at batch 1, so **width is not free — it is a linear tax on 39.8% of node time.**
* The literature's own ablations consistently report king-bucketing as worth more than raw
  accumulator width *at a fixed parameter count* — which is the trade the A=256 rows encode.
  This is the strongest imported argument **for** the change, and it is imported, so it does not
  override the measured turnover in the previous section.

The net import: the literature says the feature set is right and we are the outlier; our own
measurements say the specific step to 64 buckets is 1.5x short of the data it needs. Both can be
true, and they are: **32 buckets is where those two lines cross for us.**

### Why the static error is 331.9 / 262.6 / 184.4 cp — feature set, capacity, data, or targets?

**Mostly none of the above. Mostly the instrument.** `testing/eg_calib.py` scores 400 positions
from `overnight/eval/endgame_suite.json`, which `testing/endgame_suite.py` builds by sampling
5-16 piece positions **from our own lost games** and labelling them with **Stockfish at depth
18**. Computed here from `eg_calib.npz`:

| band | n | mean \|label\| | median \|label\| | p90 \|label\| | mean error | correlation with label |
|---|---|---|---|---|---|---|
| 5-8 | 114 | **860.5** | 572.5 | 2000.0 | 331.9 | **0.937** |
| 9-12 | 135 | 535.9 | 412.0 | 2000.0 | 262.6 | **0.901** |
| 13-16 | 151 | 404.6 | 292.0 | 779.0 | 184.4 | **0.895** |
| all | 400 | 578.8 | — | — | 252.8 | — |
| all, both sides clipped to +/-1000 cp | 400 | — | — | — | **130.1** | — |

Read that carefully:

1. **The comparison to "30-60 cp" is invalid.** Those figures describe static evaluation error
   against quiet references on roughly balanced positions. These labels are depth-18 *search*
   scores on positions selected for being hard, with a mean magnitude of 578.8 cp and a p90 that
   is a clamped mate score. No static evaluator scores KRP-vs-KR to 50 cp of a depth-18 search —
   not ours, not Stockfish's own. **331.9 cp of error against an 860.5 cp mean label at 5-8
   pieces is a 39% relative error on a distribution where the truth is frequently "+2000 or
   0".**
2. **The rank ordering is fine.** Correlations of 0.90-0.94 say the net orders these positions
   nearly correctly; what it gets wrong is the *magnitude* of decisive scores, which alpha-beta
   largely does not care about.
3. **The bands are also selection-biased.** The positions are sampled from *our own games*, and
   from the phase we lose in, so the suite is by construction the set of positions the current
   net handles worst.
4. **What error remains is a targets-and-distribution problem, not a feature-set one.** The
   evidence is `kz16w`: endgame loss weighting *improved* held-out loss on eq<=16 (7.20 ->
   6.08 x 1e-3) and *worsened* suite move quality (7.4 -> 9.1 cp, all of it in the 9-12 band).
   Human-endgame loss and engine-endgame move quality point in opposite directions. A richer
   first layer does not fix a target that points the wrong way.
5. **The one place a feature set could genuinely help is king safety and king-pawn geometry**,
   which is exactly what king buckets encode — and that is the argument for 32, not for 64.

Capacity is the least likely culprit: at 6.55 M parameters and val loss flat at 6.2-6.3 x 1e-3
across 2-20 pieces (`train-kz16r.log` strata), the error is uniform across the whole
non-trivial range rather than concentrated where capacity would bind.

---

## Task 3 — engineering cost

### The finding that changes the arithmetic

The engine's king-zone plumbing is **fully general and fully data-driven**:

* `agent.py:116` — `KING_ZONES = int(W1.shape[0]) // FEATURES`
* `agent.py:108` — `ACC_SIZE = W1.shape[1]`
* `agent._zone` already has maps for 1, 4, 8, 16 **and 32** zones
* `fastboard.zone_of` (line 637) already has maps for 1, 4, 8, 16 **and 32** zones
* `training/features.king_zone` already has 1, 4, 8, 16, 32 (and a mirrored 16)
* every kernel loop bounds on `white.shape[0]` / `out.shape[0]`, never on a literal
* `fastboard.make_full` already detects a king move crossing a zone and calls `rebuild` for that
  perspective; `fastsearch.make` (line 585) mirrors it; `unmake` restores zones from the undo
  stack; `fastsearch.sync_acc` already recomputes offsets from `zones[]`
* `export.expected_shapes` already takes `king_zones` and `accumulator` as arguments

**Consequence: `32 king zones at accumulator 256` in the current 768 feature set requires zero
edits to any file.** It is `train.py --king-zones 32 --accumulator 256`, then `export.py --half`.
The .npz shape alone reconfigures the engine. That is not a claim about how easy it *would* be —
it is what the code already does.

### What each alternative actually costs

| change | files touched | new hot-path logic | warm start? | gate work |
|---|---|---|---|---|
| **32 zones, A=256, current 768 set** | **none** | **none** | no (A changes) | check_nnue, endgame suite, 1 SPRT |
| 32 zones, A=512 | none | none | **yes** — kz32 refines kz16 (`expand_zones`) | same |
| 64 zones (= HalfKA), A=256 | 3 (`features.king_zone`, `agent._zone`, `fastboard.zone_of`) — each one added branch returning `square` | none | no | same + re-run `check_fastsearch` |
| HalfKP (drop king planes, 641/bucket) | the 3 above **plus** `features.indices`, `features.feature_index`, `features.black_from_white`, `agent._feature`, `fastboard.feature`, `pack.py`'s index space, `check_features`, `check_pack`, `check_nnue` | index-space remap in the innermost function | no | full re-verification of the packed corpus |
| **mirroring** (any bucket count) | NOTES' measured count: **9 files**, hot path | **yes** — a per-perspective flip carried through `feature`, `_acc_row`, `_acc_row_one`, `rebuild`, `make_full`, `unmake_full`, `sync_acc`, and stored in the undo stack | **no** — pilot: 58% penalty, 38% adrift at epoch 2/6 | same + `check_fastsearch --depth 4 --random 30` exactness, + `check_nnue` extended to compare mirrored indices |

**HalfKP is strictly worse engineering than HalfKA for us.** Its only advantage is 641 vs 768
rows per bucket — 16.7% of W1, i.e. **2.08 MB at A=256** — and it costs a rewrite of the index
space that every packing and checking tool depends on. Our scheme with an identity zone map
*is* HalfKA (64 x 768 = 49,152) and needs three one-line branches. If full king-square
resolution is ever taken, **take HalfKA, not HalfKP**, and pay 2 MB we demonstrably have.

### The incremental accumulator under full king-square features

Concretely, what changes in `fastboard.make_full`:

```
if piece == 5 and king_zones > 1:                     # already there
    new_zone = zone_of(to if us == 0 else to ^ 56, king_zones)
    if new_zone != zones[us]: ...                     # already there -> rebuild()
```

With `zone_of` = identity, `new_zone != zones[us]` is true for **every** king move instead of
79.8% of them. `rebuild` already exists, is already compiled, and is already called on this
path. Nothing new is written. The measured cost of the extra 20.2% of king moves is the
**+1.74 pp / ESTIMATED +0.8% node time** from task 1.

The one subtlety: `LAZY_ACC` defers the accumulator to the first `evaluate()` on a line, and
`fastsearch.make` (line 585) checks for a zone change to decide whether the deferral is safe.
With an identity map that check fires on every king move, so more nodes take the eager path.
`fastsearch.sync_acc` handles it; no new code, but it is the line to re-read if a change is made.

### How the gates apply

* `training/check_nnue.py` §1 compares `agent._feature` to `features.feature_index` over 1,536
  cases — **unchanged** by a bucket-count change, **must be extended** for a mirror (it passes
  no `flip`, so a mirrored engine would pass a check that proves nothing).
* `check_nnue.py` §1b compares `agent._zone` to `features.king_zone` square by square for
  `agent.KING_ZONES` — **already covers 32 and would cover 64** the moment both sides gain the
  branch. For a mirrored net it needs the `mirrored=True` argument threading through, which it
  does not have today.
* `check_nnue.py` §2 checks the incremental accumulator against a full rebuild after every ply
  over 6,000 plies including promotion, en passant and both castlings — **this is the gate that
  would catch a broken refresh**, and it is exactly the gate a mirror needs most, because a
  mirror changes the *feature indices themselves* mid-line when the king crosses the d/e file.
* `testing/check_fastsearch.py` must stay exact (score and node count identical to the Python
  reference at depths 1..N with the TT off). A bucket-count or width change **cannot** break it:
  no kernel source changes, and both implementations read the same .npz. A mirror **can** break
  it, in both files at once, and would need the `--random 200` run rather than `--random 30`.
* `export.py` already asserts every shape and already verifies the numpy head against torch to
  1e-3. Unchanged.

### Risks

1. **Cache.** Any row with resident W1 above ~25 MB is unmodelled. `32z @ 512` is 50.3 MB;
   `HalfKA @ 512` is 100.7 MB. The linear speed model in task 1 will over-predict for these and
   the error could be large — the discarded synthetic bench hinted at 0.53x for a 1024-wide net,
   which is far worse than linear. **Mitigation: only take bucket increases at A <= 256, where
   resident memory is flat or falling.**
2. **From-scratch shortfall.** Changing A breaks the warm start. The champion is the end of a
   60+ epoch chain. One night at 0.29 M pos/s is ~3.5 B presentations, which is a real budget,
   but it is not the same budget.
3. **Val loss cannot adjudicate this.** Established twice here. The only instrument that can is
   an SPRT, and slots are the scarce resource.
4. **A silent index mismatch has no symptom.** `features.py`'s own docstring says it: the net
   loads, the engine runs, the crash gate passes, and it plays badly. Any mirror work must be
   gated by `check_nnue` **extended**, not `check_nnue` as it stands.

---

## Task 4 — verdict, timeline, falsifier

### Is a feature-set change achievable in the window?

**A true feature-set change (HalfKP or HalfKA, 64 buckets) is not, and would not be right even
with more time.** Achievability: HalfKA at A=256 is 3 small edits + a from-scratch run + a full
re-gate — call it one engineering day, one training night, one gating day, on a **single**
laptop that is also the only gauntlet machine, with the freeze on 10 Sep. That is ~3 of the 4.5
days left, for a design our own scaling curve says is **1.5x short of the data it needs**
(18.1 M/bucket against the 27.2 M that won and the 13.6 M that lost). Wrong bet.

**Mirroring is not achievable either**, and for a better reason than time: it was piloted on
6 Sep, it starts 58% behind, it was 38% adrift at epoch 2 of 6, it cannot warm-start, and its
engine side is the 9-file hot-path change with a gate (`check_nnue`) that does not yet test what
it would need to test. Its mechanism is real — 56.2 cp of learned asymmetry in a feature set with
no castling features is pure noise — but recovering it needs a from-scratch-scale run *and* the
riskiest edit on the list. Not this week.

### THE ONE RECOMMENDATION

> **Train one net: 32 king zones, accumulator 256, current 768 feature set, from scratch on the
> 581 M Stockfish corpus with WDL-blended targets. Ship it as a single challenger. Touch no
> engine file.**

Why this and not the others:

| property | value |
|---|---|
| engine files changed | **0** — every map and width already exists and is read from `W1.shape` |
| parameters | 6.42 M vs 6.55 M today — **a reallocation, not a growth** |
| king resolution | 2x (32 buckets vs 16) |
| head MACs | **16,384 vs 32,768** |
| node rate | ESTIMATED **1.248x -> +10.2 Elo at 120 s, banked before the net proves anything** |
| net on disk | **13.11 MB vs 13.64 MB — smaller**; 27.45 MB unpacked, no book sacrifice |
| resident W1 | 25.2 MB — **identical to today**, so no cache cliff |
| data per bucket | 36.2 M at 1.16 B — **above the 27.2 M the winning kz16 run had** |
| refresh rate | unchanged from today (32 zones, not 64) |
| gates needed | `check_nnue`, endgame suite (18 min), one 8 s SPRT. **No `check_fastsearch` risk.** |

It is the only design in the table that is simultaneously (a) structurally different from what
we ship, (b) cheaper on disk **and** per node, (c) above the measured data-density line, and
(d) free of engine-code risk in a week where an engine bug is unrecoverable.

**De-risking the from-scratch requirement (ESTIMATE, untested, worth 30 minutes):** the width
change is what forbids the warm start, and it need not. `h1 = clip(x)^2` and the head is linear
in `h1`, so **dropping accumulator units is exact if the matching `W2` rows are dropped with
them**. Initialise the A=256 net from the champion by selecting the 256 accumulator units with
the largest head contribution (`sum_k |W2[k, i, :]| * std(acc_i)`), taking `W1[:, idx]` for each
of the 16 parent zones expanded to 32 by `expand_zones`, and `W2[:, idx, :]` plus
`W2[:, A+idx, :]`. That is a structured prune, not a random init, and should land far above
scratch. If it does, this becomes a **fine-tune, not a from-scratch run**, and the whole task
fits in an hour.

### Timeline

| when | what | machine |
|---|---|---|
| 6 Sep evening, ~30 min | write the prune-init helper in `training/train.py` (new flag, off by default; unmirrored path stays bit-identical, as `--mirror` already proved is achievable) | CPU |
| 6 Sep, ~2 h | launch: `--king-zones 32 --accumulator 256 --data data/sf/feb24_0{0..7}.npy --wdl-lambda <the value 149-v94wdl used>`, prune-init from the champion; **fall back to from-scratch if the prune-init's epoch-1 val is worse than a scratch epoch-1** | GPU (free) |
| overnight 6->7 Sep, ~3-4 h | 5+ passes over 581 M at 0.29 M pos/s = ~2,000 s per pass | GPU |
| 7 Sep morning, ~10 min | `export.py --half`, `check_nnue` (feature indices, 32-zone map, 6,000-ply accumulator, torch/numpy head) | CPU |
| 7 Sep morning, 18 min | endgame suite — **veto only**, per `network.md`: it caught kz16w's 15.1 cp catastrophe and mis-ranked kz16, so use it to reject disasters, not to rank | CPU |
| 7 Sep, 1-3 h | one 8 s SPRT vs the v9.5 champion, `--checkpoint` every 200 games. **>= +10 at 200 promotes; <= -10 at 400 rejects; otherwise 200 more.** | laptop, one at a time |
| 7 Sep evening | if PROMOTE: clocktest (10 min) + 40 games at 120 s (45 min), build the zip, `notify --candidate` | laptop |
| 8-10 Sep | **the remaining slots go to search and to conversion, not to nets** | laptop |

That is **one** net slot, finished 3 days before the close, leaving the rest for the thing the
leader benchmark says actually costs us the match.

### What I would rather you spent the week on, stated once

`leader_benchmark.md` measures the gap as **conversion**: their draw rate 7% vs our 26%, 0
threefold and 0 fifty-move draws against our 4 and 2, 22 of 24 decided games ending in
checkmate. Converting at their rate is ~8 extra wins in 42 games, on the order of **+130 Elo** —
more than every net change in this document combined, and more than the +10 Elo of speed the
recommendation banks. The net task above is worth doing because it is nearly free and the GPU is
idle. It is not where the 300-500 Elo is.

### The falsifier — what would have to be true for me to be wrong

I am wrong, and a real feature-set change is the biggest available gain, if **any** of these
turns out to hold:

1. **The density line is wrong.** If the recommended `32 zones @ A=256` net loses its SPRT
   *while* its held-out loss on the SF validation set is clearly better than the champion's,
   then bucket count is not data-limited in the way I have modelled and the turnover at kz32 was
   caused by something else (the epoch-1 train/val gap of +0.00054 suggests over-fitting on
   repeated shards, which more *unique* data fixes). In that case 64 buckets deserves the next
   slot, as **HalfKA (3 one-line branches), never HalfKP**.
2. **Width was carrying more than I credit.** If `32z @ A=256` loses by more than ~15 Elo, the
   confounded +89 of `021-w512-150m` was mostly width after all, the reallocation is a bad
   trade, and the right move is `32z @ A=512` (26.22 MB, 40.56 MB unpacked, still legal,
   warm-startable from the champion in ~1 h) — which tests bucket count with width held fixed.
3. **The resident-memory model is backwards.** If a 50 MB-resident W1 measures *no* slower than
   today's 25 MB, then `32z @ A=512` and `HalfKA-hm @ A=512` are both free on speed, the whole
   "reallocate width to buckets" framing collapses, and the right move is simply more buckets at
   full width. A 20-minute bench of the exported `32z @ 512` net against the champion at fixed
   nodes settles this and should be run first if a slot is cheap.
4. **The mirror is worth its 58% penalty.** If someone runs the mirrored pilot to 6 full epochs
   on the SF corpus rather than killing it at 2 on 40 M, and it crosses below control, then
   `HalfKA-hm 32 x 768 @ 256` (13.11 MB, same 1.248x, 36.2 M/bucket, one canonical king square
   per bucket) is strictly better than my recommendation and the 9-file surgery is justified —
   but only if it lands before 8 Sep, because it needs `check_nnue` extended and
   `check_fastsearch --random 200` before it can be trusted.
5. **The 331.9 cp really is a feature-set signal.** If a net trained on identical data with 32
   buckets shows no improvement in the 5-8 band of `eg_calib` while a *search* change does, my
   reading of that instrument as label-magnitude artefact is wrong — but note that the fix
   indicated would still be targets and search, not HalfKP.

None of these is settled by validation loss. All of them are settled by games.
