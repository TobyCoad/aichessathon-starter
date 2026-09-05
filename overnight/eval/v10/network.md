# v10 -- the evaluation network

Written 5 Sep ~16:00 by a read-only research pass. Nothing here was trained, run or
committed; every number below is either quoted from an existing log in this repo or
derived arithmetically from one, and derived numbers are labelled as estimates.

## Summary (10 lines)

1. Validation loss has stopped being a useful instrument. kz16 beat kz8 by 0.004659 vs
   0.004700 and won +31 Elo, but scored *worse* on the endgame suite (7.4 vs 7.0 cp).
   The 1024-wide net had 15% better loss and measured ~0 Elo. Judge nets by games.
2. The endgame problem is not data volume: 25.7% of the corpus is <=16 pieces (186M of
   725M positions) and 5.4% is near-equal-and-<=16 (39M). Volume is not the constraint.
3. It is not loss weighting either: the weighted fine-tune (kz16w) *improved* held-out
   loss on eq<=16 (7.20 -> 6.08 x1e-3) and *worsened* the suite (7.4 -> 9.1 cp), all of
   it in the 9-12 band (9.9 -> 15.1). Human-endgame loss and engine-endgame move quality
   point in opposite directions. That is a distribution problem, not an objective one.
4. The one feature-set change with a mechanism that fits our own measurements is
   **horizontal mirroring of the king buckets**: 8->16 zones was +31 Elo, 16->32 was
   worse (val 0.004690, suite 8.1) -- consistent with data-per-zone being the binding
   constraint. Mirroring doubles king resolution *and* data per weight for 0 MB and ~0 ns.
5. Width is a trap: 512->768 costs an estimated 12-18% node rate (-20 to -30 Elo) before
   the net has proved anything, and the only width test we own (-35 Elo) was run at 21.6M
   positions with no search package. Do not spend a gauntlet slot on it before 11 Sep.
6. Distillation is dominated: our labels are already Stockfish at 1.5M nodes (~depth 21,
   measured in training/relabel.py). Any teacher we can train is *less* accurate than the
   labels, so a student distilled from it is capped below what it already has.
7. Engine-distribution data is the right long answer and is unaffordable this week:
   measured 280k positions per 49 min on 6 desktop workers, and the desktop is the
   machine that produces verdicts. Also note the pilot's `--nodes 5000` labels are far
   weaker than the corpus and must not be mixed in at weight.
8. The binding constraint is not GPU time (a 145M-position epoch is 4.4 min idle, so any
   of these trains overnight) -- it is **gauntlet slots**: about three full net verdicts
   fit before the 10 Sep freeze, one of which is already spent on 104-kz16r.
9. Recommended: bundle mirroring + rebalanced output buckets into one v10 net, gate it as
   one challenger, and keep the endgame-shard rotation as the second net if the first
   passes. Everything else on the list is either negative-expectation or out of time.
10. One process gap to close first: there is **no endgame-suite baseline for the v8 tree**.
    Every suite number we own was measured under v5.5/v6 search. 104-kz16r's suite result
    is uninterpretable until the v8 champion is run through the suite once (~18 min CPU).

---

## What the evidence in this repo actually says

### Net changes measured so far

| change | val loss | endgame suite | gauntlet | source |
|---|---|---|---|---|
| kz8 (b8, 4 shards) | 0.004700 | 7.0 cp (5-8: 4.8, 9-12: 8.9, 13-16: 7.0) | champion at the time | `train-kz8c.log`, JOURNAL 5 Sep 00:06 |
| kz16 | **0.004659** | 7.4 cp (6.0 / 9.9 / 6.1) | **PROMOTE +31 +/- 23 over 524** | `train-kz16.log`, `suite-kz16.log`, `072-kz16.gauntlet.log` |
| kz32b | 0.004690 | 8.1 cp (7.0 / 9.8 / 7.5) | INCONCLUSIVE 52.2%/600 | `train-kz32b.log`, `suite-kz32b.log` |
| kz16w (endgame-weighted) | 0.004648 *weighted* | **9.1 cp** (5.2 / **15.1** / 6.7) | stopped ~44 games | `train-kz16w.log`, `suite-kz16w.log` |
| b1 (one output head) | 0.004977 | -- | closed on val loss | NOTES |
| endgame-shard fine-tune (kz8-eg) | 0.005898 = its own warm start | -- | never beat step 0, early-stopped at epoch 4 | `train-eg.log` |
| 1024-wide, 21.6M positions | 0.005545 vs 256's 0.005411 | -- | ~-35 Elo, two runs | JOURNAL 31 Aug |
| 9.1M -> 62.5M positions | -- | -- | **+151 Elo** | `relabel.py` docstring |

Three facts follow that shape everything below.

* **kz16 won the gauntlet while losing the suite.** The suite is a real instrument but it
  is not a proxy for Elo. Use it as a veto on catastrophes (kz16w's 15.1 in the 9-12 band),
  not as the ranking signal.
* **Held-out loss is close to useless at this margin.** 15% better loss bought ~0 Elo once.
  Differences of 1-2% in val loss between the candidates below are noise for our purposes.
* **Volume is the only axis with a measured large slope**, and it was measured over a 7x
  step at tiny scale. 725M -> 870M is +20% and will not repeat +151.

### Where the error actually is

Stratified held-out loss for the champion kz16 net (`train-kz16.log`, x1e-3):

```
2-8: 6.24  9-12: 6.28  13-16: 6.34  17-20: 6.31  21-24: 4.90  25-28: 3.13  29-32: 1.87
eq<=16: 7.20                                                          eq29-32: 1.04
```

Read this carefully: loss is **flat at 6.2-6.3 from 2 to 20 pieces** and only falls above 21,
where positions are near the start and material dominates. The "endgame is worse" reading
is really "everything below 21 pieces is worse". The near-equal <=16 cell is 6.9x the
near-equal 29-32 cell, which is the diagnosis the sample weighting was built for -- and
that weighting made move quality worse.

Absolute |error| in cp (JOURNAL, `bias.py` on the validation shard):
29-32: 50, 25-28: 72, 17-20: 151, 13-16: 194, 9-12: 290, 2-8: 684. Part of the low-count
blow-up is a sigmoid artefact (a 600 cp error at +1500 is a tiny probability error), which
an earlier net review already flagged -- but the 9-12 band at 290 cp is not decided
positions, and the suite agrees that 9-12 is where the moves go wrong.

Qualitatively, the post-mortems repeat one failure: *exchange-imbalance and king-activity
positions are overvalued when the attack has no follow-up*, and more time does not fix it
(JOURNAL rounds 4, 8, 9). That is a first-layer expressiveness problem conditioned on king
placement -- which is exactly what king buckets address and why 8->16 was worth +31.

### Budgets

* **Disk**: unpacked zip 27.9 MB of 50 (net 13.64 + book 9.76 + syzygy 4.40 + source).
  Headroom **22.1 MB**. Net sizes computed for every option below.
* **Import**: 31 s locally, ~45 s allowed (their box ~1.8x slower, 60 s budget in AGENTS.md).
  Numba compilation dominates; a larger `W1` adds ~0.2 s. Not a constraint for anything here.
* **GPU**: RTX 5070 Laptop. Measured epoch throughput on 145M-position shards:
  **0.55-0.67M pos/s idle (4.0-4.4 min/epoch)**, 0.26-0.31M contended (8-9 min/epoch).
  `train-kz16r.log` is running now at 0.47M pos/s, 311 s/epoch, 20 epochs -> free ~17:00.
* **CPU / gauntlets**: this is the real constraint. One net verdict = export + `check_nnue`
  + endgame suite (18 min) + SPRT[0,20] at 8 s (~500 games) + 40 games at 120 s. The laptop
  and the desktop are both already queued out. **~3 net verdicts fit before the 10 Sep
  freeze, and 104-kz16r is one of them.**
* **Self-play generation** (measured, `gen-001/002`): 6 workers -> 280k positions / 49 min
  with `--nodes 5000`. Labelling is only ~5% of that loop (SF at 5k nodes is ~3 ms against a
  ~4.4 s self-play game), so raising labels to 100k nodes costs only ~0.6x throughput --
  roughly **460k positions/hour on 12 desktop workers**.

---

## Ranked table

Elo columns: **E8** = expected Elo at 8 s vs the current champion; **E120** = at 120 s, taken
as ~half of E8 per NOTES ("120 s gains are about half of 8 s gains"). Ranges are honest
spreads, not confidence intervals. "knps" is the estimated node-rate change.

| # | Idea | E8 | E120 | GPU | Disk | knps | Risk | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | **Mirrored king buckets** (16 zones over the a-d half-board; own-king file mirror) | +10..25 | +5..12 | 0.8-1.5 h (warm start) | **0 MB** (13.64) | ~0 | med (hot path, 9 files) | **DO** |
| 2 | **Rebalanced output buckets** (12 non-uniform piece-count heads, endgame-dense) | +0..10 | +0..5 | 0.6-0.9 h (warm start) | +0.5 MB | 0 | low | **DO (bundle with 1)** |
| 3 | **Fifth month of data** (104-kz16r) | +0..10 | +0..5 | running | 0 | 0 | low | **IN FLIGHT** |
| 4 | **Endgame shard in rotation** (1 of 6 shards, not a fine-tune) | -10..+15 | -5..+8 | 0.9-1.5 h | 0 | 0 | med | **TRAIN, gate only if the suite holds** |
| 5 | Second hidden layer in the head, 32->32->1, identity warm start | +0..10 | +0..5 | 0.7 h | +0.03 MB | -3% | low | train if a slot frees |
| 6 | Sixth month (2024_12, parquet already on disk) | +0..8 | +0..4 | 1 h pack + 1 h train | 0 | 0 | low | only if 3 shows a gain |
| 7 | Pairwise-multiplied L1 (head input 2A -> A) | +20..35 speed, -0..25 accuracy | net -5..+15 | 2.5-3.5 h **from scratch** | -0.5 MB | **+12..20%** | med-high | right idea, wrong week |
| 8 | Mirrored 32 zones (2x resolution again on top of #1) | +0..15 | +0..8 | 1-1.5 h | +12.6 MB (40.5 total) | -2..-5% | med | only if #1 passes and a slot exists |
| 9 | Curated 5-man Syzygy subset (~15-20 MB) | +5..10 | +3..6 | 0 | +15..20 MB | 0 | low | competes with #8 for the headroom; not a net change |
| 10 | Width 512 -> 768 or 1024 | -30..+10 | -15..+5 | 3-4 h from scratch | +7..14 MB | -12..-25% | high | **NO** before 11 Sep |
| 11 | Engine-distribution data at scale (self-play + SF labels) | +10..40 *if you had 3 weeks* | -- | -- | 0 | 0 | high | **NO** -- costs the verdict machine |
| 12 | Knowledge distillation from a big teacher | ~0 | ~0 | 8 h | 0 | 0 | -- | **dominated, do not build** |
| 13 | int8/int16 quantised inference in numba | negative | negative | 0 | -6 MB | negative | -- | measured slower twice; closed |
| 14 | WDL / game-result lambda mixing | n/a | n/a | -- | -- | -- | -- | corpus has no result column; unavailable |
| 15 | HalfKP / HalfKA full king-square features | n/a | n/a | -- | 40960 inputs | -- | -- | over the size cap and data-starved; closed in V7_PLAN |

### Notes behind the rankings

**#1 mirroring -- why it is top.** Our own two measurements bracket it. 8->16 zones bought
+31 Elo; 16->32 lost (val 0.004690 vs 0.004659, suite 8.1 vs 7.4). The natural reading is
that resolution helps until data-per-zone runs out, and 32 zones ran out. Mirroring is the
only move that raises resolution *without* dividing the data: 16 zones over the 32 squares
of the a-d half-board is 2 squares per zone instead of 4, and every zone sees both wings'
positions. Disk is unchanged and inference costs one XOR per feature index. The counterpart
in strong engines is HalfKAv2_hm ("hm" = horizontal mirroring), and small engines mirror
their king buckets as a matter of course; our 16-zone map is the unmirrored form, which is
the unusual choice, not the mirrored one.

Two caveats stated honestly. (a) Mirroring assumes the position's value is left-right
symmetric. It is not exactly -- castling rights are asymmetric -- but the 768 feature set
does not encode castling rights at all today, so nothing is lost that the net currently has.
(b) The gain is not guaranteed to survive the "15% better loss -> 0 Elo" law; it must be
judged by SPRT, not by val loss.

**#2 rebalanced buckets -- why it is nearly free.** `bucket_of` is `(count-1)*B//32`, so the
eight heads are equal 4-piece bands: 1-4, 5-8, ..., 29-32. Half the corpus sits in 21-32 and
gets two heads; the 9-16 band where the moves go wrong gets two. Only one head is ever
evaluated, so more heads cost **nothing at inference** and 0.5 MB on disk. This is the
cheapest targeted intervention available and it has never been tried (b1 vs b8 was tried;
the *placement* of the boundaries was not).

**#4 endgame shard -- the untried variant.** `data/positions_endgame.npy` already exists:
108M positions, mean 15.4 pieces, 70% at <=16 (I checked its validation shard). It has been
used exactly once, as an *exclusive* fine-tune at lr 3e-4 (`train-eg.log`), where it never
beat its own warm start and early-stopped at epoch 4 -- a textbook catastrophic-forgetting
signature, not a verdict on the data. Rotating it as one shard in six at lr 1e-4 is a
different experiment and costs nothing but GPU time. The risk is that it fails the way
kz16w failed (better human-endgame loss, worse engine-endgame moves); the endgame suite is
the gate that catches that, and it is 18 minutes.

**#7 pairwise -- the arithmetic, and why not this week.** The head is `32 x 2A` = 32,768 MACs
per evaluation at A=512; the accumulator update is ~2-4 rows of A plus a 2A snapshot. Pairing
the accumulator (`h[i] = clip(x[i]) * clip(x[i+A/2])`, per perspective) halves the head input
to A=512 and the head to 16,384 MACs. Taking the measured split (evaluate 29.4% + accumulator
15.4% of node time, measured at A=256 and larger now at A=512) and the measured compiled
`evaluate` cost of 1.58 us at A=256, the head is roughly a third of node time at A=512, so
halving it is an estimated **+12..20% knps**. At the repo's own ~120 Elo per node doubling
that is +20..35 Elo at 8 s. The problem is that an activation change cannot warm-start: the
champion is the end of a 60+ epoch chain (kz8 -> kz16 -> kz16r), and a from-scratch net has
one night to catch up. If it lands at val 0.0048 instead of 0.00466 the speed gain and the
accuracy loss cancel and the SPRT cannot resolve it. Build it if the schedule slips; do not
lead with it.

**#10 width -- why not, despite the journal's own invitation.** JOURNAL 31 Aug says width
should be "re-tested after the search package lands", and that is fair: the -35 Elo result
was measured at 21.6M positions with no RFP to consume evaluation quality. But: no warm
start, 3-4 h from scratch, +7-14 MB, and an estimated -12..-25% node rate that the net must
pay back *before* it earns anything. It is the single most expensive experiment on the list
and the only one with a measured negative prior. It is a good week-three project.

**#12 distillation -- why it is dominated, not merely unlikely.** `training/relabel.py`
records the decisive fact: the corpus labels are fishnet at a documented 1.5M nodes, median
depth ~21, and in a 200-position adjudication at depth 18 the corpus label was closer to
truth 197/200 against a depth-10 relabel. Our labels come from a teacher far stronger than
anything we can train. A 2048-wide student-of-those-labels is by construction *less* accurate
than the labels, so distilling the shipped net from it caps it below where it already sits.
The one non-dominated variant -- generate cheap self-play positions and label them with the
big net instead of with Stockfish -- transfers the teacher's endgame errors, which is our
weak spot. Do not build it.

**#11 engine-distribution data -- the honest cost.** This is the standard answer in the
literature and I believe it is the largest remaining lever on this net. The measurement that
kills it for this week is throughput plus opportunity cost: ~460k positions/hour with decent
(100k-node) labels on 12 desktop workers, so ~10M positions is a full day of the *desktop*,
which is the machine that produces SPRT verdicts. Ten million positions is also only 1.4% of
the corpus, so it can only be used as a low-LR fine-tune, and the closest thing we have tried
to a low-LR endgame fine-tune (`train-eg.log`) failed. Two things to record for later:
the existing gen-001/002 pilot (570k positions) is labelled at `--nodes 5000` ~ depth 11-12,
**worse than the corpus**, and must not be mixed in at weight; and the gauntlet runner should
start archiving PGNs (only 478 games exist in `overnight/pgn/`) so that engine-distribution
positions accumulate for free.

---

## Scoping the top four

### 1. Mirrored king buckets (`kz16m`)

**Design.** Each perspective computes `mirror = 1 if own_king_file >= 4 else 0` (own king seen
from its own side, so black's king square is `^56` first). When `mirror` is set, every feature
square for that perspective is additionally `^7`, and the zone is computed from the mirrored
king square. `zone_of` then only ever sees files a-d, so a 16-entry map covers 32 squares.
Suggested map (refines nothing, so no `expand_zones` path -- see the warm start below):

```
rank 0-1 : zone = rank*4 + file          ->  0..7    (8 zones, the two home ranks by square)
rank 2-3 : zone = 8  + (rank-2)*2 + (file>>1)   ->  8..11
rank 4-7 : zone = 12 + ((rank-4)>>1)*2 + (file>>1) -> 12..15
```

**Files that change** (9):

| file | change |
|---|---|
| `training/features.py` | `king_zone(square, zones, mirrored=True)`; `indices()` and `feature_index()` take a `mirror` flag and XOR the file |
| `training/train.py` | `zone_of` vectorised mirror; `Net.forward` derives `mirror` per perspective from the king index and XORs the square component of every feature index; new `mirror_from_unmirrored()` warm start |
| `training/export.py` | no shape change; write `mirrored: True` into the npz so the engine cannot load a mirrored net as unmirrored |
| `agent.py` | `MIRRORED` read from the npz; `_zone(square)` -> `_zone(square, mirror)`; `_feature(..., mirror)`; `Accumulator._king_zone` returns `(zone, mirror)`; `_rebuild`, `push`, `pop` carry the mirror alongside the zone |
| `fastboard.py` | `feature(s, code, white_pov, flip)` where `flip` is 0 or 7; `zone_of` mirrored; `rebuild`, `_acc_row`, `_acc_row_one`, `make_full`, `refresh` thread the flip; pack the mirror into the existing `U_ZONE_W`/`U_ZONE_B` undo slots as bit 8 so the undo stack layout is unchanged |
| `fastsearch.py` | `sync_acc` and `make_move` pass the flip (`zones` array grows from 2 to 4 entries) |
| `training/check_nnue.py` | feature-index and zone-map checks take the mirror argument |
| `training/check_features.py` | naive reference implementation gains the mirror |
| `weights/net.npz` | new file |

**Warm start (this is what makes it affordable).** Do not start from noise. Build the
symmetrised champion: for each mirrored zone `z`, let `a` be the unmirrored zone of a
representative a-d king square and `h` the unmirrored zone of its file-flipped partner, then

```
W1_new[z][piece, sq] = 0.5 * (W1_old[a][piece, sq] + W1_old[h][piece, sq ^ 7])
```

If the champion is already approximately left-right symmetric -- which it should be, having
seen both wings -- initial validation loss lands near 0.00466 and the run only has to improve
from there. Heads, `b1` and `W3` copy across unchanged. Implement this as
`--mirror-from <unmirrored checkpoint>` in `load_checkpoint`.

**Command** (after `104-kz16r` finishes, ~17:00, so `net_w512-b8-kz16r.pt` is the start):

```
.venv/Scripts/python.exe training/train.py \
  --data data/positions_w512-150m.npy data/positions_w512-150m-b.npy \
         data/positions_2025_02.npy data/positions_2025_03.npy data/positions_2024_11.npy \
  --val data/validation_w512-150m.npy \
  --resume training/checkpoints/net_w512-b8-kz16r.pt --mirror \
  --accumulator 512 --buckets 8 --king-zones 16 \
  --lr 2e-4 --epochs 14 --patience 6 --warmup-epochs 2 --skip-sanity \
  --out training/checkpoints/net_w512-b8-kz16m.pt
```

14 epochs x 145M at 0.5M pos/s = **~1.4 h idle, ~2.5 h contended**.

**Export / inference.** `python -m training.export --checkpoint ...kz16m.pt --out
<challenger>/weights/net.npz --half`, then `python -m training.check_nnue --agent <challenger>
--checkpoint ...kz16m.pt` (this is the gate that catches a mirroring bug: it compares 1536
feature-index cases, all 64 zone-map squares, and the incremental accumulator against a full
rebuild over 6000 plies biased toward promotions, en passant and castling), then
`python -m testing.check_fastsearch --depth 4 --random 30`.

**Gate.** val loss + strata; `testing.endgame_suite run --seconds 2.5` against a **freshly
measured v8 baseline**; SPRT[0,20] at 8 s vs the champion; 40 games at 120 s on platform
openings. Kill criterion: suite worse than the v8 baseline by more than ~1.5 cp *and* the
9-12 band worse -- that is the kz16w signature.

### 2. Rebalanced output buckets (bundle into the same net)

**Design.** Replace the arithmetic `bucket_of` with a 33-entry lookup table shared verbatim
by `training/train.py` and `agent.py`, e.g. 12 heads:

```
pieces  2-4 5-6 7-8 9-10 11-12 13-14 15-16 17-19 20-22 23-25 26-28 29-32
head     0   1   2    3     4     5     6     7     8     9    10    11
```

Half the heads now sit below 17 pieces, where all of the measured move-quality loss is,
and the crowded 20-32 range keeps four. Inference cost is unchanged (one head is selected);
disk goes 13.64 -> 14.15 MB.

**Files**: `training/train.py` (`bucket_of` + a head-expansion path in `load_checkpoint`),
`agent.py` (`_bucket`), `training/export.py` (`head_numpy`'s inline bucket arithmetic and
`expected_shapes`). `fastsearch.evaluate` reads the bucket from `meta`, so check whether the
index is computed there or passed in; if computed, it changes too.

**Warm start**: each new head copies from the old head whose band contains it, so the
expanded net is *numerically identical* to the champion at initialisation. That is worth
insisting on: it means the run cannot start behind.

**Recommendation**: train this **in the same run as #1**, i.e. one v10 net carrying both, and
spend one gauntlet slot on it. Attribution is lost, which is the v8/v8.5 precedent and the
right trade when slots are the scarce resource. If it fails, the fallback run is mirroring
alone (buckets are the cheaper thing to drop, since they are also the less likely to matter).

### 3. Fifth month -- 104-kz16r (in flight, do not touch)

`overnight/month5.sh` is running: pack done 15:13, train started ~15:15 at 0.47M pos/s,
20 epochs -> **done ~17:00**, then export + `check_nnue` + suite -> ~17:20. The one thing to
add is the missing control: **run the endgame suite on the v8 champion once** so 104-kz16r's
number means something. That is one 18-minute CPU job and it is a prerequisite for judging
every net below, not just this one.

```
.venv/Scripts/python.exe -m testing.endgame_suite run --agent . --seconds 2.5 \
    > overnight/eval/v10/suite-v8-baseline.log
```

(Run it when the laptop's CPU frees, not now -- CPU was 97% busy with 16 python processes
at 15:40.)

### 4. Endgame shard in rotation (`kz16e`)

**Design.** Add `data/positions_endgame.npy` as a sixth rotating shard so one epoch in six is
endgame-dense (108M positions, 70% at <=16 pieces). This is *not* the failed experiment: that
one (`train-eg.log`) trained exclusively on this shard at lr 3e-4 from `kz8` and never beat
epoch 0. Nothing changes in any source file -- it is a command-line change only.

```
.venv/Scripts/python.exe training/train.py \
  --data data/positions_w512-150m.npy data/positions_w512-150m-b.npy \
         data/positions_2025_02.npy data/positions_2025_03.npy \
         data/positions_2024_11.npy data/positions_endgame.npy \
  --val data/validation_w512-150m.npy \
  --resume training/checkpoints/net_w512-b8-kz16m.pt \
  --accumulator 512 --buckets 8 --king-zones 16 --mirror \
  --lr 1e-4 --epochs 12 --patience 5 --warmup-epochs 1 --skip-sanity \
  --out training/checkpoints/net_w512-b8-kz16e.pt
```

Note it resumes from the **v10 net**, so if #1 fails this run is moot. Validate on the normal
held-out set (not the endgame one) so the mix cannot flatter itself, and gate primarily on the
endgame suite's 9-12 band. **Do not spend a gauntlet slot unless the suite improves**; the
prior from kz16w is that a shift toward human endgame positions makes engine endgame moves
worse.

---

## Schedule to the 10 Sep freeze

The GPU is not the constraint and the CPU is. This plan spends **two** new net gauntlet slots
and keeps a third in reserve for the v8.5 / v9 search bundle.

| when | GPU | CPU / gauntlet | human |
|---|---|---|---|
| 5 Sep now -> 17:00 | `104-kz16r` (running) | 110-v85all SPRT, desktop v85 tasks | -- |
| 5 Sep evening | free from ~17:00 | export + check + suite for 104-kz16r; **v8 suite baseline** | implement mirroring + bucket table (9 files), run `ruff`, `mypy`, `check_features`, `check_fastsearch` |
| 5->6 Sep overnight | **v10 train** (#1 + #2), ~1.4-2.5 h | 104-kz16r gauntlet if a slot frees | -- |
| 6 Sep morning | free | export v10 + `check_nnue` + suite (18 min); if the suite holds, queue the **v10 SPRT at 8 s** | fold verdicts |
| 6 Sep evening | **kz16e train** (#4), ~1.5 h | v10 SPRT running | -- |
| 7 Sep | free (spare: #5 L2 head, or #8 mirrored-32 if v10 passed) | v10 verdict lands -> 40 games at 120 s + clocktest; kz16e suite | decide v10 in/out |
| 8 Sep | -- | kz16e SPRT **only if its suite improved**; otherwise the slot goes to the search bundle | -- |
| 9 Sep | -- | full bundle gate: SPRT vs champion, 200-game crash hunt, clocktest, 40 games at 120 s, zip < 50 MB | build `submission-candidate.zip` |
| 10 Sep evening | -- | freeze | upload |

If the mirroring implementation is not passing `check_nnue` by the evening of 6 Sep, **abandon
it** and ship 104-kz16r plus the search bundle. A silently wrong feature encoding produces no
error, loads fine, passes the crash gate and merely plays badly -- which is the one failure
mode this project has documented most carefully and the one a deadline is most likely to
produce.

## Closed -- do not reopen before 11 Sep

* Accumulator width in any direction (measured -35 at 21.6M; no warm start; -12..-25% knps).
* int8/int16 quantised inference (measured slower twice in this codebase).
* HalfKP / HalfKA full king-square feature sets (over the size cap, data-starved).
* Endgame loss weighting (`--weight-endgame`): better held-out loss, worse moves. Measured.
* Distillation from a locally trained teacher (dominated by the corpus labels).
* Relabelling the corpus with our own Stockfish at shallow depth (measured 197/200 against).
* One output bucket (val 0.004977 vs 0.004659).
* 32 unmirrored king zones (val 0.004690, suite 8.1).
