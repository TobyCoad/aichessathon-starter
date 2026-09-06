# ENDGAME_SHRINK (V10_PLAN #11) -- buildable scoping

Read-only pass, 6 Sep, tree at `144-caporder` (CTRL_SIZE 47). Nothing run, nothing modified.
**Verdict up front: BUILD the switch (~35 min) and run the 2-minute offline calibration. Do NOT
spend a gauntlet slot on it. Gate on the 17-minute endgame suite; if the suite improves, ride it
into the next bundle SPRT at zero marginal slot cost.** Section 7.

## 1. Where the eval is, and what a hook costs

`fastsearch.evaluate`, **fastsearch.py:388-438** (`@njit(cache=False, fastmath=True)`), is the
only static eval in the compiled kernel; it already reads `meta[fb.PIECES]` (line 394) to pick
the output head. Seven call sites, all with `bb`, `meta`, `ctrl` in scope: **608** (QS stand-pat,
eval-cache miss -- fills `ec_val`), **614** (QS stand-pat, cache off), **700** (`MAX_PLY-8`
horizon), **788** (IMPROVING, writes `exts[MAX_PLY+ply]`), **812** (RFP), **832** (futility
d<=2), **858** (NMP_V2 margin).

**Piece count is free**: `meta[fb.PIECES]` is maintained incrementally in make/unmake
(fastboard.py:527, 539, 602, 608) and is already in a register at line 394. **Material is not,
and need not be**: ten `fb.popcount` calls (fastboard.py:184, Kernighan loop) over `bb[0..4]`,
`bb[6..10]` -- below 17 pieces, <=14 iterations plus 10 calls, ~30-50 ns against a ~1.5-3 us
`evaluate` (the head is 32x1024 MACs) and a ~5 us node. **Gate on `meta[fb.PIECES] >= 17 ->
return net`**: the middlegame pays one compare, the endgame ~1-2% of one evaluate, and
`evaluate` runs at most once per node (`cached_eval` reuse) -- well under 1% of node time. Do
**not** add a MATERIAL slot to `meta`; that touches make/unmake/undo/`Position.load`/
`check_fastboard` to save ~40 ns.

## 2. Baseline: pure material, engine values

**Recommend material only** -- `_MATERIAL`, agent.py:2468, `P 100 N 300 B 300 R 500 Q 900`,
side-to-move POV. Not tapered PSQT, not king-proximity mop-up.
* The blend's job is to **bound the magnitude of the net's claim and pull it toward the material
  verdict**, not to add knowledge. The net knows more about piece placement than any PSQT we
  would hardcode; a PSQT that disagrees injects a *new gradient* into 30% of the eval that the
  search will chase. A flat anchor cannot introduce a bad plan -- at worst it damps, and that is
  what makes this safe enough to try at all.
* These are the values `_contempt`/`ADJUDICATION` already use *and* what the referee's ply-300
  raw-material adjudication counts (games.md failure mode 3). One set of values, three uses.
* Mop-up is dead weight: it bites only at KX vs K (<=4-5 men), where `TB_MEN = 4` syzygy already
  answers exactly and games.md measured every 5-man position we reached as handled correctly.
* For a gradient later, add **only** pawn-rank and king-centralisation tables (PeSTO / Rofchade
  endgame `P` and `K`, or Michniewski's "Simplified Evaluation Function" king-endgame table from
  CPW -- both 64-entry int8), `npawns + 2` lookups, behind its own constant so section 5
  measures it first.
* Scale: `training/train.py:269` trains `sigmoid(out)` against `sigmoid(cp/400)` and `evaluate`
  returns `int(out * 400)`, so the kernel's number is already centipawns on Stockfish's scale;
  the anchor needs no rescaling.

## 3. The blend schedule

Fixed point /256, using `w*net + (256-w)*simple == net + (256-w)*(simple - net)/256`.
`EG_HI = 17`, `EG_LO = 6`, `WMIN = 179` (0.70), `CAP = 300` cp. At `pieces >= 17` the eval is
untouched, identical to today; at 6..16, `w = WMIN + (256 - WMIN) * (pieces - 6) // 11`; at
`pieces <= 6`, `w = WMIN`; and `delta = (256 - w) * (simple - net) // 256`, clamped to +/- CAP.
Continuous at 17 (the formula yields 256 there), so there is no step a search can exploit.
Calibration sweeps WMIN over {128, 153, 179, 205, 230}.

**Should the blend be suppressed when |net| is large? No -- that is backwards, and the intuition
behind it is better served by the cap.** The failure targeted *is* a large wrong net eval:
games.md round 22 has reference -23, our static **+512**, and three moves later reference -1065
against static +565, so a `|net| < T` gate would exempt every documented instance of the bug.
A gate is also a discontinuity -- two sibling leaves either side of `T` get different treatments
and the search steers toward whichever side flatters it, the shape of the correction-history
failure (-137 +/- 65, `048`). The legitimate worry behind the intuition, discarding a *genuine*
win where material misleads (fortress, wrong bishop + rook pawn, opposite bishops, pawn race),
is answered continuously by `CAP`: at most 300 cp of movement either way. Also guard
`abs(net) < DISTANCE_THRESHOLD` (19000) so a mate-range score is never blended.

Honest limit: at 13 pieces `w ~ 0.81`, so round 22's +512 becomes ~+415. **This does not fix
that position.** What it does is (a) compress the eval's dynamic range below 17 pieces, so
cross-band comparisons -- a capture/trade changes `pieces`, a quiet move does not -- become
material-anchored, which is where a move actually flips; (b) pull the score toward the quantity
the ply-300 adjudication scores on. Claim +5..15 Elo at 120 s: a damper, not a cure. The cure
is NET_V10.

## 4. Interaction risks

* **The blend MUST live inside `fastsearch.evaluate`, not at the seven call sites.**
  `QS_EVAL_CACHE` writes `ec_val[qslot]` at line 610, and the TT **stores the static eval**
  (`cached_eval` packed at 1271, unpacked at 736 -- the TT eval field is live in this tree
  despite TT_EVAL being listed as rejected). A call-site blend would store raw net evals in both
  caches and re-blend on the way out: compounding, position-dependent, invisible to
  `check_fastsearch`. Inside `evaluate` the blended value is a pure function of the position,
  both caches stay correct, and `exts[MAX_PLY+ply]` (IMPROVING) compares like with like.
* **LAZY_ACC: no interaction** -- `bb`/`sq`/`meta` are current after `make_light`, only the
  accumulator is deferred, and every call site already syncs before `evaluate`.
* **Pruning margins.** RFP 80/ply, `FUTILITY_MARGIN` 150/300, `FUTILITY_MARGIN2` 100/ply,
  `DELTA_MARGIN` 200, `BIG_DELTA` 975 and the `NMP_V2` bonus `(standing - beta)//200` are tuned
  to the net's scale. Compressing evals by up to 30% below 17 pieces makes those margins
  *relatively larger*, i.e. **less pruning at low piece counts** -- the same direction as
  `RFP_PHASE` (percent `(0, 300, 200, 160)`), which was **rejected**. Expect a small node
  increase in low-piece positions. An optional v2 scales margins by `256/w`; not before v1 has
  a suite number.
* **The bench is nearly blind to this** (`testing.bench --depth 8` is 40 mostly-middlegame
  positions): judge on the calibration and the suite instead.
* **`check_fastsearch` stays exact**: the switch is off in the tree and the test zeroes `ctrl`
  (`testing/check_fastsearch.py:106` uses `fs.CTRL_SIZE`), so the branch is dead and the kernel
  bit-identical -- 70/70 + 40/40 must still pass. CTRL_SIZE 47 -> 50 is precedented.
* **Root contempt.** `agent.py:2281` calls `_contempt(board, self.evaluate())` via
  `FastEngine.evaluate` (agent.py:1695), which shares `_eval_bucket_kernel`/`_bucket` with the
  kernel; mirror the blend there behind the same flag (~6 lines) so the +/-60 cp thresholds and
  the calibration see the number the tree plays. Leave the pure-python `Engine.evaluate`
  (agent.py:1196) alone -- it is the exactness reference.
* **The endgame suite**, `testing/endgame_suite.py`: 400 positions, 5-16 pieces, Stockfish
  depth-18 labels, chosen move's cp loss at 2.5 s. Baseline
  (`overnight/eval/v10/suite-v91-champion.log`): **mean 10.8 cp; 5-8: 17.0, 9-12: 12.0,
  13-16: 5.0**; best move 54.2%, >=100 cp on 2.0%, wall 1005 s. It scores *moves*, so it is
  partly blind to a hallucinating eval that still picks a sane move -- hence section 5.

## 5. Calibration -- one process, <2 minutes, no CPU contention

`overnight/eval/endgame_suite.json` already holds all 400 positions with `fen`, `best` and
`eval` (Stockfish depth-18 cp, side-to-move POV): a labelled static-error dataset sitting unused
in the tree, needing **no Stockfish and no search**. One throwaway script under `testing/`:
1. Import `agent` once (~40 s); build **one** `FastEngine` and reuse it (the TT is 67 MB). Per
   position: `board = chess.Board(fen)`, `eng.prepare(board, 0)`, `net = eng.evaluate()`
   (agent.py:1695, the kernel's own `_eval_bucket_kernel`/`_bucket` path). Record `net`,
   `pieces`, material from `board.piece_map()`, the label. ~1 ms each, ~0.5 s total.
2. Print **mean |net - label| by band 5-8 / 9-12 / 13-16** -- the instrument games.md asked for
   and that has never been built. Platform games predict ~475 cp at 11-16.
3. In numpy, sweep WMIN x {material, material+PK-PSQT} x CAP in {150, 300, 0}: mean
   |blended - label| per band, count of positions moved >100 cp, largest single move. Seconds
   per cell, no engine run, no recompilation.

**Acceptance before a suite run:** 9-12 and 5-8 errors must fall (target >=25% in 9-12), 13-16
must not worsen by more than ~1 cp, no position moved more than `CAP` the wrong way.
**Rejection is the useful outcome**: the hypothesis dies for 2 minutes of CPU instead of 3 hours.
Caveat: kz16 *won* its gauntlet while *losing* the suite (network.md), so this is a **veto, not
a promotion signal**. Full gate later, between gauntlets, in order: `python -m
testing.endgame_suite run --agent <challenger> --seconds 2.5` (~17 min, needs
`engines/stockfish/...`) against 10.8 / 17.0 / 12.0 / 5.0; then `testing.bench --depth 8` for
the node cost; then, only inside a bundle, the 8 s SPRT.

## 6. Implementation sketch (one ~35-minute iteration)

**fastsearch.py.** (1) After `C_NMP_MIN_PLY = 46`, in house comment style: `C_EG_SHRINK = 47`,
`C_EG_WMIN = 48` (w at `<= EG_LO`, /256), `C_EG_CAP = 49` (max |correction| cp, 0 = uncapped);
`CTRL_SIZE = 50`; constants `EG_HI = 17`, `EG_LO = 6`,
`EG_VALUES = np.array([100, 300, 300, 500, 900], dtype=np.int64)`. Do **not** add it to `FOLDED`
-- under test, so it stays a live `ctrl` read. (2) A new leaf kernel, and (3) `evaluate(...)`
grows two parameters (`bb`, `ctrl`) plus one tail block:

```python
@njit(cache=False)                       # material, side-to-move POV, agent._MATERIAL values
def simple_eval(bb, meta):
    us = meta[fb.SIDE] * 6; them = 6 - us; total = 0
    for p in range(5):
        total += EG_VALUES[p] * (fb.popcount(bb[us + p]) - fb.popcount(bb[them + p]))
    return total

if (ctrl  # tail of evaluate(), after score = int(float(out) * OUTPUT_SCALE)[C_EG_SHRINK] == 0 or meta[fb.PIECES] >= EG_HI
        or score >= DISTANCE_THRESHOLD or score <= -DISTANCE_THRESHOLD):
    return score
wmin = ctrl[C_EG_WMIN]; pieces = meta[fb.PIECES]
w = wmin if pieces <= EG_LO else wmin + (256 - wmin) * (pieces - EG_LO) // (EG_HI - EG_LO)
delta = (256 - w) * (simple_eval(bb, meta) - score) // 256
cap = ctrl[C_EG_CAP]
if cap > 0:
    delta = cap if delta > cap else (-cap if delta < -cap else delta)
return score + delta
```

(4) Add `bb, ctrl` at the seven call sites (608, 614, 700, 788, 812, 832, 858). **agent.py.**
(5) `ENDGAME_SHRINK: Final = False`, `ENDGAME_SHRINK_WMIN: Final = 179`,
`ENDGAME_SHRINK_CAP: Final = 300` with the usual comment block. (6) In `prepare()` next to
`ctrl[_fs.C_SEE_QUIET]` (agent.py:2052), three writes. (7) Mirror the blend in
`FastEngine.evaluate` (agent.py:1695) behind the same flag.

**Commit gates:** `ruff check`, `mypy agent.py fastsearch.py`,
`python -m testing.check_fastsearch --depth 4 --random 30` (70/70 + 40/40 -- guaranteed with the
switch off, and the check that the signature change broke nothing). Bench only when idle.
Challenger sed, matching `SEE_QUIET`'s precedent:
`s/^ENDGAME_SHRINK: Final = False/ENDGAME_SHRINK: Final = True/`.

## 7. Verdict: build the switch, skip the slot

**Against spending one of the ~6 remaining slots.** The net side is already moving on the same
weakness and is not ours: `150-sfnet` (Stockfish-data net) **PROMOTED +19 over 200 games**,
`152-sfnet` REJECTED, and NET_V10 belongs to the interactive session -- two changes aimed at one
475 cp defect, one of which retrains the thing that is actually wrong and would absorb part of
this one's effect. This is also the fourth member of a family measured three times negative:
correction history (-137, then rejected again), `kz16w` endgame loss weighting (suite 7.4 ->
9.1, all in 9-12), `RFP_PHASE` (rejected) -- every attempt to bound or reweight the eval below
17 pieces has cost Elo or done nothing. And an 8 s SPRT resolves about +/-12 Elo while spending
most of its decisive plies above 16 pieces, so a change confined to <17 pieces will very likely
return INCONCLUSIVE: 3 hours of laptop for no verdict, the bad trade ARCHITECTURE.md section 5
warns about. `144-caporder` (-4.2 at 328), `145-v93fill`, `146-cutnode` and `147-seequiet` plus
clocktests are already ahead of it in the queue.

**For building it anyway.** The switch is ~35 minutes with an off-state exact by construction,
and the calibration is ~2 minutes, needs no Stockfish, and produces the static-error-by-band
instrument this project has wanted since games.md -- worth having whatever happens here, since
it is also the fastest way to score NET_V10 and any future net without an 18-minute suite run.
**Plan.** Build `ENDGAME_SHRINK` off in the tree; run the 2-minute calibration. If the 9-12 band
static error does not fall by >=25% at `CAP = 300`, close the idea and write the number into
NOTES. If it does, run the 17-minute suite against 10.8 / 17.0 / 12.0 / 5.0 between gauntlets;
require mean improvement >= 2 cp **and** the 9-12 band better, with the `kz16w` kill criterion
(any band worse by >1.5 cp) as a veto. Only then fold it into the next bundle challenger already
getting an SPRT (v9.3/v9.4), where its marginal slot cost is zero. **No solo gauntlet before the
10 Sep freeze.**
