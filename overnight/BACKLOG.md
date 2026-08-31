# Experiment backlog

The overnight loop works down this file. Edit it freely — it is the steering wheel.
Take the topmost item whose `status` is `todo`, work only on that, and record the
outcome. One experiment per run.

Priorities come from measured evidence, not enthusiasm. Elo figures cited are from
published SPRT results in comparable engines and are estimates until our own SPRT
says otherwise.

---

## P0 — blocked on a human, do not start

- [ ] **Ask the organisers about `rust-chess` and `numba`.** Worth more than
  everything below combined: a pip-installable compiled move generator measured
  2.2x end-to-end and a full extra ply (~150 Elo) over python-chess, and a
  numba-JIT'd movegen benchmarks ~40x on the generator alone. Both hinge on
  whether the rules' "compiled dependencies come from PyPI" covers them.
  **Toby must email hello@aichessathon.com.** Do not build on either until answered.

---

## P2 — the learned evaluation

Mandatory: the rules require a learned model to materially drive move selection, so
the current hand-crafted evaluation is a fallback rather than a legal submission.
Published gains over a tuned hand-crafted evaluation range from **+305 down to +20**,
and the difference was training data quality, not architecture.

**The failure mode to fear is silence.** A wrong feature index or a wrong
accumulator update produces an engine that runs clean, passes the crash gate, and
merely plays worse. There is no exception and no stack trace. Every step below
therefore specifies its own correctness test, and **the test is written before the
thing it tests.** If you skip that, you will spend a night training a net that
cannot work and the SPRT will tell you only that it lost.

Everything below is specified exactly so no run has to invent a convention. Two
runs that pick different conventions produce a net and an engine that disagree,
which is the same silent failure.

### P2.1 — `training/features.py`, the encoding

`status: todo`

This file is the contract between training and inference. Get it wrong and nothing
downstream can work.

**Index formula.** For a perspective `P` (the colour whose accumulator this is), a
piece of colour `c` and type `t` (1=pawn … 6=king) standing on square `sq`
(python-chess numbering, a1 = 0):

```
rel  = sq if P == WHITE else sq ^ 56
own  = (c == P)
index = (0 if own else 384) + (t - 1) * 64 + rel
```

So indices run 0..767: the first 384 are the perspective's own pieces, the second
384 the opponent's, and the board is vertically flipped for the black perspective.

**Derivation used by the packer.** Given a white-perspective index `w`, the
black-perspective index for the same piece is:

```
black = (384 - (w // 384) * 384) + ((w % 384) // 64) * 64 + ((w % 64) ^ 56)
```

Provide this as a vectorised numpy function; the packed data stores only white
perspective and derives black at load time.

**Write this test first.** Build indices for a few thousand random positions two
independent ways -- once with the formula above, once by a deliberately naive
`board.piece_map()` loop written separately -- and assert the sorted index sets are
equal. Also assert `black_from_white(white_indices(board)) == black_indices(board)`
on the same positions. Do not proceed until both pass.

### P2.2 — `training/pack.py`, Parquet to a flat array

`status: todo`

Input: `data/standard_rated_2025_01.parquet` (CC0; columns `fen`, `cp`, `mate`,
`move`). Output: `data/positions.npy`, a structured array.

**Environment is ready.** `pyarrow 25.0.1` installed; the Parquet is complete and
size-verified at 7.50 GB with **627,353,822 rows across 599 row groups**; schema is
`fen: string, cp: int32, mate: int32, move: string`. At 627M rows you need only a
fraction, so read row groups until you have enough and stop -- do not scan the file.

**Record dtype**, exactly:

```python
np.dtype([("idx", np.uint16, 32), ("count", np.uint8),
          ("stm", np.uint8), ("cp", np.int16)])
```

67 bytes per position, so 50M positions is 3.4 GB and fits in RAM -- no out-of-core
machinery is needed. `idx` holds the white-perspective indices, padded arbitrarily
beyond `count`. `stm` is 1 for white to move.

**Filtering, in this order:**
1. Drop rows where `fen` fails to parse.
2. If `mate` is non-null, set `cp = sign(mate) * 2000`. Do not use ±10000: it
   saturates the sigmoid to zero gradient and wastes the row.
3. Clamp `cp` to [-2000, 2000]. `cp` is from **white's** point of view -- this is
   now measured, not assumed: over 4,669 positions with a material imbalance above
   200cp, correlation with material is +0.758 read as white-POV and -0.010 read as
   side-to-move. Store it white-POV; the training step flips it by `stm`.
4. Drop positions that are check-to-move or have a capture available as the best
   move, to bias toward quiet positions the evaluation can actually learn.
5. Target **at least 50% of kept rows with `|cp| <= 100`.** Quiet, near-equal
   positions are what an evaluation needs to discriminate; a set dominated by
   already-won positions teaches it nothing.
6. Deduplicate on the FEN's first four fields.

**Target 20-50M positions.** The published knee is well below 100M -- one study
reached +100 Elo at ~5M -- so do not spend the night packing 200M.

**Throughput warning.** Do not call `chess.Board(fen)` per row in a Python loop at
train time; that caps at 5-20k positions/sec and would idle the GPU completely. It
is acceptable here in `pack.py`, once, because the output is cached -- but use
`pyarrow` batch iteration and multiprocessing over row groups, and expect this step
to take a while. Print progress.

### P2.3 — `training/train.py`

`status: todo`

**Architecture**, exactly:

```
acc_own, acc_opp : 256 each, from W1 (768, 256) + b1 (256,)
x   = concat(acc_own, acc_opp)          # 512, own perspective always first
h1  = clamp(x, 0.0, 1.0) ** 2           # SCReLU
h2  = relu(h1 @ W2 + b2)                # W2 (512, 32), b2 (32,)
out = h2 @ W3 + b3                      # W3 (32, 1),  b3 (1,)
```

`out` is **centipawns from the side to move's point of view**, so the engine can use
it directly on the same scale as the current evaluation.

**Loss:** `MSE(sigmoid(out / 400), sigmoid(target_cp / 400))` where `target_cp` is
`cp` if `stm` is white else `-cp`. Sigmoid space, not raw centipawns: an error of
50cp matters enormously at 0 and not at all at 1500.

**Hyperparameters** to start from: AdamW, lr 1e-3 with cosine decay, batch 16384,
8-10 epochs, float32 throughout. **Do not quantise** -- int16 measured *slower* than
float32 in numpy, because integer paths miss BLAS. Quantisation is a C++/SIMD trick
that inverts in Python.

**The data-loader trap.** Training here is loader-bound, not GPU-bound. Do not write
a `Dataset` whose `__getitem__` returns one position. Slice the packed array
directly into batched index tensors on the GPU, and build the sparse-sum accumulator
with `index_add_` or an `EmbeddingBag`. If GPU utilisation sits below ~50%, the
loader is the problem, not the model.

CUDA torch is installed and **verified working**: `2.11.0+cu128`, RTX 5070 Laptop,
compute capability (12, 0), a real 4096-square matmul kernel measured at 3.3 TFLOP/s.
Note it is 2.11.0, not 2.13.0 -- the CUDA index's newest -- which is irrelevant to
the submission, since the engine ships hand-written numpy inference and no torch.
**Still confirm `torch.cuda.is_available()` is True before starting**, in case
something has disturbed the environment since. If either is wrong, take the one reinstall attempt described in
`PROMPT.md`, and if it still fails, mark this item `status: blocked` and drop to P3.
Training on CPU will not finish in a night and produces nothing.

**Sanity check before the long run:** overfit 10,000 positions to near-zero loss.
If it cannot, the encoding or the loss is wrong and a full run will only waste hours.

### P2.4 — `training/export.py`

`status: todo`

Write `weights/net.npz` with exactly these keys and shapes, float32:
`W1 (768,256)`, `b1 (256,)`, `W2 (512,32)`, `b2 (32,)`, `W3 (32,1)`, `b3 (1,)`.

`harness/package.py` already includes a `weights` directory in the submission zip.
At ~430k parameters this is ~1.7 MB, so the 200 MB cap is nowhere near binding.
**Commit the weights file** -- it is the one build artifact that must ship.

### P2.5 — the incremental accumulator in `agent.py`

`status: todo`

The intricate step. Maintain `acc_white` and `acc_black` alongside the search,
updating them on `push` and restoring on `pop`.

- On push, copy the current accumulator onto a stack, then apply the move's deltas;
  on pop, restore by assignment. Restoring is far cheaper than recomputing, and a
  full refresh costs ~5-10 us against ~0.6 us for an incremental update.
- Deltas per move: subtract the moving piece's from-square feature, add its
  to-square feature (using `move.promotion or piece_type` for the destination).
  Subtract a captured piece; **en passant captures on a different square from
  `move.to_square`** -- the pawn sits one rank behind. Castling moves the rook too:
  update both king and rook.
- Both accumulators change on every move; a piece's white-perspective and
  black-perspective indices are different.
- Use in-place `acc += W1[i]` on a preallocated array, never fancy indexing, which
  measured ~6x worse.

**Write this test first, and do not skip it.** Play several thousand random plies;
after every single one, assert the incrementally-maintained accumulator equals a
full rebuild to within 1e-3. The earlier hand-built prototype passed 4000/4000, so
this is achievable -- but it only passed because it was tested. Cover promotion,
en passant, both castlings, and captures explicitly.

### P2.6 — the decisive SPRT

`status: todo`

Build the challenger with the net evaluation, run
`testing.gauntlet --challenger overnight/challengers/NNN-nnue --games 800`, and obey
the exit code. This is the experiment the whole project turns on.

If it rejects, **the net is the suspect, not the search.** Check in this order:
the accumulator test, then the encoding test, then whether the sigmoid-space loss
actually converged, then the data filter. Record which one it was in the journal --
a rejected net whose cause is unknown is the worst outcome available.

## P1 — search (do these only once P2 is complete or blocked)

- [ ] `status: todo` — **Killer moves and history heuristic.** Two quiet moves per
  ply plus a from/to history table, inserted after captures in ordering. Published
  at +56 and +19 Elo, but a Python-specific measurement found only 1.1x wall-time
  gain against 1.4x node gain, because the sort key costs what it saves. Genuinely
  uncertain here — which is exactly what the SPRT is for. Keep the sort key cheap.
- [ ] `status: todo` — **Reverse futility pruning.** Best measured Elo-per-token of
  any feature in a comparable competition, and it fires at `depth < 7`, which is
  where we live. Start with margin `70 * depth`, prune when `eval - margin >= beta`
  at shallow depth with no check.
- [ ] `status: todo` — **Null-move pruning, fixed R=2.** Published +116 Elo. Skip the
  depth-scaling variant, which measured only +14. Guard against zugzwang: require
  non-pawn material for the side to move, and never in check.
- [ ] `status: todo` — **Check extension.** One ply when the move gives check.
  Measured +57 Elo in an engine of roughly our strength and ~1 Elo in Stockfish —
  a shallow-depth feature, so it should be worth more to us than to them.
- [ ] `status: todo` — **Principal variation search.** Only after ordering is solid.
  Measured +55 Elo *after* good ordering and **−33 before it**, so if the killer and
  RFP experiments above have not landed, skip this and come back.
- [ ] `status: todo` — **Time management sweep.** The `_budget` constants in
  `agent.py` were reasoned, not measured. Try expected-moves in {22, 26, 30} and
  hard-limit fractions in {0.25, 0.35, 0.45}. Cheap, and flagging is the most
  common self-inflicted loss.

## P3 — bounded, no statistical validation needed

- [ ] `status: todo` — **Syzygy 3-4-man tablebase.** Exactly 4,346,080 bytes for all
  70 WDL+DTZ files from `tablebase.lichess.ovh`. Probe WDL at leaves once material
  is down to four men (33-59 us, affordable), DTZ at the root only (up to 514 us).
  Use `get_wdl`/`get_dtz`, never `probe_*`: a missing table raises and a crash is a
  loss. Dominates hand-written mate drivers — it solves KBNvK for free.
- [ ] `status: todo` — **Opening book, 20 plies.** Polyglot `.bin`, not a pickled
  dict: measured on identical data a pickle is not smaller and costs full-file load
  plus heap residency, where `.bin` is memory-mapped. Two traps: the weight field is
  `uint16` and saturates at 65535, so scale raw game counts; and castling must be
  encoded king-takes-rook (e1h1, not e1g1). Build from human master games.

## P4 — robustness, run whenever the queue is empty

- [ ] `status: recurring` — **Crash hunt.** A few hundred games against the random
  baseline at a fast control. Random play reaches promotion, en passant and
  stalemate corners far faster than a real opponent. Any failure termination is a
  P0 bug: fix it before anything else in this file.

---

## Do not build

Measured or published as not worth it at our depth, recorded here so no run
rediscovers them: singular extensions (needs depth >= 6), internal iterative
deepening (depth >= 7), razoring (~1 Elo), aspiration windows (+23 against −25 in
two separate tests), MCTS (50k UCT iterations plays like a 2-ply minimax once you
have a real evaluation), batched leaf evaluation for alpha-beta (gives up the
ordering win to gain less), int8/int16 quantisation in numpy (slower than float32),
ONNX Runtime at batch 1 (4x slower than numpy), and rewriting move generation in
pure Python (python-chess already uses precomputed bitboards; every hand-rolled
alternative benchmarks worse).
