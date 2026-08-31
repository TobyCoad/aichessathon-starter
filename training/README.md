# Training pipeline

Everything here runs before the competition and none of it ships. The submission is
`agent.py` plus `weights/net.npz`.

## Porting to another machine

Nothing large is in git: the 7.5 GB Parquet and the packed arrays are regenerated,
and only the 0.85 MB `weights/net.npz` is committed. So a fresh machine needs the
steps below, in order.

```bash
git clone <repo> && cd aichessathon-starter
python -m venv .venv                       # Python 3.12
.venv/Scripts/python -m pip install "chess==1.11.2" numpy pyarrow ruff mypy
```

**CUDA torch.** Install it explicitly from the CUDA index; do not rely on `pip
install torch`, which will happily leave a `+cpu` build in place because the version
number matches. Use the timeout flags -- a plain `--force-reinstall` hung here for
25 minutes on a stalled socket with zero CPU time:

```bash
.venv/Scripts/python -m pip install --force-reinstall --no-cache-dir \
    --timeout 30 --retries 10 \
    --index-url https://download.pytorch.org/whl/cu128 torch
.venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`cu128` is for Blackwell (RTX 50-series, compute capability 12.0). An older card
wants a different index; check what your GPU reports before picking one.

## The pipeline

```bash
# 1. Data. ~7.5 GB, resumable, safe to interrupt. Only a size match counts as done.
.venv/Scripts/python -m training.fetch --month 2025_01

# 2. Pack. ~2 min per 16M positions on 14 cores.
.venv/Scripts/python -m training.pack --target 30000000 --val-target 500000

# 3. Check the packed data decodes back to legal chess.
.venv/Scripts/python -m training.check_pack --file data/positions.npy

# 4. Train. ~13 s per epoch per 16M positions on an RTX 5070.
.venv/Scripts/python -m training.train --epochs 12

# 5. Export to the flat .npz the engine loads.
.venv/Scripts/python -m training.export

# 6. Verify the engine's numpy inference against the torch model.
cp weights/net.npz overnight/challengers/NNN-name/weights/
.venv/Scripts/python -m training.check_nnue --agent overnight/challengers/NNN-name

# 7. The only thing that decides whether it was an improvement.
.venv/Scripts/python -m testing.gauntlet --challenger overnight/challengers/NNN-name
```

Steps 3 and 6 are not optional. Every failure mode in this pipeline is silent: a
wrong feature index, a wrong accumulator update or a wrong output scale produces an
engine that loads, plays legal moves, passes the crash gate, and merely plays worse.
There is no exception to catch.

## Things already learned the hard way

- **The network predicts a win-probability logit, not centipawns.** The engine
  multiplies by `OUTPUT_SCALE = 400`. Having it emit centipawns directly left it
  initialised five orders of magnitude below the range it needed -- output std 0.0024
  against a target std of 558 -- and the overfit check plateaued at 0.0126 instead of
  reaching 0.000300.
- **`cp` in the dataset is from white's point of view**, measured: +0.758 correlation
  with material read that way, -0.010 read as side-to-move. The trainer flips by `stm`.
- **Validation is split by row group, not by position.** Rows in the Parquet are
  consecutive plies of one game, so a random split leaks the same game into both
  sides and reports an optimistic validation loss.
- **Do not quantise.** int16 measured *slower* than float32 in numpy, because integer
  paths miss BLAS. It is a C++/SIMD trick that inverts in Python.
- **Do not use ONNX Runtime for inference.** At batch 1, which is all a depth-first
  search asks for, numpy measured ~4x faster; ORT's fixed per-call dispatch dominates
  a network this small.

## Current state

| | |
|---|---|
| Trained on | 16.4M positions, 8 epochs, final train loss 0.005883 |
| Network | (768 -> 256)x2 -> 32 -> 1, 213,313 parameters, 0.85 MB |
| Measured | **+199.5 Elo** over the tapered piece-square evaluation, 77.2% over 57 games |

The size cap is 200 MB and the network uses 0.85 MB of it, so there is room for a far
larger one. Widening the accumulator is close to free at inference, because the cost
there is numpy call overhead rather than arithmetic. Epochs are not the constraint --
the loss curve was flat by epoch 6 -- so the next experiment is more data and more
width, not more passes over the same 16M positions.
