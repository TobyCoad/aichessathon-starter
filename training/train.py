"""Train the evaluation network.

Architecture, fixed by the spec so training and inference cannot drift apart:

    acc_own, acc_opp : 256 each, from W1 (768, 256) + b1 (256,)
    x   = concat(acc_own, acc_opp)      # 512, own perspective first
    h1  = clamp(x, 0, 1) ** 2           # SCReLU
    h2  = relu(h1 @ W2 + b2)            # W2 (512, 32), b2 (32,)
    out = h2 @ W3 + b3                  # W3 (32, 1),  b3 (1,)

`out` is a **win-probability logit**, not centipawns. The engine multiplies it by
SCALE to get centipawns on the same footing as the hand-crafted evaluation.

That indirection is not cosmetic. Having the network emit centipawns directly leaves
it needing an output range of +/-2000 while initialising near zero -- measured output
std 0.0024 against a target std of 558 -- so training spends its first epochs merely
inflating the scale, and the sanity check plateaued at 0.0126 rather than converging.
In logit space the required range is about +/-5 and the problem is well conditioned.

Loss is mean squared error in win-probability space, not in raw centipawns: a 50cp
error matters enormously around equality and not at all at +1500, and training on
raw centipawns spends all its capacity on already-won positions.

Both perspectives share one weight matrix -- that is what makes the accumulator
incrementally updatable at run time, which is the whole reason for this shape.

The documented way this fails is being data-loader bound rather than GPU bound, so
there is no torch Dataset here: the packed array is sliced straight into batched
index tensors and the sparse sum is done by EmbeddingBag, which never materialises
the (batch, 32, 256) intermediate that a naive gather would.
"""

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

FEATURES = 768
MAX_PIECES = 32
ACC = 256
HIDDEN = 32
SCALE = 400.0


class Net(nn.Module):
    def __init__(self, accumulator: int = ACC, hidden: int = HIDDEN) -> None:
        super().__init__()
        # padding_idx is not used: padding is masked by per-sample weights instead,
        # because index 0 is a real feature (own pawn on a1) even if unreachable.
        self.bag = nn.EmbeddingBag(FEATURES, accumulator, mode="sum")
        self.acc_bias = nn.Parameter(torch.zeros(accumulator))
        self.l2 = nn.Linear(2 * accumulator, hidden)
        self.l3 = nn.Linear(hidden, 1)
        # Sized so the accumulator lands inside SCReLU's active band. Summing ~22
        # pieces, an accumulator std of about 0.5 needs a per-weight std near 0.1;
        # at 0.02 the accumulator sat at std 0.094 and squaring it threw away
        # another order of magnitude before the first hidden layer saw anything.
        nn.init.normal_(self.bag.weight, std=0.1)

    def forward(self, white: Tensor, black: Tensor, mask: Tensor, stm: Tensor) -> Tensor:
        acc_w = self.bag(white, per_sample_weights=mask) + self.acc_bias
        acc_b = self.bag(black, per_sample_weights=mask) + self.acc_bias
        white_to_move = stm.unsqueeze(1).bool()
        own = torch.where(white_to_move, acc_w, acc_b)
        opp = torch.where(white_to_move, acc_b, acc_w)
        x = torch.cat([own, opp], dim=1)
        h1 = torch.clamp(x, 0.0, 1.0) ** 2
        h2 = torch.relu(self.l2(h1))
        out: Tensor = self.l3(h2).squeeze(1)
        return out


def black_from_white(white: Tensor) -> Tensor:
    """Vectorised perspective swap, matching training/features.py exactly."""
    own = white // 384
    piece = (white % 384) // 64
    square = white % 64
    return (384 - own * 384) + piece * 64 + (square ^ 56)


class Batches:
    """Slices the packed array straight onto the GPU. No Dataset, no workers.

    Everything stays in its packed width on the host and is widened per batch. At
    30M positions, materialising the indices as int64 up front would cost 7.7 GB and
    the mask another 3.8 GB; per batch they are a few megabytes.
    """

    def __init__(self, records: np.ndarray, batch: int, device: torch.device) -> None:
        self.idx = np.ascontiguousarray(records["idx"])  # (N, 32) uint16
        self.counts = np.ascontiguousarray(records["count"])  # (N,) uint8
        self.stm = np.ascontiguousarray(records["stm"])
        self.cp = np.ascontiguousarray(records["cp"])
        self.batch = batch
        self.device = device
        self.count = len(records)
        self.positions = torch.arange(MAX_PIECES, device=device).unsqueeze(0)

    def __len__(self) -> int:
        return (self.count + self.batch - 1) // self.batch

    def epoch(self, generator: torch.Generator) -> Iterator[tuple[Tensor, ...]]:
        order = torch.randperm(self.count, generator=generator).numpy()
        for start in range(0, self.count, self.batch):
            rows = np.sort(order[start : start + self.batch])
            white = torch.from_numpy(self.idx[rows].astype(np.int64)).to(self.device)
            counts = torch.from_numpy(self.counts[rows].astype(np.int64)).to(self.device)
            stm = torch.from_numpy(self.stm[rows].astype(np.float32)).to(self.device)
            cp = torch.from_numpy(self.cp[rows].astype(np.float32)).to(self.device)

            mask = (self.positions < counts.unsqueeze(1)).to(torch.float32)
            # Padding is stored as index 0, which is a real feature (own pawn on a1,
            # unreachable in practice). The zero weight is what makes it contribute
            # nothing to the sum and to the gradient; the index itself is harmless.
            black = black_from_white(white)
            # cp is stored white-POV; the network predicts side-to-move-POV.
            target = torch.where(stm.bool(), cp, -cp)
            yield white, black, mask, stm, target


def loss_fn(prediction: Tensor, target: Tensor) -> Tensor:
    """MSE between predicted and target win probability.

    `prediction` is a logit; `target` is in centipawns and is squashed by SCALE.
    """
    return torch.nn.functional.mse_loss(
        torch.sigmoid(prediction), torch.sigmoid(target / SCALE)
    )


@torch.no_grad()
def evaluate_loss(net: Net, batches: "Batches", generator: torch.Generator) -> float:
    net.eval()
    total = 0.0
    seen = 0
    for white, black, mask, stm, target in batches.epoch(generator):
        total += float(loss_fn(net(white, black, mask, stm), target)) * len(target)
        seen += len(target)
    net.train()
    return total / max(seen, 1)


Source = Path | np.ndarray


def _records(source: Source, limit: int = 0) -> np.ndarray:
    """A shard as an array. Paths are memory-mapped, so only what `Batches` copies
    out is ever resident: at 145M positions that is ~10 GB instead of ~20."""
    records = np.load(source, mmap_mode="r") if isinstance(source, Path) else source
    return records[:limit] if limit else records


def train(
    sources: list[Source],
    device: torch.device,
    epochs: int,
    batch: int,
    learning_rate: float,
    seed: int = 0,
    validation: np.ndarray | None = None,
    accumulator: int = ACC,
    patience: int = 4,
    resume: Path | None = None,
    limit: int = 0,
) -> tuple[Net, dict[str, float]]:
    """Train, cycling through `sources` one shard per epoch.

    Several shards rather than one big array because the trainer holds a shard in
    RAM (`Batches` copies the index columns out of the memmap), and two shards of
    145M is the most this machine's 31 GB can rotate through; one array of 290M
    would not load at all. Returns the net and a summary the caller can write down.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    net = Net(accumulator).to(device)
    if resume is not None:
        net.load_state_dict(torch.load(resume, map_location=device, weights_only=True))
        print(f"  resumed from {resume}")
    val_batches = Batches(validation, batch, device) if validation is not None else None
    optimiser = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    sizes = [len(_records(source, limit)) for source in sources]
    steps = sum(-(-sizes[(epoch - 1) % len(sizes)] // batch) for epoch in range(1, epochs + 1))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(steps, 1))

    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale = 0
    initial = float("nan")
    if val_batches is not None and resume is not None:
        # The resumed net sets the bar: a continuation that never beats it hands
        # back the checkpoint it started from, not a worse one.
        initial = evaluate_loss(net, val_batches, torch.Generator().manual_seed(0))
        best_loss = initial
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"  initial validation loss {initial:.6f}")

    epochs_run = 0
    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        source = sources[(epoch - 1) % len(sources)]
        batches = Batches(_records(source, limit), batch, device)
        if len(sources) > 1:
            print(f"  shard {source if isinstance(source, Path) else 'array'}", flush=True)
        net.train()
        started = time.perf_counter()
        running = 0.0
        seen = 0
        for white, black, mask, stm, target in batches.epoch(generator):
            prediction = net(white, black, mask, stm)
            loss = loss_fn(prediction, target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            schedule.step()
            running += float(loss) * len(target)
            seen += len(target)
        elapsed = time.perf_counter() - started
        rate = seen / elapsed
        line = (
            f"  epoch {epoch}/{epochs}  train {running / seen:.6f}  "
            f"{rate / 1e6:.2f}M pos/s  {elapsed:.0f}s"
        )
        if val_batches is not None:
            held_out = evaluate_loss(net, val_batches, torch.Generator().manual_seed(0))
            gap = held_out - running / seen
            line += f"  val {held_out:.6f}  gap {gap:+.6f}"
            # Keep the epoch that generalised best, not the last one. Training loss
            # keeps falling long after held-out loss stops improving, and the last
            # epoch is simply the most overfit one.
            if held_out < best_loss - 1e-7:
                best_loss = held_out
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                stale = 0
                line += "  *best"
            else:
                stale += 1
        print(line, flush=True)
        del batches
        if val_batches is not None and stale >= patience:
            print(f"  early stop: validation has not improved for {patience} epochs")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"  restored the best epoch, validation loss {best_loss:.6f}")
    summary = {"best_val": best_loss, "initial_val": initial, "epochs": float(epochs_run)}
    return net, summary


def overfit_check(records: np.ndarray, device: torch.device) -> bool:
    """Overfit a small slice before committing hours to the full run.

    If the network cannot drive the loss down on ten thousand positions it has seen
    hundreds of times, the encoding or the loss is wrong and a long run would only
    produce an expensive, confidently wrong net.
    """
    print("sanity: overfitting 10,000 positions")
    subset = np.asarray(records[:10_000])
    net, _ = train([subset], device, epochs=30, batch=1024, learning_rate=3e-3)
    net.eval()
    batches = Batches(subset, 4096, device)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        total = 0.0
        seen = 0
        for white, black, mask, stm, target in batches.epoch(generator):
            total += float(loss_fn(net(white, black, mask, stm), target)) * len(target)
            seen += len(target)
    final = total / seen
    # Predicting the mean everywhere scores about 0.08 in this space; a net that has
    # actually memorised the slice should be far below that.
    verdict = final < 0.01
    print(f"sanity: final loss {final:.6f} -> {'PASS' if verdict else 'FAIL'}")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the evaluation network.")
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=[Path("data/positions.npy")],
        help="one or more packed shards, cycled one per epoch",
    )
    parser.add_argument("--val", type=Path, default=Path("data/validation.npy"))
    parser.add_argument("--out", type=Path, default=Path("weights/net.pt"))
    parser.add_argument(
        "--resume", type=Path, default=None, help="checkpoint to continue from"
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--accumulator", type=int, default=ACC, help="first-layer width")
    parser.add_argument("--patience", type=int, default=4, help="early-stop patience, epochs")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N positions")
    parser.add_argument("--skip-sanity", action="store_true")
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Training on CPU will not finish in a night; "
            "reinstall torch from the cu128 index and retry, or mark P2.3 blocked."
        )
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    for shard in arguments.data:
        if not shard.exists():
            raise SystemExit(f"missing shard {shard}")
        print(f"data: {len(_records(shard, arguments.limit)):,} positions from {shard}")

    if not arguments.skip_sanity and not overfit_check(
        _records(arguments.data[0], arguments.limit), device
    ):
        raise SystemExit("sanity check failed; not starting the full run")

    validation = None
    if arguments.val.exists():
        validation = np.asarray(np.load(arguments.val, mmap_mode="r"))
        print(f"validation: {len(validation):,} positions from games not in training")
    else:
        print(f"validation: {arguments.val} not found -- training loss only, flying blind")

    print(
        f"training {arguments.epochs} epochs, batch {arguments.batch}, "
        f"accumulator {arguments.accumulator}"
    )
    net, summary = train(
        list(arguments.data),
        device,
        arguments.epochs,
        arguments.batch,
        arguments.lr,
        validation=validation,
        accumulator=arguments.accumulator,
        patience=arguments.patience,
        resume=arguments.resume,
        limit=arguments.limit,
    )

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), arguments.out)
    print(f"wrote {arguments.out}")
    # A machine-readable verdict beside the checkpoint, so an unattended pipeline
    # can decide whether the continuation actually beat what it started from.
    report = arguments.out.with_suffix(".json")
    report.write_text(json.dumps(summary, indent=2))
    print(f"wrote {report}: {summary}")


if __name__ == "__main__":
    main()
