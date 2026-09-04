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


def bucket_of(count: Tensor, buckets: int) -> Tensor:
    """Output head for a position with `count` pieces on the board, 1..32.

    Mirrors `_bucket` in agent.py exactly: equal-width bands of piece count, so
    the endgame gets heads of its own. One shared head scored four different
    KQvK positions within 120 cp of each other and could not convert; a head
    that only ever sees few-piece positions has the capacity to tell them apart.
    """
    return torch.clamp((count - 1) * buckets // MAX_PIECES, 0, buckets - 1)


def zone_of(square: Tensor, zones: int = 8) -> Tensor:
    """Vectorised `training.features.king_zone`: the king's zone from its own side."""
    rank = square >> 3
    file = square & 7
    if zones == 1:
        return torch.zeros_like(square)
    if zones == 4:
        upper = torch.where(rank <= 3, torch.full_like(square, 2), torch.full_like(square, 3))
        return torch.where(rank <= 1, file >> 2, upper)
    if zones == 8:
        upper = torch.where(rank <= 3, 4 + (file >> 2), 6 + (file >> 2))
        return torch.where(rank <= 1, file >> 1, upper)
    if zones == 16:
        upper = torch.where(rank <= 3, 8 + (file >> 1), 12 + (file >> 1))
        return torch.where(rank <= 1, file, upper)
    if zones == 32:
        low = rank * 8 + file
        mid = 16 + (rank - 2) * 4 + (file >> 1)
        high = 24 + ((rank - 4) >> 1) * 4 + (file >> 1)
        return torch.where(rank <= 1, low, torch.where(rank <= 3, mid, high))
    raise ValueError(f"no zone map for {zones} king zones")


def zone_parents(zones: int, saved_zones: int) -> list[int]:
    """For each zone of a `zones` map, the zone of the `saved_zones` map it lies
    inside. Only defined when the new map refines the old one; a net trained on
    the coarse map then starts the fine one exactly where it finished."""
    parents: dict[int, int] = {}
    squares = torch.arange(64)
    fine_zones = zone_of(squares, zones).tolist()
    coarse_zones = zone_of(squares, saved_zones).tolist()
    for fine, coarse in zip(fine_zones, coarse_zones, strict=True):
        if parents.setdefault(fine, coarse) != coarse:
            raise SystemExit(f"{zones} zones do not refine {saved_zones}: zone {fine} straddles")
    return [parents[z] for z in range(zones)]


def expand_zones(weight: Tensor, zones: int, saved_zones: int) -> Tensor:
    """bag.weight (saved_zones * 768, A) -> (zones * 768, A) by parent zone."""
    if zones == saved_zones:
        return weight
    blocks = weight.view(saved_zones, FEATURES, -1)
    parents = torch.tensor(zone_parents(zones, saved_zones))
    return blocks[parents].reshape(zones * FEATURES, -1).contiguous()


class Net(nn.Module):
    """(K x 768 -> A)x2 -> H -> 1: `king_zones` copies of the first layer selected by
    the perspective's own king square, and `buckets` independent heads after the
    accumulator selected by piece count.

    The heads are stored stacked -- head_w2 (B, 2A, H), head_b2 (B, H),
    head_w3 (B, H), head_b3 (B,) -- and every head is evaluated for every sample,
    then the sample's own head is gathered. Gathering per-sample weight matrices
    instead would materialise (batch, 2A, H) floats, 2 GB at batch 16384.

    King zones need no new data: the white-perspective indices already say where
    both kings stand (own king 320+sq, opponent king 704+sq), so the zone offsets
    are derived per batch inside `forward`.
    """

    def __init__(
        self,
        accumulator: int = ACC,
        hidden: int = HIDDEN,
        buckets: int = 1,
        king_zones: int = 1,
    ) -> None:
        super().__init__()
        self.buckets = buckets
        self.king_zones = king_zones
        # padding_idx is not used: padding is masked by per-sample weights instead,
        # because index 0 is a real feature (own pawn on a1) even if unreachable.
        self.bag = nn.EmbeddingBag(FEATURES * king_zones, accumulator, mode="sum")
        self.acc_bias = nn.Parameter(torch.zeros(accumulator))
        self.head_w2 = nn.Parameter(torch.empty(buckets, 2 * accumulator, hidden))
        self.head_b2 = nn.Parameter(torch.empty(buckets, hidden))
        self.head_w3 = nn.Parameter(torch.empty(buckets, hidden))
        self.head_b3 = nn.Parameter(torch.empty(buckets))
        # Sized so the accumulator lands inside SCReLU's active band. Summing ~22
        # pieces, an accumulator std of about 0.5 needs a per-weight std near 0.1;
        # at 0.02 the accumulator sat at std 0.094 and squaring it threw away
        # another order of magnitude before the first hidden layer saw anything.
        nn.init.normal_(self.bag.weight, std=0.1)
        # The same uniform(+/- 1/sqrt(fan_in)) that nn.Linear uses.
        bound2 = 1.0 / (2 * accumulator) ** 0.5
        bound3 = 1.0 / hidden**0.5
        nn.init.uniform_(self.head_w2, -bound2, bound2)
        nn.init.uniform_(self.head_b2, -bound2, bound2)
        nn.init.uniform_(self.head_w3, -bound3, bound3)
        nn.init.uniform_(self.head_b3, -bound3, bound3)

    def forward(self, white: Tensor, black: Tensor, mask: Tensor, stm: Tensor) -> Tensor:
        if self.king_zones > 1:
            valid = mask > 0
            white_king = ((white - 320) * (valid & (white >= 320) & (white < 384))).sum(1)
            black_king = ((white - 704) * (valid & (white >= 704))).sum(1)
            # Each perspective's zone comes from its own king, seen from its own
            # side: the black king's square is mirrored, as in features.indices.
            white = white + (zone_of(white_king, self.king_zones) * FEATURES).unsqueeze(1)
            black = black + (zone_of(black_king ^ 56, self.king_zones) * FEATURES).unsqueeze(1)
        acc_w = self.bag(white, per_sample_weights=mask) + self.acc_bias
        acc_b = self.bag(black, per_sample_weights=mask) + self.acc_bias
        white_to_move = stm.unsqueeze(1).bool()
        own = torch.where(white_to_move, acc_w, acc_b)
        opp = torch.where(white_to_move, acc_b, acc_w)
        x = torch.cat([own, opp], dim=1)
        h1 = torch.clamp(x, 0.0, 1.0) ** 2
        # (batch, B, H): every head, then keep the one this position belongs to.
        h2 = torch.relu(torch.einsum("bi,kih->bkh", h1, self.head_w2) + self.head_b2)
        all_heads = (h2 * self.head_w3).sum(-1) + self.head_b3
        bucket = bucket_of(mask.sum(1).long(), self.buckets)
        out: Tensor = all_heads.gather(1, bucket.unsqueeze(1)).squeeze(1)
        return out


def load_checkpoint(
    path: Path, buckets: int | None = None, king_zones: int | None = None
) -> Net:
    """Build a Net from a checkpoint, whichever layout it was saved in.

    Checkpoints from before output buckets hold a single head as `l2.*`/`l3.*`.
    Those load into every head of a bucketed net, and a single-zone first layer
    loads into every king zone, so a continuation starts exactly where the smaller
    net finished rather than from noise. Asking for fewer zones or buckets than the
    file has is refused: there is no honest way to merge them.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    accumulator = int(state["bag.weight"].shape[1])
    saved_zones = int(state["bag.weight"].shape[0]) // FEATURES
    zones = king_zones or saved_zones
    if "l2.weight" in state:
        hidden = int(state["l2.weight"].shape[0])
        heads = buckets or 1
        net = Net(accumulator, hidden, heads, zones)
        with torch.no_grad():
            net.bag.weight.copy_(expand_zones(state["bag.weight"], zones, saved_zones))
            net.acc_bias.copy_(state["acc_bias"])
            net.head_w2.copy_(state["l2.weight"].t().unsqueeze(0).expand_as(net.head_w2))
            net.head_b2.copy_(state["l2.bias"].unsqueeze(0).expand_as(net.head_b2))
            net.head_w3.copy_(state["l3.weight"].expand_as(net.head_w3))
            net.head_b3.copy_(state["l3.bias"].expand_as(net.head_b3))
        return net
    saved = int(state["head_w2"].shape[0])
    if buckets is not None and buckets != saved:
        raise SystemExit(f"{path} has {saved} output buckets, asked for {buckets}")
    net = Net(accumulator, int(state["head_w2"].shape[2]), saved, zones)
    if zones != saved_zones:
        state = dict(state)
        state["bag.weight"] = expand_zones(state["bag.weight"], zones, saved_zones)
    net.load_state_dict(state)
    return net


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


def loss_fn(prediction: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
    """MSE between predicted and target win probability.

    `prediction` is a logit; `target` is in centipawns and is squashed by SCALE.
    `weight`, if given, scales each sample's squared error (mean 1 over the batch).
    """
    error = (torch.sigmoid(prediction) - torch.sigmoid(target / SCALE)) ** 2
    if weight is None:
        return error.mean()
    return (error * weight).mean()


def sample_weights(pieces: Tensor, target: Tensor) -> Tensor:
    """Up-weight the cells the net fits worst and sees least: positions with 20
    pieces or fewer, and near-equal positions. Measured on the 8-zone net, the
    near-equal <= 16-piece cell is 10x worse than the 29-32 cell and is 5% of
    the data. Normalised to mean 1 so the learning rate keeps its meaning."""
    w = 1.0 + 2.0 * (pieces <= 20).float() + 2.0 * (target.abs() < 400).float()
    return w / w.mean()


BANDS = ((2, 8), (9, 12), (13, 16), (17, 20), (21, 24), (25, 28), (29, 32))


@torch.no_grad()
def stratified_loss(net: Net, batches: "Batches", generator: torch.Generator) -> dict[str, float]:
    """Held-out loss per piece band, plus the near-equal (|target| < 150) subset
    for the <= 16 and 29-32 bands: the diagnosis a single number hides."""
    net.eval()
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for white, black, mask, stm, target in batches.epoch(generator):
        error = (torch.sigmoid(net(white, black, mask, stm)) - torch.sigmoid(target / SCALE)) ** 2
        pieces = mask.sum(1)
        equalish = target.abs() < 150
        for lo, hi in BANDS:
            sel = (pieces >= lo) & (pieces <= hi)
            key = f"{lo}-{hi}"
            sums[key] = sums.get(key, 0.0) + float(error[sel].sum())
            counts[key] = counts.get(key, 0) + int(sel.sum())
        subsets = (("eq<=16", (pieces <= 16) & equalish), ("eq29-32", (pieces >= 29) & equalish))
        for key, sel in subsets:
            sums[key] = sums.get(key, 0.0) + float(error[sel].sum())
            counts[key] = counts.get(key, 0) + int(sel.sum())
    net.train()
    return {k: sums[k] / max(counts[k], 1) for k in sums}


def format_strata(strata: dict[str, float]) -> str:
    return " ".join(f"{k}:{v * 1e3:.2f}" for k, v in strata.items())


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
    weight_endgame: bool = False,
    warmup_epochs: int = 0,
    warmup_steps: int = 1500,
    resume: Path | None = None,
    limit: int = 0,
    buckets: int = 1,
    king_zones: int = 1,
) -> tuple[Net, dict[str, float]]:
    """Train, cycling through `sources` one shard per epoch.

    Several shards rather than one big array because the trainer holds a shard in
    RAM (`Batches` copies the index columns out of the memmap), and two shards of
    145M is the most this machine's 31 GB can rotate through; one array of 290M
    would not load at all. Returns the net and a summary the caller can write down.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    if resume is not None:
        net = load_checkpoint(resume, buckets, king_zones).to(device)
        print(
            f"  resumed from {resume} into {net.buckets} output bucket(s) "
            f"and {net.king_zones} king zone(s)"
        )
    else:
        net = Net(accumulator, buckets=buckets, king_zones=king_zones).to(device)
    val_batches = Batches(validation, batch, device) if validation is not None else None
    optimiser = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    sizes = [len(_records(source, limit)) for source in sources]
    steps = sum(-(-sizes[(epoch - 1) % len(sizes)] // batch) for epoch in range(1, epochs + 1))
    # A linear warm-up, then cosine decay. A resumed net at full learning rate on
    # step one loses ground it then has to recover; two runs died that way.
    warm = min(warmup_steps, max(steps // 10, 1))
    schedule = torch.optim.lr_scheduler.SequentialLR(
        optimiser,
        [
            torch.optim.lr_scheduler.LinearLR(optimiser, start_factor=0.05, total_iters=warm),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(steps - warm, 1)),
        ],
        milestones=[warm],
    )

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
        strata = stratified_loss(net, val_batches, torch.Generator().manual_seed(0))
        print(f"  initial strata (x1e-3) {format_strata(strata)}")

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
            weight = sample_weights(mask.sum(1), target) if weight_endgame else None
            loss = loss_fn(prediction, target, weight)
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
            elif epoch > warmup_epochs:
                stale += 1
        print(line, flush=True)
        if val_batches is not None:
            strata = stratified_loss(net, val_batches, torch.Generator().manual_seed(0))
            print(f"    strata (x1e-3) {format_strata(strata)}", flush=True)
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
    parser.add_argument(
        "--buckets", type=int, default=1, help="output heads, selected by piece count"
    )
    parser.add_argument(
        "--king-zones", type=int, default=1, help="first-layer copies, selected by own king"
    )
    parser.add_argument("--patience", type=int, default=4, help="early-stop patience, epochs")
    parser.add_argument("--weight-endgame", action="store_true", help="per-sample loss weights")
    parser.add_argument(
        "--warmup-epochs", type=int, default=0, help="epochs before patience counts"
    )
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
        f"accumulator {arguments.accumulator}, buckets {arguments.buckets}, "
        f"king zones {arguments.king_zones}"
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
        weight_endgame=arguments.weight_endgame,
        warmup_epochs=arguments.warmup_epochs,
        resume=arguments.resume,
        limit=arguments.limit,
        buckets=arguments.buckets,
        king_zones=arguments.king_zones,
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
