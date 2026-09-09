"""Export a trained checkpoint to the flat .npz the engine loads.

The engine ships no torch: inference is hand-written numpy and numba, because at
batch 1 -- which is all a depth-first search ever asks for -- numpy measured about
four times faster than ONNX Runtime, whose fixed per-call dispatch overhead
dominates a network this small. So the checkpoint is transposed once here into the
exact matrices the engine multiplies, and torch never appears at run time.

Shapes are fixed and asserted below, because a silently transposed matrix would
still load, still run, and merely play badly. Two layouts exist:

    single head      W2 (2A, H)     b2 (H,)     W3 (H, 1)     b3 (1,)
    B output buckets W2 (B, 2A, H)  b2 (B, H)   W3 (B, H, 1)  b3 (B, 1)

The engine reads the number of buckets from W2's rank, so either file loads.

Float32 throughout. Quantisation is deliberately not done: int16 measured *slower*
than float32 in numpy, since integer paths miss the BLAS route. It is a C++/SIMD
trick that inverts in Python.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from training.train import BUCKET_MAP_12, FEATURES, MAX_PIECES, Net, bucket_of, load_checkpoint


def expected_shapes(
    accumulator: int,
    hidden: int,
    buckets: int,
    king_zones: int = 1,
    pairwise: bool = False,
    dual: bool = False,
) -> dict[str, tuple[int, ...]]:
    """The engine reads these shapes from the file, so width, head count and king
    zones are all training choices. W1 has one 768-row block per king zone.

    A pairwise net multiplies each perspective's two halves together, so the head
    sees `accumulator` values rather than `2 * accumulator`; a dual net's output
    layer reads twice the hidden width. The engine infers both from these shapes
    alone -- W2's input width against the accumulator, W3's against hidden -- so
    the file needs no flag for either and old files keep loading unchanged.
    """
    head_in = accumulator if pairwise else 2 * accumulator
    head_out = 2 * hidden if dual else hidden
    if buckets == 1:
        return {
            "W1": (FEATURES * king_zones, accumulator),
            "b1": (accumulator,),
            "W2": (head_in, hidden),
            "b2": (hidden,),
            "W3": (head_out, 1),
            "b3": (1,),
        }
    return {
        "W1": (FEATURES * king_zones, accumulator),
        "b1": (accumulator,),
        "W2": (buckets, head_in, hidden),
        "b2": (buckets, hidden),
        "W3": (buckets, head_out, 1),
        "b3": (buckets, 1),
    }


def convert(net: Net) -> dict[str, np.ndarray]:
    """Torch parameters to the engine's matrices.

    The heads are stored as (B, 2A, H) and the engine computes `x @ W2[k]`, which
    is already the orientation stored, so nothing is transposed here; the single
    head case squeezes the bucket axis away to keep the old file layout.
    """
    first = net.bag.weight
    if getattr(net, "has_factoriser", False):
        # Fold the shared plane into every king zone and drop it. W1 is indexed
        # [zone * 768 + feature] and the factoriser is indexed [feature] in the same
        # post-mirror-flip space, so `repeat(zones, 1)` lines block z up with it
        # exactly. The engine's file is unchanged in shape and cost, and check_nnue
        # is what proves the fold: it scores real positions through the engine's
        # folded W1 and through the torch model's bag + factor and compares.
        first = first + net.factor.weight.repeat(net.king_zones, 1)
        # Assert the tiling rather than trust it. `repeat` gives out[i] = src[i % 768],
        # which is the [zone * 768 + feature] layout; `repeat_interleave` would give
        # src[i // zones] and silently produce a net that loads, runs and plays badly.
        # main()'s self-check cannot catch this -- it feeds random accumulators, so it
        # never touches W1 -- and check_nnue only covers the king zones its positions
        # happen to visit. This covers every zone, costs microseconds, and is exact:
        # the fold is float32 here and `halve` runs after it.
        for zone in (0, net.king_zones // 2, net.king_zones - 1):
            block = first[zone * FEATURES : (zone + 1) * FEATURES]
            expected = net.bag.weight[zone * FEATURES : (zone + 1) * FEATURES] + net.factor.weight
            if not torch.equal(block, expected):
                raise SystemExit(f"factoriser fold is wrong in king zone {zone}")
    tensors = {
        "W1": first,
        "b1": net.acc_bias,
        "W2": net.head_w2,
        "b2": net.head_b2,
        "W3": net.head_w3.unsqueeze(-1),
        "b3": net.head_b3.unsqueeze(-1),
    }
    if net.buckets == 1:
        heads = ("W2", "b2", "W3", "b3")
        tensors = {name: (t[0] if name in heads else t) for name, t in tensors.items()}
    return {
        name: tensor.detach().cpu().numpy().astype(np.float32) for name, tensor in tensors.items()
    }


def halve(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """W1 as float16 on disk: the 50 MB limit is on the unpacked zip, and the
    engine casts W1 back to float32 at import. Worst-case rounding is 2**-11
    relative, far below the 1 cp the evaluation is quantised to."""
    out = dict(weights)
    out["W1"] = weights["W1"].astype(np.float16)
    return out


def first_layer(weights: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """The accumulator activation: pairwise product, or SCReLU, inferred from W2.

    `x` is own and opponent accumulators concatenated, so its width is 2A. A
    pairwise net's W2 takes A inputs -- A/2 per perspective -- and a squaring net's
    takes 2A. Shared by the exporter's self-check and by agent.py's fallback path.
    """
    w2 = weights["W2"]
    head_in = w2.shape[-2] if w2.ndim == 3 else w2.shape[0]
    if head_in == x.shape[1]:
        return np.clip(x, 0.0, 1.0) ** 2
    clipped = np.clip(x, 0.0, 1.0)
    acc = x.shape[1] // 2
    half = acc // 2
    own, opp = clipped[:, :acc], clipped[:, acc:]
    return np.concatenate(
        [own[:, :half] * own[:, half:], opp[:, :half] * opp[:, half:]], axis=1
    )


def head_numpy(weights: dict[str, np.ndarray], x: np.ndarray, count: np.ndarray) -> np.ndarray:
    """The engine's head arithmetic, in numpy, for the self-check below."""
    h1 = first_layer(weights, x)
    hidden = weights["b2"].shape[-1]
    dual = (weights["W3"].shape[-2]) == 2 * hidden

    def activate(pre: np.ndarray) -> np.ndarray:
        h2 = np.maximum(pre, 0.0)
        if not dual:
            return h2
        return np.concatenate([h2, np.clip(pre, 0.0, 1.0) ** 2], axis=-1)

    if weights["W2"].ndim == 2:
        h2 = activate(h1 @ weights["W2"] + weights["b2"])
        return np.asarray((h2 @ weights["W3"] + weights["b3"]).squeeze(1))
    buckets = weights["W2"].shape[0]
    table = weights.get("bucket_map")
    out = np.empty(len(x), dtype=np.float32)
    for i in range(len(x)):
        if table is not None:
            k = int(table[min(max(int(count[i]), 0), MAX_PIECES)])
        else:
            k = min(max((int(count[i]) - 1) * buckets // 32, 0), buckets - 1)
        h2 = activate(h1[i] @ weights["W2"][k] + weights["b2"][k])
        out[i] = (h2 @ weights["W3"][k] + weights["b3"][k])[0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint for the engine.")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/net.pt"))
    parser.add_argument("--out", type=Path, default=Path("weights/net.npz"))
    parser.add_argument("--half", action="store_true", help="store W1 as float16")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="the checkpoint is a mirrored net; load_checkpoint refuses one otherwise",
    )
    arguments = parser.parse_args()

    net = load_checkpoint(arguments.checkpoint, mirror=arguments.mirror).eval()
    weights = convert(net)
    if arguments.half:
        weights = halve(weights)
    # Self-describing metadata, so the engine never re-derives a training choice.
    # Both are read by agent.py when present and absent from every net before v10,
    # which keeps those loading exactly as they did.
    # The map always travels with the net now: the engine indexes its head arrays
    # by piece count directly, so it never evaluates a bucket formula and cannot
    # drift from the one training used. Pre-v16 files carry none and agent.py
    # falls back to the equal-width formula, which is what they were trained with.
    weights["bucket_map"] = net.bucket_map.detach().cpu().numpy().astype(np.int32)
    if net.buckets == 12 and not np.array_equal(weights["bucket_map"], BUCKET_MAP_12):
        raise SystemExit("12-head net whose map is not BUCKET_MAP_12")
    if net.is_mirrored:
        weights["mirrored"] = np.asarray(1, dtype=np.uint8)
    accumulator = int(net.bag.weight.shape[1])
    hidden = int(net.head_w2.shape[2])
    expected = expected_shapes(
        accumulator, hidden, net.buckets, net.king_zones, net.is_pairwise, net.is_dual
    )

    for name, shape in expected.items():
        actual = weights[name].shape
        if actual != shape:
            raise SystemExit(f"{name} has shape {actual}, expected {shape}")
        if not np.isfinite(weights[name]).all():
            raise SystemExit(f"{name} contains NaN or infinity")

    # Verify the exported matrices reproduce the torch model, so a transposition or
    # a missing bias cannot slip through. Random accumulators rather than real
    # positions: this checks the arithmetic, not the chess. Piece counts span every
    # bucket so each head is compared at least once.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 2 * accumulator)).astype(np.float32)
    count = np.arange(1, 65) % 32 + 1
    with torch.no_grad():
        xt = torch.from_numpy(x)
        if net.is_pairwise:
            clipped = torch.clamp(xt, 0.0, 1.0)
            own, opp = clipped[:, :accumulator], clipped[:, accumulator:]
            half = accumulator // 2
            h1_t = torch.cat(
                [own[:, :half] * own[:, half:], opp[:, :half] * opp[:, half:]], dim=1
            )
        else:
            h1_t = torch.clamp(xt, 0.0, 1.0) ** 2
        pre = torch.einsum("bi,kih->bkh", h1_t, net.head_w2) + net.head_b2
        h2_t = torch.relu(pre)
        if net.is_dual:
            h2_t = torch.cat([h2_t, torch.clamp(pre, 0.0, 1.0) ** 2], dim=-1)
        all_heads = (h2_t * net.head_w3).sum(-1) + net.head_b3
        bucket = net.bucket_map[torch.clamp(torch.from_numpy(count).long(), 0, MAX_PIECES)]
        reference = all_heads.gather(1, bucket.unsqueeze(1)).squeeze(1).numpy()
    actual_out = head_numpy(weights, x, count)
    error = float(np.abs(reference - actual_out).max())
    if error > 1e-3:
        raise SystemExit(f"numpy head disagrees with torch by {error:.4g}")

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    # np.savez's stub types its kwargs as bool; the runtime accepts arrays.
    np.savez(arguments.out, **weights)  # type: ignore[arg-type]
    size = arguments.out.stat().st_size
    parameters = sum(int(np.prod(shape)) for shape in expected.values())
    print(
        f"wrote {arguments.out} ({size / 1e6:.2f} MB, {parameters:,} parameters, "
        f"{net.buckets} output bucket(s), {net.king_zones} king zone(s), "
        f"mirrored {net.is_mirrored})"
    )
    print(f"numpy head matches torch to {error:.2g}")
    for name, shape in expected.items():
        low, high = weights[name].min(), weights[name].max()
        print(f"  {name:<3} {shape!s:<14} range [{low:+.3f}, {high:+.3f}]")


if __name__ == "__main__":
    main()
