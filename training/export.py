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

from training.train import FEATURES, Net, bucket_of, load_checkpoint


def expected_shapes(
    accumulator: int, hidden: int, buckets: int, king_zones: int = 1
) -> dict[str, tuple[int, ...]]:
    """The engine reads these shapes from the file, so width, head count and king
    zones are all training choices. W1 has one 768-row block per king zone."""
    if buckets == 1:
        return {
            "W1": (FEATURES * king_zones, accumulator),
            "b1": (accumulator,),
            "W2": (2 * accumulator, hidden),
            "b2": (hidden,),
            "W3": (hidden, 1),
            "b3": (1,),
        }
    return {
        "W1": (FEATURES * king_zones, accumulator),
        "b1": (accumulator,),
        "W2": (buckets, 2 * accumulator, hidden),
        "b2": (buckets, hidden),
        "W3": (buckets, hidden, 1),
        "b3": (buckets, 1),
    }


def convert(net: Net) -> dict[str, np.ndarray]:
    """Torch parameters to the engine's matrices.

    The heads are stored as (B, 2A, H) and the engine computes `x @ W2[k]`, which
    is already the orientation stored, so nothing is transposed here; the single
    head case squeezes the bucket axis away to keep the old file layout.
    """
    tensors = {
        "W1": net.bag.weight,
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


def head_numpy(weights: dict[str, np.ndarray], x: np.ndarray, count: np.ndarray) -> np.ndarray:
    """The engine's head arithmetic, in numpy, for the self-check below."""
    h1 = np.clip(x, 0.0, 1.0) ** 2
    if weights["W2"].ndim == 2:
        h2 = np.maximum(h1 @ weights["W2"] + weights["b2"], 0.0)
        return np.asarray((h2 @ weights["W3"] + weights["b3"]).squeeze(1))
    buckets = weights["W2"].shape[0]
    out = np.empty(len(x), dtype=np.float32)
    for i in range(len(x)):
        k = min(max((int(count[i]) - 1) * buckets // 32, 0), buckets - 1)
        h2 = np.maximum(h1[i] @ weights["W2"][k] + weights["b2"][k], 0.0)
        out[i] = (h2 @ weights["W3"][k] + weights["b3"][k])[0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint for the engine.")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/net.pt"))
    parser.add_argument("--out", type=Path, default=Path("weights/net.npz"))
    parser.add_argument("--half", action="store_true", help="store W1 as float16")
    arguments = parser.parse_args()

    net = load_checkpoint(arguments.checkpoint).eval()
    weights = convert(net)
    if arguments.half:
        weights = halve(weights)
    accumulator = int(net.bag.weight.shape[1])
    hidden = int(net.head_w2.shape[2])
    expected = expected_shapes(accumulator, hidden, net.buckets, net.king_zones)

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
        h1_t = torch.clamp(torch.from_numpy(x), 0.0, 1.0) ** 2
        h2_t = torch.relu(torch.einsum("bi,kih->bkh", h1_t, net.head_w2) + net.head_b2)
        all_heads = (h2_t * net.head_w3).sum(-1) + net.head_b3
        bucket = bucket_of(torch.from_numpy(count).long(), net.buckets)
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
        f"{net.buckets} output bucket(s), {net.king_zones} king zone(s))"
    )
    print(f"numpy head matches torch to {error:.2g}")
    for name, shape in expected.items():
        low, high = weights[name].min(), weights[name].max()
        print(f"  {name:<3} {shape!s:<14} range [{low:+.3f}, {high:+.3f}]")


if __name__ == "__main__":
    main()
