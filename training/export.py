"""Export a trained checkpoint to the flat .npz the engine loads.

The engine ships no torch: inference is hand-written numpy, because at batch 1 --
which is all a depth-first search ever asks for -- numpy measured about four times
faster than ONNX Runtime, whose fixed per-call dispatch overhead dominates a network
this small. So the checkpoint is transposed once here into the exact matrices the
engine multiplies, and torch never appears at run time.

Shapes are fixed by the spec and asserted below, because a silently transposed
matrix would still load, still run, and merely play badly.

Float32 throughout. Quantisation is deliberately not done: int16 measured *slower*
than float32 in numpy, since integer paths miss the BLAS route. It is a C++/SIMD
trick that inverts in Python.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from training.train import ACC, FEATURES, HIDDEN, Net

EXPECTED = {
    "W1": (FEATURES, ACC),
    "b1": (ACC,),
    "W2": (2 * ACC, HIDDEN),
    "b2": (HIDDEN,),
    "W3": (HIDDEN, 1),
    "b3": (1,),
}


def convert(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Torch parameters to the engine's matrices.

    torch.nn.Linear stores weights as (out, in) and computes `x @ W.T`; the engine
    computes `x @ W`, so the linear layers are transposed here rather than at every
    node of every search.
    """
    weights = {
        "W1": state["bag.weight"],
        "b1": state["acc_bias"],
        "W2": state["l2.weight"].t(),
        "b2": state["l2.bias"],
        "W3": state["l3.weight"].t(),
        "b3": state["l3.bias"],
    }
    return {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in weights.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint for the engine.")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/net.pt"))
    parser.add_argument("--out", type=Path, default=Path("weights/net.npz"))
    arguments = parser.parse_args()

    state = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    weights = convert(state)

    for name, shape in EXPECTED.items():
        actual = weights[name].shape
        if actual != shape:
            raise SystemExit(f"{name} has shape {actual}, expected {shape}")
        if not np.isfinite(weights[name]).all():
            raise SystemExit(f"{name} contains NaN or infinity")

    # Verify the exported matrices reproduce the torch model, so a transposition or
    # a missing bias cannot slip through. Random accumulators rather than real
    # positions: this checks the arithmetic, not the chess.
    net = Net().eval()
    net.load_state_dict(state)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 2 * ACC)).astype(np.float32)
    with torch.no_grad():
        h1_t = torch.clamp(torch.from_numpy(x), 0.0, 1.0) ** 2
        expected = net.l3(torch.relu(net.l2(h1_t))).squeeze(1).numpy()
    h1 = np.clip(x, 0.0, 1.0) ** 2
    h2 = np.maximum(h1 @ weights["W2"] + weights["b2"], 0.0)
    actual_out = (h2 @ weights["W3"] + weights["b3"]).squeeze(1)
    error = float(np.abs(expected - actual_out).max())
    if error > 1e-3:
        raise SystemExit(f"numpy head disagrees with torch by {error:.4g}")

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    # np.savez's stub types its kwargs as bool; the runtime accepts arrays.
    np.savez(arguments.out, **weights)  # type: ignore[arg-type]
    size = arguments.out.stat().st_size
    parameters = sum(int(np.prod(shape)) for shape in EXPECTED.values())
    print(f"wrote {arguments.out} ({size / 1e6:.2f} MB, {parameters:,} parameters)")
    print(f"numpy head matches torch to {error:.2g}")
    for name, shape in EXPECTED.items():
        low, high = weights[name].min(), weights[name].max()
        print(f"  {name:<3} {shape!s:<12} range [{low:+.3f}, {high:+.3f}]")


if __name__ == "__main__":
    main()
