"""Build a pairwise/dual/endgame-bucket checkpoint that starts life as its parent.

A new architecture normally means training from scratch, and from scratch is a
losing move here: v15 was exactly that -- 300 epochs on the wide mix -- and it came
out worse than v12 on both gates (eval_rank top-loss 248 against 201, replay 38
fixed / 3 worse against 41 / 1). So the architecture change is arranged to be an
identity at initialisation instead, and the run continues v12 rather than
restarting from noise.

Three changes, each warm-started exactly:

  pairwise   The accumulator doubles to 2A and both halves start as copies of the
             parent's single half, so `clip(a_i) * clip(a_{i+A})` is `clip(x)**2`
             -- the parent's SCReLU, exactly -- and head_w2 copies over verbatim,
             because a pairwise net's head input (2A halved) is the same width as
             a squaring net's (A doubled). The two halves are then jittered by
             +/-`--jitter` in opposite directions: identical halves receive
             identical gradients and would stay welded together for ever, and the
             product w(1+e)*w(1-e) = w**2(1-e**2) perturbs the function only to
             second order.

  dual       The second activation's output weights start at zero, so it
             contributes nothing until it learns something.

  buckets    The endgame-dense map is a different partition, so each new head is
             seeded from the parent head that covered most of its piece counts.
             This is the one approximation here; the net starts a little off on
             head boundaries and recovers in the first epochs.

Check the result rather than trusting it: `--verify` reports the parent's and the
child's validation loss on the same data, which should agree to about 1e-5.

  .venv/Scripts/python.exe -m training.warm_start --parent training/checkpoints/net_v12.pt \
      --out training/checkpoints/net_v16_init.pt --verify data/mixed_val.npy
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from training.train import Batches, Net, evaluate_loss, load_checkpoint


def head_parents(new_map: list[int], old_map: list[int]) -> list[int]:
    """For each head of the new map, the old head covering most of its counts.

    Counts are piece counts 0..32 and every one of them is a position the net will
    actually be asked about, so a plain majority over the range is the right
    seeding rule; ties go to the lower old head, which is arbitrary but fixed.
    """
    parents = []
    for head in range(max(new_map) + 1):
        counts = [old_map[c] for c in range(len(new_map)) if new_map[c] == head]
        if not counts:
            raise ValueError(f"new head {head} covers no piece count")
        tally = Counter(counts)
        best = max(tally.values())
        parents.append(min(k for k, v in tally.items() if v == best))
    return parents


def warm_start(parent: Net, jitter: float, seed: int = 0) -> tuple[Net, list[int]]:
    accumulator = int(parent.bag.weight.shape[1])
    hidden = int(parent.head_w2.shape[2])
    if parent.is_pairwise or parent.is_dual:
        raise SystemExit("parent is already a pairwise/dual net; nothing to warm-start")
    if getattr(parent, "has_factoriser", False):
        # The child is built without one and only `bag.weight` is copied below, so the
        # shared plane -- which holds most of the first layer's signal, rms 0.104 against
        # the zoned plane's 0.015 on the smoke net -- would be dropped on the floor and
        # the "identity at initialisation" this file exists for would be a fiction.
        # --verify would not reliably catch it either: its guard only trips at 1.5x worse.
        # Folding it in first is the fix, and it is one line; it is left undone rather
        # than done untested, because nothing has needed it yet.
        raise SystemExit(
            "parent has a factoriser: fold it into bag.weight before warm-starting "
            "(child = bag + factor.repeat(king_zones, 1)), or this drops it silently"
        )

    child = Net(
        accumulator=2 * accumulator,
        hidden=hidden,
        buckets=parent.buckets,
        king_zones=parent.king_zones,
        mirrored=parent.is_mirrored,
        pairwise=True,
        dual=True,
        endgame_dense=True,
    )

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        # Two halves of one parent column, pushed apart symmetrically.
        noise = torch.randn(parent.bag.weight.shape, generator=generator) * jitter
        child.bag.weight.copy_(
            torch.cat([parent.bag.weight * (1 + noise), parent.bag.weight * (1 - noise)], dim=1)
        )
        child.acc_bias.copy_(torch.cat([parent.acc_bias, parent.acc_bias]))

        old_map = parent.bucket_map.tolist()
        new_map = child.bucket_map.tolist()
        parents = head_parents(new_map, old_map)
        for head, source in enumerate(parents):
            # head_w2 needs no reshaping: the parent's head input is 2A (own and
            # opponent squared) and the child's is 2A too (own and opponent
            # pairwise-collapsed from 4A), in the same order.
            child.head_w2[head].copy_(parent.head_w2[source])
            child.head_b2[head].copy_(parent.head_b2[source])
            child.head_w3[head, :hidden].copy_(parent.head_w3[source])
            child.head_w3[head, hidden:].zero_()
            child.head_b3[head].copy_(parent.head_b3[source])
    return child, parents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jitter", type=float, default=0.02, help="symmetry-breaking scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mirror", action="store_true", default=True)
    ap.add_argument("--verify", type=Path, default=None, help="validation shard")
    ap.add_argument("--batch", type=int, default=16384)
    args = ap.parse_args()

    parent = load_checkpoint(args.parent, mirror=args.mirror).eval()
    child, parents = warm_start(parent, args.jitter, args.seed)
    child.eval()

    print(f"parent: A={parent.bag.weight.shape[1]} heads={parent.buckets} map={parent.bucket_map.tolist()}")
    print(f"child : A={child.bag.weight.shape[1]} heads={child.buckets} map={child.bucket_map.tolist()}")
    print(f"head seeding (new -> old): {list(enumerate(parents))}")

    if args.verify is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        records = np.load(args.verify, mmap_mode="r")
        batches = Batches(records, args.batch, device)
        generator = torch.Generator().manual_seed(0)
        before = evaluate_loss(parent.to(device), batches, generator)
        generator = torch.Generator().manual_seed(0)
        after = evaluate_loss(child.to(device), batches, generator)
        print(f"\nvalidation on {args.verify.name}: parent {before:.8f}  child {after:.8f}")
        print(f"  difference {after - before:+.8f} ({100 * (after - before) / before:+.3f}%)")
        if after > before * 1.5:
            raise SystemExit(
                "child is far worse than its parent: the warm start is not an identity"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child.state_dict(), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
