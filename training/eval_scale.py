"""Per-band eval SCALE, not accuracy: the quantity every pruning margin reads.

RFP (80*depth), futility (0/150/300), razoring and the NMP margin all compare the
raw static eval against fixed centipawn thresholds.  Two nets that are equally
ACCURATE but differ in slope or in how often they cross those thresholds prune
different trees at the same depth.

  .venv\\Scripts\\python.exe -m training.eval_scale --val data/sf20k/sf20k_val.npy \
      --checkpoints A.pt B.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from training.train import SCALE, Batches, load_checkpoint

BANDS = ((2, 8), (9, 12), (13, 16), (17, 20), (21, 32))
THRESH = (80, 240, 400, 800)


@torch.no_grad()
def collect(net, batches, generator):
    net.eval()
    out = {b: [[], []] for b in BANDS}
    for white, black, mask, stm, target in batches.epoch(generator):
        pred = net(white, black, mask, stm) * SCALE
        pieces = mask.sum(1)
        for lo, hi in BANDS:
            sel = (pieces >= lo) & (pieces <= hi)
            out[(lo, hi)][0].append(target[sel].cpu().numpy())
            out[(lo, hi)][1].append(pred[sel].cpu().numpy())
    net.train()
    return {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--limit", type=int, default=500_000)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = np.asarray(np.load(a.val, mmap_mode="r")[: a.limit])
    batches = Batches(records, 16384, device)
    print(f"{len(records):,} positions from {a.val.name}\n")

    data = []
    for ck in a.checkpoints:
        net = load_checkpoint(ck, 8, 16, True).to(device)  # mirrored nets since v12
        data.append((ck.stem[-18:], collect(net, batches, torch.Generator().manual_seed(0))))

    print(f"{'band':>7} {'net':>19} {'n':>9} {'slope':>7} {'sd(pred)':>9} {'sd(lab)':>8}  "
          + "  ".join(f"P|e|>{t}" for t in THRESH))
    for band in BANDS:
        for name, d in data:
            y, x = d[band]  # y = label, x = prediction
            slope = float(np.polyfit(y, x, 1)[0])  # prediction on label
            row = (f"{band[0]}-{band[1]:>2}".rjust(7) + f" {name:>19} {len(y):>9,} "
                   f"{slope:7.3f} {x.std():9.1f} {y.std():8.1f}  ")
            row += "  ".join(f"{(np.abs(x) > t).mean():7.2%}" for t in THRESH)
            print(row)
        print()


if __name__ == "__main__":
    main()
