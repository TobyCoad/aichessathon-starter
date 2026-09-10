r"""Turn training.gen_ours raw rows into sf80kf-convention shards, mixed with the SF and
human shards the champion trained on, plus a validation set that can SEE our positions.

  cp mapping  Stockfish 18 UCI cp -> corpus cp by the binned table training.uci_scale
              wrote (piecewise linear, end slopes extrapolated), mates -> +-2000.
  blend       blend_wdl(cp, white_result, lambda) with lambda 0.75, exactly the decode.
  split       the last --val-frac of each worker file is validation: rows are appended a
              game at a time, so a contiguous tail leaks at most one game.
  shards      mix_k = --sf-rows from sf80kf shard 5k + --hu-rows from a human shard + all
              of our training rows. One shard per epoch, so every epoch sees our rows
              once and a fresh slice of the old corpus.
  val         our validation rows + twice as many rows of the champion's own validation
              set (data/mixed_val.npy). Early-stopping on the old val alone would restore
              the parent -- v16g showed the old val does not move at a continuation lr --
              while ours alone would let the net drift off the SF corpus.

  .venv\Scripts\python.exe -m training.mix_ours --raw data/ours --table data/ours/uci_table.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.binpack_decode import CP_CLAMP, RECORD, blend_wdl  # noqa: E402

SF = [Path(f"data/sf80kf/sf80kf_{(5 * k) % 39:02d}.npy") for k in range(8)]
HU = [
    Path("data/positions_w512-150m.npy"),
    Path("data/positions_2025_02.npy"),
    Path("data/positions_w512-150m-b.npy"),
    Path("data/positions_2025_03.npy"),
]


def mapping(table: Path) -> tuple[np.ndarray, np.ndarray]:
    knots = np.loadtxt(table, dtype=np.float64).reshape(-1, 2)
    order = np.argsort(knots[:, 0])
    return knots[order, 0], knots[order, 1]


def corpus_cp(uci: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    lo_slope = ys[0] / xs[0] if xs[0] != 0 else 1.0
    hi_slope = ys[-1] / xs[-1] if xs[-1] != 0 else 1.0
    out = np.interp(uci, xs, ys)
    out = np.where(uci < xs[0], uci * lo_slope, out)
    out = np.where(uci > xs[-1], uci * hi_slope, out)
    out = np.where(np.abs(uci) >= 3000, np.sign(uci) * CP_CLAMP, out)
    return np.clip(np.rint(out), -CP_CLAMP, CP_CLAMP).astype(np.int64)


def convert(raw: np.ndarray, xs: np.ndarray, ys: np.ndarray, lam: float) -> np.ndarray:
    out = np.zeros(len(raw), dtype=RECORD)
    out["idx"] = raw["idx"]
    out["count"] = raw["count"]
    out["stm"] = raw["stm"]
    cp = corpus_cp(raw["cp"].astype(np.float64), xs, ys)
    res = raw["result"].astype(np.int64)
    if lam < 1.0:
        cp = np.array([blend_wdl(int(c), int(r), lam) for c, r in zip(cp, res)], dtype=np.int64)
    out["cp"] = np.clip(cp, -CP_CLAMP, CP_CLAMP)
    return out


def sample(path: Path, rows: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    picks = np.sort(rng.choice(len(arr), size=min(rows, len(arr)), replace=False))
    return np.asarray(arr[picks])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/ours"))
    parser.add_argument("--table", type=Path, default=Path("data/ours/uci_table.txt"))
    parser.add_argument("--lam", type=float, default=0.75)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("data/mix_ours"))
    parser.add_argument("--sf-rows", type=int, default=6_000_000)
    parser.add_argument("--hu-rows", type=int, default=4_000_000)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--old-val", type=Path, default=Path("data/mixed_val.npy"))
    parser.add_argument("--seed", type=int, default=0)
    a = parser.parse_args()

    xs, ys = mapping(a.table)
    print("mapping knots:", " ".join(f"{x:.0f}->{y:.0f}" for x, y in zip(xs, ys)))
    train_parts, val_parts = [], []
    games_w = games_d = 0
    for f in sorted(a.raw.glob("ours_*.npy")):
        raw = np.load(f)
        cut = len(raw) - int(len(raw) * a.val_frac)
        train_parts.append(convert(raw[:cut], xs, ys, a.lam))
        val_parts.append(convert(raw[cut:], xs, ys, a.lam))
        res = raw["result"]
        games_d += int((res == 0).sum())
        games_w += int((res != 0).sum())
        print(f"  {f.name}: {len(raw):,} rows", flush=True)
    ours_train = np.concatenate(train_parts)
    ours_val = np.concatenate(val_parts)
    print(
        f"ours: {len(ours_train):,} train / {len(ours_val):,} val rows; "
        f"rows from decisive games {games_w / max(1, games_w + games_d):.0%}; "
        f"|cp| mean {np.abs(ours_train['cp']).mean():.0f}, stm-white {ours_train['stm'].mean():.2f}",
        flush=True,
    )
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "ours_train.npy", ours_train)
    np.save(a.out / "ours_val.npy", ours_val)

    rng = np.random.default_rng(a.seed)
    old = sample(a.old_val, 2 * len(ours_val), rng)
    np.save(a.out / "val.npy", np.concatenate([ours_val, old]))
    print(f"val: {len(ours_val):,} ours + {len(old):,} old", flush=True)
    for k in range(a.shards):
        sf = sample(SF[k], a.sf_rows, rng)
        hu = sample(HU[k % 4], a.hu_rows, rng)
        shard = np.concatenate([sf, hu, ours_train])
        np.save(a.out / f"mix_{k}.npy", shard)
        print(
            f"  mix_{k}: {len(shard):,} rows = {len(sf):,} {SF[k].name} + {len(hu):,} "
            f"{HU[k % 4].name} + {len(ours_train):,} ours ({len(ours_train) / len(shard):.0%})",
            flush=True,
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
