"""Offline ENDGAME_SHRINK calibration (overnight/eval/v10/endgame_shrink.md sec 5).

No search, no Stockfish: evaluate each of the 400 labelled positions in
overnight/eval/endgame_suite.json once with the champion net (ENDGAME_SHRINK is
off in the tree, so FastEngine.evaluate returns the raw net eval), then sweep
the blend schedule in numpy. Also prints the static-error-by-band instrument
games.md asked for. One process, ~2 minutes including the agent import.

Usage: .venv/Scripts/python.exe -m testing.eg_calib
"""

from __future__ import annotations

import json
import time

import chess
import numpy as np


def main() -> None:
    t0 = time.monotonic()
    import agent
    import fastsearch as fs

    print(f"agent imported in {time.monotonic() - t0:.1f}s")
    with open("overnight/eval/endgame_suite.json", encoding="utf-8") as fh:
        positions = json.load(fh)["positions"]

    eng = agent.FastEngine()
    nets, mats, pcs, labs = [], [], [], []
    t1 = time.monotonic()
    for p in positions:
        board = chess.Board(p["fen"])
        eng.prepare(board, 0)
        nets.append(eng.evaluate())
        mats.append(int(fs.simple_eval(eng.pos.bb, eng.pos.meta)))
        pcs.append(int(eng.pos.meta[5]))
        labs.append(int(p["eval"]))
    print(f"{len(labs)} positions evaluated in {time.monotonic() - t1:.1f}s")

    net = np.array(nets, dtype=np.int64)
    mat = np.array(mats, dtype=np.int64)
    pc = np.array(pcs, dtype=np.int64)
    lab = np.array(labs, dtype=np.int64)
    np.savez(  # reusable instrument: score any future net without an engine run
        "overnight/eval/v10/eg_calib.npz", net=net, mat=mat, pieces=pc, label=lab
    )
    bands = [(5, 8), (9, 12), (13, 16)]
    base_err = np.abs(net - lab)
    mat_err = np.abs(mat - lab)

    print("\nstatic |eval - label| by piece-count band (the games.md instrument):")
    for lo, hi in bands:
        m = (pc >= lo) & (pc <= hi)
        print(
            f"  {lo:2d}-{hi:2d}: n={int(m.sum()):3d}"
            f"  net {float(base_err[m].mean()):7.1f}"
            f"  material {float(mat_err[m].mean()):7.1f}"
        )
    print(
        f"  all  : n={len(labs):3d}  net {float(base_err.mean()):7.1f}"
        f"  material {float(mat_err.mean()):7.1f}"
    )

    print("\nblend sweep (mean |blended - label| per band; moved = |blended - net|):")
    for wmin in (128, 153, 179, 205, 230):
        ramp = wmin + (256 - wmin) * (pc - fs.EG_LO) // (fs.EG_HI - fs.EG_LO)
        w = np.where(pc <= fs.EG_LO, wmin, ramp)
        w = np.where(pc >= fs.EG_HI, 256, w)
        for cap in (150, 300, 450, 600, 0):
            delta = (256 - w) * (mat - net) // 256
            if cap > 0:
                delta = np.clip(delta, -cap, cap)
            blended = net + delta
            err = np.abs(blended - lab)
            moved = np.abs(delta)
            parts = [f"WMIN {wmin:3d} CAP {cap:3d}:"]
            for lo, hi in bands:
                m = (pc >= lo) & (pc <= hi)
                parts.append(f"{lo}-{hi} {float(err[m].mean()):6.1f}")
            parts.append(f"all {float(err.mean()):6.1f}")
            parts.append(f"moved>100 {int((moved > 100).sum()):3d}")
            parts.append(f"maxmove {int(moved.max()):4d}")
            parts.append(f"worse {int((err > base_err).sum()):3d}")
            print("  " + "  ".join(parts))


if __name__ == "__main__":
    main()
