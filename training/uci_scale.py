r"""Fit the factor that turns a Stockfish UCI score into the sf80kf corpus's cp units.

The sf80kf shards were decoded from raw binpack scores (Stockfish internal units) with
`--scale 0.2479`. A label produced through UCI comes out normalised ("cp" where +100 is
a 50% win chance), a different unit, and the installed Stockfish 18 is not the engine
that scored the binpacks. Rather than derive the constant from two versions' source,
measure it: rebuild boards from held-out sf80kf rows, score them at the same node
count, and regress the stored cp on the UCI cp through the origin. The stored cp also
carries the corpus's lambda=0.75 result blend, which adds noise but no bias in
expectation; the correlation reports how tight the fit is.

  .venv\Scripts\python.exe -m training.uci_scale --rows 1500 --workers 6
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import chess
import chess.engine
import numpy as np

STOCKFISH = Path("engines/stockfish/stockfish-windows-x86-64-avx2.exe")


def board_from_row(row: np.void) -> chess.Board | None:
    """Invert `features.white_indices`: index = (0 own | 384 opp) + (type-1)*64 + square,
    white perspective so squares are unflipped. Castling and en passant are not stored."""
    board = chess.Board(None)
    for i in row["idx"][: int(row["count"])]:
        i = int(i)
        colour = chess.WHITE if i < 384 else chess.BLACK
        piece_type = (i % 384) // 64 + 1
        board.set_piece_at(i % 64, chess.Piece(piece_type, colour))
    board.turn = chess.WHITE if int(row["stm"]) == 1 else chess.BLACK
    if not board.is_valid():
        return None
    return board


def score_many(job: tuple[list[str], int]) -> list[int | None]:
    fens, nodes = job
    out: list[int | None] = []
    with chess.engine.SimpleEngine.popen_uci(str(STOCKFISH.resolve())) as sf:
        sf.configure({"Threads": 1, "Hash": 32})
        for fen in fens:
            info = sf.analyse(chess.Board(fen), chess.engine.Limit(nodes=nodes), game=object())
            score = info["score"].white()
            out.append(None if score.is_mate() else score.score())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, default=Path("data/sf80kf/sf80kf_val.npy"))
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--nodes", type=int, default=80_000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--table", type=Path, default=None, help="write the knots here")
    arguments = parser.parse_args()

    records = np.load(arguments.val, mmap_mode="r")
    rng = np.random.default_rng(arguments.seed)
    picks = rng.choice(len(records), size=arguments.rows * 2, replace=False)
    fens: list[str] = []
    stored: list[int] = []
    for p in picks:
        row = records[p]
        if abs(int(row["cp"])) >= 1900:
            continue
        board = board_from_row(row)
        if board is None or board.is_check():
            continue
        fens.append(board.fen())
        stored.append(int(row["cp"]))
        if len(fens) >= arguments.rows:
            break
    print(f"{len(fens)} boards rebuilt; scoring at {arguments.nodes} nodes ...", flush=True)
    step = (len(fens) + arguments.workers - 1) // arguments.workers
    jobs = [(fens[i : i + step], arguments.nodes) for i in range(0, len(fens), step)]
    with mp.Pool(arguments.workers) as pool:
        uci: list[int | None] = sum(pool.map(score_many, jobs), [])

    x = np.array([u for u in uci if u is not None], dtype=np.float64)
    y = np.array([s for s, u in zip(stored, uci) if u is not None], dtype=np.float64)
    np.save(arguments.val.parent.parent / "ours" / "uci_pairs.npy", np.stack([x, y], axis=1))
    keep = np.abs(x) < 1200
    x, y = x[keep], y[keep]
    slope = float((x * y).sum() / (x * x).sum())
    corr = float(np.corrcoef(x, y)[0, 1])
    resid = y - slope * x
    print(f"rows {len(x)}  slope {slope:.4f}  corr {corr:.3f}  resid sd {resid.std():.0f} cp")
    tight = np.abs(x) < 400
    slope_tight = float((x[tight] * y[tight]).sum() / (x[tight] ** 2).sum())
    print(f"|uci|<400: rows {int(tight.sum())}  slope {slope_tight:.4f}")
    print(f"UCI_SCALE={slope:.4f}")
    # The mapping is not linear (the blend and two engines' units both bend it), so also
    # report a monotone binned table for training.mix_ours to interpolate.
    edges = [-1200, -800, -500, -300, -180, -90, -30, 30, 90, 180, 300, 500, 800, 1200]
    knots = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 15:
            knots.append((float(np.median(x[m])), float(np.median(y[m])), int(m.sum())))
    print("BINS " + " ".join(f"{u:.0f}:{v:.0f}({n})" for u, v, n in knots))
    if arguments.table:
        lines = [f"{u:.1f} {v:.1f}" for u, v, _ in knots]
        arguments.table.write_text("\n".join(lines) + "\n")
        print(f"wrote {arguments.table}")


if __name__ == "__main__":
    main()
