"""Calibrate internal-score -> centipawn scale for a candidate binpack head.

Same method the 0.262 figure for test80-2024-02 came from: score a sample of the
file's own positions with our Stockfish at a fixed depth and regress.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from training.binpack_decode import VALUE_NONE, iter_chunk

ENGINE = Path(__file__).resolve().parents[4] / "engines/stockfish/stockfish-windows-x86-64-avx2.exe"


def chunks(path: Path):
    data = path.read_bytes()
    at = 0
    while at + 8 <= len(data):
        if data[at : at + 4] != b"BINP":
            return
        size = int.from_bytes(data[at + 4 : at + 8], "little")
        if at + 8 + size > len(data):
            return
        yield data[at + 8 : at + 8 + size]
        at += 8 + size


def sample(path: Path, want: int, stride: int):
    out = []
    n = 0
    for chunk in chunks(path):
        for board, _m, score, ply, _r in iter_chunk(chunk):
            n += 1
            if n % stride:
                continue
            if ply < 16 or abs(score) >= VALUE_NONE or abs(score) > 4000 or board.is_check():
                continue
            out.append((board.fen(), score))
            if len(out) >= want:
                return out
    return out


def main(path: Path, want: int, depth: int) -> None:
    rows = sample(path, want, 977)
    engine = chess.engine.SimpleEngine.popen_uci(str(ENGINE))
    engine.configure({"Threads": 4, "Hash": 128})
    pairs = []
    for fen, internal in rows:
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        pov = info["score"].pov(board.turn)
        if pov.is_mate():
            continue
        pairs.append((internal, pov.score()))
    engine.quit()
    if not pairs:
        print("no pairs")
        return
    num = sum(a * b for a, b in pairs)
    den = sum(a * a for a, _ in pairs)
    ma = statistics.mean(a for a, _ in pairs)
    mb = statistics.mean(b for _, b in pairs)
    va = sum((a - ma) ** 2 for a, _ in pairs) ** 0.5
    vb = sum((b - mb) ** 2 for _, b in pairs) ** 0.5
    cov = sum((a - ma) * (b - mb) for a, b in pairs)
    print(f"== {path.name}  n={len(pairs)} depth={depth}")
    print(f"  scale (cp per internal unit) = {num / den:.4f}")
    print(f"  pearson r = {cov / (va * vb):.4f}")
    print(f"  mean|internal|={statistics.mean(abs(a) for a, _ in pairs):.0f} "
          f"mean|sf_cp|={statistics.mean(abs(b) for _, b in pairs):.0f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
