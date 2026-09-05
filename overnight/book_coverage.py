"""Compare two Polyglot books on the platform opening pool.

For each curated FEN: does the book have any entry, and how deep does a
weighted-sample line stay in book from there (the engine's actual probe)?
"""

import sys
from pathlib import Path

import chess
import chess.polyglot

pool = [
    line.strip()
    for line in Path("testing/platform_openings.txt").read_text().splitlines()
    if line.strip()
]


def stats(path: str) -> None:
    reader = chess.polyglot.open_reader(path)
    covered = 0
    total_moves = 0
    total_depth = 0
    for fen in pool:
        board = chess.Board(fen)
        entries = list(reader.find_all(board))
        if entries:
            covered += 1
            total_moves += len(entries)
        depth = 0
        while depth < 8:
            try:
                entry = reader.weighted_choice(board)
            except IndexError:
                break
            board.push(entry.move)
            depth += 1
        total_depth += depth
    size = Path(path).stat().st_size
    print(
        f"{path}: {size / 1e6:.2f} MB, {size // 16:,} entries | "
        f"pool coverage {covered}/{len(pool)} | "
        f"moves per covered position {total_moves / max(1, covered):.1f} | "
        f"mean in-book plies from pool start {total_depth / len(pool):.2f}"
    )
    reader.close()


for book in sys.argv[1:]:
    stats(book)
