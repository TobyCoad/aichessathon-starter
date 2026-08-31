"""Build a Polyglot opening book from the games already in the Parquet.

Rows in `Lichess/fishnet-evals` are consecutive plies of the same game, so the
moves can be recovered by finding which legal move connects one row to the next.
That means a book needs no new download: the corpus is already on disk, it is CC0,
and every move in it was played by a human rather than derived from an engine.

Why a book is worth shipping at all, beyond playing better moves early: `_budget`
allocates by expected moves remaining, which front-loads badly -- about 40 seconds
of a 120 second clock goes into the first ten moves, a phase where theory already
has the answers and a depth-6 search is guessing. A book answers those instantly
and banks the time for the middlegame.

Explicitly permitted: "Books and tablebases: permitted as shipped data within the
200 MB cap; chess.polyglot and chess.syzygy are in the base image."

Two encoding traps, both of which fail silently:

  * The Polyglot weight field is uint16 and saturates at 65535. Raw game counts
    would clamp the most popular moves into a tie, so counts are scaled per
    position.
  * Polyglot encodes castling as king-takes-own-rook -- e1h1, not e1g1 -- which is
    not what `chess.Move` gives you for a standard game.
"""

import argparse
import multiprocessing as mp
import struct
from collections import Counter
from pathlib import Path

import chess
import chess.polyglot
import pyarrow.parquet as pq

BOOK_PLIES = 20
ENTRY = struct.Struct(">QHHI")


def ply_of(fen: str) -> int:
    """Ply from a FEN without constructing a Board.

    Parsing a FEN properly costs ~38 us and most rows are past book depth, so the
    cheap string form decides what is worth parsing at all.
    """
    parts = fen.split()
    if len(parts) < 2:
        return 10**6
    try:
        fullmove = int(parts[-1]) if parts[-1].isdigit() else 1
    except ValueError:
        fullmove = 1
    return (fullmove - 1) * 2 + (0 if parts[1] == "w" else 1)


def connecting_move(before: chess.Board, after: chess.Board) -> chess.Move | None:
    """The legal move taking `before` to `after`, or None if they are not adjacent."""
    target = after._transposition_key()
    for move in before.legal_moves:
        before.push(move)
        try:
            if before._transposition_key() == target:
                return move
        finally:
            before.pop()
    return None


def polyglot_move(board: chess.Board, move: chess.Move) -> int:
    """Encode a move the way Polyglot does, castling as king-takes-own-rook."""
    to_square = move.to_square
    if board.is_castling(move):
        rank = 0 if board.turn == chess.WHITE else 56
        to_square = (7 + rank) if move.to_square > move.from_square else (0 + rank)
    promotion = 0 if move.promotion is None else move.promotion - 1
    return to_square | (move.from_square << 6) | (promotion << 12)


def scan_group(job: tuple[Path, int]) -> Counter[tuple[int, int]]:
    """Count (position, move) pairs inside book depth for one row group."""
    path, group = job
    table = pq.ParquetFile(path).read_row_group(group, columns=["fen"])
    counts: Counter[tuple[int, int]] = Counter()

    previous: chess.Board | None = None
    previous_ply = -99
    for fen in table["fen"].to_pylist():
        ply = ply_of(fen)
        if ply >= BOOK_PLIES:
            previous, previous_ply = None, -99
            continue
        try:
            board = chess.Board(fen)
        except ValueError:
            previous, previous_ply = None, -99
            continue

        # A game's first recorded row is after White's opening move, so the
        # position before it is the standard start and that move is recoverable.
        if ply == 1 and previous is None:
            previous, previous_ply = chess.Board(), 0

        if previous is not None and ply == previous_ply + 1:
            move = connecting_move(previous, board)
            if move is not None:
                counts[(chess.polyglot.zobrist_hash(previous), polyglot_move(previous, move))] += 1
        previous, previous_ply = board, ply
    return counts


def write_book(counts: Counter[tuple[int, int]], destination: Path, minimum: int) -> int:
    """Write a Polyglot .bin: 16-byte entries, sorted ascending by key."""
    by_key: dict[int, list[tuple[int, int]]] = {}
    for (key, move), count in counts.items():
        if count >= minimum:
            by_key.setdefault(key, []).append((move, count))

    payload = bytearray()
    for key in sorted(by_key):
        moves = by_key[key]
        # uint16 saturates at 65535, so scale each position's counts into range
        # rather than clamping the popular moves into a tie.
        top = max(count for _, count in moves)
        scale = min(1.0, 60000.0 / top)
        for move, count in sorted(moves, key=lambda pair: -pair[1]):
            payload += ENTRY.pack(key, move, max(1, int(count * scale)), 0)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(bytes(payload))
    return len(payload) // ENTRY.size


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Polyglot opening book.")
    parser.add_argument("--source", type=Path, default=Path("data/standard_rated_2025_01.parquet"))
    parser.add_argument("--out", type=Path, default=Path("weights/book.bin"))
    parser.add_argument("--groups", type=int, default=60, help="row groups to scan")
    parser.add_argument("--min-count", type=int, default=3, help="drop rarer moves")
    parser.add_argument("--workers", type=int, default=0)
    arguments = parser.parse_args()

    workers = arguments.workers or max(1, (mp.cpu_count() or 4) - 2)
    available = pq.ParquetFile(arguments.source).metadata.num_row_groups
    groups = list(range(min(arguments.groups, available)))
    print(f"scanning {len(groups)} of {available} row groups on {workers} workers")

    counts: Counter[tuple[int, int]] = Counter()
    jobs = [(arguments.source, group) for group in groups]
    with mp.Pool(workers) as pool:
        for done, part in enumerate(pool.imap_unordered(scan_group, jobs), start=1):
            counts.update(part)
            if done % 10 == 0 or done == len(groups):
                print(f"  {done}/{len(groups)} groups: {len(counts):,} distinct moves", flush=True)

    positions = len({key for key, _ in counts})
    entries = write_book(counts, arguments.out, arguments.min_count)
    size = arguments.out.stat().st_size
    print(f"\n{len(counts):,} distinct (position, move) pairs over {positions:,} positions")
    print(f"kept {entries:,} played at least {arguments.min_count} times")
    print(f"wrote {arguments.out} ({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
