"""Correctness checks for the 768-feature encoding.

Deliberately written against a naive `board.piece_map()` loop rather than reusing
anything from `features.py`, so a misunderstanding cannot satisfy both. The point of
this file is that a wrong encoding is otherwise silent: the net trains, the engine
runs, and it just plays worse.

Run: .venv\\Scripts\\python.exe -m training.check_features
Exits non-zero on any failure.
"""

import random
import sys

import chess
import numpy as np

from training import features


def naive_indices(board: chess.Board, perspective: chess.Color) -> list[int]:
    """An independent implementation. Slow, obvious, and written from the spec."""
    out: list[int] = []
    for square, piece in board.piece_map().items():
        # Spelled out rather than a ternary: this implementation exists to be
        # obviously, independently readable against the spec, and tightening it
        # toward the one in features.py defeats the point of having two.
        if perspective == chess.WHITE:  # noqa: SIM108
            rel = square
        else:
            rel = square ^ 56
        own = piece.color == perspective
        out.append((0 if own else 384) + (piece.piece_type - 1) * 64 + rel)
    return out


def positions(count: int, seed: int = 0) -> list[chess.Board]:
    """Random positions from random play, including endgames and promotions."""
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    board = chess.Board()
    while len(boards) < count:
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            board = chess.Board()
            continue
        board.push(rng.choice(moves))
        if rng.random() < 0.3:
            boards.append(board.copy(stack=False))
        if len(board.move_stack) > 160:
            board = chess.Board()
    return boards


def main() -> None:
    boards = positions(3000)
    failures = 0

    for board in boards:
        for perspective, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
            fast = sorted(features.indices(board, perspective))
            slow = sorted(naive_indices(board, perspective))
            if fast != slow:
                print(f"FAIL bitboard vs piece_map ({name}): {board.fen()}")
                failures += 1

    for board in boards:
        white = features.white_indices(board)
        derived = sorted(features.black_from_white(np.array(white)).tolist())
        direct = sorted(features.black_indices(board))
        if derived != direct:
            print(f"FAIL black_from_white: {board.fen()}")
            failures += 1

    for board in boards:
        flat = features.white_indices(board)
        if any(not 0 <= i < features.FEATURES for i in flat):
            print(f"FAIL index out of range: {board.fen()}")
            failures += 1
        if len(flat) != len(board.piece_map()):
            print(f"FAIL wrong piece count: {board.fen()}")
            failures += 1
        if len(set(flat)) != len(flat):
            print(f"FAIL duplicate index: {board.fen()}")
            failures += 1

    # Mirroring the board must swap the two perspectives exactly. This is the
    # strongest single invariant available: it catches a wrong flip, a wrong
    # own/opponent offset, and a wrong piece-type stride all at once.
    for board in boards:
        if sorted(features.white_indices(board)) != sorted(
            features.black_indices(board.mirror())
        ):
            print(f"FAIL mirror invariance: {board.fen()}")
            failures += 1

    # Every feature must be reachable, or part of the input layer is dead weight.
    seen = set()
    for board in positions(4000, seed=7):
        seen.update(features.white_indices(board))
    coverage = len(seen) / features.FEATURES

    print(f"positions checked      : {len(boards)}")
    print("bitboard vs piece_map  : both perspectives")
    print("black_from_white       : matches direct computation")
    print("range, count, distinct : checked")
    print("mirror invariance      : white(b) == black(mirror(b))")
    print(f"feature coverage       : {coverage:.1%} of 768 reached by random play")

    if failures:
        print(f"\n{failures} FAILURES")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
