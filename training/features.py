"""The 768-feature encoding. This file is the contract between training and inference.

If the indices here disagree with the ones `agent.py` builds at run time, the net
still loads, the engine still runs, the crash gate still passes, and it simply plays
badly. There is no error to catch. So there is exactly one definition of the
encoding, it lives here, and the engine's copy is checked against it by a test.

The scheme is the plain 768 set -- 6 piece types x 2 colours x 64 squares -- taken
from each side's perspective, giving the standard `(768 -> N)x2` accumulator shape.
Not HalfKP: its 40,960 inputs need 53x the parameters and far more data than a
twelve-day project can process, and both published from-scratch nets that reported
real Elo used 768.

    rel   = sq if perspective is white else sq ^ 56
    own   = (piece colour == perspective)
    index = (0 if own else 384) + (piece_type - 1) * 64 + rel

Indices run 0..767: the first 384 are the perspective's own pieces, the second 384
the opponent's, and the board is flipped vertically for the black perspective so
that "my back rank" is always rank 1.
"""

import chess
import numpy as np
import numpy.typing as npt

FEATURES = 768
MAX_PIECES = 32


def indices(board: chess.Board, perspective: chess.Color) -> list[int]:
    """Feature indices for every piece on the board, from `perspective`.

    Bit-scanning the twelve piece bitboards rather than calling `board.piece_map()`,
    which allocates a dict of Piece objects and measured 4x slower.
    """
    out: list[int] = []
    flip = 0 if perspective == chess.WHITE else 56
    for piece_type in range(1, 7):
        base = (piece_type - 1) * 64
        for colour in (chess.WHITE, chess.BLACK):
            offset = base if colour == perspective else base + 384
            mask = board.pieces_mask(piece_type, colour)
            while mask:
                square = (mask & -mask).bit_length() - 1
                out.append(offset + (square ^ flip))
                mask &= mask - 1
    return out


def white_indices(board: chess.Board) -> list[int]:
    return indices(board, chess.WHITE)


def black_indices(board: chess.Board) -> list[int]:
    return indices(board, chess.BLACK)


def black_from_white(white: npt.NDArray[np.integer]) -> npt.NDArray[np.int64]:
    """Convert white-perspective indices to black-perspective ones, vectorised.

    The packed dataset stores only the white perspective, halving its size; the
    training step derives the black perspective with this. Swapping perspective
    swaps the own/opponent half and mirrors the square vertically.
    """
    w = np.asarray(white, dtype=np.int64)
    own = w // 384
    piece = (w % 384) // 64
    square = w % 64
    result: npt.NDArray[np.int64] = (384 - own * 384) + piece * 64 + (square ^ 56)
    return result


def feature_index(
    square: int, piece_type: int, colour: chess.Color, perspective: chess.Color
) -> int:
    """Single-piece index. The engine's incremental accumulator uses this shape."""
    rel = square if perspective == chess.WHITE else square ^ 56
    own = 0 if colour == perspective else 384
    return own + (piece_type - 1) * 64 + rel
