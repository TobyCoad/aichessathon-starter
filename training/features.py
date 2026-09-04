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

King zones. A net may carry several copies of the first layer, one per zone of the
perspective's own king, so that "knight on f5" can be worth something different
when the king is castled short than when it sits in the centre. The zone is a
property of the king's square *from that perspective* (so mirrored for black), and
the full first-layer index is `zone * 768 + index`. The zone map lives here and
`agent.py` carries a copy that check_nnue compares square by square.
"""

import chess
import numpy as np
import numpy.typing as npt

FEATURES = 768
MAX_PIECES = 32


def king_zone(square: int, zones: int = 8) -> int:
    """Zone of a king on `square`, seen from its own side, for a net with `zones`.

    Eight zones: ranks 1-2 split into four two-file bands, because that is where
    kings spend most of the game and where castled-short, castled-long and
    uncastled differ most; ranks 3-4 and 5-8 each get a queenside and a kingside
    half. Four zones: ranks 1-2 split into halves, then ranks 3-4, then 5-8.
    One zone: everything is zone 0, the plain 768 net.
    """
    rank = square >> 3
    file = square & 7
    if zones == 1:
        return 0
    if zones == 4:
        if rank <= 1:
            return file >> 2
        return 2 if rank <= 3 else 3
    if zones == 8:
        if rank <= 1:
            return file >> 1
        if rank <= 3:
            return 4 + (file >> 2)
        return 6 + (file >> 2)
    if zones == 32:
        # A refinement of the 8-zone map, so an 8-zone net warm-starts it: every
        # square of ranks 1-2 is its own zone, ranks 3-4 split by rank and file
        # pair, ranks 5-8 by rank pair and file pair.
        if rank <= 1:
            return rank * 8 + file
        if rank <= 3:
            return 16 + (rank - 2) * 4 + (file >> 1)
        return 24 + ((rank - 4) >> 1) * 4 + (file >> 1)
    raise ValueError(f"no zone map for {zones} king zones")


KING_ZONES = 8


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
