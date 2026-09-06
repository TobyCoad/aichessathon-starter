"""Decode Stockfish NNUE training data (.binpack / .binpack.zst) straight into the
packed training records that training/train.py consumes.

The binpack format (nnue-pytorch, data_loader/cpp/lib/binpack.h): a file is a
sequence of chunks, each `BINP` + little-endian uint32 size + payload. A payload is a
sequence of 32-byte entries -- 8-byte big-endian occupancy bitboard, 16 bytes of
piece nibbles in bitboard order (low nibble first; 12 = pawn that just double-pushed,
13/14 = rook with castling rights, 15 = black king and black to move), 2-byte move,
2-byte score, 2-byte ply|result<<14, 2-byte rule50 -- each followed by a 2-byte
big-endian ply count and that many (move, score-delta) pairs bit-packed as indices
into the side to move's ordered move sets, exactly as `PackedMoveScoreListReader`
reads them. Scores are side-to-move relative, in Stockfish internal units; `--scale`
converts them to white-POV centipawns (calibrate with --sample against Stockfish).

  .venv/Scripts/python.exe -m training.binpack_decode data/sf/x.binpack.zst --out data/sf/x \
      --target 580000000 --workers 8 --scale 0.48

Writes `<out>_NN.npy` shards of RECORD arrays (see training/pack.py) plus
`<out>_val.npy`, applying pack.py's quiet filters (no checks, best move not a
capture, quiet-fraction balance).
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.features import MAX_PIECES, white_indices
from training.pack import CP_CLAMP, RECORD, balance

NIBBLE_PIECE = {
    0: (chess.PAWN, chess.WHITE),
    1: (chess.PAWN, chess.BLACK),
    2: (chess.KNIGHT, chess.WHITE),
    3: (chess.KNIGHT, chess.BLACK),
    4: (chess.BISHOP, chess.WHITE),
    5: (chess.BISHOP, chess.BLACK),
    6: (chess.ROOK, chess.WHITE),
    7: (chess.ROOK, chess.BLACK),
    8: (chess.QUEEN, chess.WHITE),
    9: (chess.QUEEN, chess.BLACK),
    10: (chess.KING, chess.WHITE),
    11: (chess.KING, chess.BLACK),
}
SCORE_VLE_BLOCK = 4
VALUE_NONE = 32002  # Stockfish's 'no score' marker; mates sit just below it


def unsigned_to_signed(r: int) -> int:
    r = ((r << 15) | (r >> 1)) & 0xFFFF
    if r & 0x8000:
        r ^= 0x7FFF
    return r - 0x10000 if r & 0x8000 else r


def used_bits_safe(value: int) -> int:
    return 0 if value == 0 else (value - 1).bit_length()


def nth_set_bit(bb: int, n: int) -> int:
    """Index of the n-th (0-based) set bit from the least significant end."""
    for _ in range(n):
        bb &= bb - 1
    return (bb & -bb).bit_length() - 1


class BitReader:
    """`extractBitsLE8`: bits are consumed from the high end of each byte."""

    __slots__ = ("bits_left", "data", "offset", "start")

    def __init__(self, data: bytes | memoryview, offset: int) -> None:
        self.data = data
        self.offset = offset
        self.start = offset
        self.bits_left = 8

    def take(self, count: int) -> int:
        if count == 0:
            return 0
        if self.bits_left == 0:
            self.offset += 1
            self.bits_left = 8
        byte = (self.data[self.offset] << (8 - self.bits_left)) & 0xFF
        bits = byte >> (8 - count)
        if count > self.bits_left:
            spill = count - self.bits_left
            bits |= self.data[self.offset + 1] >> (8 - spill)
            self.bits_left += 8
            self.offset += 1
        self.bits_left -= count
        return bits

    def vle16(self, block: int) -> int:
        mask = (1 << block) - 1
        value = 0
        shift = 0
        while True:
            chunk = self.take(block + 1)
            value |= (chunk & mask) << shift
            if not (chunk >> block):
                return value & 0xFFFF
            shift += block

    def bytes_used(self) -> int:
        return self.offset - self.start + (1 if self.bits_left != 8 else 0)


def board_from_entry(
    data: memoryview, at: int
) -> tuple[chess.Board, chess.Move | None, int, int, int]:
    """Decode one 32-byte entry: board, move, side-to-move score, ply, result."""
    occ = int.from_bytes(data[at : at + 8], "big")
    board = chess.Board(None)
    board.castling_rights = 0
    turn = chess.WHITE
    ep = None
    i = 0
    bb = occ
    while bb:
        sq = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        byte = data[at + 8 + (i >> 1)]
        nib = (byte >> 4) if (i & 1) else (byte & 0xF)
        i += 1
        if nib < 12:
            pt, colour = NIBBLE_PIECE[nib]
            board._set_piece_at(sq, pt, colour)
        elif nib == 12:
            if chess.square_rank(sq) == 3:
                board._set_piece_at(sq, chess.PAWN, chess.WHITE)
                ep = sq - 8
            else:
                board._set_piece_at(sq, chess.PAWN, chess.BLACK)
                ep = sq + 8
        elif nib == 13:
            board._set_piece_at(sq, chess.ROOK, chess.WHITE)
            board.castling_rights |= chess.BB_A1 if sq == chess.A1 else chess.BB_H1
        elif nib == 14:
            board._set_piece_at(sq, chess.ROOK, chess.BLACK)
            board.castling_rights |= chess.BB_A8 if sq == chess.A8 else chess.BB_H8
        else:
            board._set_piece_at(sq, chess.KING, chess.BLACK)
            turn = chess.BLACK
    board.turn = turn
    board.ep_square = ep
    packed_move = (data[at + 24] << 8) | data[at + 25]
    score = unsigned_to_signed((data[at + 26] << 8) | data[at + 27])
    pr = (data[at + 28] << 8) | data[at + 29]
    ply = pr & 0x3FFF
    result = unsigned_to_signed(pr >> 14)
    board.halfmove_clock = (data[at + 30] << 8) | data[at + 31]
    board.fullmove_number = ply // 2 + 1
    move = decompress_move(packed_move, board)
    return board, move, score, ply, result


def decompress_move(packed: int, board: chess.Board) -> chess.Move | None:
    """None for a null / marker move (packed 0 or 0xFFFF, or from == to)."""
    if packed == 0 or packed == 0xFFFF:
        return None
    mtype = packed >> 14
    frm = (packed >> 8) & 63
    to = (packed >> 2) & 63
    if frm == to:
        return None
    if mtype == 1:  # promotion
        return chess.Move(frm, to, promotion=chess.KNIGHT + (packed & 3))
    if mtype == 2:  # castle: king square -> rook square in this library
        rank = chess.square_rank(frm)
        king_to = chess.square(6 if chess.square_file(to) > chess.square_file(frm) else 2, rank)
        return chess.Move(frm, king_to)
    return chess.Move(frm, to)


def next_move_score(
    reader: BitReader, board: chess.Board, last_score: int
) -> tuple[chess.Move, int]:
    stm = board.turn
    ours = board.occupied_co[stm]
    theirs = board.occupied_co[not stm]
    occupied = ours | theirs
    piece_id = reader.take(used_bits_safe(bin(ours).count("1")))
    frm = nth_set_bit(ours, piece_id)
    pt = board.piece_type_at(frm)
    if pt == chess.PAWN:
        forward = 8 if stm == chess.WHITE else -8
        start_rank = 1 if stm == chess.WHITE else 6
        promo_rank = 6 if stm == chess.WHITE else 1
        targets = theirs | (chess.BB_SQUARES[board.ep_square] if board.ep_square is not None else 0)
        dests = chess.BB_PAWN_ATTACKS[stm][frm] & targets
        one = frm + forward
        if not (occupied >> one) & 1:
            dests |= 1 << one
            if chess.square_rank(frm) == start_rank and not (occupied >> (one + forward)) & 1:
                dests |= 1 << (one + forward)
        count = bin(dests).count("1")
        if chess.square_rank(frm) == promo_rank:
            move_id = reader.take(used_bits_safe(count * 4))
            to = nth_set_bit(dests, move_id // 4)
            move = chess.Move(frm, to, promotion=chess.KNIGHT + (move_id % 4))
        else:
            move_id = reader.take(used_bits_safe(count))
            to = nth_set_bit(dests, move_id)
            move = chess.Move(frm, to)
    elif pt == chess.KING:
        attacks = chess.BB_KING_ATTACKS[frm] & ~ours
        size = bin(attacks).count("1")
        rights = board.castling_rights & (
            chess.BB_RANK_1 if stm == chess.WHITE else chess.BB_RANK_8
        )
        num_castlings = bin(rights).count("1")
        move_id = reader.take(used_bits_safe(size + num_castlings))
        if move_id >= size:
            idx = move_id - size
            long_right = rights & (chess.BB_A1 if stm == chess.WHITE else chess.BB_A8)
            long = idx == 0 and long_right != 0
            rank = chess.square_rank(frm)
            move = chess.Move(frm, chess.square(2 if long else 6, rank))
        else:
            move = chess.Move(frm, nth_set_bit(attacks, move_id))
    else:
        attacks = board.attacks_mask(frm) & ~ours
        move_id = reader.take(used_bits_safe(bin(attacks).count("1")))
        move = chess.Move(frm, nth_set_bit(attacks, move_id))
    score = last_score + unsigned_to_signed(reader.vle16(SCORE_VLE_BLOCK))
    score = ((score + 32768) & 0xFFFF) - 32768  # int16 arithmetic, as the writer's
    return move, score


def iter_chunk(chunk: bytes) -> Iterator[tuple[chess.Board, chess.Move | None, int, int, int]]:
    """Every (board, best move, stm score, ply, result) in one chunk, in order."""
    data = memoryview(chunk)
    n = len(chunk)
    at = 0
    while at + 34 <= n:
        board, move, score, ply, result = board_from_entry(data, at)
        at += 32
        plies = (data[at] << 8) | data[at + 1]
        at += 2
        yield board, move, score, ply, result
        if plies:
            if move is None or not board.is_pseudo_legal(move):
                raise ValueError("continuation after a null or illegal move")
            reader = BitReader(data, at)
            last_score = -score
            for _ in range(plies):
                board = board.copy(stack=False)
                board.push(move)
                move, score = next_move_score(reader, board, last_score)
                last_score = -score
                ply += 1
                result = -result
                yield board, move, score, ply, result
            at += reader.bytes_used()


def decode_chunk(job: tuple[bytes, float, int]) -> tuple[np.ndarray, int, int]:
    """One chunk -> filtered RECORD rows. Returns (records, seen, kept)."""
    chunk, scale, min_ply = job
    out = np.zeros(len(chunk) // 4 + 64, dtype=RECORD)
    kept = 0
    seen = 0
    entries = iter_chunk(chunk)
    while True:
        try:
            board, move, score, ply, _result = next(entries)
        except StopIteration:
            break
        except (ValueError, AssertionError, IndexError, KeyError):
            break  # a malformed or unsupported chain: keep what this chunk gave so far
        seen += 1
        if ply < min_ply or board.is_check() or abs(score) >= VALUE_NONE:
            continue
        if move is not None and board.is_capture(move):
            continue
        idx = white_indices(board)
        if not idx or len(idx) > MAX_PIECES:
            continue
        cp = score if board.turn == chess.WHITE else -score
        cp = round(cp * scale)
        cp = max(-CP_CLAMP, min(CP_CLAMP, cp))
        if kept >= len(out):
            out = np.concatenate([out, np.zeros(len(out), dtype=RECORD)])
        out[kept]["idx"][: len(idx)] = idx
        out[kept]["count"] = len(idx)
        out[kept]["stm"] = 1 if board.turn == chess.WHITE else 0
        out[kept]["cp"] = cp
        kept += 1
    return out[:kept], seen, kept


def chunks(path: Path) -> Iterator[bytes]:
    raw: BinaryIO
    if path.suffix == ".zst":
        import zstandard

        raw = zstandard.ZstdDecompressor().stream_reader(
            open(path, "rb")  # noqa: SIM115 -- streamed until EOF; closed at process exit
        )
    else:
        raw = open(path, "rb")  # noqa: SIM115 -- streamed until EOF; closed at process exit
    stream = io.BufferedReader(raw, buffer_size=8 << 20)
    while True:
        header = stream.read(8)
        if len(header) < 8:
            return
        if header[:4] != b"BINP":
            raise ValueError("bad chunk header")
        size = int.from_bytes(header[4:8], "little")
        payload = stream.read(size)
        if len(payload) < size:
            return
        yield payload


def sample(path: Path, limit: int) -> None:
    """Print FEN + raw stm score for the first `limit` positions (for calibration)."""
    printed = 0
    for chunk in chunks(path):
        for board, move, score, ply, result in iter_chunk(chunk):
            mv = move.uci() if move is not None else "0000"
            print(f"{board.fen()}\t{score}\t{mv}\t{ply}\t{result}")
            printed += 1
            if printed >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Stockfish binpack -> packed training shards.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="shard prefix")
    parser.add_argument("--target", type=int, default=580_000_000, help="kept positions to stop at")
    parser.add_argument("--shard", type=int, default=72_000_000, help="positions per shard")
    parser.add_argument(
        "--val", type=int, default=500_000, help="validation positions (last shard)"
    )
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) - 2))
    parser.add_argument(
        "--scale", type=float, default=0.45, help="internal score -> cp (median vs SF17.1)"
    )
    parser.add_argument("--min-ply", type=int, default=16)
    parser.add_argument("--quiet-fraction", type=float, default=0.5)
    parser.add_argument("--sample", type=int, default=0, help="print N FENs + raw scores and exit")
    parser.add_argument("--batch", type=int, default=8, help="chunks per worker task")
    arguments = parser.parse_args()
    if arguments.sample:
        sample(arguments.source, arguments.sample)
        return
    out = arguments.out or arguments.source.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)

    def jobs() -> Iterator[tuple[bytes, float, int]]:
        batch: list[bytes] = []
        for chunk in chunks(arguments.source):
            batch.append(chunk)
            if len(batch) >= arguments.batch:
                yield b"".join(batch), arguments.scale, arguments.min_ply
                batch = []
        if batch:
            yield b"".join(batch), arguments.scale, arguments.min_ply

    started = time.time()
    seen_total = kept_total = written = shard_index = 0
    pending: list[np.ndarray] = []
    pending_rows = 0
    with mp.Pool(arguments.workers) as pool:
        for task, (records, seen, kept) in enumerate(pool.imap(decode_chunk, jobs(), chunksize=1)):
            seen_total += seen
            kept_total += kept
            if task % 100 == 0:
                rate = seen_total / max(time.time() - started, 1)
                print(
                    f"  task {task}: {seen_total:,} seen, {kept_total:,} kept, {rate:,.0f} pos/s",
                    flush=True,
                )
            pending.append(records)
            pending_rows += kept
            if pending_rows >= arguments.shard or kept_total >= arguments.target:
                shard = balance(np.concatenate(pending), arguments.quiet_fraction)
                path = Path(f"{out}_{shard_index:02d}.npy")
                np.save(path, shard)
                written += len(shard)
                shard_index += 1
                print(
                    f"  wrote {path.name}: {len(shard):,} positions"
                    f" (balanced from {pending_rows:,});"
                    f" {kept_total:,} kept of {seen_total:,} seen, {time.time() - started:.0f} s",
                    flush=True,
                )
                pending, pending_rows = [], 0
                if kept_total >= arguments.target:
                    pool.terminate()
                    break
    if pending:
        shard = balance(np.concatenate(pending), arguments.quiet_fraction)
        path = Path(f"{out}_{shard_index:02d}.npy")
        np.save(path, shard)
        written += len(shard)
        print(f"  wrote {path.name}: {len(shard):,} positions", flush=True)
    # validation: split off the tail of the last shard
    last = Path(f"{out}_{shard_index if pending else shard_index - 1:02d}.npy")
    if last.exists():
        arr = np.load(last)
        if len(arr) > 2 * arguments.val:
            np.save(Path(f"{out}_val.npy"), arr[-arguments.val :])
            np.save(last, arr[: -arguments.val])
    print(
        f"done: {written:,} positions in {shard_index + (1 if pending else 0)} shards from"
        f" {seen_total:,} decoded in {(time.time() - started) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
