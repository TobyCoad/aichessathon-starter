"""A compiled bitboard board for the search.

python-chess is bitboard-based but interpreted: a legal move list costs ~28 us and
push/pop ~2.6 us, together about six of the ten microseconds a search node costs.
This module is the same representation compiled with numba, which the organisers
name as the supported fast path. Everything the search does per node lives here:
legal move generation with pins and check masks, make and unmake with an undo
stack, Zobrist keys, and the network's first-layer accumulator updated inside
make so the whole per-node board cost is a few compiled calls.

python-chess is still used at the root, where cost does not matter and its
correctness is the reference: parsing the FEN, the opening book, the tablebase,
and a final legality check of the move that goes out.

Conventions:
  squares    a1 = 0 .. h8 = 63, as python-chess
  colours    0 white, 1 black
  pieces     0 P 1 N 2 B 3 R 4 Q 5 K, code = colour * 6 + piece, -1 empty
  moves      from | to << 6 | promotion << 12, promotion 0 or 1..4 for N B R Q
  keys       polyglot Zobrist, identical to chess.polyglot.zobrist_hash so the
             test suite can compare every key against python-chess

State is a handful of numpy arrays rather than an object, because that is what
compiled functions take:
  bb    uint64[12]      piece bitboards by code
  sq    int8[64]        piece code per square
  meta  int64[8]        SIDE, CASTLING, EP, HALFMOVE, PLY, PIECES, FULLMOVE
  undo  int64[MAX, 8]   per-ply: move, captured, castling, ep, halfmove, zones
  keys  uint64[MAX]     key of the position at each stack depth
"""

from __future__ import annotations

from typing import Any

import chess
import chess.polyglot
import numpy as np
from numba import njit

MAX_PLY = 128
MOVE_CAP = 256

SIDE, CASTLING, EP, HALFMOVE, PLY, PIECES, FULLMOVE = 0, 1, 2, 3, 4, 5, 6
U_MOVE, U_CAPTURED, U_CASTLING, U_EP, U_HALFMOVE, U_ZONE_W, U_ZONE_B = 0, 1, 2, 3, 4, 5, 6

WK_RIGHT, WQ_RIGHT, BK_RIGHT, BQ_RIGHT = 1, 2, 4, 8

ONE = np.uint64(1)
ZERO = np.uint64(0)
ALL = np.uint64(0xFFFFFFFFFFFFFFFF)
FILE_A = np.uint64(0x0101010101010101)
FILE_H = np.uint64(0x8080808080808080)
RANK_1 = np.uint64(0xFF)
RANK_2 = np.uint64(0xFF00)
RANK_4 = np.uint64(0xFF000000)
RANK_5 = np.uint64(0xFF00000000)
RANK_7 = np.uint64(0xFF000000000000)
RANK_8 = np.uint64(0xFF00000000000000)
NOT_FILE_A = ALL ^ FILE_A
NOT_FILE_H = ALL ^ FILE_H

# ------------------------------------------------------------------ tables ----

DEBRUIJN = np.uint64(0x03F79D71B4CB0A89)
_INDEX64 = np.zeros(64, dtype=np.int64)
for _i in range(64):
    _INDEX64[(((1 << _i) * 0x03F79D71B4CB0A89) & 0xFFFFFFFFFFFFFFFF) >> 58] = _i

# Ray directions: 0 N +8, 1 NE +9, 2 E +1, 3 SE -7, 4 S -8, 5 SW -9, 6 W -1, 7 NW +7.
# Positive directions (0, 1, 2, 7) find the first blocker with the lowest bit,
# negative ones (3, 4, 5, 6) with the highest.
_DELTAS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
RAYS = np.zeros((8, 64), dtype=np.uint64)
KNIGHT = np.zeros(64, dtype=np.uint64)
KING = np.zeros(64, dtype=np.uint64)
PAWN_ATTACKS = np.zeros((2, 64), dtype=np.uint64)  # squares a pawn of colour on sq attacks
BETWEEN = np.zeros((64, 64), dtype=np.uint64)
LINE = np.zeros((64, 64), dtype=np.uint64)
CASTLE_MASK = np.full(64, 15, dtype=np.int64)  # rights that survive a move touching sq

for _s in range(64):
    _f, _r = _s & 7, _s >> 3
    for _d, (_df, _dr) in enumerate(_DELTAS):
        _x, _y = _f + _df, _r + _dr
        _ray = 0
        while 0 <= _x < 8 and 0 <= _y < 8:
            _ray |= 1 << (_y * 8 + _x)
            _x += _df
            _y += _dr
        RAYS[_d, _s] = _ray
    _k = 0
    for _df, _dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
        _x, _y = _f + _df, _r + _dr
        if 0 <= _x < 8 and 0 <= _y < 8:
            _k |= 1 << (_y * 8 + _x)
    KNIGHT[_s] = _k
    _k = 0
    for _df, _dr in _DELTAS:
        _x, _y = _f + _df, _r + _dr
        if 0 <= _x < 8 and 0 <= _y < 8:
            _k |= 1 << (_y * 8 + _x)
    KING[_s] = _k
    _w = _b = 0
    for _df in (-1, 1):
        _x = _f + _df
        if 0 <= _x < 8:
            if _r + 1 < 8:
                _w |= 1 << ((_r + 1) * 8 + _x)
            if _r - 1 >= 0:
                _b |= 1 << ((_r - 1) * 8 + _x)
    PAWN_ATTACKS[0, _s] = _w
    PAWN_ATTACKS[1, _s] = _b

for _a in range(64):
    for _d in range(8):
        _ray = int(RAYS[_d, _a])
        _opp = int(RAYS[(_d + 4) % 8, _a])
        _bits = _ray
        while _bits:
            _low = _bits & -_bits
            _b = _low.bit_length() - 1
            _bits ^= _low
            BETWEEN[_a, _b] = _ray & ~int(RAYS[_d, _b]) & ~(1 << _b)
            LINE[_a, _b] = _ray | _opp | (1 << _a)

CASTLE_MASK[chess.E1] = 15 ^ (WK_RIGHT | WQ_RIGHT)
CASTLE_MASK[chess.H1] = 15 ^ WK_RIGHT
CASTLE_MASK[chess.A1] = 15 ^ WQ_RIGHT
CASTLE_MASK[chess.E8] = 15 ^ (BK_RIGHT | BQ_RIGHT)
CASTLE_MASK[chess.H8] = 15 ^ BK_RIGHT
CASTLE_MASK[chess.A8] = 15 ^ BQ_RIGHT

# Polyglot Zobrist keys, so every key is checkable against python-chess.
_POLY = np.array(chess.polyglot.POLYGLOT_RANDOM_ARRAY, dtype=np.uint64)
ZOB_PIECE = np.zeros((12, 64), dtype=np.uint64)
for _code in range(12):
    _colour, _piece = divmod(_code, 6)
    _kind = 2 * _piece + (1 if _colour == 0 else 0)
    for _s in range(64):
        ZOB_PIECE[_code, _s] = _POLY[64 * _kind + 8 * (_s >> 3) + (_s & 7)]
ZOB_CASTLE = _POLY[768:772].copy()  # WK, WQ, BK, BQ
ZOB_EP = _POLY[772:780].copy()
ZOB_TURN = _POLY[780]

# ------------------------------------------------------------- primitives ----


@njit(cache=False)
def lsb(b: Any) -> Any:
    return _INDEX64[((b & (~b + ONE)) * DEBRUIJN) >> np.uint64(58)]


@njit(cache=False)
def msb(b: Any) -> Any:
    n = 0
    if b >> np.uint64(32):
        b >>= np.uint64(32)
        n += 32
    if b >> np.uint64(16):
        b >>= np.uint64(16)
        n += 16
    if b >> np.uint64(8):
        b >>= np.uint64(8)
        n += 8
    if b >> np.uint64(4):
        b >>= np.uint64(4)
        n += 4
    if b >> np.uint64(2):
        b >>= np.uint64(2)
        n += 2
    if b >> np.uint64(1):
        n += 1
    return n


@njit(cache=False)
def popcount(b: Any) -> Any:
    n = 0
    while b:
        b &= b - ONE
        n += 1
    return n


@njit(cache=False)
def bit(s: Any) -> Any:
    return ONE << np.uint64(s)


@njit(cache=False)
def rook_attacks(s: Any, occ: Any) -> Any:
    out = ZERO
    for d in (0, 2):
        ray = RAYS[d, s]
        block = ray & occ
        if block:
            ray ^= RAYS[d, lsb(block)]
        out |= ray
    for d in (4, 6):
        ray = RAYS[d, s]
        block = ray & occ
        if block:
            ray ^= RAYS[d, msb(block)]
        out |= ray
    return out


@njit(cache=False)
def bishop_attacks(s: Any, occ: Any) -> Any:
    out = ZERO
    for d in (1, 7):
        ray = RAYS[d, s]
        block = ray & occ
        if block:
            ray ^= RAYS[d, lsb(block)]
        out |= ray
    for d in (3, 5):
        ray = RAYS[d, s]
        block = ray & occ
        if block:
            ray ^= RAYS[d, msb(block)]
        out |= ray
    return out


@njit(cache=False)
def attackers_to(bb: Any, occ: Any, s: Any, by: Any) -> Any:
    """Pieces of colour `by` attacking square `s`, given occupancy `occ`."""
    base = by * 6
    out = PAWN_ATTACKS[1 - by, s] & bb[base]
    out |= KNIGHT[s] & bb[base + 1]
    out |= KING[s] & bb[base + 5]
    diag = bishop_attacks(s, occ)
    out |= diag & (bb[base + 2] | bb[base + 4])
    ortho = rook_attacks(s, occ)
    out |= ortho & (bb[base + 3] | bb[base + 4])
    return out


@njit(cache=False)
def is_attacked(bb: Any, occ: Any, s: Any, by: Any) -> Any:
    return attackers_to(bb, occ, s, by) != ZERO


@njit(cache=False)
def occupancy(bb: Any, colour: Any) -> Any:
    base = colour * 6
    return bb[base] | bb[base + 1] | bb[base + 2] | bb[base + 3] | bb[base + 4] | bb[base + 5]


@njit(cache=False)
def in_check(bb: Any, meta: Any) -> Any:
    us = meta[SIDE]
    occ = occupancy(bb, 0) | occupancy(bb, 1)
    return is_attacked(bb, occ, lsb(bb[us * 6 + 5]), 1 - us)


# ------------------------------------------------------------- generation ----


@njit(cache=False)
def _add(out: Any, n: Any, frm: Any, to: Any) -> Any:
    out[n] = frm | (to << 6)
    return n + 1


@njit(cache=False)
def _add_promotions(out: Any, n: Any, frm: Any, to: Any) -> Any:
    for promo in (4, 1, 3, 2):  # queen first, knight second
        out[n] = frm | (to << 6) | (promo << 12)
        n += 1
    return n


@njit(cache=False)
def gen_legal(bb: Any, sqa: Any, meta: Any, out: Any, captures_only: Any) -> Any:
    """Legal moves into `out`; returns the count.

    Pins and check masks make every move legal by construction, except en passant,
    which is verified by making it: the captured pawn leaves its own square, so a
    horizontal pin through both pawns is only visible afterwards.
    """
    us = meta[SIDE]
    them = 1 - us
    base = us * 6
    occ_us = occupancy(bb, us)
    occ_them = occupancy(bb, them)
    occ = occ_us | occ_them
    ksq = lsb(bb[base + 5])
    checkers = attackers_to(bb, occ, ksq, them)
    n = 0

    # King moves, with the king removed from the occupancy so it cannot shelter
    # behind itself along a slider's ray.
    occ_no_king = occ ^ bit(ksq)
    targets = KING[ksq] & ~occ_us
    if captures_only:
        targets &= occ_them
    while targets:
        t = lsb(targets)
        targets &= targets - ONE
        if not is_attacked(bb, occ_no_king, t, them):
            n = _add(out, n, ksq, t)

    if checkers & (checkers - ONE):
        return n  # double check: only the king may move

    # Pinned pieces: an enemy slider aligned with the king with exactly one of our
    # pieces between them.
    pinned = ZERO
    snipers = (rook_attacks(ksq, ZERO) & (bb[them * 6 + 3] | bb[them * 6 + 4])) | (
        bishop_attacks(ksq, ZERO) & (bb[them * 6 + 2] | bb[them * 6 + 4])
    )
    while snipers:
        s = lsb(snipers)
        snipers &= snipers - ONE
        blockers = BETWEEN[ksq, s] & occ
        if blockers and not (blockers & (blockers - ONE)) and (blockers & occ_us):
            pinned |= blockers

    if checkers:
        csq = lsb(checkers)
        mask = BETWEEN[ksq, csq] | bit(csq)
    else:
        mask = ALL
    quiet_mask = ZERO if captures_only else (mask & ~occ)
    capture_mask = mask & occ_them

    # Knights: a pinned knight never moves.
    pieces = bb[base + 1] & ~pinned
    while pieces:
        frm = lsb(pieces)
        pieces &= pieces - ONE
        targets = KNIGHT[frm] & (quiet_mask | capture_mask)
        while targets:
            t = lsb(targets)
            targets &= targets - ONE
            n = _add(out, n, frm, t)

    # Sliders: a pinned slider moves only along the pin line, and never in check.
    for kind in (2, 3, 4):
        pieces = bb[base + kind]
        while pieces:
            frm = lsb(pieces)
            pieces &= pieces - ONE
            if kind == 2:
                attacks = bishop_attacks(frm, occ)
            elif kind == 3:
                attacks = rook_attacks(frm, occ)
            else:
                attacks = bishop_attacks(frm, occ) | rook_attacks(frm, occ)
            targets = attacks & (quiet_mask | capture_mask)
            if bit(frm) & pinned:
                if checkers:
                    continue
                targets &= LINE[ksq, frm]
            while targets:
                t = lsb(targets)
                targets &= targets - ONE
                n = _add(out, n, frm, t)

    # Pawns.
    pawns = bb[base]
    if us == 0:
        promo_rank = RANK_8
        start_rank = RANK_2
        forward = 8
    else:
        promo_rank = RANK_1
        start_rank = RANK_7
        forward = -8
    while pawns:
        frm = lsb(pawns)
        pawns &= pawns - ONE
        fb = bit(frm)
        is_pinned = (fb & pinned) != ZERO
        if is_pinned and checkers:
            continue
        allowed = LINE[ksq, frm] if is_pinned else ALL
        # Captures.
        targets = PAWN_ATTACKS[us, frm] & capture_mask & allowed
        while targets:
            t = lsb(targets)
            targets &= targets - ONE
            n = _add_promotions(out, n, frm, t) if bit(t) & promo_rank else _add(out, n, frm, t)
        # Pushes.
        to = frm + forward
        tb = bit(to)
        if not (tb & occ):
            if tb & promo_rank:
                # A quiet promotion is not a capture: python-chess's capture
                # generator omits it, and quiescence keeps the same semantics.
                if not captures_only and (tb & mask) and (tb & allowed):
                    n = _add_promotions(out, n, frm, to)
            elif not captures_only:
                if (tb & mask) and (tb & allowed):
                    n = _add(out, n, frm, to)
                if fb & start_rank:
                    to2 = to + forward
                    t2b = bit(to2)
                    if not (t2b & occ) and (t2b & mask) and (t2b & allowed):
                        n = _add(out, n, frm, to2)

    # En passant, verified by making it.
    ep = meta[EP]
    if ep >= 0:
        epb = bit(ep)
        capturers = PAWN_ATTACKS[them, ep] & bb[base]
        while capturers:
            frm = lsb(capturers)
            capturers &= capturers - ONE
            captured = ep - forward
            occ_after = (occ ^ bit(frm) ^ bit(captured)) | epb
            # Remove the captured pawn from the attacker set as well as occupancy.
            saved = bb[them * 6]
            bb[them * 6] = saved ^ bit(captured)
            attacked = is_attacked(bb, occ_after, ksq, them)
            bb[them * 6] = saved
            if not attacked:
                n = _add(out, n, frm, ep)

    # Castling: never out of check, rook and rights present, path empty and safe.
    if not captures_only and not checkers:
        rights = meta[CASTLING]
        if us == 0:
            if (rights & WK_RIGHT) and not (occ & np.uint64(0x60)) and (bb[3] & bit(7)) and (
                not is_attacked(bb, occ, 5, 1) and not is_attacked(bb, occ, 6, 1)
            ):
                n = _add(out, n, 4, 6)
            if (rights & WQ_RIGHT) and not (occ & np.uint64(0x0E)) and (bb[3] & bit(0)) and (
                not is_attacked(bb, occ, 3, 1) and not is_attacked(bb, occ, 2, 1)
            ):
                n = _add(out, n, 4, 2)
        else:
            if (
                (rights & BK_RIGHT)
                and not (occ & np.uint64(0x6000000000000000))
                and (bb[9] & bit(63))
                and not is_attacked(bb, occ, 61, 0)
                and not is_attacked(bb, occ, 62, 0)
            ):
                n = _add(out, n, 60, 62)
            if (
                (rights & BQ_RIGHT)
                and not (occ & np.uint64(0x0E00000000000000))
                and (bb[9] & bit(56))
                and not is_attacked(bb, occ, 59, 0)
                and not is_attacked(bb, occ, 58, 0)
            ):
                n = _add(out, n, 60, 58)
    return n


# ------------------------------------------------------------ make / unmake ----


@njit(cache=False)
def _ep_hash(bb: Any, meta: Any, ep: Any) -> Any:
    """Polyglot hashes the en passant file only if a pawn of the side to move
    stands beside the pushed pawn -- the same rule python-chess applies."""
    if ep < 0:
        return ZERO
    us = meta[SIDE]
    pushed = ep - 8 if us == 0 else ep + 8
    beside = ((bit(pushed) & NOT_FILE_A) >> ONE) | ((bit(pushed) & NOT_FILE_H) << ONE)
    if beside & bb[us * 6]:
        return ZOB_EP[ep & 7]
    return ZERO


@njit(cache=False)
def compute_key(bb: Any, sqa: Any, meta: Any) -> Any:
    key = ZERO
    for s in range(64):
        code = sqa[s]
        if code >= 0:
            key ^= ZOB_PIECE[code, s]
    rights = meta[CASTLING]
    for i in range(4):
        if rights & (1 << i):
            key ^= ZOB_CASTLE[i]
    key ^= _ep_hash(bb, meta, meta[EP])
    if meta[SIDE] == 0:
        key ^= ZOB_TURN
    return key


@njit(cache=False)
def make_light(bb: Any, sqa: Any, meta: Any, undo: Any, keys: Any, move: Any) -> Any:
    """Apply `move` to the board and key. Legality is the generator's job."""
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    us = meta[SIDE]
    them = 1 - us
    ply = meta[PLY]
    code = sqa[frm]
    piece = code - us * 6
    captured = sqa[to]
    key = keys[ply]

    undo[ply, U_MOVE] = move
    undo[ply, U_CAPTURED] = captured
    undo[ply, U_CASTLING] = meta[CASTLING]
    undo[ply, U_EP] = meta[EP]
    undo[ply, U_HALFMOVE] = meta[HALFMOVE]

    key ^= _ep_hash(bb, meta, meta[EP])
    halfmove = meta[HALFMOVE] + 1
    ep_new = -1

    # Lift the mover.
    bb[code] ^= bit(frm)
    sqa[frm] = -1
    key ^= ZOB_PIECE[code, frm]

    if captured >= 0:
        bb[captured] ^= bit(to)
        key ^= ZOB_PIECE[captured, to]
        meta[PIECES] -= 1
        halfmove = 0

    if piece == 0:
        halfmove = 0
        if to == meta[EP]:
            # En passant: the captured pawn sits behind the target square.
            behind = to - 8 if us == 0 else to + 8
            pcode = them * 6
            bb[pcode] ^= bit(behind)
            sqa[behind] = -1
            key ^= ZOB_PIECE[pcode, behind]
            meta[PIECES] -= 1
            undo[ply, U_CAPTURED] = -2  # marker: en passant
        elif (to - frm == 16) or (frm - to == 16):
            ep_new = (frm + to) >> 1
    elif piece == 5 and (to - frm == 2 or frm - to == 2):
        # Castling: move the rook too.
        rook = us * 6 + 3
        if to > frm:
            rfrom, rto = to + 1, to - 1
        else:
            rfrom, rto = to - 2, to + 1
        bb[rook] ^= bit(rfrom) | bit(rto)
        sqa[rfrom] = -1
        sqa[rto] = rook
        key ^= ZOB_PIECE[rook, rfrom] ^ ZOB_PIECE[rook, rto]

    # Drop the mover, possibly promoted.
    landing = code if promo == 0 else us * 6 + promo
    bb[landing] ^= bit(to)
    sqa[to] = landing
    key ^= ZOB_PIECE[landing, to]

    # Castling rights lost by touching e1/h1/a1/e8/h8/a8.
    rights = meta[CASTLING] & CASTLE_MASK[frm] & CASTLE_MASK[to]
    if rights != meta[CASTLING]:
        changed = rights ^ meta[CASTLING]
        for i in range(4):
            if changed & (1 << i):
                key ^= ZOB_CASTLE[i]
    meta[CASTLING] = rights
    meta[EP] = ep_new
    meta[HALFMOVE] = halfmove
    meta[SIDE] = them
    if them == 0:
        meta[FULLMOVE] += 1
    key ^= ZOB_TURN
    key ^= _ep_hash(bb, meta, ep_new)
    meta[PLY] = ply + 1
    keys[ply + 1] = key


@njit(cache=False)
def unmake_light(bb: Any, sqa: Any, meta: Any, undo: Any, keys: Any) -> Any:
    ply = meta[PLY] - 1
    move = undo[ply, U_MOVE]
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    them = meta[SIDE]  # the side that did not move
    us = 1 - them
    captured = undo[ply, U_CAPTURED]
    landing = sqa[to]
    code = landing if promo == 0 else us * 6

    bb[landing] ^= bit(to)
    sqa[to] = -1
    bb[code] ^= bit(frm)
    sqa[frm] = code
    piece = code - us * 6

    if captured >= 0:
        bb[captured] ^= bit(to)
        sqa[to] = captured
        meta[PIECES] += 1
    elif captured == -2:
        behind = to - 8 if us == 0 else to + 8
        pcode = them * 6
        bb[pcode] ^= bit(behind)
        sqa[behind] = pcode
        meta[PIECES] += 1
    if piece == 5 and (to - frm == 2 or frm - to == 2):
        rook = us * 6 + 3
        if to > frm:
            rfrom, rto = to + 1, to - 1
        else:
            rfrom, rto = to - 2, to + 1
        bb[rook] ^= bit(rfrom) | bit(rto)
        sqa[rto] = -1
        sqa[rfrom] = rook

    meta[CASTLING] = undo[ply, U_CASTLING]
    meta[EP] = undo[ply, U_EP]
    meta[HALFMOVE] = undo[ply, U_HALFMOVE]
    meta[SIDE] = us
    if them == 0:
        meta[FULLMOVE] -= 1
    meta[PLY] = ply


# ------------------------------------------------------ accumulator fusion ----
# The network's first layer is a sum of W1 rows, one per piece, from each side's
# perspective, in the block of that side's king zone. make_full updates both sums
# with the move's deltas, or rebuilds one perspective when its king crosses a zone.

FEATURES = 768


@njit(cache=False)
def zone_of(square: Any, zones: Any) -> Any:
    """Mirrors training.features.king_zone for 1, 4, 8 or 32 zones."""
    rank = square >> 3
    file = square & 7
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
    if zones == 16:
        if rank <= 1:
            return file
        if rank <= 3:
            return 8 + (file >> 1)
        return 12 + (file >> 1)
    if zones == 32:
        if rank <= 1:
            return rank * 8 + file
        if rank <= 3:
            return 16 + (rank - 2) * 4 + (file >> 1)
        return 24 + ((rank - 4) >> 1) * 4 + (file >> 1)
    return 0


@njit(cache=False)
def feature(square: Any, code: Any, white_pov: Any) -> Any:
    colour = code // 6
    piece = code - colour * 6
    if white_pov:
        own = colour == 0
        rel = square
    else:
        own = colour == 1
        rel = square ^ 56
    return (0 if own else 384) + piece * 64 + rel


@njit(cache=False)
def rebuild(sqa: Any, w1: Any, b1: Any, out: Any, white_pov: Any, zone: Any) -> Any:
    width = out.shape[0]
    for i in range(width):
        out[i] = b1[i]
    offset = zone * FEATURES
    for s in range(64):
        code = sqa[s]
        if code >= 0:
            row = w1[offset + feature(s, code, white_pov)]
            for i in range(width):
                out[i] += row[i]


@njit(cache=False)
def _acc_row(
    w1: Any, white: Any, black: Any, s: Any, code: Any, off_w: Any, off_b: Any, sign: Any
) -> Any:
    width = white.shape[0]
    rw = w1[off_w + feature(s, code, True)]
    rb = w1[off_b + feature(s, code, False)]
    if sign > 0:
        for i in range(width):
            white[i] += rw[i]
            black[i] += rb[i]
    else:
        for i in range(width):
            white[i] -= rw[i]
            black[i] -= rb[i]


@njit(cache=False)
def _acc_row_one(
    w1: Any, acc: Any, s: Any, code: Any, off: Any, white_pov: Any, sign: Any
) -> Any:
    """_acc_row for one perspective only."""
    width = acc.shape[0]
    row = w1[off + feature(s, code, white_pov)]
    if sign > 0:
        for i in range(width):
            acc[i] += row[i]
    else:
        for i in range(width):
            acc[i] -= row[i]


@njit(cache=False)
def make_full(
    bb: Any,
    sqa: Any,
    meta: Any,
    undo: Any,
    keys: Any,
    move: Any,
    w1: Any,
    b1: Any,
    white: Any,
    black: Any,
    astack: Any,
    zones: Any,
    king_zones: Any,
) -> Any:
    """make_light plus the accumulator update, in one compiled call."""
    ply = meta[PLY]
    width = white.shape[0]
    for i in range(width):
        astack[ply, 0, i] = white[i]
        astack[ply, 1, i] = black[i]
    undo[ply, U_ZONE_W] = zones[0]
    undo[ply, U_ZONE_B] = zones[1]

    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    us = meta[SIDE]
    them = 1 - us
    code = sqa[frm]
    piece = code - us * 6
    captured = sqa[to]
    off_w = zones[0] * FEATURES
    off_b = zones[1] * FEATURES

    crossing = 0
    new_zone = 0
    if piece == 5 and king_zones > 1:
        new_zone = zone_of(to if us == 0 else to ^ 56, king_zones)
        if new_zone != zones[us]:
            crossing = 1

    if crossing == 0:
        _acc_row(w1, white, black, frm, code, off_w, off_b, -1)
        if captured >= 0:
            _acc_row(w1, white, black, to, captured, off_w, off_b, -1)
        elif piece == 0 and to == meta[EP]:
            behind = to - 8 if us == 0 else to + 8
            _acc_row(w1, white, black, behind, them * 6, off_w, off_b, -1)
        landing = code if promo == 0 else us * 6 + promo
        _acc_row(w1, white, black, to, landing, off_w, off_b, 1)
        if piece == 5 and (to - frm == 2 or frm - to == 2):
            rook = us * 6 + 3
            if to > frm:
                rfrom, rto = to + 1, to - 1
            else:
                rfrom, rto = to - 2, to + 1
            _acc_row(w1, white, black, rfrom, rook, off_w, off_b, -1)
            _acc_row(w1, white, black, rto, rook, off_w, off_b, 1)
    else:
        # A king move across a zone boundary: the mover's own perspective changes
        # block and is rebuilt below; the other perspective keeps its block and
        # only needs the ordinary deltas of a king move (capture, castling rook).
        if us == 0:
            other, off_o, pov = black, off_b, False
        else:
            other, off_o, pov = white, off_w, True
        _acc_row_one(w1, other, frm, code, off_o, pov, -1)
        if captured >= 0:
            _acc_row_one(w1, other, to, captured, off_o, pov, -1)
        _acc_row_one(w1, other, to, code, off_o, pov, 1)
        if to - frm == 2 or frm - to == 2:
            rook = us * 6 + 3
            if to > frm:
                rfrom, rto = to + 1, to - 1
            else:
                rfrom, rto = to - 2, to + 1
            _acc_row_one(w1, other, rfrom, rook, off_o, pov, -1)
            _acc_row_one(w1, other, rto, rook, off_o, pov, 1)

    make_light(bb, sqa, meta, undo, keys, move)

    if crossing:
        zones[us] = new_zone
        if us == 0:
            rebuild(sqa, w1, b1, white, True, zones[0])
        else:
            rebuild(sqa, w1, b1, black, False, zones[1])


@njit(cache=False)
def unmake_full(
    bb: Any,
    sqa: Any,
    meta: Any,
    undo: Any,
    keys: Any,
    white: Any,
    black: Any,
    astack: Any,
    zones: Any,
) -> Any:
    unmake_light(bb, sqa, meta, undo, keys)
    ply = meta[PLY]
    width = white.shape[0]
    for i in range(width):
        white[i] = astack[ply, 0, i]
        black[i] = astack[ply, 1, i]
    zones[0] = undo[ply, U_ZONE_W]
    zones[1] = undo[ply, U_ZONE_B]


@njit(cache=False)
def refresh(
    bb: Any,
    sqa: Any,
    meta: Any,
    w1: Any,
    b1: Any,
    white: Any,
    black: Any,
    zones: Any,
    king_zones: Any,
) -> Any:
    if king_zones > 1:
        zones[0] = zone_of(lsb(bb[5]), king_zones)
        zones[1] = zone_of(lsb(bb[11]) ^ 56, king_zones)
    else:
        zones[0] = 0
        zones[1] = 0
    rebuild(sqa, w1, b1, white, True, zones[0])
    rebuild(sqa, w1, b1, black, False, zones[1])


# ----------------------------------------------------------- null moves ----


@njit(cache=False)
def make_null(bb: Any, meta: Any, undo: Any, keys: Any) -> Any:
    """Pass the move: flip the side, drop the en passant square, keep the key
    consistent. The accumulators do not change -- only whose turn it is."""
    ply = meta[PLY]
    key = keys[ply] ^ _ep_hash(bb, meta, meta[EP]) ^ ZOB_TURN
    undo[ply, U_MOVE] = 0
    undo[ply, U_EP] = meta[EP]
    undo[ply, U_HALFMOVE] = meta[HALFMOVE]
    meta[EP] = -1
    meta[SIDE] = 1 - meta[SIDE]
    meta[HALFMOVE] += 1
    if meta[SIDE] == 0:
        meta[FULLMOVE] += 1
    meta[PLY] = ply + 1
    keys[ply + 1] = key


@njit(cache=False)
def unmake_null(meta: Any, undo: Any) -> Any:
    ply = meta[PLY] - 1
    meta[EP] = undo[ply, U_EP]
    meta[HALFMOVE] = undo[ply, U_HALFMOVE]
    if meta[SIDE] == 0:
        meta[FULLMOVE] -= 1
    meta[SIDE] = 1 - meta[SIDE]
    meta[PLY] = ply


# --------------------------------------------------------- move ordering ----

MVV = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int64)
CAPTURE_BONUS = 1 << 20
KILLER_FIRST = (1 << 20) - 1
KILLER_SECOND = (1 << 20) - 2
PROMOTION_BONUS = 1 << 19


@njit(cache=False)
def order_moves(
    out: Any,
    n: Any,
    sqa: Any,
    hash_move: Any,
    killer1: Any,
    killer2: Any,
    history: Any,
    scores: Any,
) -> Any:
    """Score and sort `out[:n]` in place: hash move, captures by MVV-LVA, killers,
    then quiet moves by history. Stable insertion sort, so equal scores keep the
    generator's order, as Python's sort would."""
    for i in range(n):
        m = out[i]
        frm = m & 63
        to = (m >> 6) & 63
        promo = (m >> 12) & 7
        if m == hash_move:
            s = 1 << 30
        else:
            victim = sqa[to]
            if victim >= 0:
                attacker = sqa[frm]
                s = CAPTURE_BONUS + MVV[victim % 6] * 16 - MVV[attacker % 6]
            elif m == killer1:
                s = KILLER_FIRST
            elif m == killer2:
                s = KILLER_SECOND
            else:
                s = history[frm * 64 + to]
            if promo:
                s += PROMOTION_BONUS + promo * 100
        scores[i] = s
    for i in range(1, n):
        m = out[i]
        s = scores[i]
        j = i - 1
        while j >= 0 and scores[j] < s:
            out[j + 1] = out[j]
            scores[j + 1] = scores[j]
            j -= 1
        out[j + 1] = m
        scores[j + 1] = s


@njit(cache=False)
def score_moves(
    out: Any,
    n: Any,
    sqa: Any,
    hash_move: Any,
    killer1: Any,
    killer2: Any,
    history: Any,
    scores: Any,
) -> Any:
    """order_moves' scoring pass alone; pick_move then draws moves best-first."""
    for i in range(n):
        m = out[i]
        frm = m & 63
        to = (m >> 6) & 63
        promo = (m >> 12) & 7
        if m == hash_move:
            s = 1 << 30
        else:
            victim = sqa[to]
            if victim >= 0:
                attacker = sqa[frm]
                s = CAPTURE_BONUS + MVV[victim % 6] * 16 - MVV[attacker % 6]
            elif m == killer1:
                s = KILLER_FIRST
            elif m == killer2:
                s = KILLER_SECOND
            else:
                s = history[frm * 64 + to]
            if promo:
                s += PROMOTION_BONUS + promo * 100
        scores[i] = s


@njit(cache=False)
def pick_move(out: Any, scores: Any, i: Any, n: Any) -> Any:
    """Bring the best of out[i:n] to position i, shifting the rest down so equal
    scores keep their order -- exactly the sequence the stable sort produces --
    and return it. Most nodes cut off after one or two picks, which is what
    sorting every move would have paid for nothing."""
    best = i
    best_score = scores[i]
    for j in range(i + 1, n):
        if scores[j] > best_score:
            best = j
            best_score = scores[j]
    m = out[best]
    j = best
    while j > i:
        out[j] = out[j - 1]
        scores[j] = scores[j - 1]
        j -= 1
    out[i] = m
    scores[i] = best_score
    return m


SEE_VALUE = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int64)


@njit(cache=False)
def see(bb: Any, sqa: Any, meta: Any, move: Any) -> Any:
    """Static exchange evaluation of `move`, a capture, for the side to move:
    the material won or lost if both sides keep recapturing on the target
    square with their least valuable attacker, each free to stop. Sliders
    behind a piece that has just captured are found because attackers_to is
    recomputed against the shrinking occupancy. Promotions count the piece
    gained; en passant counts the pawn."""
    us = meta[SIDE]
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    attacker = sqa[frm]
    victim = sqa[to]
    gain = np.zeros(32, dtype=np.int64)
    if victim >= 0:
        gain[0] = SEE_VALUE[victim % 6]
    elif attacker % 6 == 0 and to == meta[EP]:
        gain[0] = SEE_VALUE[0]
    else:
        gain[0] = 0
    # Value of the piece standing on the target after each capture: first the
    # mover (as promoted), then each recapturer in turn.
    on_square = SEE_VALUE[attacker % 6]
    if promo:
        gain[0] += SEE_VALUE[promo] - SEE_VALUE[0]
        on_square = SEE_VALUE[promo]
    occ = (occupancy(bb, 0) | occupancy(bb, 1)) & ~bit(frm)
    if victim < 0 and attacker % 6 == 0 and to == meta[EP]:
        occ &= ~bit(to + (-8 if us == 0 else 8))
    side = 1 - us
    d = 0
    while d < 30:
        attackers = attackers_to(bb, occ, to, side) & occ
        if attackers == ZERO:
            break
        # least valuable attacker of `side`; pieces already used are gone from occ
        base = side * 6
        chosen = -1
        lva_value = 0
        for p in range(6):
            candidates = attackers & bb[base + p]
            if candidates != ZERO:
                chosen = lsb(candidates)
                lva_value = SEE_VALUE[p]
                break
        if chosen < 0:
            break
        d += 1
        gain[d] = on_square - gain[d - 1]
        if max(-gain[d - 1], gain[d]) < 0:
            # whichever side is to choose here already prefers to stop
            break
        on_square = lva_value
        occ &= ~bit(chosen)
        side = 1 - side
    while d > 0:
        d -= 1
        if -gain[d + 1] < gain[d]:
            gain[d] = -gain[d + 1]
    return gain[0]


# ------------------------------------------------------------ repetition ----


@njit(cache=False)
def repeats(meta: Any, keys: Any) -> Any:
    """True if the current position occurred earlier on the search stack, within
    the reversible window."""
    ply = meta[PLY]
    key = keys[ply]
    limit = ply - meta[HALFMOVE]
    if limit < 0:
        limit = 0
    i = ply - 2
    while i >= limit:
        if keys[i] == key:
            return True
        i -= 2
    return False


@njit(cache=False)
def non_pawn_material(bb: Any, colour: Any) -> Any:
    base = colour * 6
    return (bb[base + 1] | bb[base + 2] | bb[base + 3] | bb[base + 4]) != ZERO


@njit(cache=False)
def pawn_index(bb: Any, bits: Any) -> Any:
    """Hash of the pawn structure, both colours, reduced to `bits` bits."""
    h = bb[0] * np.uint64(0x9E3779B97F4A7C15) ^ bb[6] * np.uint64(0xC2B2AE3D27D4EB4F)
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(32)
    return h >> np.uint64(64 - bits)


@njit(cache=False)
def gives_check_after(bb: Any, meta: Any) -> Any:
    """Whether the side to move is in check (call after make)."""
    return in_check(bb, meta)


# ------------------------------------------------------ Python-side helpers ----


class Position:
    """The arrays a compiled search needs, built from a python-chess board."""

    __slots__ = ("bb", "keys", "meta", "sq", "undo")

    def __init__(self, board: chess.Board) -> None:
        self.bb = np.zeros(12, dtype=np.uint64)
        self.sq = np.full(64, -1, dtype=np.int8)
        self.meta = np.zeros(8, dtype=np.int64)
        self.undo = np.zeros((MAX_PLY, 8), dtype=np.int64)
        self.keys = np.zeros(MAX_PLY + 1, dtype=np.uint64)
        self.load(board)

    def load(self, board: chess.Board) -> None:
        self.bb[:] = 0
        self.sq[:] = -1
        for square, piece in board.piece_map().items():
            code = (0 if piece.color == chess.WHITE else 6) + piece.piece_type - 1
            self.bb[code] |= np.uint64(1) << np.uint64(square)
            self.sq[square] = code
        rights = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            rights |= WK_RIGHT
        if board.has_queenside_castling_rights(chess.WHITE):
            rights |= WQ_RIGHT
        if board.has_kingside_castling_rights(chess.BLACK):
            rights |= BK_RIGHT
        if board.has_queenside_castling_rights(chess.BLACK):
            rights |= BQ_RIGHT
        self.meta[SIDE] = 0 if board.turn == chess.WHITE else 1
        self.meta[CASTLING] = rights
        self.meta[EP] = -1 if board.ep_square is None else board.ep_square
        self.meta[HALFMOVE] = board.halfmove_clock
        self.meta[PLY] = 0
        self.meta[PIECES] = len(board.piece_map())
        self.meta[FULLMOVE] = board.fullmove_number
        self.keys[0] = compute_key(self.bb, self.sq, self.meta)

    def to_board(self) -> chess.Board:
        """A python-chess board of the current position, for the root and the
        tablebase. Set through the bitboard attributes: ~5 us, not a FEN round trip."""
        board = chess.Board(None)
        bb = [int(x) for x in self.bb]
        board.pawns = bb[0] | bb[6]
        board.knights = bb[1] | bb[7]
        board.bishops = bb[2] | bb[8]
        board.rooks = bb[3] | bb[9]
        board.queens = bb[4] | bb[10]
        board.kings = bb[5] | bb[11]
        white = bb[0] | bb[1] | bb[2] | bb[3] | bb[4] | bb[5]
        black = bb[6] | bb[7] | bb[8] | bb[9] | bb[10] | bb[11]
        board.occupied_co[chess.WHITE] = white
        board.occupied_co[chess.BLACK] = black
        board.occupied = white | black
        board.promoted = 0
        board.turn = chess.WHITE if self.meta[SIDE] == 0 else chess.BLACK
        rights = int(self.meta[CASTLING])
        castling = 0
        if rights & WK_RIGHT:
            castling |= chess.BB_H1
        if rights & WQ_RIGHT:
            castling |= chess.BB_A1
        if rights & BK_RIGHT:
            castling |= chess.BB_H8
        if rights & BQ_RIGHT:
            castling |= chess.BB_A8
        board.castling_rights = castling
        ep = int(self.meta[EP])
        board.ep_square = None if ep < 0 else ep
        board.halfmove_clock = int(self.meta[HALFMOVE])
        board.fullmove_number = int(self.meta[FULLMOVE])
        return board


def move_to_uci(move: int) -> str:
    frm = move & 63
    to = (move >> 6) & 63
    promo = (move >> 12) & 7
    out = chess.SQUARE_NAMES[frm] + chess.SQUARE_NAMES[to]
    if promo:
        out += "nbrq"[promo - 1]
    return out


def move_from_chess(move: chess.Move) -> int:
    promo = 0 if move.promotion is None else move.promotion - 1
    return move.from_square | (move.to_square << 6) | (promo << 12)


def warm_up() -> None:
    """Compile every kernel now, inside the init budget, never on the clock."""
    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    pos = Position(board)
    out = np.zeros(MOVE_CAP, dtype=np.int32)
    n = gen_legal(pos.bb, pos.sq, pos.meta, out, False)
    gen_legal(pos.bb, pos.sq, pos.meta, out, True)
    width = 8
    w1 = np.zeros((FEATURES * 8, width), dtype=np.float32)
    b1 = np.zeros(width, dtype=np.float32)
    white = np.zeros(width, dtype=np.float32)
    black = np.zeros(width, dtype=np.float32)
    astack = np.zeros((MAX_PLY, 2, width), dtype=np.float32)
    zones = np.zeros(2, dtype=np.int64)
    refresh(pos.bb, pos.sq, pos.meta, w1, b1, white, black, zones, 8)
    for i in range(n):
        make_full(
            pos.bb,
            pos.sq,
            pos.meta,
            pos.undo,
            pos.keys,
            int(out[i]),
            w1,
            b1,
            white,
            black,
            astack,
            zones,
            8,
        )
        in_check(pos.bb, pos.meta)
        repeats(pos.meta, pos.keys)
        non_pawn_material(pos.bb, 0)
        unmake_full(pos.bb, pos.sq, pos.meta, pos.undo, pos.keys, white, black, astack, zones)
    make_light(pos.bb, pos.sq, pos.meta, pos.undo, pos.keys, int(out[0]))
    unmake_light(pos.bb, pos.sq, pos.meta, pos.undo, pos.keys)
    make_null(pos.bb, pos.meta, pos.undo, pos.keys)
    unmake_null(pos.meta, pos.undo)
    compute_key(pos.bb, pos.sq, pos.meta)
    pawn_index(pos.bb, 14)
    popcount(pos.bb[0])
    history = np.zeros(4096, dtype=np.int32)
    scores = np.zeros(MOVE_CAP, dtype=np.int64)
    order_moves(out, n, pos.sq, int(out[0]), 0, 0, history, scores)
    score_moves(out, n, pos.sq, int(out[0]), 0, 0, history, scores)
    pick_move(out, scores, 0, n)


__all__ = [
    "MAX_PLY",
    "MOVE_CAP",
    "Position",
    "gen_legal",
    "in_check",
    "make_full",
    "make_light",
    "make_null",
    "move_from_chess",
    "move_to_uci",
    "non_pawn_material",
    "order_moves",
    "refresh",
    "repeats",
    "unmake_full",
    "unmake_light",
    "unmake_null",
    "warm_up",
]
