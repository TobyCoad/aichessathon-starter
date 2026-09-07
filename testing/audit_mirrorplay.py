"""Independent end-to-end audit of the MIRRORED net path.

`training/check_nnue.py` exercises `agent.Accumulator` and the numpy/torch agreement.
It does not drive `fastboard.make_full` / `fastsearch` -- the code that actually plays.
This module does, and it checks everything against a ground truth built from the RAW
(undoubled) W1 table plus `training/features.py`, so it cannot be fooled by an error
that is shared between `agent._zone` and `fastboard.zone_of`.

    python -m testing.audit_mirrorplay --engine <dir-with-agent.py-and-weights>

`--engine` defaults to the repo root. Point it at a copy of the engine whose
weights/net.npz carries mirrored=1 to audit the mirrored path.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FEATURES = 768


def load_engine(engine_dir: Path) -> Any:
    """Import agent/fastboard/fastsearch from `engine_dir`, not from the repo root."""
    for name in ("agent", "fastboard", "fastsearch"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(engine_dir))
    import agent

    thread = getattr(agent, "_WARM_THREAD", None)
    if thread is not None:
        thread.join()
        if agent._WARM_FAILED:
            raise RuntimeError("warm_up failed; nothing below would measure the fast path")
    return agent


# ------------------------------------------------------------------ ground truth ----


def truth_acc(agent: Any, board: chess.Board, colour: chess.Color) -> np.ndarray:
    """One perspective's first-layer sum, from the RAW table and training/features.py.

    Deliberately independent of W1's doubling, `_block`, `_mirror_flip` and `_zone`.
    """
    from training import features as F  # noqa: N812

    raw = agent._W1_RAW
    mirrored = agent.MIRRORED
    zones = agent.KING_ZONES
    king = board.king(colour)
    if zones == 1 or king is None:
        zone = 0
    else:
        own = king if colour == chess.WHITE else king ^ 56
        flip = F.mirror_flip(own) if mirrored else 0
        zone = F.king_zone(own ^ flip, zones, mirrored=mirrored)
    acc = np.array(agent.B1, dtype=np.float64)
    off = zone * FEATURES
    for idx in F.indices(board, colour, mirrored=mirrored):
        acc += raw[off + idx]
    return acc


def gap(a: Any, b: Any) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def truth_gap(agent: Any, board: chess.Board, white: Any, black: Any) -> float:
    return max(
        gap(white, truth_acc(agent, board, chess.WHITE)),
        gap(black, truth_acc(agent, board, chess.BLACK)),
    )


# ------------------------------------------------------------------- helpers ----


def mirror_board(board: chess.Board) -> chess.Board:
    """The left-right reflection of `board`. Castling rights are dropped (a file
    reflection is not a legal castling transform); en passant is reflected."""
    out = chess.Board(None)
    for square, piece in board.piece_map().items():
        out.set_piece_at(square ^ 7, piece)
    out.turn = board.turn
    out.castling_rights = 0
    out.ep_square = (board.ep_square ^ 7) if board.ep_square is not None else None
    out.halfmove_clock = board.halfmove_clock
    out.fullmove_number = board.fullmove_number
    return out


def random_positions(count: int, seed: int, plies: int = 40) -> list[chess.Board]:
    """Random-playout positions, biased to keep kings wandering near the e-file."""
    rng = random.Random(seed)
    out: list[chess.Board] = []
    while len(out) < count:
        board = chess.Board()
        for _ in range(rng.randint(4, plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over() or board.king(chess.WHITE) is None:
            continue
        out.append(board)
    return out


def king_wing_positions(count: int, seed: int) -> list[chess.Board]:
    """Legal random positions with both kings, chosen so kings sit on both wings
    and often adjacent to the e-file, where a wing crossing is one move away."""
    rng = random.Random(seed)
    out: list[chess.Board] = []
    files = [2, 3, 4, 5]
    guard = 0
    while len(out) < count and guard < count * 400:
        guard += 1
        board = chess.Board(None)
        wk = rng.choice(files) + 8 * rng.randint(0, 2)
        bk = rng.choice(files) + 8 * rng.randint(5, 7)
        if chess.square_distance(wk, bk) < 2:
            continue
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        for colour in (chess.WHITE, chess.BLACK):
            for ptype in (chess.PAWN, chess.PAWN, chess.KNIGHT, chess.ROOK, chess.QUEEN):
                for _ in range(20):
                    sq = rng.randrange(64)
                    if board.piece_at(sq) is not None:
                        continue
                    if ptype == chess.PAWN and not (8 <= sq < 56):
                        continue
                    board.set_piece_at(sq, chess.Piece(ptype, colour))
                    break
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        board.castling_rights = 0
        if not board.is_valid():
            continue
        out.append(board)
    return out


# How many played-out positions the wide static-symmetry gate samples. The audit's own
# probe set is 16 positions and check_nnue's is 8; a W1 block index that is wrong for a
# rare king square would pass both. This walks random legal games instead, which is the
# distribution the search actually visits.
SYMMETRY_SAMPLE = 400

# --------------------------------------------------------------------- checks ----


class Report:
    def __init__(self) -> None:
        self.fails = 0

    def line(self, name: str, ok: bool, detail: str) -> None:
        if not ok:
            self.fails += 1
        print(f"{name:<34} : {'OK  ' if ok else 'FAIL'}  {detail}")


def check_w1_doubling(agent: Any, rep: Report) -> None:
    raw, w1 = agent._W1_RAW, agent.W1
    if not agent.MIRRORED:
        rep.line("W1 doubling", w1 is raw, "unmirrored: W1 is the raw table, undoubled")
        return
    n = raw.shape[0]
    ok = w1.shape[0] == 2 * n and np.array_equal(w1[:n], raw)
    ok = ok and np.array_equal(w1[n:], raw[np.arange(n) ^ 7])
    # And the property that actually matters: block z+K, feature f == raw block z, f^7.
    worst = 0.0
    zones = agent.KING_ZONES
    for z in range(zones):
        lo, hi = z * FEATURES, (z + 1) * FEATURES
        worst = max(worst, gap(w1[n + lo : n + hi], raw[lo:hi][np.arange(FEATURES) ^ 7]))
    rep.line(
        "W1 doubling", bool(ok) and worst == 0.0,
        f"{w1.shape[0]} rows = 2 x {n}, block z+{zones} row f == raw block z row f^7 "
        f"over all {zones * FEATURES} rows, worst {worst}",
    )


def check_zone_maps(agent: Any, rep: Report) -> None:
    """agent._block(_zone,_mirror_flip) and fastboard.zone_of vs training/features."""
    import fastboard as fb
    from training import features as F  # noqa: N812

    zones = agent.KING_ZONES
    mirrored = agent.MIRRORED
    bad_agent = bad_fb = 0
    blocks = set()
    for sq in range(64):
        flip = F.mirror_flip(sq) if mirrored else 0
        want_zone = F.king_zone(sq ^ flip, zones, mirrored=mirrored) if zones > 1 else 0
        want_block = want_zone + (zones if flip else 0)
        got_agent = agent._block(
            agent._zone(sq ^ agent._mirror_flip(sq)) if mirrored else agent._zone(sq),
            agent._mirror_flip(sq) if mirrored else 0,
        ) if zones > 1 else 0
        got_fb = int(fb.zone_of(sq, zones)) if zones > 1 else 0
        bad_agent += got_agent != want_block
        bad_fb += got_fb != want_block
        blocks.add(want_block)
    limit = agent.W1.shape[0] // FEATURES
    rep.line("zone map: agent", bad_agent == 0, f"{64 - bad_agent}/64 blocks match features.py")
    rep.line("zone map: fastboard", bad_fb == 0, f"{64 - bad_fb}/64 blocks match features.py")
    rep.line(
        "block range", max(blocks) < limit,
        f"blocks {min(blocks)}..{max(blocks)}, W1 holds {limit} blocks",
    )


def check_static(agent: Any, boards: list[chess.Board], rep: Report, label: str) -> None:
    """agent.Accumulator.refresh and fastboard.refresh vs the ground truth."""
    import fastboard as fb

    acc = agent.Accumulator()
    worst_py = 0.0
    for board in boards:
        acc.refresh(board)
        worst_py = max(worst_py, truth_gap(agent, board, acc.white, acc.black))
    rep.line(
        f"Accumulator.refresh {label}", worst_py < 1e-2,
        f"{len(boards)} pos, worst {worst_py:.2e}",
    )

    engine = agent._FAST
    if engine is None:
        rep.line(f"fastboard.refresh {label}", not agent.FAST_BOARD, "no FastEngine: skipped")
        return
    worst_fb = 0.0
    for board in boards:
        engine.pos.load(board)
        fb.refresh(
            engine.pos.bb, engine.pos.sq, engine.pos.meta, agent.W1, agent.B1,
            engine.white, engine.black, engine.zones, agent.KING_ZONES,
        )
        worst_fb = max(worst_fb, truth_gap(agent, board, engine.white, engine.black))
    rep.line(
        f"fastboard.refresh {label}", worst_fb < 1e-2,
        f"{len(boards)} pos, worst {worst_fb:.2e}",
    )


def check_incremental_tree(
    agent: Any, boards: list[chess.Board], depth: int, seed: int, rep: Report, label: str
) -> None:
    """Recursive make/unmake through the COMPILED board, ground-truthed at every node,
    with the zone stack checked for exact restore on the way back up."""
    import fastboard as fb

    engine = agent._FAST
    if engine is None:
        rep.line(f"fastboard tree {label}", not agent.FAST_BOARD, "no FastEngine: skipped")
        return
    rng = random.Random(seed)
    nodes = 0
    crossings = 0
    worst = 0.0
    zone_bad = 0
    acc_restore_bad = 0

    def walk(board: chess.Board, d: int) -> None:
        nonlocal nodes, worst, crossings, zone_bad, acc_restore_bad
        if d == 0 or board.is_game_over():
            return
        moves = list(board.legal_moves)
        rng.shuffle(moves)
        for move in moves[:4]:
            before_zones = engine.zones.copy()
            before_w = engine.white.copy()
            before_b = engine.black.copy()
            packed = fb.move_from_chess(move)
            engine._make(packed)
            board.push(move)
            nodes += 1
            if not np.array_equal(engine.zones, before_zones):
                crossings += 1
            worst = max(worst, truth_gap(agent, board, engine.white, engine.black))
            walk(board, d - 1)
            board.pop()
            engine._unmake()
            if not np.array_equal(engine.zones, before_zones):
                zone_bad += 1
            if not (
                np.array_equal(engine.white, before_w) and np.array_equal(engine.black, before_b)
            ):
                acc_restore_bad += 1

    for board in boards:
        work = board.copy()
        engine.pos.load(work)
        fb.refresh(
            engine.pos.bb, engine.pos.sq, engine.pos.meta, agent.W1, agent.B1,
            engine.white, engine.black, engine.zones, agent.KING_ZONES,
        )
        walk(work, depth)
    rep.line(
        f"fastboard tree {label}", worst < 1e-2 and zone_bad == 0 and acc_restore_bad == 0,
        f"{nodes} nodes, {crossings} block changes, worst {worst:.2e}, "
        f"zone-restore fails {zone_bad}, acc-restore fails {acc_restore_bad}",
    )


def check_pytree(
    agent: Any, boards: list[chess.Board], depth: int, seed: int, rep: Report
) -> None:
    """The same torture over agent.Accumulator.push/pop -- the FAST_BOARD=off path."""
    acc = agent.Accumulator()
    rng = random.Random(seed)
    nodes = 0
    crossings = 0
    worst = 0.0
    restore_bad = 0

    def walk(board: chess.Board, d: int) -> None:
        nonlocal nodes, worst, crossings, restore_bad
        if d == 0 or board.is_game_over():
            return
        moves = list(board.legal_moves)
        rng.shuffle(moves)
        for move in moves[:4]:
            state = (acc.zone_white, acc.zone_black, acc.flip_white, acc.flip_black)
            before_w = acc.white.copy()
            before_b = acc.black.copy()
            acc.push(board, move)
            board.push(move)
            nodes += 1
            if (acc.zone_white, acc.zone_black, acc.flip_white, acc.flip_black) != state:
                crossings += 1
            worst = max(worst, truth_gap(agent, board, acc.white, acc.black))
            walk(board, d - 1)
            board.pop()
            acc.pop()
            if (acc.zone_white, acc.zone_black, acc.flip_white, acc.flip_black) != state:
                restore_bad += 1
            if not (
                np.array_equal(acc.white, before_w) and np.array_equal(acc.black, before_b)
            ):
                restore_bad += 1

    for board in boards:
        work = board.copy()
        acc.refresh(work)
        walk(work, depth)
    rep.line(
        "Accumulator push/pop tree", worst < 1e-2 and restore_bad == 0,
        f"{nodes} nodes, {crossings} zone/flip changes, worst {worst:.2e}, "
        f"restore fails {restore_bad}",
    )


def check_symmetry(agent: Any, boards: list[chess.Board], depth: int, rep: Report) -> None:
    """A mirrored net must score a position and its left-right reflection IDENTICALLY:
    the reflection is exactly the transform the net folds away. This is the strongest
    end-to-end statement available, because it goes through the compiled search.
    Meaningless (and skipped) for an unmirrored net, which has no such invariance."""
    if not agent.MIRRORED:
        rep.line("mirror symmetry", True, "skipped: net is not mirrored")
        return
    engine = agent._FAST
    if engine is None:
        rep.line("mirror symmetry", not agent.FAST_BOARD, "no FastEngine: skipped")
        return
    import fastsearch as fs

    def scored(board: chess.Board, d: int) -> tuple[int, int]:
        import time

        engine.prepare(board, 0)
        engine.deadline = time.monotonic() + 3600.0
        engine.ctrl[fs.C_STOP] = 0
        # Neutralise everything that carries between calls, so the only difference
        # between a position and its reflection is the position itself.
        engine.ctrl[fs.C_TT_OFF] = 1
        engine.butterfly[:] = 0
        engine.killers2[:] = 0
        engine.counter[:] = 0
        engine.conthist1[:] = 0
        return engine.evaluate(), engine.root_search(d, -agent.INFINITY, agent.INFINITY, 0)

    worst_eval = 0
    worst_search = 0
    checked = 0
    ladder: dict[int, int] = {}
    for board in boards:
        flipped = mirror_board(board)
        if not flipped.is_valid():
            continue
        for d in range(1, depth + 1):
            _, s1 = scored(board, d)
            _, s2 = scored(flipped, d)
            ladder[d] = max(ladder.get(d, 0), abs(s1 - s2))
        e1, s1 = scored(board, depth)
        e2, s2 = scored(flipped, depth)
        worst_eval = max(worst_eval, abs(e1 - e2))
        worst_search = max(worst_search, abs(s1 - s2))
        checked += 1
    rep.line(
        "mirror symmetry (static eval)", worst_eval == 0,
        f"{checked} positions, worst gap {worst_eval} cp",
    )
    # A mirrored net's invariant is that the EVALUATION is reflection-symmetric. That is
    # what a wrong W1 block index breaks, and it is gated hard above and widely below.
    #
    # The SEARCH is not reflection-symmetric and gating it was a mistake I made and then
    # measured out. A position and its reflection have different square numbers, so move
    # generation order differs, and every order-dependent heuristic follows it. Evidence:
    #   * "ordering cannot change a root value at depth 1-2" is FALSE. Position
    #     qN2k3/3P4/5p2/3p2Q1/7r/4K2R/1n4P1/8 b has identical static eval (-33) and a
    #     65 cp gap at depth 2 that SURVIVES turning the prune ctrl slots off (_FOLD is
    #     False, so those slots are live -- the control was real).
    #   * The largest gap, 27,593 cp on 3k2R1/p2r4/3N4/1QP2P1p/8/4K3/1q5n/8 b, is one
    #     orientation seeing a mate the other misses. Stockfish confirms BOTH are mate
    #     in 3, and the UNMIRRORED shipped net shows the same split on the same position
    #     at depths 4-10. So it is a pre-existing mate-finding weakness of this search,
    #     not a mirroring defect, and gating it here would block a net over a bug it
    #     does not have.
    # Printed, loudly, because it is worth looking at -- but not a failure.
    print(
        f"{'mirror symmetry ladder':<34} : INFO  worst search gap by depth "
        + ", ".join(f"d{d}={ladder[d]}" for d in sorted(ladder))
    )
    print(
        f"{'mirror symmetry (depth ' + str(depth) + ')':<34} : INFO  "
        f"worst search gap {worst_search} cp -- the search is not reflection-symmetric; "
        "see the note in check_symmetry before reading anything into this"
    )


def check_symmetry_wide(agent: Any, count: int, seed: int, rep: Report) -> None:
    """Reflection-symmetry of the EVALUATION over positions from random played-out games.

    This is the gate that a wrong W1 block index cannot survive, and it is the one worth
    having: the search-gap checks above are not invariants (see the note there), but this
    is. Castling rights are skipped -- the feature set has no castling features, so the
    net is blind to them, but the BOARD is not, and a flipped position with rights is a
    different position rather than a reflection of the same one.
    """
    if not agent.MIRRORED:
        rep.line("mirror symmetry (wide)", True, "skipped: net is not mirrored")
        return
    engine = agent._FAST
    if engine is None:
        rep.line("mirror symmetry (wide)", not agent.FAST_BOARD, "no FastEngine: skipped")
        return
    rng = random.Random(seed)
    worst, worst_fen, checked, bad = 0, None, 0, 0
    for _ in range(count):
        board = chess.Board()
        for _ in range(rng.randint(2, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if board.is_game_over():
                break
        if board.is_game_over() or board.castling_rights:
            continue
        engine.prepare(board, 0)
        left = engine.evaluate()
        engine.prepare(board.transform(chess.flip_horizontal), 0)
        right = engine.evaluate()
        checked += 1
        if left != right:
            bad += 1
            if abs(left - right) > worst:
                worst, worst_fen = abs(left - right), board.fen()
    rep.line(
        "mirror symmetry (wide)", bad == 0,
        f"{checked} played-out positions, {bad} disagreements, worst {worst} cp"
        + (f" at {worst_fen}" if worst_fen else ""),
    )


def check_selfplay(agent: Any, games: int, nodes_depth: int, seed: int, rep: Report) -> None:
    """Play whole games with the real search and ground-truth the engine's live
    accumulators against a full rebuild after every move actually played."""
    engine = agent._FAST
    if engine is None:
        rep.line("self-play vs rebuild", not agent.FAST_BOARD, "no FastEngine: skipped")
        return
    import time

    import fastboard as fb
    import fastsearch as _fs
    from training import features as F  # noqa: N812

    def blocks_of(board: chess.Board) -> tuple[int, int]:
        out = []
        for colour in (chess.WHITE, chess.BLACK):
            king = board.king(colour)
            # Every position this audit builds has both kings; a missing one means the
            # caller handed us a corrupt board, and guessing a square would hide it.
            assert king is not None, f"no {'white' if colour else 'black'} king on {board.fen()}"
            own = king if colour == chess.WHITE else king ^ 56
            flip = F.mirror_flip(own) if agent.MIRRORED else 0
            zone = F.king_zone(own ^ flip, agent.KING_ZONES, mirrored=agent.MIRRORED)
            out.append(zone + (agent.KING_ZONES if flip else 0))
        return out[0], out[1]

    worst = 0.0
    plies = 0
    crossings = 0
    zone_drift = 0
    rng = random.Random(seed)
    for _g in range(games):
        board = chess.Board()
        for _ in range(rng.randint(0, 6)):
            board.push(rng.choice(list(board.legal_moves)))
        while not board.is_game_over(claim_draw=False) and board.fullmove_number < 100:
            engine.prepare(board, 0)
            engine.deadline = time.monotonic() + 3600.0
            engine.ctrl[_fs.C_STOP] = 0
            want = blocks_of(board)
            engine.root_search(nodes_depth, -agent.INFINITY, agent.INFINITY, 0)
            # A completed search must leave the engine standing at the root, with the
            # accumulators and the zone/block pair it started with.
            worst = max(worst, truth_gap(agent, board, engine.white, engine.black))
            if tuple(int(x) for x in engine.zones) != want:
                zone_drift += 1
            now = time.monotonic()
            try:
                move = engine.choose(now + 0.06, now + 0.10)
            except agent.Timeout:
                move = fb.move_from_chess(rng.choice(list(board.legal_moves)))
            mv = chess.Move.from_uci(fb.move_to_uci(move))
            if mv not in board.legal_moves:
                rep.line("self-play vs rebuild", False, f"illegal move {mv} in {board.fen()}")
                return
            before = blocks_of(board)
            board.push(mv)
            if blocks_of(board) != before:
                crossings += 1
            plies += 1
    rep.line(
        "self-play vs rebuild", worst < 1e-2 and zone_drift == 0,
        f"{plies} plies over {games} games, worst {worst:.2e}, "
        f"{crossings} block changes played, post-search zone drift {zone_drift}",
    )


def check_lazy_acc(agent: Any, boards: list[chess.Board], seed: int, rep: Report) -> None:
    """Drive fastsearch's deferred-accumulator path (make_move/unmake_move with
    C_LAZY_ACC set) and force a sync at every node, ground-truthing the result."""
    engine = agent._FAST
    if engine is None or not agent.COMPILED_SEARCH:
        rep.line("fastsearch LAZY_ACC path", not agent.FAST_BOARD, "no compiled search: skipped")
        return
    import fastboard as fb
    import fastsearch as fs

    rng = random.Random(seed)
    worst = 0.0
    nodes = 0
    restore_bad = 0
    for board in boards:
        work = board.copy()
        engine.pos.load(work)
        fb.refresh(
            engine.pos.bb, engine.pos.sq, engine.pos.meta, agent.W1, agent.B1,
            engine.white, engine.black, engine.zones, agent.KING_ZONES,
        )
        ctrl = engine.ctrl
        ctrl[fs.C_LAZY_ACC] = 1
        ctrl[fs.C_ACC_PLY] = int(engine.pos.meta[fb.PLY])
        root_w, root_b = engine.white.copy(), engine.black.copy()
        stack: list[chess.Move] = []
        for _ in range(12):
            moves = list(work.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            packed = fb.move_from_chess(move)
            fs.make_move(
                engine.pos.bb, engine.pos.sq, engine.pos.meta, engine.pos.undo,
                engine.pos.keys, packed, agent.W1, agent.B1, engine.white, engine.black,
                engine.astack, engine.zones, agent.KING_ZONES, ctrl,
            )
            work.push(move)
            stack.append(move)
            # Force the deferred updates through, exactly as a leaf evaluation does.
            fs.sync_acc(
                engine.pos.undo, agent.W1, engine.white, engine.black, engine.astack,
                engine.zones, ctrl, int(engine.pos.meta[fb.PLY]),
            )
            nodes += 1
            worst = max(worst, truth_gap(agent, work, engine.white, engine.black))
        while stack:
            fs.unmake_move(
                engine.pos.bb, engine.pos.sq, engine.pos.meta, engine.pos.undo,
                engine.pos.keys, engine.white, engine.black, engine.astack,
                engine.zones, ctrl,
            )
            work.pop()
            stack.pop()
        if not (
            np.array_equal(engine.white, root_w) and np.array_equal(engine.black, root_b)
        ):
            restore_bad += 1
        ctrl[fs.C_LAZY_ACC] = 0
    rep.line(
        "fastsearch LAZY_ACC path", worst < 1e-2 and restore_bad == 0,
        f"{nodes} synced plies, worst {worst:.2e}, root-restore fails {restore_bad}",
    )


def check_pyengine(agent: Any, games: int, seed: int, rep: Report) -> None:
    """The FAST_BOARD=off play path: agent.Engine's own search over python-chess,
    ground-truthing its Accumulator after every move it actually plays."""
    import time

    eng = agent.Engine()
    rng = random.Random(seed)
    worst = 0.0
    plies = 0
    stack_bad = 0
    for _g in range(games):
        board = chess.Board()
        for _ in range(rng.randint(0, 6)):
            board.push(rng.choice(list(board.legal_moves)))
        for _ in range(40):
            if board.is_game_over(claim_draw=False):
                break
            now = time.monotonic()
            # `_get_move` refreshes before calling choose; choose itself assumes the
            # accumulator is already synced to `board`.
            eng.acc.refresh(board)
            move = eng.choose(board, now + 0.08, now + 0.12)
            if move not in board.legal_moves:
                rep.line("Engine self-play vs rebuild", False, f"illegal {move} in {board.fen()}")
                return
            # Every push must have been popped: the zone stack is empty again.
            if eng.acc.zones:
                stack_bad += 1
            worst = max(worst, truth_gap(agent, board, eng.acc.white, eng.acc.black))
            board.push(move)
            plies += 1
    rep.line(
        "Engine self-play vs rebuild", worst < 1e-2 and stack_bad == 0,
        f"{plies} plies, worst {worst:.2e}, unbalanced push/pop {stack_bad}",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(ROOT))
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--tree-depth", type=int, default=4)
    ap.add_argument("--search-depth", type=int, default=6)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--skip", default="",
        help="comma-separated: selfplay,symmetry,lazy,pytree,pyengine",
    )
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    engine_dir = Path(args.engine).resolve()
    agent = load_engine(engine_dir)
    print(f"engine   : {engine_dir}")
    print(
        f"net      : zones={agent.KING_ZONES} mirrored={agent.MIRRORED} "
        f"W1={agent.W1.shape} acc={agent.ACC_SIZE}"
    )
    print(
        f"paths    : FAST_BOARD={agent.FAST_BOARD} _FAST_OK={agent._FAST_OK} "
        f"_COMPILED={agent._COMPILED} FastEngine={'yes' if agent._FAST else 'no'} "
        f"LAZY_ACC={agent.LAZY_ACC}"
    )
    print()

    rep = Report()
    random_pos = random_positions(args.positions, args.seed)
    wing_pos = king_wing_positions(args.positions, args.seed + 1)

    check_w1_doubling(agent, rep)
    check_zone_maps(agent, rep)
    check_static(agent, random_pos, rep, "(random)")
    check_static(agent, wing_pos, rep, "(both wings)")
    if "pytree" not in skip:
        check_pytree(agent, wing_pos[:12] + random_pos[:12], args.tree_depth, args.seed, rep)
    check_incremental_tree(
        agent, wing_pos[:12] + random_pos[:12], args.tree_depth, args.seed, rep, "(wings)"
    )
    if "lazy" not in skip:
        check_lazy_acc(agent, wing_pos[:20] + random_pos[:20], args.seed, rep)
    if "symmetry" not in skip:
        check_symmetry(agent, wing_pos[:16], args.search_depth, rep)
        check_symmetry_wide(agent, SYMMETRY_SAMPLE, args.seed + 2, rep)
    if "selfplay" not in skip:
        check_selfplay(agent, args.games, args.search_depth, args.seed, rep)
    if "pyengine" not in skip:
        check_pyengine(agent, max(1, args.games // 2), args.seed, rep)

    print()
    print("RESULT: " + ("all checks passed" if rep.fails == 0 else f"{rep.fails} CHECK(S) FAILED"))
    sys.exit(1 if rep.fails else 0)


if __name__ == "__main__":
    main()
