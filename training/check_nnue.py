r"""Verify the engine's learned evaluation against the trained model.

Three things must agree or the net is silently useless:

  1. the engine's feature indices and training/features.py;
  2. the incrementally maintained accumulator and a full rebuild, after every
     single ply -- including promotion, en passant and both castlings;
  3. the engine's numpy forward pass and the torch model it was trained as.

None of these produce an error when they are wrong. The engine loads, plays legal
moves, passes the crash gate, and merely plays worse.

Run: .venv\Scripts\python.exe -m training.check_nnue --agent overnight/challengers/002-nnue
"""

import argparse
import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType

import chess
import numpy as np
import torch

from training import features
from training.train import Net


def load_agent(directory: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("challenger_agent", directory / "agent.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {directory / 'agent.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["challenger_agent"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the engine's learned evaluation.")
    parser.add_argument("--agent", type=Path, default=Path("overnight/challengers/002-nnue"))
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/net.pt"))
    parser.add_argument("--plies", type=int, default=6000)
    arguments = parser.parse_args()

    agent = load_agent(arguments.agent)
    failures = 0

    # 1. Feature indices must match the training definition exactly.
    for square in range(64):
        for piece_type in range(1, 7):
            for colour in (chess.WHITE, chess.BLACK):
                for white_pov in (True, False):
                    mine = agent._feature(square, piece_type, colour, white_pov)
                    theirs = features.feature_index(
                        square, piece_type, colour, chess.WHITE if white_pov else chess.BLACK
                    )
                    if mine != theirs:
                        failures += 1
    print(f"feature indices        : {'MATCH' if failures == 0 else 'MISMATCH'} over 1536 cases")

    # 2. Incremental accumulator versus full rebuild, after every ply.
    rng = random.Random(0)
    acc = agent.Accumulator()
    reference = agent.Accumulator()
    board = chess.Board()
    acc.refresh(board)
    checked = drift = 0
    seen = {"promotion": 0, "en passant": 0, "castling": 0, "capture": 0}

    for _ in range(arguments.plies):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            board = chess.Board()
            acc.refresh(board)
            continue
        # Bias toward the rare paths: they are where accumulator bugs live.
        special = [
            m for m in moves
            if m.promotion
            or board.is_en_passant(m)
            or board.is_castling(m)
            or board.is_capture(m)
        ]
        move = rng.choice(special) if special and rng.random() < 0.5 else rng.choice(moves)
        if move.promotion:
            seen["promotion"] += 1
        if board.is_en_passant(move):
            seen["en passant"] += 1
        if board.is_castling(move):
            seen["castling"] += 1
        if board.is_capture(move):
            seen["capture"] += 1

        acc.push(board, move)
        board.push(move)
        checked += 1

        reference.refresh(board)
        if not (
            np.allclose(acc.white, reference.white, atol=1e-3)
            and np.allclose(acc.black, reference.black, atol=1e-3)
        ):
            drift += 1
            if drift <= 3:
                print(f"  DRIFT after {move.uci()}: {board.fen()}")
            acc.refresh(board)

    print(f"accumulator            : {checked - drift}/{checked} plies match a full rebuild")
    print("  coverage             : " + ", ".join(f"{k} {v}" for k, v in seen.items()))
    failures += drift

    # 2b. Random play produced only one en passant in 6000 plies, which is not
    # coverage of the case most likely to be wrong: the captured pawn is not on
    # the destination square. These positions force each rare path explicitly.
    forced = [
        ("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", "d4e3", "en passant, black"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", "en passant, white"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "castling, white kingside"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1", "castling, white queenside"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8", "castling, black kingside"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", "castling, black queenside"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", "promotion to queen"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n", "underpromotion to knight"),
        ("4k3/8/8/8/8/8/7p/6K1 b - - 0 1", "h2h1q", "promotion, black"),
        ("2r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7c8q", "capture-promotion"),
    ]
    forced_failures = 0
    for fen, uci, label in forced:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            print(f"  SKIP {label}: {uci} not legal in {fen}")
            forced_failures += 1
            continue
        acc.refresh(board)
        acc.push(board, move)
        board.push(move)
        reference.refresh(board)
        ok = np.allclose(acc.white, reference.white, atol=1e-3) and np.allclose(
            acc.black, reference.black, atol=1e-3
        )
        if not ok:
            print(f"  DRIFT on {label}: {fen} {uci}")
            forced_failures += 1
    print(
        f"forced rare paths      : {len(forced) - forced_failures}/{len(forced)} exact"
        " (en passant, castling, promotion, capture-promotion)"
    )
    failures += forced_failures

    # 3. The engine's numpy forward pass versus the torch model.
    state = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    net = Net().eval()
    net.load_state_dict(state)

    board = chess.Board()
    worst = 0.0
    compared = 0
    for _ in range(400):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            board = chess.Board()
            continue
        board.push(rng.choice(moves))
        acc.refresh(board)
        engine_cp = acc.evaluate(board.turn)

        white = features.white_indices(board)
        black = features.black_indices(board)
        pad = features.MAX_PIECES - len(white)
        with torch.no_grad():
            w = torch.tensor([white + [0] * pad])
            b = torch.tensor([black + [0] * pad])
            mask = torch.tensor([[1.0] * len(white) + [0.0] * pad])
            stm = torch.tensor([1.0 if board.turn == chess.WHITE else 0.0])
            logit = float(net(w, b, mask, stm))
        torch_cp = logit * agent.OUTPUT_SCALE
        worst = max(worst, abs(torch_cp - engine_cp))
        compared += 1

    print(f"numpy vs torch         : {compared} positions, worst gap {worst:.2f} cp")
    if worst > 2.0:
        failures += 1

    if failures:
        print(f"\n{failures} FAILURES")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
