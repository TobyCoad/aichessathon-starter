"""The attack-bias instrument: does the net overvalue positions where we are attacking?

Measured on our own games (overnight/eval/v10/static_vs_ref.json, n=258), training source
moves this monotonically -- Lichess-only -452 cp, the shipped 50/50 mix -209, pure Stockfish
-120 -- while quiet positions barely move. That is the defect behind our blown wins: we
believe an attack is worth ~450 cp more than it is, commit to it, and it does not break through.

That sample is 42 attacking positions drawn from games we LOST, so it is small and selected.
This builds the same measurement on a neutral sample straight from the Stockfish corpus,
where every position already carries an engine label, so no Stockfish run is needed.

Usage: python -m training.attack_bias --checkpoint training/checkpoints/<net>.pt [--n 4000]
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import chess
import torch

from training.features import indices
from training.train import SCALE, load_checkpoint

FEATURES_PER_POS = 32


def attackers_on_enemy_king(board: chess.Board) -> int:
    """How many of the side-to-move's pieces bear on the enemy king's zone."""
    enemy_king = board.king(not board.turn)
    if enemy_king is None:
        return 0
    zone = {enemy_king} | set(chess.SquareSet(chess.BB_KING_ATTACKS[enemy_king]))
    return sum(len(board.attackers(board.turn, square)) for square in zone)


def evaluate(net: torch.nn.Module, boards: list[chess.Board], device: torch.device) -> list[float]:
    n = len(boards)
    white = torch.zeros(n, FEATURES_PER_POS, dtype=torch.long)
    black = torch.zeros(n, FEATURES_PER_POS, dtype=torch.long)
    mask = torch.zeros(n, FEATURES_PER_POS)
    stm = torch.zeros(n)
    for i, board in enumerate(boards):
        w = indices(board, chess.WHITE)
        b = indices(board, chess.BLACK)
        white[i, : len(w)] = torch.tensor(w)
        black[i, : len(b)] = torch.tensor(b)
        mask[i, : len(w)] = 1.0
        stm[i] = 1.0 if board.turn == chess.WHITE else 0.0
    out: list[float] = []
    with torch.no_grad():
        for lo in range(0, n, 8192):
            hi = min(lo + 8192, n)
            batch = net(white[lo:hi].to(device), black[lo:hi].to(device),
                        mask[lo:hi].to(device), stm[lo:hi].to(device))
            out.extend((batch * SCALE).cpu().tolist())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fens", type=Path, default=Path("overnight/eval/v10/attack_probe.json"))
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--king-zones", type=int, default=16)
    args = parser.parse_args()

    import json

    probe = json.loads(args.fens.read_text())
    boards = [chess.Board(r["fen"]) for r in probe]
    refs = [r["cp"] for r in probe]
    atk = [attackers_on_enemy_king(b) for b in boards]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_checkpoint(args.checkpoint, args.buckets, args.king_zones).to(device).eval()
    got = evaluate(net, boards, device)

    hi = [refs[i] - got[i] for i in range(len(probe)) if atk[i] >= 4]
    lo = [refs[i] - got[i] for i in range(len(probe)) if atk[i] < 4]
    print(f"{args.checkpoint.name}")
    print(f"  attacking (>=4 attackers, n={len(hi)}): signed {st.mean(hi):+7.0f} cp   "
          f"|err| {st.mean([abs(x) for x in hi]):6.0f}")
    print(f"  quiet     (<4 attackers, n={len(lo)}): signed {st.mean(lo):+7.0f} cp   "
          f"|err| {st.mean([abs(x) for x in lo]):6.0f}")
    weighted = (sum(abs(x) for x in hi) + sum(abs(x) for x in lo)) / max(1, len(hi) + len(lo))
    print(f"  weighted overall |err|: {weighted:6.0f} cp        "
          f"(negative signed = the net OVERvalues)")


if __name__ == "__main__":
    main()
