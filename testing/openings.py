"""Balanced opening positions, so repeated games are not the same game.

`harness/referee.py` starts every game from the initial position. Two deterministic
agents therefore replay one identical game, and a 20-game score is a single sample
dressed up as twenty. These lines give the arena somewhere else to start.

The set is deliberately quiet and roughly balanced: mainline openings at 6-10 plies,
where neither side is already better. Unbalanced positions inflate the decisive rate
and measure the book rather than the engine.
"""

from functools import lru_cache

import chess

# Mainline openings, taken to the point where theory has a real choice and neither
# side stands better. Written in SAN because that is the form a human can check.
LINES: tuple[str, ...] = (
    # --- Open games, 1.e4 e5
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6",  # Ruy Lopez, Morphy
    "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4",  # Ruy Lopez, Berlin
    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6",  # Italian, Giuoco Piano
    "e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5",  # Italian, Giuoco Pianissimo
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6",  # Scotch
    "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4",  # Petrov
    "e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4",  # Vienna
    "e4 e5 f4 exf4 Nf3 g5 h4 g4",  # King's Gambit
    # --- Sicilian
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6",  # Najdorf
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6",  # Accelerated Dragon
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6",  # Dragon
    "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6",  # Taimanov
    "e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7",  # Closed Sicilian
    "e4 c5 c3 d5 exd5 Qxd5 d4 Nf6",  # Alapin
    # --- Other 1.e4
    "e4 e6 d4 d5 Nc3 Bb4 e5 c5",  # French, Winawer
    "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7",  # French, Tarrasch
    "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5",  # Caro-Kann, Classical
    "e4 c6 d4 d5 e5 Bf5 Nf3 e6",  # Caro-Kann, Advance
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6",  # Scandinavian
    "e4 Nf6 e5 Nd5 d4 d6 Nf3 g6",  # Alekhine
    "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7",  # Pirc
    "e4 g6 d4 Bg7 Nc3 d6 Nf3 a6",  # Modern
    # --- Queen's pawn
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7",  # QGD, Classical
    "d4 d5 c4 e6 Nc3 c6 Nf3 Nf6",  # Semi-Slav
    "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4",  # Slav, Main
    "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6",  # QGA
    "d4 d5 Nf3 Nf6 c4 e6 Nc3 Be7",  # QGD, Transposition
    "d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5",  # Exchange QGD
    # --- Indian defences
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6",  # King's Indian
    "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5",  # Gruenfeld
    "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O",  # Nimzo-Indian, Rubinstein
    "d4 Nf6 c4 e6 Nf3 b6 g3 Ba6",  # Queen's Indian
    "d4 Nf6 c4 c5 d5 b5 cxb5 a6",  # Benko Gambit
    "d4 Nf6 c4 e6 Nf3 Bb4 Bd2 Qe7",  # Bogo-Indian
    "d4 f5 g3 Nf6 Bg2 e6 Nf3 Be7",  # Dutch
    # --- Flank
    "Nf3 Nf6 c4 e6 Nc3 d5 d4 Be7",  # English into QGD
    "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5",  # English, Reversed Sicilian
    "c4 c5 Nf3 Nf6 d4 cxd4 Nxd4 e6",  # Symmetrical English
    "Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7",  # Reti / King's Indian Attack
    "b3 e5 Bb2 Nc6 e3 Nf6 Bb5 Bd6",  # Nimzo-Larsen
)


@lru_cache(maxsize=1)
def opening_fens() -> tuple[str, ...]:
    """Distinct positions from LINES, each line also cut short at several depths.

    One position per line is not enough. A 600-game SPRT is 300 pairs, so 40
    openings would each be replayed sixteen times and counted as sixteen
    independent samples. Correlated repeats inflate the likelihood ratio -- the
    very failure this module exists to prevent, reintroduced at 80 games rather
    than at 1. Cutting each line short multiplies the set without inventing
    positions that are unbalanced or off-book.
    """
    seen: set[str] = set()
    fens: list[str] = []
    for line in LINES:
        moves = line.split()
        board = chess.Board()
        for played, san in enumerate(moves, start=1):
            board.push_san(san)
            # Both parities. Whether White or Black is to move in the starting
            # position does not bias a paired match, because the pair swaps which
            # engine holds which colour, not which colour moves first.
            if played >= 6:
                fen = board.fen()
                if fen not in seen:
                    seen.add(fen)
                    fens.append(fen)
    return tuple(fens)


def pairs(games: int) -> list[tuple[str, bool]]:
    """Return `games` (fen, agent_plays_white) entries, colours paired.

    Consecutive entries share a FEN and swap colours, so every opening is played
    from both sides. Pentanomial statistics need that pairing; so does anyone who
    does not want a colour bias reported as an improvement.
    """
    fens = opening_fens()
    pairs_needed = (games + 1) // 2
    if pairs_needed > len(fens):
        print(
            f"  note: {pairs_needed} pairs over {len(fens)} openings, "
            f"{pairs_needed / len(fens):.1f} repeats each -- the SPRT will read "
            "more confident than the evidence warrants"
        )
    out: list[tuple[str, bool]] = []
    for index in range(games):
        out.append((fens[(index // 2) % len(fens)], index % 2 == 0))
    return out
