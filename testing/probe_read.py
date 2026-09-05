"""Read the pondering probe from a platform PGN: our time spent on moves 8-10."""
import glob
import io
import re
import sys
from pathlib import Path

import chess.pgn

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("overnight/pgn/platform/*.pgn"))[-1]
text = Path(path).read_text(encoding="utf-8")
game = chess.pgn.read_game(io.StringIO(text))
colour = "white" if "-white-" in Path(path).name else "black"
stamps = re.findall(r"%clk (\d+):(\d+):([\d.]+)", text)
clocks = [float(h) * 3600 + float(m) * 60 + float(s) for h, m, s in stamps]
board = game.board()
ours = chess.WHITE if colour == "white" else chess.BLACK
previous = {chess.WHITE: None, chess.BLACK: None}
print(f"{Path(path).name}: we are {colour}")
for index, move in enumerate(game.mainline_moves()):
    side = board.turn
    if index < len(clocks):
        spent = None if previous[side] is None else previous[side] - clocks[index] + 0.5
        if side == ours and 7 <= board.fullmove_number <= 11:
            shown = "?" if spent is None else f"{spent:.2f}"
            print(f"  move {board.fullmove_number:>2} {board.san(move):<7} spent {shown} s")
        previous[side] = clocks[index]
    board.push(move)
