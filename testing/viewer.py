# ruff: noqa: E501  -- an HTML/JS template lives in this file
"""Build a self-contained HTML page to step through saved games.

The arena keeps every game as a PGN; this turns a directory of them into one
page with a board, the move list, the result and termination, and keyboard
navigation. No dependencies at view time: the positions are computed here with
python-chess and embedded as FENs, and the board is drawn with Unicode pieces.

Run: .venv\\Scripts\\python.exe -m testing.viewer overnight/pgn/<match-dir> --out games.html
"""

import argparse
import html
import io
import json
from pathlib import Path

import chess
import chess.pgn

PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { font: 14px/1.4 system-ui, sans-serif; margin: 0; background: #f4f1ea; color: #222; }
  .wrap { display: grid; grid-template-columns: 300px 520px 1fr; gap: 16px; padding: 16px; }
  .list { max-height: 95vh; overflow: auto; }
  .list button { display: block; width: 100%; text-align: left; margin: 2px 0; padding: 6px;
                 border: 1px solid #ccc; background: #fff; cursor: pointer; font: inherit; }
  .list button.active { background: #ffe9a8; }
  .board { display: grid; grid-template-columns: repeat(8, 62px); grid-template-rows: repeat(8, 62px);
           border: 2px solid #444; width: max-content; }
  .sq { display: flex; align-items: center; justify-content: center; font-size: 44px; }
  .light { background: #f0d9b5; } .dark { background: #b58863; }
  .from, .to { box-shadow: inset 0 0 0 4px rgba(255, 200, 0, .8); }
  .moves { max-height: 80vh; overflow: auto; }
  .moves span { display: inline-block; padding: 2px 5px; cursor: pointer; border-radius: 3px; }
  .moves span.cur { background: #ffe9a8; }
  .meta { margin: 8px 0; }
  .ctl button { font: inherit; padding: 4px 10px; margin-right: 4px; }
  .eval { font-family: ui-monospace, monospace; color: #555; }
</style>
<div class="wrap">
  <div class="list" id="list"></div>
  <div>
    <div class="meta" id="meta"></div>
    <div class="board" id="board"></div>
    <div class="ctl" style="margin-top:8px">
      <button onclick="go(0)">|&lt;</button><button onclick="go(cur-1)">&lt;</button>
      <button onclick="go(cur+1)">&gt;</button><button onclick="go(1e9)">&gt;|</button>
      <button onclick="flip=!flip;draw()">flip</button>
      <span class="eval" id="clock"></span>
    </div>
    <div style="color:#666;margin-top:6px">keys: left/right step, home/end, f flips, up/down change game</div>
  </div>
  <div class="moves" id="moves"></div>
</div>
<script>
const GAMES = __GAMES__;
let gi = 0, cur = 0, flip = false;
function go(i) { const g = GAMES[gi]; cur = Math.max(0, Math.min(g.fens.length - 1, i)); draw(); }
function pick(i) { gi = i; cur = 0; flip = !GAMES[gi].agentWhite; draw(); }
function draw() {
  const g = GAMES[gi];
  const fen = g.fens[cur].split(' ')[0];
  const rows = fen.split('/');
  const cells = [];
  for (let r = 0; r < 8; r++) {
    let f = 0;
    for (const ch of rows[r]) {
      if (ch >= '1' && ch <= '8') { for (let k = 0; k < +ch; k++) { cells.push(''); f++; } }
      else { cells.push(ch); f++; }
    }
  }
  const board = document.getElementById('board');
  board.innerHTML = '';
  const last = cur > 0 ? g.moves[cur - 1] : null;
  for (let i = 0; i < 64; i++) {
    const idx = flip ? 63 - i : i;
    const r = Math.floor(idx / 8), f = idx % 8;
    const sqName = 'abcdefgh'[f] + (8 - r);
    const d = document.createElement('div');
    d.className = 'sq ' + (((r + f) % 2) ? 'dark' : 'light');
    if (last && (sqName === last.uci.slice(0, 2))) d.classList.add('from');
    if (last && (sqName === last.uci.slice(2, 4))) d.classList.add('to');
    d.textContent = g.pieces[cells[idx]] || '';
    board.appendChild(d);
  }
  document.getElementById('meta').innerHTML =
    `<b>${g.white}</b> vs <b>${g.black}</b> &nbsp; result <b>${g.result}</b> (${g.termination}), ${g.moves.length} plies`;
  const mv = document.getElementById('moves');
  mv.innerHTML = g.moves.map((m, i) => `${i % 2 === 0 ? '<b>' + (Math.floor(i / 2) + 1) + '.</b> ' : ''}` +
    `<span class="${i + 1 === cur ? 'cur' : ''}" onclick="go(${i + 1})">${m.san}</span>`).join(' ');
  const c = mv.querySelector('.cur'); if (c) c.scrollIntoView({block: 'nearest'});
  document.getElementById('clock').textContent = cur > 0 && g.moves[cur - 1].comment ? g.moves[cur - 1].comment : '';
  document.querySelectorAll('#list button').forEach((b, i) => b.classList.toggle('active', i === gi));
}
document.getElementById('list').innerHTML = GAMES.map((g, i) =>
  `<button onclick="pick(${i})">${i + 1}. ${g.label}</button>`).join('');
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') go(cur + 1); else if (e.key === 'ArrowLeft') go(cur - 1);
  else if (e.key === 'Home') go(0); else if (e.key === 'End') go(1e9);
  else if (e.key === 'f') { flip = !flip; draw(); }
  else if (e.key === 'ArrowDown') pick(Math.min(GAMES.length - 1, gi + 1));
  else if (e.key === 'ArrowUp') pick(Math.max(0, gi - 1));
});
pick(0);
</script>
"""


def load(path: Path, agent_name: str) -> dict[str, object] | None:
    game = chess.pgn.read_game(io.StringIO(path.read_text(encoding="utf-8")))
    if game is None:
        return None
    board = game.board()
    fens = [board.fen()]
    moves: list[dict[str, str]] = []
    for node in game.mainline():
        move = node.move
        moves.append({"san": board.san(move), "uci": move.uci(), "comment": node.comment})
        board.push(move)
        fens.append(board.fen())
    white = game.headers.get("White", "white")
    black = game.headers.get("Black", "black")
    result = game.headers.get("Result", "*")
    termination = game.headers.get("Termination", "")
    agent_white = agent_name in white.lower()
    label = f"{result} {termination} ({len(moves)} plies) {path.stem}"
    return {
        "white": white, "black": black, "result": result, "termination": termination,
        "fens": fens, "moves": moves, "agentWhite": agent_white, "label": label, "pieces": PIECES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a game viewer from a PGN directory.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, default=Path("games.html"))
    parser.add_argument("--agent-name", default="starter", help="substring naming our side")
    arguments = parser.parse_args()

    games = []
    for path in sorted(arguments.directory.glob("*.pgn")):
        loaded = load(path, arguments.agent_name)
        if loaded is not None:
            games.append(loaded)
    if not games:
        raise SystemExit(f"no games under {arguments.directory}")
    page = (
        PAGE.replace("__TITLE__", html.escape(arguments.directory.name))
        .replace("__GAMES__", json.dumps(games))
    )
    arguments.out.write_text(page, encoding="utf-8")
    print(f"wrote {arguments.out} with {len(games)} games")


if __name__ == "__main__":
    main()
