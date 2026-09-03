# ruff: noqa: E501  -- an HTML/JS page lives in this file
"""Play the champion by hand in a browser.

A tiny standard-library HTTP server: the page holds the game as a FEN, sends your
move, and the server validates it with python-chess, asks `agent.get_move` for the
reply with whatever think time you set, and returns the new position. The engine
keeps its per-game state (transposition table, repetition history) in the process,
as it does on the platform.

Run: .venv\\Scripts\\python.exe -m testing.play [--port 5610] [--agent .]
"""

import argparse
import importlib.util
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from urllib.parse import parse_qs, urlparse

import chess

PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Play the champion</title>
<style>
  body { font: 15px/1.4 system-ui, sans-serif; margin: 0; background: #f4f1ea; color: #222; }
  .wrap { display: grid; grid-template-columns: 560px 1fr; gap: 20px; padding: 16px; }
  .board { display: grid; grid-template-columns: repeat(8, 68px); grid-template-rows: repeat(8, 68px); border: 2px solid #444; width: max-content; user-select: none; }
  .sq { display: flex; align-items: center; justify-content: center; font-size: 48px; cursor: pointer; position: relative; }
  .light { background: #f0d9b5; } .dark { background: #b58863; }
  .sel { box-shadow: inset 0 0 0 4px #e0a800; }
  .tgt::after { content: ''; position: absolute; width: 18px; height: 18px; border-radius: 50%; background: rgba(0,0,0,.25); }
  .cap.tgt::after { width: 58px; height: 58px; border: 5px solid rgba(0,0,0,.25); background: none; box-sizing: border-box; }
  .last { box-shadow: inset 0 0 0 4px rgba(255, 200, 0, .7); }
  .ctl { margin: 10px 0; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  button, select, input { font: inherit; padding: 5px 10px; }
  .status { font-weight: 600; margin: 8px 0; min-height: 1.4em; }
  .moves { max-height: 70vh; overflow: auto; line-height: 1.8; }
  .moves span { padding: 1px 4px; }
  .think { color: #666; font-family: ui-monospace, monospace; }
</style>
<div class="wrap">
  <div>
    <div class="ctl">
      <button onclick="newGame('white')">New game as White</button>
      <button onclick="newGame('black')">New game as Black</button>
      <label>bot thinks for <select id="ms"><option value="1000">1 s</option><option value="3000" selected>3 s</option><option value="10000">10 s</option><option value="30000">30 s</option></select></label>
      <button onclick="undo()">Undo</button>
    </div>
    <div class="board" id="board"></div>
    <div class="status" id="status"></div>
    <div class="think" id="think"></div>
  </div>
  <div class="moves" id="moves"></div>
</div>
<script>
const PIECES = __PIECES__;
let S = null, sel = null, busy = false, flip = false, history = [];
function squares(fen) {
  const rows = fen.split(' ')[0].split('/'); const cells = [];
  for (const row of rows) for (const ch of row) { if (ch >= '1' && ch <= '8') for (let k = 0; k < +ch; k++) cells.push(''); else cells.push(ch); }
  return cells;  // index 0 = a8 .. 63 = h1
}
function name(idx) { return 'abcdefgh'[idx % 8] + (8 - Math.floor(idx / 8)); }
function draw() {
  const b = document.getElementById('board'); b.innerHTML = '';
  const cells = squares(S.fen);
  const targets = sel ? S.legal.filter(m => m.startsWith(sel)).map(m => m.slice(2, 4)) : [];
  for (let i = 0; i < 64; i++) {
    const idx = flip ? 63 - i : i; const r = Math.floor(idx / 8), f = idx % 8;
    const d = document.createElement('div'); const sq = name(idx);
    d.className = 'sq ' + (((r + f) % 2) ? 'dark' : 'light');
    if (sq === sel) d.classList.add('sel');
    if (targets.includes(sq)) { d.classList.add('tgt'); if (cells[idx]) d.classList.add('cap'); }
    if (S.last && (sq === S.last.slice(0, 2) || sq === S.last.slice(2, 4))) d.classList.add('last');
    d.textContent = PIECES[cells[idx]] || '';
    d.onclick = () => click(sq);
    b.appendChild(d);
  }
  document.getElementById('status').textContent = S.status;
  document.getElementById('moves').innerHTML = S.moves.map((m, i) => (i % 2 === 0 ? '<b>' + (i / 2 + 1) + '.</b> ' : '') + '<span>' + m + '</span>').join(' ');
}
async function api(path, body) {
  const r = await fetch(path, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {});
  return r.json();
}
async function newGame(colour) {
  flip = colour === 'black'; sel = null; history = [];
  busy = true; document.getElementById('think').textContent = '';
  S = await api('/api/new?colour=' + colour + '&ms=' + document.getElementById('ms').value);
  busy = false; draw(); showThink();
}
function showThink() { if (S.think_ms != null) document.getElementById('think').textContent = 'bot: ' + S.bot_move + ' in ' + (S.think_ms / 1000).toFixed(1) + ' s, eval ' + S.eval + ' cp, ' + S.nodes.toLocaleString() + ' nodes'; }
async function click(sq) {
  if (!S || busy || S.over) return;
  if (sel && sel !== sq) {
    let uci = sel + sq;
    const promo = S.legal.filter(m => m.startsWith(uci));
    if (promo.length && promo[0].length === 5) uci = uci + (prompt('Promote to (q/r/b/n)?', 'q') || 'q')[0];
    if (S.legal.includes(uci)) {
      history.push(S); busy = true; document.getElementById('status').textContent = 'thinking...';
      const s = await api('/api/move', {fen: S.fen, move: uci, moves: S.moves, ms: +document.getElementById('ms').value});
      busy = false; sel = null; S = s; draw(); showThink(); return;
    }
  }
  sel = S.legal.some(m => m.startsWith(sq)) ? sq : null; draw();
}
function undo() { if (history.length && !busy) { S = history.pop(); sel = null; draw(); } }
newGame('white');
</script>
"""


def load_agent(directory: Path) -> ModuleType:
    # The agent imports fastboard from its own directory; make that resolvable
    # however this script was launched.
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("play_agent", directory / "agent.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["play_agent"] = module
    spec.loader.exec_module(module)
    return module


AGENT: ModuleType | None = None


def status_of(board: chess.Board) -> tuple[str, bool]:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return ("White" if board.turn else "Black") + " to move" + (" (check)" if board.is_check() else ""), False
    if outcome.winner is None:
        return f"Draw by {outcome.termination.name.lower().replace('_', ' ')}", True
    return ("White" if outcome.winner else "Black") + " wins by " + outcome.termination.name.lower(), True


def bot_reply(board: chess.Board, ms: int) -> dict[str, object]:
    assert AGENT is not None
    started = time.perf_counter()
    uci = AGENT.get_move(board.fen(), ms)
    elapsed = (time.perf_counter() - started) * 1000.0
    move = chess.Move.from_uci(uci)
    san = board.san(move)
    board.push(move)
    # What the engine thinks of the position it just left us in, from its side.
    engine = AGENT._FAST
    evaluation: int | None = None
    nodes = 0
    if engine is not None:
        try:
            evaluation = -engine.evaluate()
            nodes = engine.nodes
        except Exception:
            evaluation = None
    return {"bot_move": san, "think_ms": round(elapsed), "eval": evaluation, "nodes": nodes}


def state(board: chess.Board, moves: list[str], extra: dict[str, object] | None = None) -> dict[str, object]:
    text, over = status_of(board)
    out: dict[str, object] = {
        "fen": board.fen(),
        "legal": [m.uci() for m in board.legal_moves],
        "moves": moves,
        "status": text,
        "over": over,
        "last": board.peek().uci() if board.move_stack else None,
    }
    if extra:
        out.update(extra)
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def _json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/api/new":
            query = parse_qs(url.query)
            colour = query.get("colour", ["white"])[0]
            ms = int(query.get("ms", ["3000"])[0])
            board = chess.Board()
            moves: list[str] = []
            extra: dict[str, object] = {}
            if colour == "black":
                reply = bot_reply(board, ms)
                moves.append(str(reply["bot_move"]))
                extra = reply
            self._json(state(board, moves, extra))
            return
        body = PAGE.replace("__PIECES__", json.dumps(PIECES)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        board = chess.Board(payload["fen"])
        moves = list(payload.get("moves", []))
        move = chess.Move.from_uci(payload["move"])
        if move not in board.legal_moves:
            self._json(state(board, moves, {"status": "illegal move"}))
            return
        moves.append(board.san(move))
        board.push(move)
        extra: dict[str, object] = {}
        if board.outcome(claim_draw=True) is None:
            reply = bot_reply(board, int(payload.get("ms", 3000)))
            moves.append(str(reply["bot_move"]))
            extra = reply
        self._json(state(board, moves, extra))


def main() -> None:
    global AGENT
    parser = argparse.ArgumentParser(description="Play the champion in a browser.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--port", type=int, default=5610)
    arguments = parser.parse_args()
    AGENT = load_agent(arguments.agent.resolve())
    server = ThreadingHTTPServer(("127.0.0.1", arguments.port), Handler)
    print(f"play the champion at http://127.0.0.1:{arguments.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
