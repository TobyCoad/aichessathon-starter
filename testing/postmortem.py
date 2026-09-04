# ruff: noqa: E501  -- an HTML/SVG template lives in this file
"""Post-mortem of played games: where the points went, and why.

Feed it the PGNs the platform gives you (they carry clock comments). For each game it

  1. works out which side we were, by replaying our engine briefly on every
     position and seeing which side's moves it reproduces;
  2. evaluates every position with a reference engine (a local Stockfish, used
     only for analysis -- nothing here ships), from our point of view;
  3. flags every move of ours that lost value, and classifies each one:
       book       -- the move came from the opening book, the search never ran
       time       -- played with under a second, in the low-clock regime
       horizon    -- our engine finds the reference move given more time
       evaluation -- our engine still prefers its move with more time, and its
                     static evaluation disagrees with the reference by a lot
       search     -- everything else: our engine prefers its move with more time
                     but the evaluations agree, so the search missed it
  4. writes a Markdown report and an HTML page with the evaluation and clock curves.

Run:  .venv\\Scripts\\python.exe -m testing.postmortem game.pgn [more.pgn ...] [--out overnight/postmortem]
      .venv\\Scripts\\python.exe -m testing.postmortem --dir ~/Downloads --glob "aichessathon-*.pgn"
"""

import argparse
import importlib.util
import io
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType

import chess
import chess.engine
import chess.pgn

STOCKFISH = Path("engines/stockfish/stockfish-windows-x86-64-avx2.exe")
CLOCK = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")
INCREMENT_MS = 500
MATE_CP = 2000
BLUNDER, MISTAKE, INACCURACY = -150, -70, -30


def load_agent(directory: Path) -> ModuleType:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("postmortem_agent", directory / "agent.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["postmortem_agent"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Ply:
    number: int
    ours: bool
    san: str
    uci: str
    clock_ms: int | None
    spent_ms: int | None
    eval_before: int  # reference, our POV, before the move
    eval_after: int  # reference, our POV, after the move
    best: str  # reference's preferred move in this position
    delta: int = 0
    kind: str = ""  # blunder / mistake / inaccuracy / ""
    cause: str = ""  # book / time / horizon / evaluation / search
    engine_short: str = ""  # our engine's move with the time it actually had
    engine_long: str = ""  # our engine's move with 10 s
    static: int | None = None  # our net's static evaluation, our POV


@dataclass
class Report:
    file: str
    colour: str
    result: str
    termination: str
    plies: int
    score: float
    match_rate: float
    min_clock_ms: int
    time_trouble_moves: int
    counts: dict[str, int] = field(default_factory=dict)
    causes: dict[str, int] = field(default_factory=dict)
    turning: str = ""
    moves: list[Ply] = field(default_factory=list)


def clock_ms(comment: str) -> int | None:
    m = CLOCK.search(comment or "")
    if not m:
        return None
    return int((int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) * 1000)


def cp(score: chess.engine.PovScore, colour: chess.Color) -> int:
    s = score.pov(colour)
    if s.is_mate():
        m = s.mate() or 0
        return MATE_CP if m > 0 else -MATE_CP
    return max(-MATE_CP, min(MATE_CP, s.score() or 0))


def detect_side(agent: ModuleType, game: chess.pgn.Game, probe_ms: int) -> tuple[chess.Color, float]:
    """Which side plays like us: replay both, count agreements at a short budget."""
    agree = {chess.WHITE: 0, chess.BLACK: 0}
    total = {chess.WHITE: 0, chess.BLACK: 0}
    board = game.board()
    for move in game.mainline_moves():
        try:
            ours = agent.get_move(board.fen(), probe_ms)
        except Exception:
            ours = ""
        total[board.turn] += 1
        agree[board.turn] += ours == move.uci()
        board.push(move)
    rate = {c: agree[c] / max(1, total[c]) for c in (chess.WHITE, chess.BLACK)}
    colour = chess.WHITE if rate[chess.WHITE] >= rate[chess.BLACK] else chess.BLACK
    return colour, rate[colour]


def analyse(agent: ModuleType, sf: chess.engine.SimpleEngine, path: Path, depth: int, colour: chess.Color | None, probe_ms: int) -> Report:
    game = chess.pgn.read_game(io.StringIO(path.read_text(encoding="utf-8")))
    assert game is not None, path
    if colour is None:
        colour, rate = detect_side(agent, game, probe_ms)
    else:
        rate = float("nan")
    board = game.board()
    limit = chess.engine.Limit(depth=depth)

    # Reference evaluation of every position, plus the reference's move.
    info = sf.analyse(board, limit)
    evals = [cp(info["score"], colour)]
    bests = [info["pv"][0].uci() if info.get("pv") else ""]
    nodes = list(game.mainline())
    for node in nodes:
        board.push(node.move)
        if board.is_game_over():
            outcome = board.outcome()
            if outcome is None or outcome.winner is None:
                evals.append(0)
            else:
                evals.append(MATE_CP if outcome.winner == colour else -MATE_CP)
            bests.append("")
            break
        info = sf.analyse(board, limit)
        evals.append(cp(info["score"], colour))
        bests.append(info["pv"][0].uci() if info.get("pv") else "")

    # Walk again, building the per-ply record.
    board = game.board()
    previous_clock: dict[chess.Color, int | None] = {chess.WHITE: None, chess.BLACK: None}
    plies: list[Ply] = []
    for index, node in enumerate(nodes):
        if index + 1 >= len(evals):
            break
        mover = board.turn
        clock = clock_ms(node.comment)
        spent = None
        before = previous_clock[mover]
        if clock is not None and before is not None:
            spent = before - clock + INCREMENT_MS
        previous_clock[mover] = clock
        ply = Ply(
            number=index + 1,
            ours=mover == colour,
            san=board.san(node.move),
            uci=node.move.uci(),
            clock_ms=clock,
            spent_ms=spent,
            eval_before=evals[index],
            eval_after=evals[index + 1],
            best=bests[index],
        )
        if ply.ours:
            ply.delta = ply.eval_after - ply.eval_before
            if ply.delta <= BLUNDER:
                ply.kind = "blunder"
            elif ply.delta <= MISTAKE:
                ply.kind = "mistake"
            elif ply.delta <= INACCURACY:
                ply.kind = "inaccuracy"
            if ply.kind:
                ply.cause, ply.engine_short, ply.engine_long, ply.static = diagnose(agent, board, ply, colour)
        plies.append(ply)
        board.push(node.move)

    ours = [p for p in plies if p.ours]
    result = game.headers.get("Result", "*")
    won = (result == "1-0") == (colour == chess.WHITE) and result != "1/2-1/2"
    score = 0.5 if result == "1/2-1/2" else (1.0 if won else 0.0)
    counts = {k: sum(1 for p in ours if p.kind == k) for k in ("blunder", "mistake", "inaccuracy")}
    causes: dict[str, int] = {}
    for p in ours:
        if p.kind:
            causes[p.cause] = causes.get(p.cause, 0) + 1
    clocks = [p.clock_ms for p in ours if p.clock_ms is not None]
    worst = min(ours, key=lambda p: p.delta) if ours else None
    turning = f"move {worst.number} {worst.san} ({worst.delta:+d} cp, {worst.cause})" if worst and worst.kind else "none"
    return Report(
        file=path.name,
        colour="white" if colour == chess.WHITE else "black",
        result=result,
        termination=game.headers.get("Termination", ""),
        plies=len(plies),
        score=score,
        match_rate=rate,
        min_clock_ms=min(clocks) if clocks else 0,
        time_trouble_moves=sum(1 for p in ours if p.spent_ms is not None and p.spent_ms < 1000 and (p.clock_ms or 0) < 20_000),
        counts=counts,
        causes=causes,
        turning=turning,
        moves=plies,
    )


def diagnose(agent: ModuleType, board: chess.Board, ply: Ply, colour: chess.Color) -> tuple[str, str, str, int | None]:
    """Why did we play a losing move here? Replay our engine at two budgets."""
    fen = board.fen()
    book = agent._book_move(board) is not None if hasattr(agent, "_book_move") else False
    static: int | None = None
    try:
        engine = agent.Engine()
        engine.acc.refresh(board)
        static = engine.evaluate(board)
        if board.turn != colour:
            static = -static
    except Exception:
        static = None
    short_budget = max(200, min(ply.spent_ms or 3000, 20_000))
    try:
        short = agent.get_move(fen, short_budget * 4)  # the budget is a fraction of the clock handed in
    except Exception:
        short = ""
    try:
        long = agent.get_move(fen, 120_000)  # ~5 s of search under the tournament budget
    except Exception:
        long = ""
    if book:
        return "book", short, long, static
    if ply.spent_ms is not None and ply.spent_ms < 1000 and (ply.clock_ms or 0) < 20_000:
        return "time", short, long, static
    if long == ply.best and long != ply.uci:
        return "horizon", short, long, static
    # "evaluation" means our *search*, given time, still scores the position far
    # from the reference. The static score alone proves nothing: the search exists
    # to correct it, and often does.
    searched = search_score(agent, board, colour)
    if searched is not None and abs(searched - ply.eval_before) >= 150:
        return "evaluation", short, long, static
    return "search", short, long, static


def search_score(agent: ModuleType, board: chess.Board, colour: chess.Color, seconds: float = 3.0) -> int | None:
    """Our engine's own score for `board` after a short search, from our side."""
    engine_class = getattr(agent, "FastEngine", None)
    fastboard = getattr(agent, "_fb", None)
    if engine_class is None or fastboard is None or getattr(agent, "_FAST", None) is None:
        return None
    try:
        engine = engine_class()
        engine.pos.load(board)
        fastboard.refresh(
            engine.pos.bb, engine.pos.sq, engine.pos.meta, agent.W1, agent.B1,
            engine.white, engine.black, engine.zones, agent.KING_ZONES,
        )
        engine.root_side = int(engine.pos.meta[0])
        engine.draw_root = 0
        engine.deadline = time.monotonic() + seconds
        score = None
        try:
            for depth in range(1, 40):
                score = engine.search(depth, -agent.INFINITY, agent.INFINITY, 0)
        except agent.Timeout:
            pass
        if score is None:
            return None
        return int(score) if board.turn == colour else -int(score)
    except Exception:
        return None


PAGE = """<!doctype html>
<meta charset="utf-8"><title>Post-mortem __FILE__</title>
<style>body{font:14px system-ui;margin:16px;background:#f4f1ea;color:#222} table{border-collapse:collapse} td,th{padding:2px 8px;border-bottom:1px solid #ddd;font-family:ui-monospace,monospace;font-size:13px} .blunder{background:#f8c6c6} .mistake{background:#fbe2b6} .inaccuracy{background:#fff5c2} svg{background:#fff;border:1px solid #ccc}</style>
<h2>__TITLE__</h2>
<p>__SUMMARY__</p>
<svg width="900" height="260" viewBox="0 0 900 260">__EVAL__</svg>
<svg width="900" height="160" viewBox="0 0 900 160">__CLOCK__</svg>
<table><tr><th>#</th><th>move</th><th>eval before</th><th>after</th><th>delta</th><th>spent</th><th>clock</th><th>reference</th><th>cause</th><th>engine short</th><th>engine 5 s</th><th>our static</th></tr>__ROWS__</table>
"""


def svg_curve(values: list[float], height: int, lo: float, hi: float, colour: str, marks: list[int]) -> str:
    n = max(1, len(values) - 1)
    pts = " ".join(f"{20 + 860 * i / n:.1f},{height - 20 - (height - 40) * (v - lo) / (hi - lo):.1f}" for i, v in enumerate(values))
    zero = height - 20 - (height - 40) * (0 - lo) / (hi - lo)
    out = f'<line x1="20" y1="{zero:.1f}" x2="880" y2="{zero:.1f}" stroke="#999" stroke-dasharray="4"/>'
    out += f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="{pts}"/>'
    for i in marks:
        x = 20 + 860 * i / n
        out += f'<line x1="{x:.1f}" y1="20" x2="{x:.1f}" y2="{height - 20}" stroke="#d33" stroke-width="1"/>'
    return out


def write_html(report: Report, out: Path) -> None:
    evals = [report.moves[0].eval_before] + [p.eval_after for p in report.moves]
    marks = [i for i, p in enumerate(report.moves) if p.ours and p.kind in ("blunder", "mistake")]
    clocks = [(p.clock_ms or 0) / 1000 for p in report.moves if p.ours]
    rows = []
    for p in report.moves:
        cls = p.kind if p.ours else ""
        rows.append(
            f'<tr class="{cls}"><td>{p.number}</td><td>{"" if p.ours else "&nbsp;&nbsp;"}{p.san}</td><td>{p.eval_before:+d}</td><td>{p.eval_after:+d}</td>'
            f'<td>{f"{p.delta:+d}" if p.ours else ""}</td><td>{(p.spent_ms or 0) / 1000:.1f}</td><td>{(p.clock_ms or 0) / 1000:.1f}</td>'
            f'<td>{p.best}</td><td>{p.cause}</td><td>{p.engine_short}</td><td>{p.engine_long}</td><td>{"" if p.static is None else f"{p.static:+d}"}</td></tr>'
        )
    summary = (
        f"we were {report.colour}, result {report.result} by {report.termination}, {report.plies} plies, "
        f"side-detection agreement {report.match_rate:.0%}; blunders {report.counts.get('blunder', 0)}, mistakes {report.counts.get('mistake', 0)}, "
        f"inaccuracies {report.counts.get('inaccuracy', 0)}; causes {report.causes}; lowest clock {report.min_clock_ms / 1000:.1f} s, "
        f"moves under 1 s in time trouble {report.time_trouble_moves}; turning point {report.turning}"
    )
    page = (
        PAGE.replace("__FILE__", report.file).replace("__TITLE__", report.file).replace("__SUMMARY__", summary)
        .replace("__EVAL__", svg_curve([float(v) for v in evals], 260, -MATE_CP, MATE_CP, "#2a6", marks))
        .replace("__CLOCK__", svg_curve(clocks, 160, 0, 125, "#26a", []))
        .replace("__ROWS__", "".join(rows))
    )
    out.write_text(page, encoding="utf-8")


def write_markdown(reports: list[Report], out: Path) -> None:
    lines = ["# Post-mortem", "", "| game | we | result | plies | blunders | mistakes | causes | lowest clock | time-trouble moves | turning point |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        lines.append(
            f"| {r.file} | {r.colour} | {r.result} {r.termination} | {r.plies} | {r.counts.get('blunder', 0)} | {r.counts.get('mistake', 0)} | "
            f"{', '.join(f'{k} {v}' for k, v in sorted(r.causes.items()))} | {r.min_clock_ms / 1000:.1f} s | {r.time_trouble_moves} | {r.turning} |"
        )
    for r in reports:
        bad = [p for p in r.moves if p.ours and p.kind]
        if not bad:
            continue
        lines += ["", f"## {r.file} ({r.colour}, {r.result})", "", "| # | move | delta | reference | cause | engine short | engine 5 s | our static | ref before |", "|---|---|---|---|---|---|---|---|---|"]
        for p in bad:
            lines.append(f"| {p.number} | {p.san} | {p.delta:+d} | {p.best} | {p.cause} | {p.engine_short} | {p.engine_long} | {'' if p.static is None else f'{p.static:+d}'} | {p.eval_before:+d} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-mortem of played games.")
    parser.add_argument("pgns", nargs="*", type=Path)
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--glob", default="*.pgn")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--stockfish", type=Path, default=STOCKFISH)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--colour", choices=["white", "black"], default=None)
    parser.add_argument("--probe-ms", type=int, default=1500, help="clock handed to our engine when detecting our side")
    parser.add_argument("--out", type=Path, default=Path("overnight/postmortem"))
    arguments = parser.parse_args()

    paths = list(arguments.pgns)
    if arguments.dir:
        paths += sorted(arguments.dir.glob(arguments.glob))
    if not paths:
        raise SystemExit("no PGNs given")
    arguments.out.mkdir(parents=True, exist_ok=True)
    agent = load_agent(arguments.agent.resolve())
    sf = chess.engine.SimpleEngine.popen_uci(str(arguments.stockfish.resolve()))
    sf.configure({"Threads": 1, "Hash": 64})
    colour = None if arguments.colour is None else (chess.WHITE if arguments.colour == "white" else chess.BLACK)
    reports: list[Report] = []
    try:
        for path in paths:
            started = time.perf_counter()
            report = analyse(agent, sf, path, arguments.depth, colour, arguments.probe_ms)
            reports.append(report)
            write_html(report, arguments.out / (path.stem + ".html"))
            (arguments.out / (path.stem + ".json")).write_text(json.dumps(asdict(report), indent=1), encoding="utf-8")
            print(
                f"{path.name}: we were {report.colour}, {report.result} by {report.termination}, "
                f"blunders {report.counts.get('blunder', 0)} mistakes {report.counts.get('mistake', 0)} "
                f"causes {report.causes} lowest clock {report.min_clock_ms / 1000:.1f} s  ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    finally:
        sf.quit()
    write_markdown(reports, arguments.out / "REPORT.md")
    print(f"wrote {arguments.out / 'REPORT.md'} and one html/json per game")


if __name__ == "__main__":
    main()
