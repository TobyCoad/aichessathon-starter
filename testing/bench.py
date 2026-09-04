"""Search benchmark: the numbers the depth work is judged by.

Forty fixed positions -- openings, middlegames, endings, tactics. Two modes:

  fixed depth  nodes, time, best move and score per position: the determinism
               reference. Two builds with the same semantics must agree exactly.
  fixed time   depth reached per position: the number we are trying to move.

Run:  .venv\\Scripts\\python.exe -m testing.bench --depth 6
      .venv\\Scripts\\python.exe -m testing.bench --seconds 5
      .venv\\Scripts\\python.exe -m testing.bench --depth 6 --agent overnight/challengers/050-x
Add --json PATH to save the per-position results for a later diff.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import chess

# fmt: off
POSITIONS = [
    # openings
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 1 4",
    "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkb1r/pp3ppp/4pn2/2pp4/3P1B2/4P3/PPP2PPP/RN1QKBNR w KQkq - 0 5",
    # middlegames
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QK2R w KQ - 0 8",
    "r2q1rk1/1b1nbppp/pp2pn2/2pp4/2PP4/1PN1PN2/PB2BPPP/R2Q1RK1 w - - 0 11",
    "r1bqr1k1/pp1n1pbp/2pp1np1/8/2PPP3/2N2N1P/PP2BPP1/R1BQR1K1 b - - 0 10",
    "2rq1rk1/pb1nbppp/1p2pn2/2pp4/3P4/1P1BPN2/PBPN1PPP/R2QR1K1 w - - 0 12",
    "r1b2rk1/2q1bppp/p1nppn2/1p6/3NPP2/2N1B3/PPPQ2PP/2KR1B1R w - - 0 12",
    "r2qkb1r/pp1n1ppp/2p1pn2/3p1b2/2PP4/1QN1PN2/PP3PPP/R1B1KB1R w KQkq - 0 7",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "rnb1k2r/pp2bppp/4pn2/q1pp4/3P4/2P1PN2/P1QNBPPP/R3K2R w KQkq - 0 9",
    "r3k2r/pbppqppp/1pn2n2/4p3/1bB1P3/2NP1N2/PPPB1PPP/R2QK2R w KQkq - 0 8",
    "r1bqk2r/pp1nbppp/2p1pn2/3p4/2PP4/2N1PN2/PPQ2PPP/R1B1KB1R w KQkq - 0 7",
    # tactics (WAC / classic)
    "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
    "8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1",
    "5rk1/1ppb3p/p1pb4/6q1/3P1p1r/2P1R2P/PP1BQ1P1/5RKN w - - 0 1",
    "r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - 0 1",
    "5rk1/pp4pp/4p3/2R3Q1/3n4/2q4r/P1P2PPP/5RK1 b - - 0 1",
    "r1b1k2r/ppppqppp/8/2bP4/3p4/6P1/PPQPPPBP/R1B2RK1 b kq - 0 1",
    "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1",
    "3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - - 0 1",
    "2q1rr1k/3bbnnp/p2p1pp1/2pPp3/PpP1P1P1/1P2BNNP/2BQ1PRK/7R b - - 0 1",
    "rnbqkb1r/p3pppp/1p6/2ppP3/3N4/2P5/PPP1QPPP/R1B1KB1R w KQkq - 0 1",
    # endings
    "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
    "8/p7/1p6/1P4k1/6p1/6K1/P7/8 w - - 0 1",
    "8/8/1p1r1k2/p1pPN1p1/P3KnP1/1P6/8/3R4 b - - 0 1",
    "6k1/5p2/6p1/8/7p/8/6PP/6K1 w - - 0 1",
    "8/5pk1/6p1/1R6/8/6P1/r4PK1/8 w - - 0 1",
    "5k2/8/8/8/8/8/3Q4/4K3 w - - 0 1",
    "8/2k5/8/8/8/8/1R6/1K6 w - - 0 1",
    "r3k3/8/8/8/8/8/8/4K2R w Kq - 0 1",
    "8/1b3k2/8/3N4/8/8/2K5/8 w - - 0 1",
    "6k1/pp3ppp/8/2b5/8/2N5/PPP2PPP/6K1 w - - 0 1",
    # the platform losses, at the decisive moment
    "5rk1/1b5p/pp4r1/2P5/4p2R/2R1NB2/PP3K2/8 w - - 1 33",
    "7k/1b5p/pP2r3/8/6R1/2R1Np2/PP1r1K2/8 w - - 1 39",
    "1r2br1k/2q2pp1/pN1p1b1p/P2Bp3/pP2P3/2PRP2P/6P1/3Q1RK1 w - - 1 29",
    "3b4/5pp1/R1n2k1p/P7/8/7P/5PP1/5BK1 b - - 2 45",
]
# fmt: on


def load_agent(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_agent", path / "agent.py")
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engine_for(agent: ModuleType, board: chess.Board):
    """A fresh FastEngine on `board`; None if the agent has no compiled board."""
    if getattr(agent, "_FAST", None) is None:
        return None
    fb = agent._fb
    engine = agent.FastEngine()
    if hasattr(engine, "prepare"):
        engine.prepare(board, 0)
        return engine
    engine.pos.load(board)
    fb.refresh(
        engine.pos.bb, engine.pos.sq, engine.pos.meta, agent.W1, agent.B1,
        engine.white, engine.black, engine.zones, agent.KING_ZONES,
    )
    engine.root_side = int(engine.pos.meta[0])
    engine.draw_root = 0
    return engine


def root_search(engine, depth: int, alpha: int, beta: int) -> int:
    if hasattr(engine, "root_search"):
        return int(engine.root_search(depth, alpha, beta, 0))
    return int(engine.search(depth, alpha, beta, 0))


def best_move(agent: ModuleType, engine, board: chess.Board) -> str:
    entry = engine.table.get(int(engine.pos.keys[0])) if hasattr(engine.table, "get") else None
    if entry and entry[3]:
        return agent._fb.move_to_uci(entry[3])
    tt = getattr(engine, "tt", ())
    if tt:
        key = engine.pos.keys[0]
        slot = int(key & agent._fs.TT_MASK)
        if tt[0][slot] == key:
            move = int(agent._fs.unpack_move(tt[1][slot]))
            if move:
                return agent._fb.move_to_uci(move)
    return "?"


def run_depth(agent: ModuleType, board: chess.Board, depth: int) -> dict:
    engine = engine_for(agent, board)
    engine.deadline = time.monotonic() + 3600
    started = time.perf_counter()
    score = 0
    for d in range(1, depth + 1):
        score = root_search(engine, d, -agent.INFINITY, agent.INFINITY)
    elapsed = time.perf_counter() - started
    return {
        "depth": depth, "score": int(score), "nodes": int(engine.nodes),
        "seconds": round(elapsed, 4), "best": best_move(agent, engine, board),
    }


def run_time(agent: ModuleType, board: chess.Board, seconds: float) -> dict:
    engine = engine_for(agent, board)
    engine.deadline = time.monotonic() + seconds
    started = time.perf_counter()
    reached = 0
    score = 0
    try:
        for d in range(1, 64):
            score = root_search(engine, d, -agent.INFINITY, agent.INFINITY)
            reached = d
    except agent.Timeout:
        pass
    elapsed = time.perf_counter() - started
    return {
        "depth": reached, "score": int(score), "nodes": int(engine.nodes),
        "seconds": round(elapsed, 3), "best": best_move(agent, engine, board),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search benchmark.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="first N positions only")
    arguments = parser.parse_args()
    if not arguments.depth and not arguments.seconds:
        raise SystemExit("give --depth or --seconds")
    agent = load_agent(arguments.agent.resolve())
    positions = POSITIONS[: arguments.limit] if arguments.limit else POSITIONS

    results = []
    total_nodes = 0
    total_seconds = 0.0
    depths = []
    for index, fen in enumerate(positions, start=1):
        board = chess.Board(fen)
        if arguments.depth:
            r = run_depth(agent, board, arguments.depth)
        else:
            r = run_time(agent, board, arguments.seconds)
        r["fen"] = fen
        results.append(r)
        total_nodes += r["nodes"]
        total_seconds += r["seconds"]
        depths.append(r["depth"])
        print(
            f"{index:>2} d{r['depth']:<2} {r['score']:>+6} {r['best']:<6} "
            f"{r['nodes']:>9} nodes {r['seconds']:>7.3f}s  {fen[:40]}",
            flush=True,
        )
    knps = total_nodes / max(total_seconds, 1e-9) / 1000
    if arguments.depth:
        print(
            f"\nfixed depth {arguments.depth}: {total_nodes:,} nodes "
            f"in {total_seconds:.1f}s = {knps:.0f} knps"
        )
    else:
        print(
            f"\nfixed time {arguments.seconds}s: mean depth {sum(depths) / len(depths):.2f}, "
            f"{knps:.0f} knps"
        )
    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"wrote {arguments.json}")


if __name__ == "__main__":
    main()
