r"""Evaluate a net by replaying the games we lost, at the real time control.

The gauntlet answers "is this net better against a copy of ourselves at 8 s a game".
That is not the question. This answers "in the positions where we actually lost, at the
clock we actually had, does this net choose better?" -- which is what a rating is made of.

For every one of our moves in a lost or drawn game it calls the platform's own entry
point, `agent.get_move(fen, time_left_ms)`, with the clock we really had at that move, so
the budget logic, contempt and the transposition table all behave as they did in the game.
Each chosen move is then scored against a Stockfish reference with a fresh hash.

Two honest limits, both measured:
  * The engine is time-limited, not depth-limited, so a replay does NOT reproduce the game
    move for move -- on one game it matched 28 of 42. Read differences in AGGREGATE.
  * This machine is roughly 1.8-2.1x faster than the platform, so a replay searches deeper
    than the real game did. That bias is shared by every candidate, so it cancels in an A/B.
    Do NOT try to model it by dividing the clock: the time manager is not scale-invariant
    (LOW_CLOCK_V6 = 12 s and the move horizon are absolute), so halving the clock puts a
    quarter of all moves into the wrong branch and a 20x divisor puts every move into the
    emergency path.
  * Selecting only lost and drawn games conditions the sample on games a DIFFERENT net
    played, so a win here does not map cleanly onto rating.

Run:  .venv\Scripts\python.exe -m testing.replay_eval --agent DIR --dir overnight/pgn/platform
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics as st
import sys
import time
from pathlib import Path
from types import ModuleType

import chess
import chess.engine
import chess.pgn

STOCKFISH = Path("engines/stockfish/stockfish-windows-x86-64-avx2.exe")
MATE_CP = 20000
# One "reference mates, we do not" is ~19,500 cp and would own the whole mean:
# a game measured at ACPL 288.7 had a MEDIAN loss of 0.0. Clamp each side before
# differencing (the lichess convention) and count mate events separately.
CLAMP = 1000
CLOCK = re.compile(r"clk (\d+):(\d+):([\d.]+)")


def load_agent(directory: Path) -> ModuleType:
    """Load a candidate engine, kernel included.

    `agent.py` does a plain `import fastsearch` / `import fastboard`, so without putting the
    candidate first on sys.path -- and evicting any already-imported copy -- you get the
    candidate's Python layer running on the ROOT kernel. Worse, `fastsearch._scan_agent_flags`
    reads the agent.py sitting next to ITSELF and folds those switches into the numba kernel
    as compile-time constants, so you would measure the candidate's evaluation through the
    root's search switches: a chimera that is neither build.
    """
    directory = directory.resolve()
    for name in ("fastsearch", "fastboard", "replay_agent"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("replay_agent", directory / "agent.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    for name in ("fastsearch", "fastboard"):
        bound = sys.modules.get(name)
        if bound is not None and Path(bound.__file__ or "").parent != directory:
            raise RuntimeError(
                f"{name} resolved to {bound.__file__}, not the candidate's copy in {directory}"
            )
    return module


def our_colour(path: Path) -> chess.Color | None:
    """Filenames encode it: round-NN-result-COLOUR-vs-opponent. Never guess it -- the
    engine-replay heuristic postmortem.py uses was only 72-79% accurate."""
    match = re.match(r"round-\d+-(?:win|loss|draw|unknown)-(white|black)-", path.name)
    if match is None:
        return None
    return chess.WHITE if match.group(1) == "white" else chess.BLACK


def clock_ms(comment: str) -> int | None:
    found = CLOCK.search(comment or "")
    if found is None:
        return None
    hours, minutes, seconds = found.groups()
    return int((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def score_after(engine: chess.engine.SimpleEngine, board: chess.Board, uci: str, depth: int) -> int:
    """Reference score after `uci`, from the mover's point of view. A fresh `game` token
    per call: without one python-chess never sends `ucinewgame`, the hash carries between
    positions, and the same move can score 339 or 585 depending on what preceded it."""
    child = board.copy()
    child.push(chess.Move.from_uci(uci))
    # claim_draw=True, as harness/referee.py uses: a move that makes a threefold or a
    # claimable fifty-move draw ENDS the game on the platform. Without this those positions
    # are handed to Stockfish as live and scored as whatever the eval says -- and they occur
    # in six of our games, all of them drawn ones, i.e. exactly what we are trying to explain.
    if child.is_game_over(claim_draw=True):
        outcome = child.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0
        return MATE_CP if outcome.winner == board.turn else -MATE_CP
    info = engine.analyse(child, chess.engine.Limit(depth=depth), game=object())
    return int(info["score"].pov(board.turn).score(mate_score=MATE_CP))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--dir", type=Path, default=Path("overnight/pgn/platform"))
    parser.add_argument("--glob", default="round-*.pgn")
    parser.add_argument("--results", default="loss,draw", help="which games to replay")
    parser.add_argument("--depth", type=int, default=20, help="reference depth")
    parser.add_argument("--base-ms", type=int, default=120_000)
    parser.add_argument("--increment-ms", type=int, default=500)
    parser.add_argument("--dump", type=Path, default=None,
                        help="write per-move JSONL; write fresh every run, never read one back")
    parser.add_argument("--max-games", type=int, default=0)
    arguments = parser.parse_args()

    wanted = set(arguments.results.split(","))
    games = [
        p for p in sorted(arguments.dir.glob(arguments.glob))
        if any(f"-{w}-" in p.name for w in wanted) and our_colour(p) is not None
    ]
    if arguments.max_games:
        games = games[: arguments.max_games]
    agent = load_agent(arguments.agent.resolve())
    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH.resolve()))
    engine.configure({"Threads": 1, "Hash": 128})

    losses: list[float] = []
    per_move: list[dict[str, object]] = []
    matched = total = mates = negative = crashes = flags = illegal = 0
    per_game: list[tuple[str, float, int]] = []
    for path in games:
        colour = our_colour(path)
        with path.open(encoding="utf-8", errors="ignore") as handle:
            game = chess.pgn.read_game(handle)
        if game is None:
            continue
        board = game.board()
        game_loss: list[float] = []
        clock = float(arguments.base_ms)
        increment = float(arguments.increment_ms)
        for node in game.mainline():
            move = node.move
            if board.turn != colour:
                board.push(move)
                continue
            if clock <= 1000:
                board.push(move)
                continue
            # Hand the engine the clock the REFEREE would have: base, minus this engine's own
            # spends, plus the increment. Reading %clk instead looks faithful and is not --
            # `agent._note_clock` infers the increment from the clock delta against its own
            # last spend, so a PGN clock (the OLD net's spends) makes that inference noise.
            # Measured: inferred increment 735-1451 ms against a true 500, think time swinging
            # 1.8x on the same game, and ACPL 5.51/7.72/12.59 across three identical runs.
            # Simulating it makes the inference exact (500/500/500), collapses think-time
            # spread to 0.5%, cuts ACPL sd from 3.62 to 1.21 -- and still ends the game within
            # 0.5 s of the historical final clock, so nothing is lost.
            started = time.monotonic()
            try:
                chosen = agent.get_move(board.fen(), int(clock))
            except Exception as error:  # a crash loses the game
                crashes += 1
                per_move.append({"game": path.name, "ply": board.ply(), "event": "crash",
                                 "detail": repr(error)})
                break
            spent = (time.monotonic() - started) * 1000.0
            clock = clock - spent + increment
            if clock <= 0:
                flags += 1
                per_move.append({"game": path.name, "ply": board.ply(), "event": "flag"})
                break
            if chosen not in {m.uci() for m in board.legal_moves}:
                # python-chess's push() does NOT validate, so an illegal move would be applied
                # and mis-scored in silence. That is one of the three ways we lose a whole game.
                illegal += 1
                per_move.append({"game": path.name, "ply": board.ply(), "event": "illegal",
                                 "detail": chosen})
                break
            best = engine.analyse(board, chess.engine.Limit(depth=arguments.depth), game=object())
            reference = best["pv"][0].uci() if best.get("pv") else None
            if reference is None:
                board.push(move)
                continue
            total += 1
            if chosen == reference:
                matched += 1
                game_loss.append(0.0)
            else:
                best_cp = score_after(engine, board, reference, arguments.depth)
                got_cp = score_after(engine, board, chosen, arguments.depth)
                if abs(best_cp) >= MATE_CP or abs(got_cp) >= MATE_CP:
                    mates += 1
                gap = max(-CLAMP, min(CLAMP, best_cp)) - max(-CLAMP, min(CLAMP, got_cp))
                if gap < 0:
                    # The reference's own "best" scored worse than our move at the depth the
                    # children were searched. Counting these as zero hides how much of the
                    # metric is reference noise, so track them.
                    negative += 1
                game_loss.append(float(max(0, gap)))
                per_move.append({"game": path.name, "ply": board.ply(), "played": chosen,
                                 "reference": reference, "loss": float(max(0, gap)),
                                 "raw_gap": gap, "eval_before": best_cp})
            board.push(move)
        if game_loss:
            losses.extend(game_loss)
            per_game.append((path.name, st.mean(game_loss), len(game_loss)))
            print(f"  {path.name[:52]:54s} ACPL {st.mean(game_loss):6.1f}  n={len(game_loss)}",
                  flush=True)
    engine.quit()

    if not losses:
        print("no moves scored")
        return
    print(f"\n{arguments.agent}: {len(games)} games, {total} of our moves")
    print(f"  ACPL                {st.mean(losses):7.1f}")
    print(f"  median loss         {st.median(losses):7.1f}")
    print(f"  matched reference   {matched / total:7.1%}")
    print(f"  moves losing >=100  {sum(1 for x in losses if x >= 100) / total:7.2%}")
    print(f"  moves losing >=200  {sum(1 for x in losses if x >= 200) / total:7.2%}")
    print(f"  mate events         {mates:7d}  (clamped to +/-{CLAMP}, counted not averaged)")
    print(f"  reference was worse {negative / total:7.2%}  (raw gap negative: reference noise)")
    print(f"  crashes {crashes}  flags {flags}  illegal {illegal}   <- each loses a whole game")
    if arguments.dump:
        with arguments.dump.open("w", encoding="utf-8") as handle:
            for row in per_move:
                handle.write(json.dumps(row) + chr(10))
        print(f"  per-move dump       {arguments.dump}")


if __name__ == "__main__":
    main()
