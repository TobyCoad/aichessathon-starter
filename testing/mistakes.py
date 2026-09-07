r"""Did we blunder here, and does the new net still blunder here?

Two stages, deliberately separate, because only the second one depends on the net:

  find    Walk our moves in the real platform games, score each one we actually played
          against a Stockfish reference, and write out the ones that gave material away.
          Net-independent, so it is computed ONCE and cached. The screen runs at
          --screen-depth and every hit is re-scored at --depth, so a shallow false
          positive costs one deep confirmation rather than a place in the set.

  retest  Hand each of those positions to a candidate at the clock we really had, and
          ask whether it repeats the mistake, fixes it, or finds something worse.

What this can and cannot tell you:
  * It answers "does this net stop making the mistakes we actually made", which is the
    question worth asking before shipping. It does NOT estimate rating: the set is
    conditioned on positions a DIFFERENT net misplayed, so a clean sweep here is
    evidence, not a number that can be added to an Elo.
  * Each position is replayed in isolation: fresh transposition table, fresh repetition
    history. In the game the engine arrived with a warm table and a search already
    pointed somewhere, so the replay can find moves the game did not.
  * This machine is ~1.8-2.1x faster than the platform, so a replay searches deeper than
    the real game did. That bias is shared by every candidate, so it cancels in an A/B --
    and it is the reason a fixed set of positions beats a fresh gauntlet here.

Run:
  .venv\Scripts\python.exe -m testing.mistakes find --out overnight/eval/mistakes.json
  .venv\Scripts\python.exe -m testing.mistakes retest --agent . --set overnight/eval/mistakes.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import chess
import chess.engine
import chess.pgn

from testing.replay_eval import (
    CLAMP,
    STOCKFISH,
    clock_ms,
    load_agent,
    our_colour,
    score_after,
)


def clamp(cp: int) -> int:
    return max(-CLAMP, min(CLAMP, cp))


# Every piece of search state that survives a get_move call and would otherwise carry
# into an unrelated position. Named explicitly rather than discovered, because the first
# version of this function looked for `holder.table` and called `.fill(0)` on it -- and
# BOTH engines' `table` is a plain dict, while the real compiled transposition table is
# `FastEngine.tt`, a tuple of two 4M-entry arrays. It silently cleared nothing at all:
# measured 277,262 live TT entries and 204,313 eval-cache entries surviving a "reset".
_ARRAY_STATE = ("conthist1", "counter", "corr", "killers2", "ec_key", "ec_val")
_TUPLE_STATE = ("tt",)
_DICT_STATE = ("history", "table")

# agent.py's MODULE-level game state, which no engine object owns and which the second
# version of this function also missed. These are not caches -- they steer the clock:
# `_DRAW_SCORES` feeds _convert_budget (two scoring positions in a row double the third
# one's budget) and _draw_budget_soft (six near-equal in a row caps it near 0.3 s), and
# `_MATCH_BASE_PLY` pins to the FIRST position's ply and understates every later one.
# Left alone, a position's verdict depends on the two positions before it, so reordering
# the set or passing --max-positions changes the answer.
_MODULE_STATE: dict[str, object] = {
    "_INC_SAMPLES": [],
    "_DRAW_SCORES": [],
    "_DRAW_LAST_PLY": -1,
    "_MATCH_BASE_PLY": -1,
    "_LAST_GAME_PLY": -1,
    "_LAST_CLOCK_MS": -1.0,
    "_LAST_SPENT_MS": -1.0,
    "_MAX_CLOCK_MS": 0.0,
    "_SEARCHED_MOVES": 0,
    "_PONDER_LAST_NODES": 0,
}


def reset_state(agent: ModuleType) -> None:
    """Between two unrelated positions, forget the last one.

    The engine keeps a repetition history, a transposition table, an eval cache and the
    history heuristics across calls, because within one game that is correct. Here
    consecutive positions come from different games: a carried-over repetition history
    can fabricate a threefold, and a warm table hands position N+1 a search that was
    aimed at position N.

    Returns nothing, but raises if it finds an engine it does not know how to clear --
    a reset that silently does nothing is worse than no reset, because the docstring
    then lies about the measurement.
    """
    for holder in (getattr(agent, "_ENGINE", None), getattr(agent, "_FAST", None)):
        if holder is None:
            continue
        cleared = 0
        for name in _DICT_STATE:
            value = getattr(holder, name, None)
            if isinstance(value, dict):
                value.clear()
                cleared += 1
        for name in _ARRAY_STATE:
            value = getattr(holder, name, None)
            if value is not None and hasattr(value, "fill"):
                value.fill(0)
                cleared += 1
        for name in _TUPLE_STATE:
            for part in getattr(holder, name, ()) or ():
                if hasattr(part, "fill"):
                    part.fill(0)
                    cleared += 1
        # `butterfly` is an ndarray on FastEngine but a list-of-lists on the python
        # Engine, so the array loop above skips the latter entirely.
        butterfly = getattr(holder, "butterfly", None)
        if butterfly is not None and hasattr(butterfly, "fill"):
            butterfly.fill(0)
            cleared += 1
        elif isinstance(butterfly, list):
            for row in butterfly:
                for i in range(len(row)):
                    row[i] = 0
            cleared += 1
        # Killers hold packed ints on FastEngine and `chess.Move | None` on Engine, so
        # the empty value differs. Writing 0 into the python engine's list is a type
        # violation in the one function whose job is to restore the constructed state.
        killers = getattr(holder, "killers", None)
        if isinstance(killers, list):
            empty: Any = 0 if type(holder).__name__ == "FastEngine" else None
            for row in killers:
                if isinstance(row, list):
                    for i in range(len(row)):
                        row[i] = empty
            cleared += 1
        if cleared == 0:
            raise RuntimeError(
                f"reset_state cleared nothing on {type(holder).__name__}: the engine's "
                "state attributes have been renamed, and every measurement after the "
                "first position would be contaminated by the one before it"
            )
    for name, empty in _MODULE_STATE.items():
        if not hasattr(agent, name):
            raise RuntimeError(
                f"reset_state expected agent.{name} and did not find it: the module's "
                "game state has been renamed and the clock would carry between positions"
            )
        setattr(agent, name, list(empty) if isinstance(empty, list) else empty)


def our_moves(path: Path) -> tuple[chess.Color, list[tuple[str, str, int | None]]]:
    """(fen, played_uci, clock_before_the_move_ms) for every move WE made in this game."""
    colour = our_colour(path)
    assert colour is not None
    with path.open() as handle:
        game = chess.pgn.read_game(handle)
    assert game is not None
    board = game.board()
    out: list[tuple[str, str, int | None]] = []
    # The clock comment on a node is what remained AFTER that node's move, so ours before
    # move N is the one attached to our previous move. None until we have made one.
    ours_last: int | None = None
    for node in game.mainline():
        if node.move is None:
            continue
        if board.turn == colour:
            out.append((board.fen(), node.move.uci(), ours_last))
            ours_last = clock_ms(node.comment)
        board.push(node.move)
    return colour, out


def loss_of(
    engine: chess.engine.SimpleEngine, board: chess.Board, played: str, depth: int
) -> tuple[int, str]:
    """How much the played move gave away, and what the reference wanted instead.

    Both sides are scored AFTER a move at the same depth, so the two numbers are
    commensurable; scoring `board` itself against a child would compare different
    horizons and read every quiet move as a small blunder.
    """
    best = engine.analyse(board, chess.engine.Limit(depth=depth), game=object())
    reference = best["pv"][0].uci()
    if reference == played:
        return 0, reference
    got = clamp(score_after(engine, board, played, depth))
    want = clamp(score_after(engine, board, reference, depth))
    return max(0, want - got), reference


def cmd_find(arguments: argparse.Namespace) -> None:
    games = [
        p
        for p in sorted(arguments.dir.glob(arguments.glob))
        if any(f"-{w}-" in p.name for w in arguments.results.split(","))
        and our_colour(p) is not None
    ]
    if arguments.max_games:
        games = games[: arguments.max_games]
    # The screen must be LOOSER than the verdict. Measured: screening at depth 14 with the
    # same 100 cp bar missed 16 of the 36 moves that a depth-20 confirmation calls >= 100 cp
    # -- a 44% false-negative rate, including one move worth 281 cp. Shallow and deep search
    # disagree in both directions, so a screen set at the verdict's own threshold silently
    # throws away the cases where depth changed its mind.
    screen_threshold = (
        arguments.screen_threshold
        if arguments.screen_threshold is not None
        else arguments.threshold // 2
    )
    print(
        f"{len(games)} games, screen depth {arguments.screen_depth} at {screen_threshold} cp, "
        f"confirm depth {arguments.depth} at {arguments.threshold} cp"
    )

    found: list[dict[str, Any]] = []
    scanned = 0
    started = time.monotonic()
    with chess.engine.SimpleEngine.popen_uci(str(STOCKFISH)) as sf:
        sf.configure({"Threads": 1, "Hash": 256})
        for index, path in enumerate(games, 1):
            colour, moves = our_moves(path)
            hits = 0
            for fen, played, clock in moves:
                board = chess.Board(fen)
                scanned += 1
                shallow, _ = loss_of(sf, board, played, arguments.screen_depth)
                if shallow < screen_threshold:
                    continue
                # Confirm at full depth: the shallow screen is cheap but wrong often
                # enough that an unconfirmed set would be padded with fine moves.
                deep, reference = loss_of(sf, board, played, arguments.depth)
                if deep < arguments.threshold:
                    continue
                hits += 1
                found.append(
                    {
                        "game": path.name,
                        "colour": "white" if colour == chess.WHITE else "black",
                        "fen": fen,
                        "played": played,
                        "reference": reference,
                        "loss_cp": deep,
                        "clock_ms": clock,
                    }
                )
            print(f"  [{index:>2}/{len(games)}] {path.name[:52]:52s} {hits:>2} mistake(s)")

    elapsed = time.monotonic() - started
    losses = [int(f["loss_cp"]) for f in found]
    payload = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dir": str(arguments.dir),
        "results": arguments.results,
        "screen_depth": arguments.screen_depth,
        "depth": arguments.depth,
        "threshold": arguments.threshold,
        "screen_threshold": screen_threshold,
        "moves_scanned": scanned,
        "mistakes": found,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1))
    share = 100.0 * len(found) / max(scanned, 1)
    print(
        f"\n{len(found)} mistakes of {scanned} moves ({share:.1f}%) in {elapsed / 60:.0f} min"
    )
    if losses:
        print(
            f"  loss cp: median {st.median(losses):.0f}  "
            f"mean {st.mean(losses):.0f}  max {max(losses)}"
        )
        for band in (100, 200, 300):
            print(f"  losing >= {band:>3} cp: {sum(1 for x in losses if x >= band)}")
    print(f"  wrote {arguments.out}")


def cmd_retest(arguments: argparse.Namespace) -> None:
    payload = json.loads(arguments.set.read_text())
    mistakes = payload["mistakes"]
    if arguments.max_positions:
        mistakes = mistakes[: arguments.max_positions]
    depth = int(payload["depth"])
    print(f"{len(mistakes)} mistakes from {arguments.set}, scored at the set's depth {depth}")

    agent = load_agent(arguments.agent)
    rows: list[dict[str, Any]] = []
    same = better = worse = fixed = crashed = illegal = 0
    started = time.monotonic()
    with chess.engine.SimpleEngine.popen_uci(str(STOCKFISH)) as sf:
        sf.configure({"Threads": 1, "Hash": 256})
        for index, item in enumerate(mistakes, 1):
            board = chess.Board(str(item["fen"]))
            # The clock we really had. Only a MISSING sample falls back to the base
            # clock -- `or` would also swallow a recorded 0 and hand the candidate a full
            # 120 s, flattering it on exactly the time-trouble moves most likely to be
            # blunders. Measured: 31 missing samples over 2,426 moves, one per game, always
            # the first move, where the base clock is the right answer.
            recorded = item.get("clock_ms")
            clock = arguments.base_ms if recorded is None else int(recorded)
            reset_state(agent)
            try:
                chosen = agent.get_move(board.fen(), int(clock))
            # A crash here is a lost game on the platform, so it is counted, not raised.
            except Exception as exc:
                crashed += 1
                print(f"  [{index:>3}] CRASH {type(exc).__name__}: {exc}")
                continue
            # from_uci raises on "" and on a malformed string, and "0000" parses to a
            # null move that is simply not in legal_moves. All three are one lost game on
            # the platform, so all three are counted here rather than ending the run.
            try:
                legal = chess.Move.from_uci(chosen) in board.legal_moves
            except ValueError:
                legal = False
            if not legal:
                illegal += 1
                print(f"  [{index:>3}] ILLEGAL {chosen!r}")
                continue
            old_loss = int(item["loss_cp"])
            if chosen == item["played"]:
                new_loss = old_loss  # same move, same position, same depth
                verdict = "SAME MOVE"
                same += 1
            else:
                new_loss, _ = loss_of(sf, board, chosen, depth)
                delta = new_loss - old_loss
                # FIXED needs the mistake to be GONE and the move to be materially
                # better. Testing `new_loss < fixed_below` alone let a 50 cp mistake
                # scored at 49 cp count as fixed -- a 1 cp gain, inside the reference's
                # own noise, headlined as a repaired blunder.
                if new_loss < arguments.fixed_below and delta < -arguments.margin:
                    fixed += 1
                    verdict = "FIXED"
                elif delta < -arguments.margin:
                    better += 1
                    verdict = "better"
                elif delta > arguments.margin:
                    worse += 1
                    verdict = "WORSE"
                else:
                    same += 1
                    verdict = "no change"
            rows.append({**item, "chosen": chosen, "new_loss_cp": new_loss, "verdict": verdict})
            print(
                f"  [{index:>3}] {str(item['game'])[:34]:34s} played {item['played']} "
                f"(-{old_loss:>4}) -> {chosen} (-{new_loss:>4})  {verdict}"
            )

    elapsed = time.monotonic() - started
    print(f"\n{arguments.agent}  {len(rows)} positions in {elapsed / 60:.0f} min")
    print(f"  FIXED (now under {arguments.fixed_below} cp)  {fixed}")
    print(f"  better by >{arguments.margin} cp                {better}")
    print(f"  same move or within margin            {same}")
    print(f"  WORSE by >{arguments.margin} cp                 {worse}")
    print(f"  crashes {crashed}  illegal {illegal}   <- each is a lost game on the platform")
    if rows:
        old = [int(r["loss_cp"]) for r in rows]
        new = [int(r["new_loss_cp"]) for r in rows]
        print(
            f"  total cp given away: was {sum(old):,}  now {sum(new):,}  ({sum(new) - sum(old):+,})"
        )
        print(f"  median loss:         was {st.median(old):.0f}      now {st.median(new):.0f}")
    if arguments.dump:
        arguments.dump.parent.mkdir(parents=True, exist_ok=True)
        arguments.dump.write_text("\n".join(json.dumps(r) for r in rows))
        print(f"  per-position dump    {arguments.dump}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    find = sub.add_parser("find", help="build the mistake set from the PGNs (net-independent)")
    find.add_argument("--dir", type=Path, default=Path("overnight/pgn/platform"))
    find.add_argument("--glob", default="round-*.pgn")
    find.add_argument("--results", default="loss,draw")
    find.add_argument("--screen-depth", type=int, default=14)
    find.add_argument("--depth", type=int, default=20)
    find.add_argument("--threshold", type=int, default=100, help="cp given away to count")
    find.add_argument("--screen-threshold", type=int, default=None,
                      help="cp bar for the shallow screen; default half of --threshold")
    find.add_argument("--max-games", type=int, default=0)
    find.add_argument("--out", type=Path, default=Path("overnight/eval/mistakes.json"))
    find.set_defaults(func=cmd_find)

    retest = sub.add_parser("retest", help="ask a candidate the same questions")
    retest.add_argument("--agent", type=Path, required=True)
    retest.add_argument("--set", type=Path, default=Path("overnight/eval/mistakes.json"))
    retest.add_argument("--base-ms", type=int, default=120_000)
    retest.add_argument("--margin", type=int, default=25, help="cp change we call no change")
    retest.add_argument("--fixed-below", type=int, default=50, help="cp under which it is gone")
    retest.add_argument("--max-positions", type=int, default=0)
    retest.add_argument("--dump", type=Path, default=None)
    retest.set_defaults(func=cmd_retest)

    arguments = parser.parse_args()
    arguments.func(arguments)


if __name__ == "__main__":
    main()
