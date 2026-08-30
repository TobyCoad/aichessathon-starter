# Overnight run

You are continuing work on an AI Chessathon submission, unattended. This file is
your entire briefing: you have no memory of previous runs. Everything you need to
know is in this repository.

Working directory: `C:\dev\aichessathon\starter`
Python: `.\.venv\Scripts\python.exe` (always use this, never bare `python`)

## Do exactly one experiment, then stop

1. Read `overnight/JOURNAL.md` to see what previous runs did and learned.
2. Read `overnight/BACKLOG.md`. Take the **topmost item marked `status: todo`**.
   Skip anything under "P0 — blocked on a human". If every item is done, run the
   P4 crash hunt instead.
3. Do that one item. Not two. A finished experiment with a recorded result is worth
   more than three half-finished ones, because the next run starts from nothing.
4. Append what happened to `overnight/JOURNAL.md` — including failures, which are
   the more useful half. Update the item's `status` in `BACKLOG.md` to `done`,
   `rejected` or leave `todo` with a note if you ran out of time.
5. Commit. Then stop.

## Build steps versus experiments

The backlog holds two kinds of item and they are handled differently.

**Build steps** (everything in P2 except the last one) produce a file and a passing
test. They do not go near the gauntlet, because there is nothing to play yet. Done
means: the file exists, its correctness test passes, `ruff` and `mypy` are green,
and it is committed. Several of these are sequential — P2.2 needs P2.1 — so if the
one above yours is not finished, finish that instead.

**Experiments** (P1, P3, and P2.6) change how the engine plays, and go through the
gauntlet below.

If a build step is too big for one run, that is expected: do part of it, commit
working code with its test passing, and write in the journal exactly where you got
to and what the next run should do first. Never commit a half-written file whose
test does not run.

## How to test a change

Never edit `agent.py` directly to try something out. `agent.py` is the champion and
must remain shippable at all times.

```
mkdir overnight\challengers\NNN-short-name
copy agent.py overnight\challengers\NNN-short-name\agent.py
# edit the copy, then:
.\.venv\Scripts\python.exe -m testing.gauntlet --challenger overnight\challengers\NNN-short-name
```

The gauntlet runs a crash gate and then an SPRT against the champion. Obey its exit
code and nothing else:

- **0 PROMOTE** — copy the challenger over `agent.py`, then commit both.
- **1 REJECT** — leave `agent.py` alone. Record why in the journal.
- **2 INCONCLUSIVE** — leave `agent.py` alone. Record it; a re-run with more games
  is a legitimate next experiment if the Elo estimate looked promising.

A 40-game score is not evidence. A change worth +20 Elo needs roughly 1,300 games to
resolve. Do not promote anything on a hunch, a short match, or a plausible argument.
The gauntlet's verdict is the only thing that promotes.

## Hard rules

- **Never `git push`.** The fork is public and the engine must stay private.
- **Never edit `harness/`.** It mirrors the platform's protocol and clock; changing
  it makes every local result meaningless. `testing/` is the editable copy.
- **Never commit a red gate.** `.\.venv\Scripts\python.exe -m ruff check .` and
  `.\.venv\Scripts\python.exe -m mypy` must both pass before you commit.
- **Never leave `agent.py` broken.** If you run out of time mid-edit, revert it.
  Losing an experiment is fine; shipping a broken champion is not.
- **No network calls from `agent.py`**, no engine dependencies (Stockfish, Lc0, Maia
  in any form), no `numba` or `rust-chess` until the organisers have answered.
- Large data files go in `data/` and are gitignored. Never commit training data.

## Rules constraints that bound every change

- `agent.py` at the zip root exposing `get_move(fen: str, time_left_ms: int) -> str`.
- 1 CPU core, 2 GB RAM, no network, no GPU, 200 MB expanded, 60 s import budget.
- 120 s + 0.5 s per move, wall time. Illegal move, crash, or flag loses the game.
- A learned model must materially drive move selection. The current hand-crafted
  evaluation is a fallback, not a final submission — P2 in the backlog is mandatory.

## Time budget

You have about four hours. Arena runs are the slow part and cost almost no tokens,
so prefer starting a long, statistically meaningful match over reasoning at length
about a short one. If you are running low, stop cleanly: record the state, commit,
and leave the next run a good starting point. Do not start an experiment you cannot
finish and record.
