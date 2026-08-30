# Overnight loop

One engine experiment every five hours, unattended, gated by an SPRT.

## How it works

A Windows scheduled task runs `run.ps1`, which pipes `PROMPT.md` into
`claude --print`. Each run:

1. reads `JOURNAL.md` for what previous runs learned,
2. takes the topmost `status: todo` item from `BACKLOG.md`,
3. builds a challenger in `challengers/`, never editing `agent.py` directly,
4. runs `testing.gauntlet`, which crash-gates then SPRTs it against the champion,
5. promotes only on exit code 0, records the outcome, commits, and stops.

**Runs are stateless and state is durable.** Nothing is carried in memory between
runs. If a run dies partway through -- a usage limit, a crash, a reboot -- the next
one reads the same files and continues. There is nothing to resume, which is what
makes the five-hourly cadence robust rather than fragile.

## Why the SPRT gate exists

A change worth +20 Elo needs roughly 1,300 games to resolve, and a 20-game score
carries a +/-135 Elo confidence interval. An unattended loop without a stopping rule
will promote regressions on lucky short matches and report success. Verified
against known-bad challengers:

| Challenger | Verdict | Exit | Measured |
|---|---|---|---|
| quiescence removed | REJECT | 1 | -330 Elo |
| identical to champion | INCONCLUSIVE | 2 | champion stays |

## Operating it

```powershell
.\overnight\install-task.ps1                                  # register, first run in 10 min
Start-ScheduledTask     -TaskName "AIChessathon-Overnight"    # run one now
Disable-ScheduledTask   -TaskName "AIChessathon-Overnight"    # pause
Unregister-ScheduledTask -TaskName "AIChessathon-Overnight"   # remove
Get-ScheduledTaskInfo   -TaskName "AIChessathon-Overnight"    # last result, next run
```

Each morning: read `JOURNAL.md` for what happened and `git log` for what changed.
Every promotion is one commit with the game count and Elo estimate in the message,
so anything that looks wrong reverts cleanly.

## What it will not do

- **Never pushes.** The fork is public; the engine stays local.
- **Never edits `harness/`**, which mirrors the platform's clock and protocol.
- **Never commits a red gate** -- `ruff` and `mypy` must pass first.
- **Never promotes on anything but an SPRT pass.**

## Limits worth knowing

The machine must be awake; the task sets `WakeToRun`, which wakes a sleeping
machine but cannot help a shut-down one. `run.ps1` aborts if the working tree is
dirty, on the assumption that a previous run died mid-edit and a human should look.
And the loop cannot answer the P0 question in `BACKLOG.md` -- whether `rust-chess`
and `numba` are permitted -- which is worth more than everything it can do.
