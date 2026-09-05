You are one iteration of a continuous improvement loop for a chess engine (AI
Chessathon entry) in this repository. Work for one iteration, then stop.

Start by reading, in this order:
1. overnight/continuous/NOTES.md  (state, rules, backlog, next step -- authoritative)
2. `bash overnight/desktop_status.sh` output and `tail -n 30 overnight/eval/night3.log`
   (new verdicts from the desktop and the laptop)
3. The tail of overnight/JOURNAL.md (last 60 lines)

Then do exactly ONE unit of work, end to end:
- If new verdicts landed: record them in NOTES.md and JOURNAL.md; promote a PASS
  into the tree (flip the switch default to True, or copy the net) only when the
  rules in NOTES.md allow, and re-run the exactness check afterwards.
- Otherwise take the top backlog item that is not already running: implement it
  as a switch (off by default), lint + mypy + exactness check, measure it briefly
  (node counts or the endgame suite), and queue its gauntlet by appending a task
  to overnight/desktop/tasks.json or overnight/laptop/tasks.json (whichever has
  the shorter queue; the laptop runs one gauntlet at a time).
- Or, if two or more switches have passed and no bundle is pending, build and
  gate the bundle as the rules describe.

Hard rules (NOTES.md has the full list): never upload, never touch harness/,
never edit worker result files, switches off in the tree, exactness check must
pass before any commit that touches agent.py/fastsearch.py/fastboard.py, at most
~12 busy processes on this laptop, no gauntlet started here while one runs.

Finish by: committing your work with a clear message (git add specific files;
the loop pushes), updating NOTES.md ("Running now", "Verdicts", "Backlog", and a
concrete "Next step" for the next iteration), and appending one dated paragraph
to overnight/JOURNAL.md. If nothing can progress (all queues full, nothing
landed), say so in NOTES.md's "Next step" and stop.
