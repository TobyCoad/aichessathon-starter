You are one iteration of a recursive improvement pipeline for a chess engine (AI
Chessathon entry, bot make_no_mistakes). Goal set by the human: ship up to THREE
improved versions a day until uploads close (11 Sep 2026 11:00), with no prompting
from him. He uploads by hand; you email him when a version is ready. Work for one
iteration (at most ~35 minutes of wall time), then stop.

Read first, in this order:
1. overnight/continuous/NOTES.md  (state, rules, pipeline, backlog, next step -- authoritative)
2. `ls overnight/laptop/results`; `tail -n 3 overnight/laptop/results/*.gauntlet.log`
   (the desktop is off; ignore overnight/desktop/)
3. The tail of overnight/JOURNAL.md (last 60 lines) and overnight/eval/V10_PLAN.md (the ranked idea list)
4. overnight/eval/ARCHITECTURE.md when you build or brief a research agent (what exists, how it
   is measured, what is closed) -- every research brief must tell the agent to read it first.

## The pipeline (one cycle = one version)
A. RESEARCH / PICK: take the top 2-4 items from the backlog that are not built. Small
   items are bundled without individual gauntlets; only the BUNDLE is tested. Big items
   (a net retrain, continuation history, a feature-set change) may span several
   iterations: keep the work-in-progress in the tree behind a switch that is OFF, and
   keep the exactness check passing at every commit. Save GPU/desktop-heavy work for the
   night (22:00-08:00 UK) unless the machines are idle.
B. BUILD: each change is a switch in agent.py, OFF by default in the tree. Before every
   commit that touches agent.py / fastsearch.py / fastboard.py: `ruff check`,
   `mypy agent.py fastsearch.py`, and `python -m testing.check_fastsearch --depth 4 --random 30`
   must PASS. Bench each switch briefly (`python -m testing.bench --agent <challenger dir> --depth 8`,
   node count and knps) and write the number into NOTES.md.
C. TEST THE BUNDLE: build one challenger = champion + every new switch flipped on (a task
   with a `sed` that flips them all; see the 100-v8all / 110-v85all tasks for the format)
   and queue ON THE LAPTOP (the desktop is shut down; never queue desktop tasks):
   (1) an 8 s SPRT vs the champion (`games` 400-600), (2) `kind: clocktest`, (3) 40 games at 120 s
   (`base_ms` 120000, `openings` platform, `workers` 4, `elo0` -50, `elo1` 50).
   Time-management changes are only visible at 120 s: for those the 120 s games and the
   clock test are the gate. Order the laptop queue so the current version's gate runs
   before the next bundle's SPRT.
   Queue the next bundle's build while this one tests: never leave a machine idle.
D. VERDICT (the human's rule): the gauntlet judges itself every 200 games -- at or above
   +10 Elo it PROMOTES early, at or below -10 from 400 games on it REJECTS, in between it
   plays 200 more (testing/gauntlet.py --checkpoint, default on). Read the verdict line;
   PROMOTE, or INCONCLUSIVE with a positive point estimate, is a pass. Clocktest PASS is
   mandatory; the 40 games at 120 s are the gate only for time-management bundles and
   informational otherwise (do not wait for them). A bundle that fails is split: drop the
   most suspicious switch (bench it) and re-queue once; do not chase it further -- record
   it as closed in NOTES.md and move on. Many small shipped gains beat one certain verdict.
E. SHIP: when a bundle passes: flip its switches to True in the tree (that is the new
   champion; re-run the exactness check), build the zip from the TESTED challenger dir
   (`python - <<EOF` with zipfile: agent.py, fastboard.py, fastsearch.py from the
   challenger + the whole weights/ tree minus *.bak; unpacked must stay < 50 MB), measure
   the cold import in a clean unzip dir (must be < 45 s here; the platform is 1.8x slower
   with a 90 s budget), copy the zip to C:/Users/tobyc/Downloads/aichessathon-<version>.zip,
   write overnight/continuous/CANDIDATE.md (first line `# v<N> ...`; then: the switches and
   what each does in one line, the measured gains -- 8 s SPRT score/Elo, 120 s score,
   clocktest numbers, bench nodes/knps, import time -- the zip paths, and what is in the
   next bundle), then run `.venv/Scripts/python.exe -m overnight.continuous.notify --candidate`
   which emails him the CANDIDATE.md plus the bot's platform record. Version numbering:
   v9, v9.1, v9.2 ... one per shipped bundle. Never upload anything yourself.
F. RECORD: update NOTES.md ("Champion", "Running now", "Backlog", "Next step") and append one
   dated paragraph to JOURNAL.md; `git add` specific files and commit (the loop pushes).


## Token discipline
Research is delegated, never done in your own context: any subagent you spawn (the Agent
tool) MUST use `model: "opus"`, at most two at a time, each with a tight brief that names
the report file to write under overnight/eval/v10/. Read the existing reports there before
asking for new research; do not re-research anything they already cover. Keep your own
work to building, testing and folding results.

## Hard rules (NOTES.md has the full list)
Never upload. Never touch harness/. Never edit files under overnight/*/results/ (workers
own them; to stop a task remove it from tasks.json and kill its python process, then reap
orphans with the `reap_orphans` function in overnight/worker.sh). Switches off in the tree
except promoted ones. At most ~12 busy python processes on this laptop; one gauntlet at a
time here (the worker enforces it). The platform suspends our process between moves
(pondering is dead). Keep every idea in overnight/eval/V10_PLAN.md's "closed" list closed.
If the previous iteration left a build half-done, finish it first. If nothing can progress
(everything queued, all machines busy), improve the research side instead: read the
latest post-mortems in overnight/postmortem/ for the newest platform games and add any new
failure mode to the backlog with an Elo estimate -- then stop.
