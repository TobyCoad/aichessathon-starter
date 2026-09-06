# Handover: AI Chessathon engine, 6 Sep 2026 07:20

You are picking up from Claude Fable, who ran this project for Toby (bot `make_no_mistakes`)
through 5-6 Sep. Read this file, then `overnight/continuous/NOTES.md` (live state, the
authoritative baton), then `overnight/eval/V10_PLAN.md` (ranked ideas) and
`overnight/eval/ARCHITECTURE.md` (engine + evaluation framework + closed list). Everything
else is detail. Dated history of every verdict: `overnight/JOURNAL.md`.

## What the job is
Ship the strongest chess engine we can before uploads close **11 Sep 2026 11:00** (then a
13-round Swiss on the locked build). Toby's standing instruction: a recursive loop that
researches, builds, bundles, tests and ships **up to three versions a day**, and **emails him**
when a version is ready; he uploads by hand. Big changes may span iterations or run overnight.
Never upload anything yourself. Currently the bot sits mid-table; the goal is top 5.

## State right now
- **Live build**: whatever Toby last uploaded (v9.2 or v9.3 -- check his reply / the platform).
  Shipped by email so far: v8 (5 Sep 14:00), v8.5 (18:00), v9 (19:53), v9.1 (22:00),
  v9.2 (03:30), v9.3 (07:15). Each has a zip in `C:/Users/tobyc/Downloads/aichessathon-<v>.zip`.
- **Tree = champion = v9.3**: `agent.py` switches on = COMPILED_SEARCH, LMR, ASPIRATION(15), SEE,
  REPETITION_TWOFOLD, HISTORY2, ROOT_ORDER, TT_BUCKETS, QS_CAP 14, SAFE_BITS, SEE_MAIN,
  LMR_AGGRESSIVE(+PVS), LAZY_ACC, TIME_V5, PRUNE_V2, SINGULAR, QS_EVAL_CACHE, ADJUDICATION,
  HISTORY2_FIX, KILLER_CLEAR, TIME_V6, NMP_V2; `weights/net.npz` = the mixed Stockfish/Lichess
  net (md5 45f73c3f). Off in the tree (built, tested or pending): IMPROVING, CUTNODE (REJECT),
  CONT_HIST (REJECT), CAPTURE_ORDER (flat), QS_TT, ASP_WIDE, SEE_QUIET, DRAW_BUDGET (queued),
  INIT_FOLD (exact, -5 s import, ships with the next version after a clean-unzip import check).
- **Running**: the laptop worker is on `155-mixnet2s` (the v9.3 net with its output head
  scaled x1.31 to unit slope), then the queue in `overnight/laptop/tasks.json`. The loop
  (`overnight/continuous/loop.sh`) iterates every 25 min. The desktop is OFF.
- **Platform record**: 30 rated games as of 5 Sep 22:00; all losses/draws reached <= 16
  pieces (evaluation weakness) -- the reason for the Stockfish-data net.

## The machinery (all in the repo)
- `overnight/continuous/loop.sh` -- every 25 min: pull, fetch platform games
  (`testing.fetch_games`, post-mortems into `overnight/postmortem/`), run one Claude iteration
  (`claude --print --model opus ... PROMPT.md`), push. Start: `nohup bash overnight/continuous/loop.sh &`.
  Check: `tail overnight/continuous/loop.log`; per-iteration transcripts `iter-N.log`. It pauses
  1 h after three failures in a row (usage limits). ONE instance only -- check for
  `loop.sh` bash processes before starting another (two ran at once on 5 Sep).
- `overnight/continuous/PROMPT.md` -- the iteration's instructions (pipeline A-F, token
  discipline: research subagents are Opus only, two at a time, must read ARCHITECTURE.md).
- `overnight/continuous/NOTES.md` -- shared memory between iterations. Update it every
  iteration: Champion, Running now, Backlog, Next step.
- `overnight/worker.sh laptop` -- the gauntlet worker: takes the first task in
  `overnight/laptop/tasks.json` without a `overnight/laptop/results/<name>.txt`, builds
  `overnight/challengers/<name>/` from the tree + the task's `sed` (switch flips) or `net`,
  waits while other gauntlets / clocktests / the binpack decode / the endgame suite run, runs
  it, writes the result and commits. Task kinds: switch (default; SPRT at 8 s, judged every
  200 games: >= +10 Elo promotes, <= -10 from 400 rejects, else 200 more), clocktest,
  generate. Start: `nohup bash overnight/worker.sh laptop > overnight/laptop/worker.log &`.
  To stop a task: remove it from tasks.json FIRST, then kill its `testing.gauntlet` python,
  then reap orphaned pool workers (dead parent, command line `multiprocessing|harness`).
  Net tasks: put the net under `overnight/nets/` (gitignored); a net equal to the tree's aborts.
- `overnight/continuous/notify.py` -- email Toby (Gmail SMTP config borrowed from
  `C:/dev/quant-role-scout/config.json`): `--candidate` sends `CANDIDATE.md` + the platform
  record; `--text "..."` a note.
- Shipping a version (PROMPT.md step E): flip the passed switches in the tree (or copy the
  net), re-run the exactness check, build the zip FROM THE TESTED CHALLENGER DIR
  (agent.py, fastboard.py, fastsearch.py + weights/; unpacked < 50 MB), measure the cold
  import in a clean unzip (must be < 45 s here; the platform is ~1.55-1.8x slower with a 90 s
  budget -- v8.5 took 63 s there), copy to Downloads, write CANDIDATE.md, run notify.
- Gates before any engine commit: `ruff check`, `mypy agent.py fastsearch.py`,
  `python -m testing.check_fastsearch --depth 4 --random 30` (flags-off kernel bit-identical
  to the Python reference). Bench: `python -m testing.bench --agent <dir> --depth 8`.
  Endgame suite: `python -m testing.endgame_suite run --agent <dir> --seconds 2.5`
  (baseline under the v9.1 search: 10.8 cp; older numbers were under an older search).
  Clock replay: `python -m testing.clocktest --agent <dir> --workers 3` (must pass with a
  floor >= 5 s; only 120 s tests can judge time-management changes).

## The Stockfish-data line (Toby's overnight priority, continue it)
- `training/binpack_decode.py` decodes Stockfish NNUE training binpacks
  (https://huggingface.co/datasets/linrock/test80-2024, monthly, ~7-12 GB zst, ~6B
  positions each) into RECORD shards for `training/train.py`. Validated exact. Scale:
  **0.262 cp per internal unit** (0.45 was wrong: nets came out 1.7x too loud and lost).
  Shards on disk: `data/sf/feb24_00..08.npy` + `feb24_val.npy` (581M positions, rescaled).
- Results: pure SF retrain REJECT (-76; forgot human positions). Mixed 1:1 SF+Lichess
  warm start (`overnight/sf_train_mix.sh`, `net_w512-b8-kz16-mix2.pt`) PROMOTE +19 -> v9.3.
  Its Lichess-val is still 72% worse than the old net's while SF-val is 49% better; slope
  on Lichess targets 0.76 (155-mixnet2s tests a x1.31 output rescale).
- Next ideas, in order: (1) more SF months (Jan/Mar 2024) with the same recipe;
  (2) a lower SF share (1:2) or lr 3e-5 fine-tune if Lichess-val matters in games;
  (3) from-scratch training on SF + Lichess (~3 GPU h) once the recipe is settled;
  (4) NET_V10 architecture (mirrored king buckets, endgame-dense output buckets).
  Always check a new net's slope on Lichess val (`sum(pred*target)/sum(target^2)` ~ 1.0)
  and md5-compare the challenger's net with the tree's before believing a verdict.

## Lessons that cost time (do not repeat)
- A gauntlet under heavy load fails its crash gate on init timeouts (the worker now waits
  for the decode/suite; the gate replays once on <= 2 init timeouts).
- Killing a gauntlet orphans its pool workers; reap them.
- A net task whose path was inside its own challenger dir tested the champion against itself.
- An iteration killed mid-edit left kernel changes uncommitted while agent.py referenced
  them; commit kernel + agent together, check `git status` on engine files before shipping.
- The same-generator bug: never reuse a torch Generator across two nets when comparing
  predictions position-by-position.
- Pondering is dead (the platform suspends the process); IIR, LMP, PVS-alone, correction
  history, int8 inference, a shipped numba cache, wider nets, staged movegen: all closed.

## Daily rhythm
Ladder games hourly 08:00-22:00 UK (post-mortems land automatically); upload cap resets
12:00; Toby uploads from the emails. Nights are for GPU training and long gauntlets.
Freeze target: 10 Sep evening. Uploads close 11 Sep 11:00.
