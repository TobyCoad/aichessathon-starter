# Continuous engine loop -- shared notes (the baton)

Read this first every iteration. Keep it short: state, verdicts, backlog, next step.
Update it at the end of every iteration. Facts here override anything older in
JOURNAL.md; add the iteration's line to JOURNAL.md too.

## Fixed rules (never relax)
- Uploads are the human's. The loop never uploads. A passed bundle is left as
  `submission-candidate.zip` + `overnight/continuous/CANDIDATE.md` for the human.
- Never edit `harness/`. Never run `git push --force`. Never edit files under
  `overnight/desktop/results/` or `overnight/laptop/results/` (workers own them).
- Every engine change is a switch in `agent.py`, OFF by default in the tree. The
  tree must always be the current champion: it is what the workers copy.
- Before committing an engine change: `ruff check`, `mypy agent.py fastsearch.py`,
  and `python -m testing.check_fastsearch --depth 4 --random 30` must PASS
  (flags-off kernel identical to the Python reference).
- One switch per challenger. Judged by SPRT[0,20] at 8 s vs the champion; nets and
  time changes also on `testing/endgame_suite.py` and 40 games at 120 s.
- A bundle (champion + every passed switch) repeats the full gate before it is a
  candidate: SPRT vs champion, 200-game crash hunt (`--elo0 900 --elo1 950`),
  `testing.clocktest`, 40 games at 120 s. Unpacked zip must stay under 50 MB.
- Keep at most ~12 busy processes on the laptop. Do not start a gauntlet on the
  laptop while another runs (the worker enforces this).
- The platform SUSPENDS our process between moves: pondering is dead. Init on
  their box is ~1.8x ours; stay under ~45 s of local import.

## Machines
- Laptop (this machine): editing, GPU training, `overnight/laptop/tasks.json` ->
  worker `bash overnight/worker.sh laptop` -> `overnight/laptop/results/`.
- Desktop (E:/dev/aichessathon-starter, 16 cores): `overnight/desktop/tasks.json`
  -> `bash overnight/desktop_worker.sh` -> `overnight/desktop/results/`.
  Both pull/push `main`. Results are one `<task>.txt` per task; a task without a
  result file is pending. Heartbeat: `overnight/desktop/heartbeat.txt`.
- Task fields: name, sed (switch flip), kind (switch|clocktest), champion (default
  `.`), games, openings (default|platform), workers, elo0, elo1, base_ms.

## Champion
- v7.1 = compiled search + LMR + ASPIRATION + SEE + REPETITION_TWOFOLD + 16-zone net
  (weights/net.npz, float16 W1). Uploaded 5 Sep 12:00 slot. Ladder ~14th.
- Measured stack (8 s self-play): compiled +67, LMR +47, aspiration +41, SEE +25,
  16-zone net +31. 120 s gains are about half of 8 s gains.

## Verdicts so far (vs the champion of the time)
PROMOTE: 050 compiled, 052 LMR, 054 aspiration, 055 SEE, 057 twofold, 072 kz16.
REJECT/closed: PVS (x2), LMP, RFP_PHASE (suite), correction history (x2), TT_EVAL,
book-off (inconclusive), book-verify (-94), IIR (inconclusive), NMP_GUARD (flat),
pondering (platform freezes the process), endgame fine-tune of the old net,
QS_EVAL_CACHE (exact, +2% only), CHECK_EXT_CAP (no effect),
091 TT_KEEP (stopped at 108 games, -32 +/- 52, llr -1.21, leaning reject),
b1-kz16 net (1 output bucket: val 0.004977 vs the champion b8-kz16's 0.004659,
clearly worse, closed on val loss without a challenger).
Bundle evidence: v8-120s (the 7-switch probe at 120 s, platform openings) scored
67.5% (+18 =18 -4) over 40 games -- labelled INCONCLUSIVE only because 40 games
cannot close an SPRT. Together with v8-clocktest PASS this says the v8 switches
as a group are strongly positive at long TC; single-switch verdicts still decide
what enters a bundle.

## Running now (5 Sep 14:50)
- desktop: on 092-qscap14 (62 games, llr -0.68, leaning negative at 14:32);
  then 093-safe, 095-asp15, 096-clocktest-v71, 097-seemain, 103-lazyacc,
  105-bookprune (NEW: champion + book-mc20-md10.bin, platform openings,
  SPRT[0,20] at 8 s). worker.sh gained a `book` task field (mirrors `net`,
  swaps weights/book.bin in the challenger dir); if the book file is not yet
  in the tree the worker logs "book ... missing" and retries in 120 s -- it
  does NOT run with the champion book by mistake.
- laptop CPU: book rebuild running detached (backlog 4):
  `training.build_book --min-count 20 --max-drop 10 --workers 2` over 60 row
  groups of 2025_01 -> overnight/books/book-mc20-md10.bin, log
  overnight/eval/book2-build.log. Started 14:39; 2 workers is slow (~30+ min
  expected). The book file must be COMMITTED before the desktop reaches
  105-bookprune (hours away). Prior art: --max-drop 30 tested exactly 50.0%
  over 400 games on DEFAULT openings (book barely fires there); this one is
  stricter and judged on platform openings per the backlog.
- laptop gauntlet: 100-v8all RESTARTED -- the 11:43 gauntlet died silently at ~453
  games (llr -0.45); the worker found no verdict line, discarded the result and
  re-ran the task at 13:49, now ~50 games. That evidence is lost; the rerun
  starts from zero. Only ONE worker chain is alive (PID check: the second
  worker.sh in the process list is a subshell of the first). Known race in
  run_task: the challenger dir is rebuilt BEFORE the busy-CPU wait, so
  overlapping worker restarts clobber files (the 11:43 rm/cp errors). Queue
  after 100-v8all: 101-lmraggr, 073-kz16w, 098-rootorder, 099-ttbuckets,
  102-timev5-clock, 102-timev5-120s.
- laptop background: overnight/month5.sh (detached, idempotent, logs to
  night3.log with "month5" lines): fetch 2024_11 (6.49 GB, running) -> pack on
  4 workers -> train kz16r (GPU, 5 shards, lr 1e-4, resumes from b8-kz16) ->
  export + check_nnue + suite into challengers/104-kz16r. It does NOT queue the
  gauntlet; a later iteration does that after reading the suite. NOTE: the
  fishnet-evals dataset ENDS at 2025_03 (2025_04/05 are 404), so the fifth
  month is 2024_11.
- laptop GPU: idle until the month5 train step (b1-kz16 finished 11:44,
  verdict recorded above).

## Backlog (ranked; take the top item that is not running)
1. Fold every landed verdict: promote passes into the tree (switch -> True, or
   the net), reject the rest, note it here and in JOURNAL.md. NOTE: 100-v8all
   is a BUNDLE probe (7 switches at once) vs v7.1 -- it cannot promote a single
   switch by itself; if it PASSES, treat it as evidence for backlog 2 and gate
   a proper bundle from the single-switch passes only.
2. When >= 2 switches have passed: build the bundle challenger, run its gate
   (see rules), and if green write CANDIDATE.md + submission-candidate.zip.
3. RUNNING as overnight/month5.sh (see above). When suite-104-kz16r.log lands,
   queue 104-kz16r as a net task ({"name": "104-kz16r", "net":
   "overnight/challengers/104-kz16r/weights/net.npz", "sed": ""}) on the
   shorter queue. The self-play parquets (gen-001/002, ~570k positions) were
   deliberately left OUT of this retrain (a sixth rotating shard would get
   ~250x per-position weight); a self-play mix is a separate future experiment.
4. RUNNING as the detached book build + queued 105-bookprune (see above).
   When book-mc20-md10.bin exists: run the coverage compare
   (`python overnight/book_coverage.py weights/book.bin overnight/books/book-mc20-md10.bin`),
   record the numbers, and COMMIT the book binary so the desktop can build the
   challenger. Champion baseline: 9.76 MB, 610,028 entries, pool coverage
   28/80, 2.6 moves per covered position, mean 1.25 in-book plies from a pool
   start.
5. Anything from overnight/eval/V7_PLAN.md not listed as closed.

## Next step
Iteration 9: FIRST check overnight/eval/book2-build.log -- if it wrote
overnight/books/book-mc20-md10.bin, measure coverage vs weights/book.bin on
testing/platform_openings.txt (79 FENs: coverage, moves per position, mean
in-book plies), record the numbers here and in JOURNAL.md, and `git add`
overnight/books/book-mc20-md10.bin (the desktop's 105-bookprune waits on that
commit; if the build FAILED, fix or remove 105-bookprune from
overnight/desktop/tasks.json). Then the usual sweep: desktop results for
092-qscap14 / 093 / 095; laptop 100-v8all rerun (if it dies again near 450
games with no verdict, read the tail of
overnight/laptop/results/100-v8all.gauntlet.log BEFORE the worker truncates
it, and consider capping games at 400). Check the month5 chain: grep "month5"
overnight/eval/night3.log (pack running since 14:15; on PACK/TRAIN FAILED, fix
and relaunch `nohup bash overnight/month5.sh &`, idempotent). If
suite-104-kz16r.log exists, record its numbers and queue the 104-kz16r net
task per backlog 3.
Verdict context for the queued challengers: 098-rootorder +3.5% nodes (REJECT
likely); 099-ttbuckets node-neutral at depth 8 (only long searches can show a
gain); 101-lmraggr 0.92x nodes at depth 8 (modest); 102-timev5 (floor 18 +
stable refund) is 120 s only -- judged by its clocktest + the 40-game 120 s
match, NOT an 8 s SPRT (below LOW_CLOCK the floor never binds, 8 s play is
byte-identical), and the fixed-movetime endgame suite cannot see a budget
change so it is waived for this one. 103-lazyacc is EXACT (same nodes, same
scores, verified lazy==eager on 40 random positions and identical bench node
counts): +2% knps at depth 8, +5% at depth 10, and the gain should grow at
long TC where hash cutoffs are denser -- a small-positive SPRT or even
inconclusive-positive is fine to promote per the exact-change precedent ONLY
if it does not slow anything (QS_EVAL_CACHE at +2% was closed as not worth it;
lazyacc differs in that it also helps every non-evaluating node, judge on the
SPRT).
