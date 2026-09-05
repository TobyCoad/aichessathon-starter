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
- v8 (5 Sep 14:55) = v7.1 + HISTORY2 + ROOT_ORDER + TT_BUCKETS + QS_CAP 14 + SAFE_BITS
  + ASPIRATION_WINDOW 15 + SEE_MAIN, all now True in the tree. Evidence: 40 games at
  120 s on platform openings 67.5% (+18 =18 -4); 8 s SPRT flat (+3.5 +/- 25 at 502,
  rerun +30 +/- 38 at 210, stopped by hand so 101-lmraggr could start vs v8).
  Uploaded by the human for the 15:00 slot. Every challenger is judged vs v8 now.
  Single-switch tasks for the bundle parts (093/095/097/098/099) removed as moot;
  092-qscap14 on the desktop began vs v7.1 and its verdict is VOID (champion changed
  under it) -- record it as void, do not fold it. Next target: v9 = v8 + LMR_AGGRESSIVE
  (101-lmraggr, laptop, running next).
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
clearly worse, closed on val loss without a challenger),
105-bookprune (closed on coverage without a gauntlet: book-mc20-md10.bin built
fine at 15:06 -- 31,200 entries, 0.50 MB -- but scanning only 60/599 row groups
with min-count 20 collapsed pool coverage to 7/80 with 0.26 mean in-book plies
vs the champion book's 28/80 and 1.38; the challenger would play book-less in
~91% of platform games, so SPRT[0,20] over 600 games cannot resolve it. Task
removed from the desktop queue; the book stays uncommitted in overnight/books/.
092-qscap14 VOID (started vs v7.1; champion changed under it)).
Bundle evidence: v8-120s (the 7-switch probe at 120 s, platform openings) scored
67.5% (+18 =18 -4) over 40 games -- labelled INCONCLUSIVE only because 40 games
cannot close an SPRT. Together with v8-clocktest PASS this says the v8 switches
as a group are strongly positive at long TC; single-switch verdicts still decide
what enters a bundle.

## v8.5 plan (5 Sep 16:10, human's call)
- v8.5 bundle = v8 + LMR_AGGRESSIVE + LAZY_ACC + TIME_V5 + PRUNE_V2 + SINGULAR, tested as
  ONE challenger vs v8 like v8 was: 110-v85all (laptop, 8 s SPRT), v85-clocktest + v85-120s
  (desktop, 40 games at 120 s platform openings), plus 111-singular alone on the desktop
  for attribution. 101-lmraggr stopped at 42 games (+25 +/- 90, nothing learned) to make
  room. Bench to depth 8 vs v8: PRUNE_V2 0.93x nodes, SINGULAR 1.55x (extensions cost
  nodes at fixed depth; judged at fixed time), both 1.50x. Do NOT queue single-switch
  tasks for the bundle parts; fold the bundle verdicts when they land.

## Running now (5 Sep 15:15)
- desktop: finishing 092-qscap14 (VOID -- record and discard when its result
  file lands, do not fold); then v85-clocktest, v85-120s, 111-singular.
  105-bookprune REMOVED from the queue (closed on coverage, see Verdicts) --
  if the desktop already logged "book ... missing" retries, the pull clears it.
- laptop gauntlet: 110-v85all (v8.5 bundle probe: LMR_AGGRESSIVE + LAZY_ACC +
  TIME_V5 + PRUNE_V2 + SINGULAR vs v8, SPRT[0,20] at 8 s), in its crash-gate
  stage at 15:08. Queue after it: 073-kz16w, 102-timev5-clock, 102-timev5-120s
  (102 tasks are moot if the v8.5 bundle passes as one; drop them then).
- laptop CPU: month5 pack running (4 workers, group 270/509 at 15:09, log
  overnight/eval/pack-2024_11.log) -> train kz16r (GPU) -> export + check_nnue
  + suite into challengers/104-kz16r. The chain does NOT queue the gauntlet;
  the iteration that reads suite-104-kz16r.log does (backlog 3).
- laptop GPU: idle until the month5 train step.
- Process check 15:08: one worker chain, one month5 chain, pack 4 workers,
  gauntlet runners -- at the ~12-busy budget; do not start more CPU work.

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
4. CLOSED (was: book rebuild + 105-bookprune). The mc20-md10 build collapsed
   coverage (7/80, see Verdicts). Optional future item, LOW priority given
   every book experiment so far was flat-or-worse: rebuild scanning ALL 599
   row groups of 2025_01 (~4.5 h on 2 workers; only when the CPU budget
   allows) so min-count 20 keeps coverage, then re-run the coverage compare
   before queueing any gauntlet. Champion baseline for that compare: 9.76 MB,
   610,028 entries, coverage 28/80, 2.6 moves per covered position, mean 1.38
   in-book plies from a pool start.
5. Anything from overnight/eval/V7_PLAN.md not listed as closed.

## Next step
Iteration 10: the usual sweep. (1) Laptop: check
overnight/laptop/results/110-v85all.gauntlet.log -- if the v8.5 bundle verdict
landed, record it; a PASS makes v8.5 the bundle candidate pending the
desktop's v85-clocktest + v85-120s (fold all three together per the v8.5 plan
above; then drop the moot 102-timev5 tasks from the laptop queue and
111-singular stays for attribution only). If 110-v85all dies near 450 games
with no verdict again, read the log tail BEFORE the worker truncates it.
(2) Desktop: when 092-qscap14.txt lands, log it as VOID (champion changed
under it) and nothing else. (3) month5: tail
overnight/eval/pack-2024_11.log (group 270/509 at 15:09, ~roughly an hour to
go) and grep "month5" overnight/eval/night3.log; on PACK/TRAIN FAILED, fix and
relaunch `nohup bash overnight/month5.sh &` (idempotent). If
suite-104-kz16r.log exists, record its numbers and queue 104-kz16r as a net
task on the shorter queue per backlog 3. If nothing has landed anywhere and
the CPU budget is full (it was at 15:08), say so and stop -- do NOT start new
CPU work while the pack and the gauntlet share the laptop.
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
