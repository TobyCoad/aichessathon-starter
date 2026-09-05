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
091 TT_KEEP (stopped at 108 games, -32 +/- 52, llr -1.21, leaning reject).

## Running now (5 Sep 13:45)
- desktop: gen-002 DONE (287,103 positions / 3000 games -> results/data/), and
  v8-clocktest PASS (0/6 errors, lowest clock 10.0 s, longest move 12.8 s) --
  the v8 bundle keeps time. Now on v8-120s; then 092-qscap14, 093-safe,
  095-asp15, 096-clocktest-v71, 097-seemain, 103-lazyacc (queued this iter).
- laptop CPU: 100-v8all gauntlet at 453 games, llr -0.45, +6 +/- 26 --
  inconclusive-leaning; the 600-game cap should close it soon. Worker queue
  after it: 101-lmraggr, 073-kz16w, 098-rootorder, 099-ttbuckets,
  102-timev5-clock, 102-timev5-120s.
- laptop GPU: net_w512-b1-kz16 training (if still up -- not re-verified this iter).
- download: fetch 2025_04 idling ("nothing new to analyse").

## Backlog (ranked; take the top item that is not running)
1. Fold every landed verdict: promote passes into the tree (switch -> True, or
   the net), reject the rest, note it here and in JOURNAL.md. NOTE: 100-v8all
   is a BUNDLE probe (7 switches at once) vs v7.1 -- it cannot promote a single
   switch by itself; if it PASSES, treat it as evidence for backlog 2 and gate
   a proper bundle from the single-switch passes only.
2. When >= 2 switches have passed: build the bundle challenger, run its gate
   (see rules), and if green write CANDIDATE.md + submission-candidate.zip.
3. Pack the fifth month; retrain the 16-zone net on five months (lr 1e-4).
   The two desktop self-play parquets (gen-001/002, ~570k positions) can join
   the training mix when a retrain happens.
4. Book rebuild with `--max-drop 10 --min-count 20`, judged on platform openings.
5. Anything from overnight/eval/V7_PLAN.md not listed as closed.

## Next step
Iteration 6: check results (laptop for 100-v8all, 453 games at llr -0.45 and
capped at 600, closing soon; desktop for v8-120s, then 092 / 093 / 095). Fold
any verdicts (100-v8all is a BUNDLE probe -- see the caveat in backlog 1).
Otherwise take backlog item 3 (pack the fifth month + retrain), but first
check whether the GPU training run (net_w512-b1-kz16) is still occupying the
GPU -- if it finished, record its checkpoint and suite numbers before starting
a new train.
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
