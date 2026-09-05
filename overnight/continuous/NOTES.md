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
- Bundles, not single switches (since 5 Sep). Judged at 8 s vs the champion by the
  gauntlet's every-200-games rule: >= +10 Elo promotes early, <= -10 from 400 games
  rejects, in between 200 more (testing/gauntlet.py --checkpoint). Clocktest PASS is
  mandatory; 40 games at 120 s gate time-management bundles only. Nets also need the
  endgame suite.
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

## Pipeline (5 Sep 18:30, the human's standing instruction)
- Ship up to THREE versions a day, each a BUNDLE of several switches tested together
  (like v8 and v8.5): 8 s SPRT on the laptop + clocktest and 40 games at 120 s on the
  desktop. Small changes are never gauntleted alone. Big changes span iterations behind an
  OFF switch, or run overnight. See overnight/continuous/PROMPT.md for the cycle.
- When a bundle passes: promote in the tree, build the zip from the tested challenger,
  write overnight/continuous/CANDIDATE.md, run
  `.venv/Scripts/python.exe -m overnight.continuous.notify --candidate` (emails him the
  change list, the measured gain and the platform record). He uploads by hand. Versions:
  v9, v9.1, v9.2, ...
- Speed report landed (overnight/eval/v10/speed.md): (a) QS_EVAL_CACHE=True is EXACT and
  +4.2% knps under v8.5 (identical node count) -> put it in the v9 bundle, no gauntlet of
  its own; (b) init is 31.7 s idle (fastsearch.search alone 23 s): constant-folding the
  settled switches cuts fs.warm_up 18% and eager signatures in fastboard remove 61
  redundant specialisations (7 -> ~4 s) -> INIT_FOLD, a v9.1 item, source-only; a shipped
  numba cache / AOT is REJECTED (segfaults on rebuild, and AGENTS.md forbids native
  binaries); (c) int8/int16 inference is 1.9x SLOWER than float32 (closed for good);
  (d) the only big node-rate lever left is a 32->16 output head (+15% knps, needs a
  retrain) -> fold into NET_V10.
- Bundle in flight for v9 -- gate SPLIT 18:28 (the parallel TIME_V6 session, ec3b65e):
  132-v9core on the laptop (600 games at 8 s vs v8.5) tests the FOUR core switches:
  QS_EVAL_CACHE (exact, +4.2% knps) + ADJUDICATION + HISTORY2_FIX + KILLER_CLEAR
  (593053f; bench vs champion 1.00x / 1.02x / 0.97x nodes, adjudication smoke test
  passed, exact 70/70). Its predecessor 131-v9all was stopped at 98 games (+28 +/- 57,
  llr +0.46) for the re-queue. TIME_V6 was REVIVED 18:29 (71b7b50, after two earlier
  cuts failed the clocktest): final constants = low clock rem/18 exact stop, horizon
  56-0.4m floor 30, hard min(10%, 2.5x soft), factor cap 1.5; local clock replay PASS
  (lowest 5.8 s, longest 11.9 s at 1.5x charge). TIME_V6 is judged ONLY by the desktop
  v9-clocktest + v9-120s (five-switch sed, queued after v85-120s-b); the four-switch
  build has its own v9core-clocktest + v9core-120s (renamed 18:45 from duplicate names
  that would never have run). Ship paths: 132-v9core PROMOTE + v9core-clocktest PASS
  ships v9 without TIME_V6; the TIME_V6 pair passing adds it to v9.1.
- CONT_HIST (V10_PLAN #2) BUILT 5 Sep 18:45, off in the tree: 1-ply continuation
  history (C_CONT_HIST=38, conthist1 as the one new kernel arg) in ordering, the LMR
  history term (continuous hist//6000 clamped +/-2) and the prune2 test, gravity
  update on cutoffs, halved under HYGIENE. Bench vs champion: depth 8 0.890x nodes
  (1,327,419 vs 1,491,095) at 249 vs 258 knps; depth 10 0.900x nodes at 250 vs 262
  knps -- hits the spec's <= 0.90x target. ruff/mypy/exact 70/70 PASS. Queued as
  133-conthist on the laptop (600 games at 8 s) behind 132-v9core; SPRT verdict
  decides whether it anchors the v9.1 bundle. CONT_HIST2 (2-ply) only if 1-ply passes.
- 111-singular (desktop) is attribution only: if it ends REJECT, consider dropping
  SINGULAR from the tree in the next bundle (bench first); the v8.5 bundle passed with it.

## Champion
- v8.5 (uploaded by the human 5 Sep ~17:30, 18:00 slot) = v8 + LMR_AGGRESSIVE + LAZY_ACC +
  TIME_V5 + PRUNE_V2 + SINGULAR, all True in the tree since 5 Sep 17:40. Evidence: 8 s
  SPRT PROMOTE +36 over 477 games (55.2%), clocktest PASS 0/6; the 40 games at 120 s
  (v85-120s-b, desktop) are still to come and now measure the live build. Import ~41 s
  here (~75 s platform, budget 90) -- init time is a live risk, see V10_PLAN #10.
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
TIME_V6 (two clocktest FAILs at 120 s x1.5 charge: untamed drained to 1.6 s,
tamed f8286b8 to 2.0 s vs the 5 s floor -- overnight/eval/clocktest-timev6c.log;
closed, TIME_V5 stays; time-management ideas must pass a solo clocktest first),
104-kz16r retrain (month5.sh, 2024_11 data): initial_val == best_val 0.0046589 --
nine epochs never improved on the champion checkpoint and early-stop restored the
initial weights, so the exported net IS the champion net; closed WITHOUT a gauntlet
(it would measure noise). More of the same data is exhausted: NET_V10 must change
the architecture (mirrored king buckets, rebalanced output buckets, 16-out head),
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

## v8.5 plan (5 Sep 15:05, human's call) -- DONE, shipped 18:00
- v8.5 bundle = v8 + LMR_AGGRESSIVE + LAZY_ACC + TIME_V5 + PRUNE_V2 + SINGULAR, tested as
  ONE challenger vs v8 like v8 was: 110-v85all (laptop, 8 s SPRT), v85-clocktest + v85-120s
  (desktop, 40 games at 120 s platform openings), plus 111-singular alone on the desktop
  for attribution. 101-lmraggr stopped at 42 games (+25 +/- 90, nothing learned) to make
  room. Bench to depth 8 vs v8: PRUNE_V2 0.93x nodes, SINGULAR 1.55x (extensions cost
  nodes at fixed depth; judged at fixed time), both 1.50x. Do NOT queue single-switch
  tasks for the bundle parts; fold the bundle verdicts when they land.

## Running now (5 Sep 18:50)
- laptop worker: 132-v9core (crash gate started 18:29, then 600 games at 8 s vs v8.5),
  then 133-conthist (600 games at 8 s). month5.sh chain COMPLETED 16:47 -- folded.
- desktop worker: 111-singular (attribution, -6 +/- 36 at 272, heartbeat 18:21), then
  v85-120s-b (40 games at 120 s of the live v8.5 build), then v9-clocktest + v9-120s
  (five-switch sed WITH TIME_V6 -- these judge TIME_V6), then v9core-clocktest +
  v9core-120s (four-switch sed, the mandatory clocktest for shipping v9core).

## Backlog (ranked; take the top item that is not running) -- see overnight/eval/V10_PLAN.md
0. Fold verdicts. v8.5 (110-v85all) PROMOTED at 8 s (+36 over 477 games); its 120 s gate
   is v85-120s-b on the desktop (the first run hit 6/24 init timeouts under leftover
   load, not an engine fault). When v85-120s-b passes: flip the five v8.5 switches in
   the tree (that is v8.5 = the new champion), note it, and submission-v85.zip is
   already built from the tested challenger for the human.
1. TIME_V6: REVIVED 18:29 with final constants (71b7b50) after a local clock replay
   PASS; judged only by the desktop v9-clocktest + v9-120s. If that clocktest fails
   again, TIME_V6 is closed for good (third strike).
2. CONT_HIST (V10_PLAN #2): BUILT, queued as 133-conthist (see bundle notes above).
3. ADJUDICATION (V10_PLAN #3): BUILT, in the v9 bundle (132-v9core).
4. NET_V10 (V10_PLAN #4): mirrored king buckets + rebalanced output buckets (+ the
   16-out head from speed.md, +15% knps). 104-kz16r showed more same-style data adds
   NOTHING (val flat at 0.0046589): only an architecture change can move the net now.
   Prereq left: the v8/v8.5 endgame-suite baseline for comparison. One gauntlet slot.
5. IMPROVING + CUTNODE, NMP_V2 (V10_PLAN #5-6) as one bundle.
6. Exact kernel speed (see allocation, evaluate blocking) and init-time insurance per
   overnight/eval/v10/speed.md when it lands.
Closed by the research pass (do not reopen): staged movegen, multi-cut, IID, TT
replacement, QS checks, correction history, wider nets, distillation, int8, self-play at
scale, 6-man TB, book rescan, HalfKA.

## Next step
Iteration next: (1) fold verdicts as they land: 132-v9core (laptop),
111-singular, v85-120s-b, v9-clocktest/v9-120s (TIME_V6's gate), v9core-clocktest/
v9core-120s, 133-conthist. If 132-v9core PROMOTES and v9core-clocktest PASSES: ship
v9 (flip the four switches in the tree, re-run the exactness check, zip from the
TESTED challenger dir, CANDIDATE.md, notify). If the TIME_V6 pair passes, TIME_V6
joins the v9.1 bundle; if the clocktest fails, close it for good. (2) If 133-conthist
lands: a PROMOTE or positive-inconclusive + node win makes it the anchor of v9.1
(with TIME_V6 if passed); consider CONT_HIST2 (2-ply, lane-2 exts writes) only after
1-ply passes. (3) If a machine is idle after 22:00, start the NET_V10 prerequisite
(v8.5 endgame-suite baseline) and the NET_V10 architecture work (mirrored king
buckets, rebalanced output buckets, 16-out head) -- 104-kz16r proved more same-style
data is worthless, do NOT restart month-data chains.
