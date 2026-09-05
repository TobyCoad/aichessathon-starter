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
- Bundle in flight for v9 -- ALL FIVE SWITCHES BUILT (off in the tree), the 8 s SPRT of
  the whole bundle is queued as 130-v9all on the laptop (500 games vs v8.5): TIME_V6
  (tamed values, f8286b8; its solo clocktest is rerunning in
  overnight/eval/clocktest-timev6b.log) + QS_EVAL_CACHE (exact, +4.2% knps) +
  ADJUDICATION + HISTORY2_FIX + KILLER_CLEAR (593053f; bench vs champion 1.00x / 1.02x /
  0.97x nodes, smoke test of the adjudication arming passed, exact 70/70). Desktop
  v9-clocktest + v9-120s to queue once v85-120s-b finishes. If clocktest-timev6b FAILS
  again, pull 130-v9all and re-queue it without the TIME_V6 flip. CONT_HIST (V10_PLAN #2,
  the biggest search item) is the multi-iteration build for v9.1; follow search.md 3.1
  exactly (widen `exts` to 4*MAX_PLY instead of new arrays; conthist1 as new kernel args).
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

## Running now (5 Sep 18:20)
- laptop worker: 130-v9all (8 s SPRT of the five-switch v9 bundle vs v8.5, 500 games;
  120-timev6 was pulled after the untamed clocktest drained to 1.6 s). Laptop background:
  month5.sh (pack 2024_11 -> train kz16r -> suite); clocktest of tamed TIME_V6 restarted
  ~17:50 (overnight/eval/clocktest-timev6b.log).
- desktop worker: 111-singular (attribution, -18 +/- 42 at 170), then v85-120s-b (40 games at
  120 s of the live v8.5 build).

## Backlog (ranked; take the top item that is not running) -- see overnight/eval/V10_PLAN.md
0. Fold verdicts. v8.5 (110-v85all) PROMOTED at 8 s (+36 over 477 games); its 120 s gate
   is v85-120s-b on the desktop (the first run hit 6/24 init timeouts under leftover
   load, not an engine fault). When v85-120s-b passes: flip the five v8.5 switches in
   the tree (that is v8.5 = the new champion), note it, and submission-v85.zip is
   already built from the tested challenger for the human.
1. TIME_V6 (V10_PLAN #1): reserve 0.10 -> 0.04, LOW_CLOCK 15 -> 9, node-effort +
   stability + score-drop factors, no next-iteration prediction; absorbs TIME_V5.
   Judged by clocktest + 40 games at 120 s on the desktop only.
2. CONT_HIST (V10_PLAN #2): continuation history alone -- HISTORY2_FIX and KILLER_CLEAR
   are built and in the v9 bundle (130-v9all). 8 s SPRT on the laptop.
3. ADJUDICATION (V10_PLAN #3): BUILT, in the v9 bundle (130-v9all).
4. NET_V10 (V10_PLAN #4): mirrored king buckets + rebalanced output buckets, after the
   v8 endgame-suite baseline and 104-kz16r. One gauntlet slot.
5. IMPROVING + CUTNODE, NMP_V2 (V10_PLAN #5-6) as one bundle.
6. Exact kernel speed (see allocation, evaluate blocking) and init-time insurance per
   overnight/eval/v10/speed.md when it lands.
Closed by the research pass (do not reopen): staged movegen, multi-cut, IID, TT
replacement, QS checks, correction history, wider nets, distillation, int8, self-play at
scale, 6-man TB, book rescan, HalfKA.

## Next step
Iteration next: (1) fold verdicts: clocktest-timev6b (tamed TIME_V6; if it FAILS pull
130-v9all from the laptop queue and re-queue it without the TIME_V6 flip), 111-singular,
v85-120s-b, and 130-v9all when it lands. (2) When v85-120s-b is done, queue v9-clocktest
+ v9-120s on the desktop (same seds as 130-v9all; see v85-120s-b's task for the 120 s
format). (3) Start the CONT_HIST build (V10_PLAN #2, multi-iteration, switch OFF;
search.md 3.1: widen `exts` to 4*MAX_PLY instead of new arrays, conthist1 as new kernel
args) -- HISTORY2_FIX and KILLER_CLEAR are already built and in the v9 bundle, so
CONT_HIST is the only piece left of that bundle item. (4) If a machine is idle after
22:00, start the NET_V10 prerequisites (v8 endgame-suite baseline; 104-kz16r fold).
