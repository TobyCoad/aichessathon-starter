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
- DESKTOP IS SHUT DOWN (5 Sep 20:05, the human's message). Do NOT queue anything in
  overnight/desktop/tasks.json and do not wait for desktop results (v9core-120s,
  111-singular, v85-120s-b will never land). Everything runs on the laptop: the worker
  takes gauntlets, clocktests and 120 s games in order, one at a time. Budget the queue:
  an 8 s bundle SPRT to its 200-game checkpoint ~1 h, clocktest ~10 min, 40 games at
  120 s ~45 min with 4 workers. Keep the laptop plugged in and awake.
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
- v9 SHIPPED 5 Sep 19:55 (this iteration): 132-v9core PROMOTE +23 at checkpoint 200
  (+70 =73 -57, 53.2%), v9core-clocktest-l PASS (0/6, lowest 11.3 s, longest 13.2 s).
  Four switches flipped in the tree (QS_EVAL_CACHE + ADJUDICATION + HISTORY2_FIX +
  KILLER_CLEAR); exact 70/70 PASS after the flip. Zip from the tested challenger
  132-v9core: C:/Users/tobyc/Downloads/aichessathon-v9.zip + submission-v9.zip
  (27.9 MB unpacked, cold import 33.6 s under load). CANDIDATE.md written, notify
  sent 19:54. TIME_V6: laptop clocktest v9-clocktest-l PASS (lowest 5.7 s); its
  remaining gate is the 120 s runs -- v9-120s-l on the laptop (started before the
  tree flip, so it measures v9+TIME_V6 vs v8.5) and desktop v9-clocktest + v9-120s
  (their seds now only flip TIME_V6, i.e. champion+TIME_V6 vs v9 champion -- exactly
  the right test). TIME_V6 joins v9.1 only if those pass.
- QUEUE HYGIENE after the tree flip (19:55): v85-120s-b, v9core-clocktest and
  v9core-120s removed from the desktop queue -- their seds now no-op against the v9
  tree, so they would have been champion-vs-champion noise. 111-singular (running,
  desktop) is VOID for the same reason if its challenger was built after the 17:40
  v8.5 flip (SINGULAR already True in the tree -> sed no-op); its -4.3 +/- 29 at 404
  looks exactly like self-play noise. Do NOT fold its verdict as attribution.
- CONT_HIST (V10_PLAN #2) BUILT 5 Sep 18:45, off in the tree: 1-ply continuation
  history (C_CONT_HIST=38, conthist1 as the one new kernel arg) in ordering, the LMR
  history term (continuous hist//6000 clamped +/-2) and the prune2 test, gravity
  update on cutoffs, halved under HYGIENE. Bench vs champion: depth 8 0.890x nodes
  (1,327,419 vs 1,491,095) at 249 vs 258 knps; depth 10 0.900x nodes at 250 vs 262
  knps -- hits the spec's <= 0.90x target. ruff/mypy/exact 70/70 PASS. Queued as
  133-conthist on the laptop (600 games at 8 s) behind 132-v9core; SPRT verdict
  decides whether it anchors the v9.1 bundle. CONT_HIST2 (2-ply) only if 1-ply passes.
- INIT_FOLD (speed.md section 2) BUILT 5 Sep 19:25, off in the tree behind agent.INIT_FOLD:
  fastsearch scans agent.py's flag lines at import and, when INIT_FOLD is True, compiles the
  18 settled switch slots (FOLDED in fastsearch.py) as constants via `_F_X if _FOLD else
  ctrl[C_X] != 0` ternaries that numba prunes before typing (pruning verified for all four
  condition shapes used). In-flight slots (C_QS_CACHE, C_HIST2_FIX, C_KILLER_CLEAR,
  C_CONT_HIST) and all value/state slots stay live reads, so challenger seds still work;
  C_PVS folds as PVS-or-LMR_AGGRESSIVE (mirrors agent line 1958). prepare() raises on any
  fold/ctrl mismatch. Measured back-to-back under gauntlet load: import 43.2 s (tree) ->
  38.4 s (folded), -4.8 s; bench depth 8 EXACTLY 1,491,095 nodes (bit-identical);
  ruff/mypy/check_fastsearch 70/70 PASS with the fold off. Exact by construction, so no
  gauntlet: it ships inside v9.1 (INIT GUARD rule: CONT_HIST pairs with INIT_FOLD). At v9.1
  zip time flip INIT_FOLD True in the tested challenger, re-check bench nodes + import.
  Scratch build kept in overnight/challengers/initfold.
## Stockfish-data net (the human's overnight priority, run by the interactive session -- NOT the loop)
- Source: linrock/test80-2024 binpacks (Stockfish self-play, engine-distribution positions).
  Decoder training/binpack_decode.py (validated: 435k sample entries, 0 invalid boards /
  illegal moves; scores are side-to-move internal units, int16-wrapped, VALUE_NONE 32002
  dropped; scale 0.45 cp per unit = median vs SF 17.1 depth 12, corr 0.90).
- MISMATCH MEASURED 23:30: the champion net (b8-kz16, Lichess-trained) scores validation
  loss 0.006879 on the Stockfish self-play validation set vs 0.00465 on Lichess data --
  48% worse overall, and the endgame strata are 2-3x worse (x1e-3: 9-12 pieces 11.39,
  13-16 9.99, eq<=16 5.10). That is the distribution gap games.md predicted. Training on
  the SF shards is running (24 epochs, ~2.5 min each). Baseline endgame suite of the champion
  net under the v9.1 search: 10.8 cp mean (5-8 pieces 17.0, 9-12 12.0, 13-16 5.0) --
  compare 150-sfnet's suite against THIS number, not the old 7.0.
- DONE 23:07: 581,225,450 positions in data/sf/feb24_00..08.npy (+ feb24_val.npy 500k) from
  1.03B decoded in 45.6 min (370k pos/s, 8 workers); the chain is now on the baseline suite.
- Plan was: decode data/sf/test80-2024-02-feb-2tb7p.min-v2.v6.binpack.zst to ~580M kept
  positions in RECORD shards data/sf/feb24_NN.npy (+ feb24_val.npy), then train the same
  architecture (512x16 zones, 8 buckets) warm-started from training/checkpoints/
  net_w512-b8-kz16.pt on those shards -> export --half -> check_nnue -> endgame suite ->
  net task gauntlet vs the champion -> compare with the Lichess-trained net. THE GPU IS
  RESERVED for this until it finishes; the loop must not start NET_V10 or any training
  before then (search bundles continue as normal). Results will be recorded here.

## Overnight programme (5->6 Sep; the human's instruction: keep iterating all night)
The last ladder game is 22:00; there are no games 22:00-08:00 UK, so the laptop is free
for testing and the GPU for training. Do NOT stop. Emails still go out per candidate (he
uploads in the morning; the upload cap resets 12:00). Order of work:
1. Finish v9.1: fold the TIME_V6 120 s verdict and 133-conthist; ship v9.1 = TIME_V6 (if the
   120 s games are not negative) + CONT_HIST (if positive) + INIT_FOLD, with the init guard
   (clean-unzip import < 45 s) -- INIT_FOLD is what makes CONT_HIST affordable.
2. Bigger search bundle for v9.2 (V10_PLAN #5-#6-#7): IMPROVING + CUTNODE flags gating RFP /
   futility / LMR / NMP, NMP_V2 (R = 3 + d/4 + eval margin, verification at depth >= 10),
   CAPTURE_ORDER (SEE-ordered captures below quiets + capture history). Build all three as
   switches, bench each, one SPRT for the bundle.
3. GPU IS FREE: the month5 chain finished 16:47 -- kz16r (five months) restored a best epoch
   with validation loss 0.004659, IDENTICAL to the b8-kz16 start, i.e. the fifth month added
   nothing measurable; run its suite only if cheap, otherwise skip 104-kz16r and go straight
   to NET_V10. NET_V10 on the GPU (V10_PLAN #4, network.md scoping): first the v8/v9 endgame-suite
   baseline (~18 min CPU, run while a gauntlet is NOT at its checkpoint), then train the
   mirrored-king-bucket net with rebalanced output buckets warm-started from
   training/checkpoints/net_w512-b8-kz16.pt (check the month5 chain first: if kz16r is
   training, let it finish and evaluate it; the GPU takes one job at a time). Export
   --half, check_nnue, suite, then a net task ({"net": ...}) SPRT. A net that passes ships
   as its own version.
4. If everything above is queued and the laptop is saturated: QS transposition table
   (V10_PLAN #9), aspiration widening 1.5x, ENDGAME_SHRINK scoping against the suite.
Rules of the night: one gauntlet at a time (the worker enforces it); the endgame suite and
training count as load -- do not start them while a gauntlet is within 30 games of a
checkpoint; keep the exactness check green at every commit; never edit results files.

- 5 Sep 22:40 (session): iteration 7 died on the usage limit mid-edit, leaving the IMPROVING /
  CUTNODE kernel code (C_IMPROVING, C_CUTNODE = 39, 40; CTRL_SIZE 42) uncommitted while the
  committed agent.py already referenced it -- origin was inconsistent. Verified exact
  (70/70, 40/40) and committed as-is; 141-v92prune tests the pair. ALWAYS commit kernel +
  agent together; never leave the tree with a half-done build at the end of an iteration.

## Champion
- v9.1 (5 Sep 21:55, uploaded by the human when he reads the email) = v9 + TIME_V6, all True
  in the tree (exact 70/70). Evidence: 40 games at 120 s vs v9 55.0% (+11 =22 -7), clocktest
  PASS (floor 5.8 s by design). v9 (19:53) = v8.5 + QS_EVAL_CACHE + ADJUDICATION +
  HISTORY2_FIX + KILLER_CLEAR (+23 at the 200-game checkpoint). CONT_HIST REJECTED at 8 s
  (133-conthist) -- closed. IMPROVING + CUTNODE built (off, by the loop) -- the v9.2 bundle.
- v9 (candidate emailed 5 Sep 19:54, awaiting the human's upload) = v8.5 +
  QS_EVAL_CACHE + ADJUDICATION + HISTORY2_FIX + KILLER_CLEAR, all True in the tree
  since 5 Sep 19:49. Evidence: 8 s SPRT PROMOTE +23 at 200 games (53.2%), clocktest
  PASS 0/6 (lowest 11.3 s), cold import 33.6 s. Every challenger is judged vs v9 now.
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
INFRA, not engine verdicts: 140-v92prune (1/24 init fail) and 141-v92prune (19/24
init fail) both died to init timeouts under the 8-worker binpack decode; the worker
now waits while a decode or endgame suite runs, and the pair re-queued as
142-v92prune. Do not count these as IMPROVING/CUTNODE rejects.
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

## Running now (5 Sep 23:45)
- Decode DONE (581M positions, 23:07). Interactive session now runs the v9.1
  endgame-suite baseline (overnight/eval/suite-v91-champion.log, 300/400 at
  23:34, ~5 min left) -- the NET_V10 prereq. GPU stays reserved for the SF
  retrain (NOT the loop's job).
- laptop worker: picked 142-v92prune at 22:42 and WAITS in the load-check loop
  until the suite exits (the fix after 140/141 died to init timeouts under the
  decode -- infra rejects, not engine verdicts). Queue after it:
  v92prune-clocktest-l, 143-v92nmp (IMPROVING + CUTNODE + NMP_V2), 
  v92nmp-clocktest-l, then 144-caporder + caporder-clocktest-l (new, this
  iteration). NMP_V2's build (previous iteration) was left uncommitted; this
  iteration's commit carries it together with CAPTURE_ORDER (exact check ran
  on the combined tree). training/binpack_decode.py left uncommitted -- it
  belongs to the interactive session's chain.
- desktop: OFF. Queue nothing there.

## Backlog (ranked; take the top item that is not running) -- see overnight/eval/V10_PLAN.md
0. Fold 142-v92prune and 143-v92nmp when they land; ship v9.2 from whichever bundle
   shape passed (pair or trio; clocktest for each is queued). INIT_FOLD (built,
   exact, -4.8 s import) rides in the v9.2 zip: flip it True in the tested
   challenger at zip time, re-check bench nodes + clean-unzip import < 45 s.
1. NMP_V2 (V10_PLAN #6): BUILT 5 Sep 23:00, off in the tree (C_NMP_V2=41, fills
   CTRL_SIZE 42's last slot). Dynamic R = 3 + depth//4 + min((standing-beta)//200, 3),
   null tried only when standing >= beta, skipped on a TT upper bound < beta
   (tt_depth >= 0 guards the no-hit default tt_flag=2/tt_score=0). Verification
   search deferred to NMP_V2B. Queued in 143-v92nmp.
2. CAPTURE_ORDER (V10_PLAN #7): BUILT 5 Sep 23:40, off in the tree
   (C_CAPTURE_ORDER=42, CTRL_SIZE 43). Rescore pass after score_moves: SEE < 0
   captures below every quiet (band -(1<<21) + see*16), SEE >= 0 keep the
   MVV-LVA band; capture history (gravity bonus on capture cutoffs, >>=1 decay)
   in the first 4608 entries of the conthist1 buffer (CONT_HIST is closed;
   both-on raises at init) so the kernel signature is unchanged. Bench depth 8:
   1,574,873 vs 1,445,087 nodes (1.090x) at 378 vs 391 knps -- ordering must
   buy back ~9% nodes at fixed time; the SPRT decides. ruff/mypy/exact 70/70 +
   40/40 PASS. Queued as 144-caporder (600 games, 8 s) + caporder-clocktest-l
   at the end of the laptop queue; its sed flips only CAPTURE_ORDER, so it
   stays champion+CAPTURE_ORDER even if the tree flips to v9.2 under it.
3. NET_V10 (V10_PLAN #4): mirrored king buckets + rebalanced output buckets + the
   16-out head. The interactive session owns the GPU and the SF-data retrain;
   the loop only folds results. Prereq left: the v9 endgame-suite baseline
   (run when the laptop is not near a gauntlet checkpoint).
4. Init/speed leftovers from speed.md: eager signatures on the fastboard leaves
   (~-3 s more), see allocation, evaluate blocking.
Closed by the research pass (do not reopen): staged movegen, multi-cut, IID, TT
replacement, QS checks, correction history, wider nets, distillation, int8, self-play at
scale, 6-man TB, book rescan, HalfKA.

## Next step
(1) Fold 142-v92prune when it lands (the worker starts it as soon as the
endgame suite exits; SPRT to checkpoint 200 ~1 h + the clocktest behind it). A
pass makes IMPROVING+CUTNODE the v9.2 anchor; 143-v92nmp then tests the trio
with NMP_V2. Ship v9.2 from the best passing shape, with INIT_FOLD flipped in
the zip (item 0 above -- re-check import < 45 s). (2) 144-caporder (queued this
iteration) decides CAPTURE_ORDER for the v9.3 bundle. (3) While waiting: next
build item is QS transposition table (V10_PLAN #9) or aspiration widening 1.5x
(#12) as filler switches for v9.3. (4) Do NOT start GPU work; the SF retrain
belongs to the interactive session.
