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
- SF NET RESULT 6 Sep 00:45 (150-sfnet, warm start from b8-kz16 on 581M SF positions, best
  epoch 15/24): SF-val 0.003917 (champion 0.006891, -43%) but Lichess-val 0.011299 (champion
  0.004633, +144%: it forgot the human distribution). Endgame suite 10.8 cp = champion's 10.8
  (5-8: 16.7 vs 17.0, 9-12: 8.4 vs 12.0 BETTER, 13-16: 8.4 vs 5.0 WORSE). check_nnue PASS.
  Gauntlet 150-sfnet queued next on the laptop -- the games decide. A MIXED-DATA net
  (SF shards interleaved with the Lichess shards, warm start from the champion) is training
  now as 151-mixnet (overnight/sf_train_mix.sh) to keep both distributions; GPU still reserved.
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
142-v92prune (IMPROVING + CUTNODE) REJECT at 8 s: 40.0% (+33 =46 -61) over 140
games, llr -2.18 -- the pair is out. Split by bench: IMPROVING alone halves the
d8 tree (0.506x nodes, big eval swings) -- over-pruning, CLOSED as the culprit;
CUTNODE alone is benign (0.995x) and gets the one allowed requeue (146-cutnode).
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

## Running now (6 Sep 02:20, iter 14)
- 150-sfnet at 104 games, +33.5 +/- 56.4 (02:04) -- looking positive; its
  200-game checkpoint lands ~03:00, then nmp-clocktest-l (the v9.2 gate).
- Iter 14 finished the half-done eager-signatures build (backlog item 4, see
  below) and committed it with the binpack_decode ruff fixes. No verdicts
  landed this iteration; nothing new to fold.
- 143-nmp PROMOTED 01:25: +26 Elo at the 200-game checkpoint (+76 =63 -62,
  53.5%, llr +1.09). NMP_V2 is the v9.2 anchor. Remaining gate: nmp-clocktest-l,
  queued directly after 150-sfnet (running, crash-gate stage as of 01:32).
  When it PASSES: flip NMP_V2 True in the tree, re-run exact, build the v9.2
  zip from overnight/challengers/143-nmp with INIT_FOLD flipped True in the
  zip copy (clean-unzip import < 45 s check), CANDIDATE.md, notify. Do NOT
  wait for 120 s games (NMP_V2 is not time management).
- 142-v92prune REJECTED 00:13 (40.0%, +33 =46 -61 over 140 games, llr -2.18):
  the IMPROVING+CUTNODE pair is out. Its clocktest passed (5.6 s) but is moot.
  Per the split rule the trio 143-v92nmp (which stacked NMP_V2 on the failed
  pair) was killed ~20 min in and requeued as 143-nmp (NMP_V2 ALONE, 600
  games) + nmp-clocktest-l -- the worker picked it up 00:30; PROMOTED, above.
  Split re-queue of the pair (benched 00:50, champion baseline 1,445,087 nodes
  d8): IMPROVING alone is 731,904 nodes (0.506x!) with big eval swings -- it
  over-prunes and is the -50 Elo culprit; CLOSED without its own gauntlet.
  CUTNODE alone is 1,438,246 (0.995x, benign) -> requeued as 146-cutnode +
  cutnode-clocktest-l at the queue tail (the one allowed requeue; if it fails,
  both are closed for good). New-switch benches d8: QS_TT 1,435,882 (0.994x,
  229 knps under load); ASP_WIDE 1,445,087 -- node-identical to the champion
  on the bench suite (no aspiration fails at d8), judged by the SPRT only.
- QS_TT + ASP_WIDE (the previous iteration's half-done build) FINISHED and
  committed 00:35 (1472326): ruff/mypy PASS, exact 70/70 + table-on 40/40.
  Their bundle task 145-v93fill + v93fill-clocktest-l was already queued.
- Interactive session's SF-net chain is at check_nnue on 150-sfnet (GPU/net
  work stays theirs; the loop only folds results).
- laptop queue order now: 143-nmp (running, +50 +/- 49 at 98 games 01:00),
  150-sfnet (interactive session's net task), nmp-clocktest-l, 144-caporder,
  caporder-clocktest-l, 145-v93fill, v93fill-clocktest-l, 146-cutnode,
  cutnode-clocktest-l, 147-seequiet, seequiet-clocktest-l.
- desktop: OFF. Queue nothing there.
- INIT_FOLD CAVEAT CLEARED (6 Sep 01:15, iter 11): scratch rebuilt from the
  current tree (now CTRL_SIZE 45), bench d8 with the fold on = 1,445,087 nodes,
  bit-identical to the champion; fold + NMP_V2 flipped = 279,188 nodes at d6,
  identical to no-fold + NMP_V2 (the v9.2 zip scenario); prepare()'s FOLDED
  check passed in every run (FOLDED holds only the 18 settled slots, so growing
  CTRL_SIZE does not touch it). INIT_FOLD is clear to ride in the v9.2 zip.
- SEE_QUIET BUILT 6 Sep 01:20 (iter 11), off in the tree (C_SEE_QUIET=44,
  CTRL_SIZE 45): skip a late quiet at depth <= 6 when fb.see on its destination
  square loses more than 30*depth^2 (fb.see already handles quiets; no board
  change). Guards: not in check, searched > 0, alpha away from mate. Bench d8:
  1,098,731 vs 1,445,087 nodes (0.760x) at similar knps under load -- healthy,
  not IMPROVING's 0.506x. Score watch: bench pos 39 swings +534 -> +214 (same
  best move); the rest within ~35 cp. ruff/mypy/exact 70/70 + table-on 40/40
  PASS. Queued as 147-seequiet (600 games, 8 s) + seequiet-clocktest-l at the
  queue tail.

## Backlog (ranked; take the top item that is not running) -- see overnight/eval/V10_PLAN.md
0. SHIP v9.2 when nmp-clocktest-l PASSES (queued right after 150-sfnet): champion
   + NMP_V2 (PROMOTE +26 at 201 games), INIT_FOLD + the fastboard eager
   signatures ride in the zip. Then fold 144-caporder, 145-v93fill, 147-seequiet
   as they land; ship v9.3 from the union of later passes. INIT_FOLD FOLDED-map
   caveat CLOSED 6 Sep 01:15 (bit-identical bench vs CTRL_SIZE-45 tree; scratch
   in overnight/challengers/initfold).
1. NMP_V2 (V10_PLAN #6): PROMOTED 6 Sep 01:25 as 143-nmp (+26 Elo, 53.5% over
   201 games). Dynamic R = 3 + depth//4 + min((standing-beta)//200, 3),
   null tried only when standing >= beta, skipped on a TT upper bound < beta.
   Verification search deferred to NMP_V2B (a later idea, now with evidence the
   base works). Awaiting nmp-clocktest-l only.
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
   DONE 6 Sep 02:15 (iter 14, committed in-tree): 14 leaf helpers (lsb/msb/popcount/
   bit/attacks/attackers/occupancy/_add/_add_promotions/feature/_acc_row/...) get
   eager numba signatures -- source-only, exact by construction (bench d8
   1,445,087 nodes bit-identical to the champion; exact 70/70 + table-on 40/40,
   ruff/mypy PASS). Import benefit (~-3 s idle per speed.md) not separable under
   gauntlet load (one under-load sample 46.0 s vs 43.2 s tree earlier -- noise);
   the clean-unzip import check at v9.2 zip time is the real measure. Remaining
   leftovers: see allocation, evaluate blocking (small, take only if idle).
Closed by the research pass (do not reopen): staged movegen, multi-cut, IID, TT
replacement, QS checks, correction history, wider nets, distillation, int8, self-play at
scale, 6-man TB, book rescan, HalfKA.

## Next step
(1) SHIP v9.2 the moment nmp-clocktest-l PASSES (it runs right after 150-sfnet,
which was at 104 games +33.5 at 02:04; checkpoint 200 ~03:00, clocktest ~10 min
after): flip NMP_V2 True in the tree, re-run exact, zip from
overnight/challengers/143-nmp with INIT_FOLD flipped True in the zip copy,
clean-unzip import < 45 s check, CANDIDATE.md, notify. The eager fastboard
signatures are already in the tree (committed iter 14) so the challenger copy
step picks them up ONLY if 143-nmp is rebuilt -- prefer zipping agent.py/
fastsearch.py from the tested 143-nmp dir and fastboard.py from the tree ONLY
if bench in the zip dir is re-verified bit-identical; otherwise ship the tested
trio as-is (the signatures then ride in v9.3). (2) Fold 150-sfnet (interactive
session's verdict), then 144-caporder, 145-v93fill, 146-cutnode, 147-seequiet
in turn; ship v9.3 from the union of passes. After 146, IMPROVING and CUTNODE
are closed for good. (3) If idle: speed leftovers (see allocation, evaluate
blocking) or read the newest overnight/postmortem/ for failure modes. Keep the
exactness check green. (4) Do NOT start GPU work; the SF retrain and
151-mixnet belong to the interactive session -- fold its suite/gauntlet
results when they appear in overnight/laptop/results/.
