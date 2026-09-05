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
QS_EVAL_CACHE (exact, +2% only), CHECK_EXT_CAP (no effect).

## Running now (5 Sep 11:30)
- desktop: 091-ttkeep (108 games at 11:01 heartbeat, llr -1.21, trending REJECT),
  then gen-001/002, 092-qscap14, 093-safe, 095-asp15, 096-clocktest-v71, 097-seemain.
- laptop CPU: 090-history2 rerun gauntlet (queue8b, outside the worker); the worker
  holds 098-rootorder then 099-ttbuckets (tasks.json) until that gauntlet clears.
- laptop GPU: net_w512-b1-kz16 (no output buckets, from scratch) -> export/check/suite/gauntlet.
- download: fetch 2025_04 running -> pack when done.

## Backlog (ranked; take the top item that is not running)
1. Fold every landed verdict: promote passes into the tree (switch -> True, or
   the net), reject the rest, note it here and in JOURNAL.md.
2. When >= 2 switches have passed: build the bundle challenger, run its gate
   (see rules), and if green write CANDIDATE.md + submission-candidate.zip.
3. Time: expected-moves floor 26 -> 18 and shrink when stable (120 s only).
4. Lazy accumulator update (defer make_full's NNUE update until evaluate).
5. Pack the fifth month; retrain the 16-zone net on five months (lr 1e-4).
6. Book rebuild with `--max-drop 10 --min-count 20`, judged on platform openings.
7. Anything from overnight/eval/V7_PLAN.md not listed as closed.

## Next step
Iteration 3: check results (desktop_status.sh, night3.log, laptop results/ for
the 090 rerun, 098-rootorder, 099-ttbuckets; desktop results/ for 091-ttkeep,
which was trending REJECT at llr -1.21). Fold any verdicts. Otherwise take
backlog item 3 (time: expected-moves floor 26 -> 18, 120 s only -- needs the
clocktest and 40 games at 120 s per the rules, not just the 8 s SPRT).
Verdict context: 098-rootorder measured +3.5% nodes (weakly negative, REJECT
likely); 099-ttbuckets measured exactly neutral on nodes at depth 8 (table not
saturated at bench sizes; only long searches can show a gain, and the double
probe costs ~5% speed, so INCONCLUSIVE/REJECT would not surprise).
