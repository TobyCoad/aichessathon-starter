# Journal

One entry per run, newest last. Record what was tried, what the gauntlet said, and
what the next run should know. **Failures are the more valuable half** — an
experiment recorded as rejected saves a future run from repeating it, and there is
no other memory between runs.

Format:

```
## YYYY-MM-DD HH:MM — <experiment>
Verdict: PROMOTE / REJECT / INCONCLUSIVE / not finished
Result:  <the gauntlet's numbers: games, Elo, LLR>
Notes:   <what surprised you; what the next run should do differently>
```

---

## 2026-08-30 22:40 — foundation laid by hand

Verdict: n/a — this was the setup run, not an experiment.

Built:
- `testing/` — the measurement rig. `harness/referee.py` hardcodes the starting
  position, so two deterministic agents replay one identical game; `testing/`
  forks it for start-FEN support only and plays 40 balanced openings from both
  sides, across cores, terminated by SPRT.
- `testing/sprt.py` — SPRT[0, 20], alpha = beta = 0.05. Needed a `MIN_PAIRS = 25`
  guard: unguarded, over 200 simulated matches it promoted a true 0-Elo change 14%
  of the time against a nominal 5%, because the variance estimate is wild early.
  Guarded, it measures 5.2% at 0 Elo and 0.0% at −50.
- `testing/gauntlet.py` — the promotion interlock. Crash gate, then SPRT.
- `agent.py` — iterative-deepening alpha-beta, TT used primarily for move
  ordering, MVV-LVA, quiescence with delta pruning, tapered PeSTO evaluation,
  conservative clock.
- `tools/winshim.py` — the harness uses `selectors` on pipes, which Windows only
  accepts for sockets, so no local game ran at all before this.

Measured baseline for the next run to beat:
- depth 4–5 middlegame, 7 endgame, ~50–86k nps in a 2 s budget
- vs `baselines/minimax`: +46 =3 −1 over 50 games at 5s+0.05s
- vs `baselines/minimax`: +5 =1 −0 over 6 games at the full 120s+0.5s
- no crashes, no illegal moves, no flags
- evaluation is mirror-symmetric on 299/299 random positions

Notes: the evaluation is hand-crafted, so this is a fallback rather than a legal
final submission — the rules require a learned model to materially drive move
selection. P2 in the backlog is therefore mandatory, not optional.

## 2026-08-30 23:45 — environment prepared for P2

Verdict: n/a — setup, not an experiment.

- **CUDA torch working.** `2.11.0+cu128`, RTX 5070 Laptop, capability (12, 0),
  3.3 TFLOP/s on a real matmul. The first two install attempts failed silently:
  pip treated the existing `2.13.0+cpu` as satisfying `torch` and downloaded
  nothing, then a `--force-reinstall` hung for 25 minutes on a stalled socket with
  zero CPU time. `--timeout 30 --retries 10` fixed it. If torch ever needs
  reinstalling, go straight to those flags.
- **Training data downloading**, `standard_rated_2025_01.parquet`, 7.5 GB, resumable
  via Range requests. Almost certainly incomplete — check the size and re-run
  `training.fetch` to resume before P2.2 touches it.
- **`pyarrow` is not installed.** P2.2 needs it.
- Fixed a cosmetic SPRT bug: identical early pairs collapsed the sample variance and
  printed an LLR of 41 million. Floored the variance at 0.01, far below any real
  match variance, so it binds only in the degenerate case. Error rates unchanged.

Notes for the next run: P2.1 needs neither the GPU nor the data, so it can proceed
regardless of how the download went. Do that first.

## 2026-08-31 09:55 — 1024-wide net on 21.6M positions

Verdict: **INCONCLUSIVE** — champion stays.

```
+182 =259 -159   score 51.9%   over 600 games
llr +0.59 [-2.94, 2.94]   elo +13.3 +/- 21.0
```

Scaled up on every axis that looked promising and got nothing:

| | champion | challenger |
|---|---|---|
| training positions | 9.1M | 21.6M |
| accumulator width | 256 | 1024 |
| parameters | 213,313 | 853,057 |
| best held-out loss | 0.006527 | **0.005545** |
| measured strength | — | **+13.3 +/- 21.0** |

**A 15% better validation loss converted to approximately zero Elo.** That is the
finding worth keeping. Held-out loss says the network predicts Stockfish's
centipawns more accurately; it does not say the engine picks better moves. Two
candidate explanations, not mutually exclusive: the wider net costs node rate, and
at depth 4-6 a ply is worth ~150 Elo, so the evaluation gain may have bought back
exactly what the slowdown cost; or ranking candidate moves correctly is a different
problem from scoring them precisely, and the extra capacity went into the latter.

The confidence interval is [-8, +34], so it is not *proven* worthless -- but
resolving a true +13 would need roughly 5,000 games, which is not affordable before
the 11 September lock. Treat width as spent.

**Do not repeat this experiment.** If the net is revisited, change the training
signal or the data, not the parameter count.

Process notes:
- Early stopping earned its place on the first run: best epoch was 14, and epochs
  15-18 all had lower *training* loss with worse validation. Without it we would
  have shipped the most overfit epoch and had no way to see it.
- `check_nnue.py` had the same hardcoded-256 width bug already fixed in
  `export.py`. It failed loudly rather than silently comparing against a
  differently-shaped net, which is what that file is for.

## 2026-08-31 — audit fixes, tablebase, and the search package

Three adversarial audits (rules compliance, literature gap analysis, bug hunt) plus
the fixes that came out of them.

### Measured

| change | verdict | games |
|---|---|---|
| 3-4-man Syzygy tablebase | +18.3 +/- 21.7, inconclusive | 400 |
| **Reverse futility pruning** | **PASS, +62.1 +/- 38.7** | **164** |

RFP is the first clearly positive result since the network itself, and it lands in
the published range (+57.1 Blunder, +145.8 int0x80).

### Bugs fixed, all of them silent

- **Delta pruning was inert**: the margin was a queen's value, so it fired on 2 of
  15,540 capture candidates. At 200 the same trace prunes 11.7%.
- **Tablebase scores were not rebased across the transposition table.** The mate
  rebasing keyed off MATE_THRESHOLD and TB_WIN sits below it, so 106 of 1,075
  entries carried a ply-dependent score that survived unchanged.
- **Repetition history was off by one**: one prior sighting is not a draw.
- **Checkmate lost to the fifty-move counter.**
- **Syzygy cursed wins (+/-1) were scored as wins.** They are draws in play.
- **The transposition table was unbounded**: 0.47 GB in a full game against a 2 GB
  cap, and every SPRT so far ran at 8 s where it reaches 39k entries, not 627k.

### The measurement rig was overstating confidence

There were 40 openings. A 600-game match is 300 pairs, so each was replayed sixteen
times and counted as sixteen independent samples. Now 119, with a warning when a
match outruns the set. **Verdicts already recorded stay valid -- inconclusive stays
inconclusive -- but the error bars were too tight.**

### The insight worth keeping

Before RFP, the engine had **no consumer of evaluation quality outside quiescence
leaves**. No reverse futility, no futility, no razoring -- every leaf score reached
the result through one narrow channel. That is a credible explanation for the 4x
wider net measuring +13 +/- 21, and it means the net may have been judged before it
had anywhere to deposit an improvement. Worth re-testing a bigger net *after* the
search package lands, rather than treating width as settled.

## 2026-08-31 — search package promoted, width settled, data experiment begun

### Measured

| change | verdict | games |
|---|---|---|
| **Full search package** (RFP, killers, history, check ext, null-move) | **PASS, +199.2 +/- 104.1** | **56** |
| Reverse futility pruning alone | PASS, +62.1 +/- 38.7 | 164 |
| 1024-wide net vs 256 control, same data + search | ~-35, two runs, not promoted | 222 + 79 |

The search package is worth about as much as the neural network was. At depth
4-6 one ply is ~150 Elo, and these are the features that buy plies.

### Width is settled: the net is data-starved, not capacity-starved

The missing control finally exists -- 256 wide on the *same* 21.6M positions with
the *same* early stopping -- and it beats the 1024 on held-out loss (0.005411 vs
0.005545) and in games (~-35 Elo for the wider net, two runs). Earlier I called the
"15% better loss" for the 1024 a reason to revisit width; that figure compared a
*train* loss on one dataset to a *val* loss on another and did not survive the
control.

### A rig bug that was present in every measurement

`--workers` counts *games*, and each game runs **two** agent processes. The default
was `cores - 2`, so 16 cores ran 28 processes -- 1.75x oversubscribed. Agents time
themselves in wall clock but poll only every 1024 nodes, so a descheduled process
sails past its budget: a 222-game match produced **46 flags**, while the same
engines single-threaded overshot on **zero of 59 moves**. Default is now
`cores//2 - 1`.

Load was symmetric so the verdicts stand, but every depth figure so far was
measured on ~0.6 of a core rather than the dedicated core the platform gives.

### External calibration exists now

`tools/uci_opponent.py` wraps any UCI engine as an agent directory;
`testing/calibrate.py` plays a Stockfish Skill Level ladder (1320/1608/1923/2363,
2,000-5,400 calibration games per rung) and reports a **bracket**. Deliberately not
a single number: a skill-limited engine has no scalar rating -- DeepMind measured
one unchanged model at 2895 Elo vs humans and 2299 vs bots.

Use `UCI_Elo` for nothing. It is a search-depth ladder whose polynomial has not
been refit since Jan 2023, and whose same setting was labelled ~2200 in SF15 and
~2711 in SF16.

### Next: the label problem

We use 7% of one downloaded file that holds 480M unique positions, and the
+/-100cp balancing discards a third of what we do collect to satisfy a floor that
its source paper never ablated. Repacking 80M with balancing off. If more data at
current quality moves the needle, better labels will move it further; if it does
not, relabelling with our own Stockfish at fixed depth would not have helped either
and the bottleneck is elsewhere.

## 2026-09-01 — the unrecorded day, and the night that follows

Verdict: mixed; recorded after the fact from the logs, which is the process failure.

Runs on 1 Sep that never reached this file:
- **021-w512-150m**: 512-wide accumulator on 145M positions, min-ply 16, balancing
  off. PROMOTE, +89 +/- 49 over 96 games (pinned run). A second, slower run reached
  the same +89 but was REJECTED on 15 failures: 32 flags across both sides at
  20 s + 0.2 s. That flag count is the finding, not the Elo.
- **022-reserve / 023-timefix**: clock reserve and 256-node polling. Inconclusive
  and unfinished. An 8 s SPRT cannot measure a 120 s time manager; wrong tool.
- **024-lmr**: late move reductions with null-window scouting. ~0 +/- 30 after
  226 games, inconclusive at 8 s. Retest at 30 s after staged move generation.
- **025-tune**: four changes bundled (losing-capture demotion, check evasions in
  quiescence, history decay, fail-soft mate). First run trending -22 +/- 33; the
  rerun was 223 of 250 games both-failed, mean length 13 plies -- an environment
  failure. **Verdict invalid**, and a bundle teaches nothing either way.

Evening review (`overnight/REVIEW_2026-09-01.md`): the champion spends 98% of a
120 s game and single moves at 4x soft budget; move generation is now ~46% of node
time (fresh benchmark 107 knps); the skill-ladder rating is inflated. Four switches
added to `agent.py`, all off, flags-off proven identical to fa10f1c on 12 probes.

Tonight `overnight/night.sh` runs unattended: TIME_V2 clock replay + not-worse SPRT,
second 145M shard, 512 net continued on both shards, then QS_EVASIONS, STAGED_MOVEGEN
and HYGIENE one at a time, then a Claude Fable review into `overnight/night/REVIEW.md`.
The five-hourly Claude loop is disabled for the night; re-enable it in the morning.

## 2026-09-02 — first clean night: one promotion, one near-miss, one refuted schedule

Verdict: night.sh ran end to end with zero flags, crashes, or init failures across
~1,240 games; one promotion (030-hygiene); the engine is still not clock-safe.

**Clock.** At 120 s + 0.5 s charged x1.5, the champion bottomed at 1.2 s (longest
move 20.9 s) — confirmation of the review's flaw #1. TIME_V2 bottomed at 4.6 s,
longest move 16.3 s, and failed the ≥5 s bar by 0.4 s; at x1.0 it passes (6.0 s
floor). Its floor is near-constant across games, so this is a reserve constant,
not variance. SPRT[-25, 0] at 8 s: inconclusive at 300 games, −4.6 ± 25, llr +1.21
toward pass. Not promoted (safety exit 1). Fix the reserve, re-replay, promote by
hand — the SPRT regime cannot see what this switch does anyway.

**Shard.** Packed 145,283,816 positions from row groups 279+ (quiet-fraction 0 by
design, 29.2% quiet), 4.9M duplicates dropped, disjoint validation of 532,307.
check_pack green; its "balancer may not have run" warning is intentional-off noise.

**Train.** Resumed net_w512-150m (val 0.0051766) at lr 3e-4 over both shards.
Val rose to 0.005262 then descended monotonically to 0.005185; patience 6 against
the initial value fired at epoch 6/24 and best_val = initial_val. Nothing exported,
nothing tested — correctly skipped. The run refutes "resume at 3e-4", not "more
data": the early stop was structurally guaranteed by the warm restart. Next attempt
should be output buckets, not a re-tuned continuation.

**028-qs-evasions.** REJECT at 267 games, −18.3 ± 26.5, llr −3.08. Consistent
with 025's trend. The stand-pat-in-check flaw is real but the fix needs an eval
trained or corrected for in-check positions; parked.

**029-staged-movegen.** Ran out of its 400-game cap at +9.6 ± 25.0, llr −0.05 —
inconclusive, plausibly positive, and the least informative ending possible.
Retest with 800–1000 games; movegen is 46% of node time, so the mechanism is there.

**030-hygiene.** PASS on SPRT[-25, 0] at 246 games, +15.5 ± 30.5, llr +2.95 vs
bound 2.94, crash gate clean. Promoted (e96222f, backup taken). It is a not-worse
result, not a +16 Elo result — and it is a three-change bundle, against our own
rule; risk accepted because all three are correctness fixes.

**Gate.** ruff/mypy green, submission.zip 10,482,842 bytes with HYGIENE on. Do not
upload it: the shipped time manager is the one that bottomed at 1.2 s. Before
4 Sep: fix TIME_V2 reserve + re-replay, then one 120 s + 0.5 s match against the
pre-512 champion, then a 500-game crash hunt.

## 2026-09-02 — the day session: clock, buckets, and the list worked in order

Verdict: two promotions (032-time-v2-lowclock, 034-buckets), three rejections or
inconclusives, and the first validation of the whole chain at 120 s + 0.5 s.

**Clock (item 1).** 031 (reserve 0.15) bottomed at 4.4 s under a 1.5x charge --
lower than reserve 0.10, so the reserve was never what set the floor. The floor was
an equilibrium: below 5 s the old rule credited half the increment, and 1.5x that
spend equalled the 0.5 s income at ~4.5 s. 032 adds LOW_CLOCK = 15 s below which the
budget is remaining/30 with no credit; floor 9.4 s charged, longest move 16.1 s, no
flags. 031 vs champion in the real harness at 120 s: 48.8% over 40, -9 +/- 77, no
failures. Promoted cce7b3c. The champion itself beat the pre-512 build 57.5% over 40
games at 120 s (+52 +/- 80), so the promotion chain holds at the real control.

**Staged move generation (item 2).** 800-game budget, REJECT at 245 games,
-18.5 +/- 26.8. Two readings at 8 s now agree (400-game run: +10 +/- 25). The
is_legal check and the generator overhead cost more than the skipped generations
save at this depth. Parked.

**Output buckets (item 3).** Eight heads by piece count, warm-started from the 512
net (every head initialised to the old head; verified identical to 1e-6). 16 epochs
on both 145M shards at lr 1.5e-4: val 0.005177 -> 0.005025, still falling. Engine
side reads either layout; head selection verified against a checkpoint with heads
offset by 20 cp steps. SPRT[0, 20]: INCONCLUSIVE at 600, +150 =323 -127, +13.3
+/- 19.2. Promoted dac2dae on the not-worse bar plus the independent loss gain and
zero inference cost -- stated as such in the commit, not as a [0, 20] pass.
Node rate 101 knps, unchanged within noise.

**LMR at 30 s + 0.3 s (item 4).** Ported onto the bucketed champion. INCONCLUSIVE
at 300 games, +11.6 +/- 27.8. Not promoted; the third inconclusive reading for LMR.

**Book pruning by child evaluation (item 5).** build_book --max-drop 30 pruned
84,911 of ~459k moves (18%); mainlines intact, Stafford-style traps reduced to the
refutation. SPRT[-10, 10]: exactly 50.0% over 400 games. Not promoted; the paired
openings start at ply 6-10 so the book barely features in the rig. Kept as an option.

**Crash hunt (item 6).** 500 games vs random at 4 s + 0.05 s: +498 =2 -0, zero
failures. (First attempt stopped at 51 games because SPRT[0, 20] accepted; rerun
with unreachable bounds.)

**Process.** Matches at 5 workers ran alongside GPU training without flags; the
loader-bound trainer slowed from 235 s to 480 s per epoch while sharing the CPU.
The five-hourly Claude loop stayed disabled all day; re-enable it when the machine
is not being used for matches.

**Final validation of the shipped build (buckets + TIME_V2 + HYGIENE) at 120 s.**
Stress replay charged x1.5: 0 flags, lowest clock 10.0 s, longest move 15.2 s.
40 games vs the pre-bucket champion in the real harness: 48.8%, -9 +/- 73, no
failures. The bucket net is neutral within noise at the real control; the 8 s
evidence (+13 +/- 19 over 600) remains the only positive strength reading for it.
submission.zip (11.3 MB compressed) is built from this build and is the upload.
