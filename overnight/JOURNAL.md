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

## 2026-09-02 evening — the compiled board

Verdict: PROMOTE candidate (040-fastboard), pending the 120 s match and crash hunt.

**King zones (037).** REJECT, -19.1 +/- 27.3 over 312 games, despite validation
loss 0.005025 -> 0.004843 (-3.6%). The eight-block first layer (12.6 MB) misses
cache: 81 knps against 92 on an idle core, and that 12% outweighed the evaluation
gain. A 4-zone variant is the fallback if the idea is revisited.

**LMR (036), 1000-game budget.** REJECT at 729 games, -11.5 +/- 17.3. Closed.

**fastboard.py.** A numba bitboard board: legal generation with pins and check
masks, make/unmake with an undo stack, polyglot Zobrist keys, null moves, move
ordering in-kernel, the first-layer accumulator updated inside make. FastEngine in
agent.py is Engine's search over it; python-chess keeps the root and checks every
move before it leaves, and any exception falls back to Engine for that move.

Tests, all green: perft exact on the six standard positions; differential fuzz
against python-chess on 5,006 real positions and 149,358 random plies (moves,
captures, keys, check status, round trips) with zero discrepancies; fused
accumulator identical to the engine's own; static evaluation identical on 150
positions; depth-4 best move agrees 72/80. Node rate 241 knps against 106 (2.3x);
cold import 7.0 s. A head-selecting evaluation kernel removed per-call slicing of
the bucketed heads and gave the old engine +25% too.

Gates: clock replay at 120 s + 0.5 s charged x1.5, 0 flags, floor 10.1 s, longest
move 16.9 s. Crash gate 24/24. SPRT[0, 20]: **PASS at 96 games, +35 =50 -11,
+88.7 +/- 49.1.** The largest single gain since the search package.

Process note: a closed laptop lid suspended the first validation run mid-flight
and produced six fake flags and seven fake init failures; the rerun on an awake
machine was clean. Sleep is the one failure mode no test in this repo can survive.

## 2026-09-02 night — upload validated, king zones revisited, contempt built

**Compiled board shipped.** 040-fastboard promoted (94744cb): SPRT PASS +89 +/- 49 at
8 s; 120 s match +17 =19 -4 (66.2%, +117 +/- 82), 500-game crash hunt +500 =0 -0,
clock replay floor 10.1 s. Independent code review found nothing critical; its
three fixes landed. Fresh-context rules judge and reliability auditor both said
upload as is (66 awkward-position probes legal, 490k lockstep nodes vs python-chess
exact, four fallback tiers verified, RSS peak 0.5 GB). **Platform validation
passed**: `compiled board: on` in both smoke games, both won by checkmate in 20
plies, upload 16.85 MB.

**Tournament control, 30 games each, opponents on 500 ms increment:**
vs Stockfish skill 10: +10 =12 -8, 53.3%, +23 +/- 98, 11 repetition draws.
vs Weiss d8: +24 =5 -1, 88.3%. PGNs under overnight/pgn/, viewer testing/viewer.py.

**King zones on the compiled board.** The 8-zone net that lost -19 +/- 27 on the
python-chess engine PASSED on the compiled one: +144 =342 -104 over 590 games,
+23.6 +/- 18.8. Promoted (weights only). The 4-zone net (val 0.004889) trained
after a CUDA index assert exposed that the zone map was hard-wired to eight; the
map is now selected by zone count in trainer, features, engine and board, and
cross-checked. Its match against the 8-zone champion runs tonight.

**Strength assessment (fresh context):** bracket 2300-2500 CCRL-like, centre ~2400;
the ladder's 2700 is Stockfish's recalibration scale, not CCRL. Field estimate:
top-48 ~97%, top-10 ~80%, top-3 ~40-50%. Main Swiss-relevant weakness: draws by
repetition from equal or better positions, zero contempt. Built as CONTEMPT switch
(level -10, ahead -25..-50 toward ply 300, behind +20); tested vs weiss-d6.

Also: arena saves PGNs by default; get_move coerces the clock argument; the zip
target includes fastboard.py.

## 2026-09-03 — night 2: futility promoted, three switches settled, upload refreshed

Verdict: clean night, one promotion (042-futility), zero failures across ~1,200 games.

**042-futility.** PROMOTE (56e01da): SPRT[0, 20] PASS at 217 games, +48 =143 -26,
+33.9 +/- 24.8, llr +3.09. The mirror of RFP finally lands. At 120 s the final
build scored exactly 50.0% against the night-start build, so the 8 s gain does
not visibly survive the control -- the familiar shrinkage, promoted anyway on
the SPRT.

**043-tt-age.** INCONCLUSIVE at the 600-game cap: exactly 50.0%, -0.0 +/- 18.0,
llr -2.37 leaning reject. The once-a-minute full TT clear costs nothing at this
node rate. Closed unless time controls lengthen.

**044-pvs.** REJECT at 356 games, -12.7 +/- 24.0, llr -3.03. PVS without LMR buys
null-window re-searches with weak ordering to justify them. If revisited, bundle
with LMR as one challenger.

**039-kz4 vs kz8.** INCONCLUSIVE at the 600-game cap, +13.3 +/- 19.0 for 4 zones.
The smaller net (7.35 MB vs ~12.6) trending positive is the night's loose end;
rerun at 1,200 games.

**041-contempt.** Uninformative, not rejected: both builds beat weiss-d6 at 94-96%
with exactly 4 repetition draws each -- the metric contempt exists to move did not
move. Retest against the champion at a real control, after reading the drawn PGNs.

**Final gates.** Clock replay x1.5: 0/6 flags, floor 10.4 s, longest move 16.8 s.
120 s vs night start: +9 =22 -9. submission.zip 21,624,823 bytes -- 5 MB over the
16.85 MB validated upload; check the platform cap before replacing seat 1. Seat 2
keeps the 2 Sep build.

Process: the orchestrator waited 1.3 h for the evening contempt run to free the
machine, and the kingzones stage only verified a promotion (b98a213) made before
night2 started. Recurring: 50-62% of self-play gauntlet games end in threefold
repetition, throttling every SPRT the rig runs.

## 2026-09-03 morning — contempt measured against Stockfish and promoted

The Weiss d6 test was uninformative (every ahead-draw a perpetual check). Against
Stockfish skill 10 at 20 s + 0.2 s, 51 games a side on the same openings:
champion +15 =28 -7 (58.0%), 28 draws, 13 from ahead of which 9 chosen;
contempt +21 =18 -12 (58.8%), 18 draws, 7 from ahead, all chosen. Same score, ten
fewer draws, more decisive games both ways. Three flags in the challenger's run
(two ours, one the opponent's) all ended at 08:22:32 in three concurrent games:
a machine stall, not the engine. Clock replay x1.5 clean, floor 10.0 s.
Promoted d029237. New tool: testing/draws.py classifies drawn games by who was
ahead at the end. submission.zip rebuilt (21.6 MB, 27.9 MB unzipped): compiled
board + 8-king-zone net + futility + contempt. Not yet uploaded.

## 2026-09-04 — the ladder opens: two losses read, a post-mortem harness, the clock reworked

Rounds 1-3 won by checkmate (Sunfish, Imperial Knights, Danya's Disciple); round 3
showed contempt deviating from a repetition while winning. Round 4 lost as Black
to Checkers, round 5 as White to Blunder Buss.

**testing/postmortem.py** (new): detects our side by replaying the engine, scores
every position with a reference engine, flags our losing moves and classifies
each as book / time / horizon / evaluation / search, with per-game HTML curves
and a cross-game table. Round 4: -140 by our 11th move, 17...Kh8 (-110, horizon,
played in 1.6 s) and 21...c6 (-119); our static evaluation 100-200 cp too
optimistic throughout (exchange up vs bishop pair and passed pawns); clock 17 s at
move 40, thirty moves at half a second. Round 5: +127 before 14.bxc6, -350 after
(horizon; the engine finds Nxh3 today at any budget); the platform spent 9.2 s on
it. Instrumenting the driver showed why: a table warm from the previous move
makes iterations 1-8 finish in milliseconds, the cost predictor launches the next
depth blind, hits the hard cap, and the move comes from the shallow verdict.

**TIME_V3** promoted (cab76dd): expected moves 46 - 0.4/move floored at 26, and an
extension to 2.5 soft budgets when the best move or score changes between
iterations. Profile at 120 s: 62-80 s left at ply 30 (was 36-64), floor 12.5 s,
longest 9.6 s; x1.5 charge floor 9.8 s; 40 games at 120 s vs the previous build
+10 =27 -3 (58.8%), no failures. Zip rebuilt for the next upload.

**TIME_V4** first form (prediction floor + keep a proven-better root move) 46.2% at
120 s (-26 +/- 80): the floor stops iterations a warm table would have finished.
Reduced to the proven-better rule alone (047-time-v4b), under validation.

**TIME_V4 (reduced) promoted** (b014d7e): keep a root move proven better in an
unfinished iteration. 40 games at 120 s vs the TIME_V3 build +10 =23 -7 (53.8%),
no failures; replay floor 10.0 s. Zip rebuilt 12:35Z with V3 + V4: the build to
upload after the next round. testing/fetch_games.py pulls our rated games from
the public team page (no API exists; uploads are dashboard-only) and post-mortems
the non-wins; an hourly self-paced check at :15 runs it and reports.

## 2026-09-04 afternoon: round-8 loss, correction history

Round 8 (White vs No More Ammo, v3 build) lost in a rook-and-knight ending, moves
33-42. Static eval there was +400..+1010 against a reference of -12..-477, but our
own search scored the same positions -37..-164, and the 1-zone backup net is just as
optimistic statically -- so not a king-zone defect; a depth-8 search trusting a wrong
static score through RFP/futility. Harness fix 965cf0e: "evaluation" now compares a
3 s search of ours with the reference instead of the static score.

Two switches added, both off in the champion:
- CORRECTION (8cc308b): pawn-structure correction history (2 x 16384, grain 256,
  weight min(d+1,16)/256, cap 400 cp), applied to RFP, futility and the quiescence
  stand-pat. Sanity at the round-8 positions: finds Kf1 at move 39 (was Kg3), drops
  Ne5 at move 42; ~7% slower. Gauntlet 048-correction SPRT[0,20] running.
- TT_EVAL (9af1f44): a bounded transposition score replaces the static score for
  pruning when its bound allows. To be gauntleted after 048.

048-correction REJECT -137±65 (56 games); 048b (cap 100, scale 1024, quiet-only,
no QS) REJECT -33±33 (208 games). Correction history CLOSED: the net's errors are
too large and too position-specific for a pawn-structure bias to help.

Round 9 (White vs FuzzyBot, v4 build): DRAW by our own perpetual from +166.
29.Rxf6?! gxf6 30.Qh5 ... The search scored Rxf6 at +526 (3 s), +307 (10 s and 30 s,
depth 10) with pv Rxf6 gxf6 Qh5 Kg7 Qg4+ Kh8 Qh4 Kh7: the net scores the exposed
black king at +300 when the attack has no follow-up (reference ~+50 → perpetual).
Then, down the exchange, contempt made the draw welcome. Same family as rounds 4
and 8: exchange-imbalance / activity positions overvalued by the net, and more
time does not fix it. fetch_games now passes the site's colour to the post-mortem
(side detection had misread round 9 as Black).

049-tt-eval: INCONCLUSIVE +4.6±18.2 over 600 games (50.7%) -> not promoted.
Endgame fine-tune (net_w512-b8-kz8-eg, endgame-weighted shard 108M, lr 3e-4):
no epoch beat the starting net on the endgame-weighted validation (0.005898 both)
-> closed. Net bias diagnostic (scratch bias.py on the validation shard): no
material-class bias (all within ±10 cp, Q-vs-R+minor ±35), but |err| by piece
count: 29-32: 50, 25-28: 72, 17-20: 151, 13-16: 194, 9-12: 290, 2-8: 684 cp.

SEARCH DEPTH PROJECT started (overnight/SEARCH_PLAN.md). Stage 0: testing/bench.py
(40 positions; baseline d5 = 990,248 nodes, overnight/eval/bench-baseline-d5.json).
Stage 1: fastsearch.py = negamax+quiescence numba kernels, array TT (2^20), objmode
clock poll; agent.py COMPILED_SEARCH switch (off) + FastEngine.root_search/prepare.
First result: identical scores and node counts to the Python search at d1-6 with
the TT off, but only ~1.8x faster -- the evaluation kernel dominates the node.

Stage 1 gate: check_fastsearch PASS (140/140 identical with TT off at d4; 40/40 best
moves with TT on at d6, node ratio 1.00). Idle bench d6: champion 225 knps, compiled
380 knps (1.7x). Micro-costs inside numba: evaluate 1.58 us (float, vectorised;
int16 naive was slower), make+unmake 0.41, gen_legal 0.19. Stage 2a (bef420b):
packed 2-array TT + lazy stable move picking, still exact. Stage 3 switches
(183161a) PVS/LMR/LMP in the kernel, off by default, flags-off kernel exact.
Fixed-depth-6 nodes vs 050: PVS 0.96x, LMR 0.14x, LMP 0.38x. Gauntlets queued:
050 vs champion, then 052-lmr, 051-pvs, 053-lmp each vs 050.

050-compiled-search: PROMOTE +67±40 over 110 games (59.5%), crash gate clean.
Clocktest (120s+0.5s, x1.5 +20ms, 6 games): 0 flags, lowest clock 9.7 s, longest
move 14.4 s -> PASS. COMPILED_SEARCH promoted (e41ddcb); zip built 17:36 with
fastboard.py + fastsearch.py (27 MB unpacked, import 17.6 s, "compiled board: on"),
handed to the user for upload as v5. Stage-3 gauntlets running vs 050.

Round 11 (Black vs SaucyBeans): DRAW declared by the referee with mate on the board
for us (+2000). harness/referee uses board.outcome(claim_draw=True): python-chess lets
the side to move claim as soon as ONE legal move would make a third occurrence, so
after 39.Ke1 (two positions had occurred twice) the game ended before we chose.
Our search only scored the THIRD occurrence as a draw. Fix: REPETITION_TWOFOLD
(abbb3e9): any position seen once before in the game is a draw in the search
(_REPEAT_LIMIT). Replay with the switch: 37...Bg4/Bg3+ and 38...Bg3+/Nh4 instead of
the repeats. Challenger 057-twofold queued after 052b-lmr. Also explains the
50%+ repetition-draw rate in every self-play gauntlet (same referee rule).
052-lmr PROMOTE +47 over 232 games vs 050. Review agent: no correctness bugs;
three efficiency fixes applied (41fa599). night3.sh running.

v5.5 (a3c6cfe): COMPILED_SEARCH + LMR (reviewed variant) + REPETITION_TWOFOLD.
Clocktest 6 games x1.5: 0 flags, lowest clock 10.6 s, longest move 13.0 s -> PASS.
Zip built 18:40 (27 MB unpacked, import 21.5 s under load), handed to the user.
night3.sh + night3b.sh (assemble v6 from verdicts, gate, report) running.

NOTE 056-kz32 "PROMOTE +70": VOID as a net test. The kz32 training never beat its
warm start, so its net.npz is the kz8 net tiled (verified block-identical by the net
review agent); the challenger dir was built from the tree AFTER v5.5's switches, so
+70 over 136 games = v5.5 engine vs v5 (050). The retrained net is 059-kz32b, judged
vs 058-v5.5 (same engine, old net). 057-twofold and night3b gates run vs 058-v5.5.

Research agents (Opus, 4 Sep night): landscape = pondering is allowed and unused
(rules: "your process keeps its core while the opponent thinks"; runner keeps the
process alive) -> PONDER switch built. Search review = double null move restores
the key and fakes a repetition, so NMP was inert at depth >= 6 -> NMP_GUARD switch;
also halfmove>=4 guard on repetition scans (exact). Net review = the endgame error
is mostly a sigmoid artefact in cp; the real weak cell is near-equal positions at
<=16 pieces (5% of data, 10x worse mse) -> per-sample loss weighting; make_full
rebuilds both accumulators on a zone crossing (only one is stale) -> fix before
judging 32 zones; int8/int16 heads measured slower again (closed); 16 zones over 32.

4 Sep late: 057-twofold PROMOTE +66 over 160 games (vs 050). Review findings applied
as switches (all off in the tree): PONDER (nogil kernels + C_STOP; functional test:
892k nodes pondered in 4 s, correct predicted reply, stops on request), NMP_GUARD
(double null fakes a repetition: 35/3598 nulls at d7, -2% nodes), RFP_PHASE
(piece-count-scaled margins), IIR, BOOK_ENABLED. Exact speed (92720d0): eval scratch
buffer + 4-wide blocked head = +28% knps at identical nodes; one-sided zone rebuild
(check_nnue 10/10 exact). Trainer: --weight-endgame, strata printout, LR warm-up,
--warmup-epochs. 16-zone map added. testing/endgame_suite.py (400 positions, SF d18)
and GAUNTLET_OPENINGS=platform (80 curated FENs). V7 plan: overnight/eval/V7_PLAN.md.
Queues: night3b (v6 gates vs 058-v5.5) -> queue5 (061 ponder, 064 rfpphase, 065 nobook
on platform openings, 063 pvs, 066 iir, 062 nmpguard; each vs 060-v6) ; queue6 after
kz32b (kz8c control, kz8w weighted, kz16; export/check/suite/gauntlet each).

5 Sep 00:06 RFP_PHASE CLOSED on the endgame suite (400 positions 5-16 pieces, SF d18,
2.5 s/move): baseline 7.0 cp mean loss (5-8: 4.8, 9-12: 8.9, 13-16: 7.0); margins
x(0,300,200,160) 11.6; (0,200,150,130) 15.8; (100,150,130,115) 7.6; (100,130,120,110)
7.5. Wider margins cost depth in the 9-12 band every time. The pruning margins are
not the endgame problem. Switch left in the code, off; dropped from queue5.

5 Sep 03:25 v6 BUILT (4df72cf): compiled search + LMR (reviewed) + ASPIRATION + SEE +
REPETITION_TWOFOLD + exact speed patches, old 8-zone net. Gates vs 058-v5.5: SPRT
PROMOTE +90 over 167 (62.6%); crash hunt 500 games, 0 failures (62.7%); clocktest
0 flags, lowest 10.2 s; 120 s 40 games 56.2%. Zip 27 MB unpacked, import 25 s under
load. 059-kz32b (32 zones, val 0.004690 vs 0.004843) INCONCLUSIVE 52.2%/600 vs the
v5.5 engine with the old net; endgame suite 8.1 vs 7.0 cp -> not shipped.
Handed to the user. queue5 (v7 switches vs 060-v6) started 03:25 with 061-ponder.

5 Sep 07:46 v7 BUILT (a7cf304) = v6 + PONDER. Evidence: 061-ponder PROMOTE +109 over 92
games vs 060-v6 at 8 s (65.2%, no failures, 6 workers); quick 120 s check 4/4 games no
flags (separate processes); crash gate 24 clean. Full gates (night4.sh: SPRT vs v6,
200-game hunt, 40 x 120 s) running behind the upload. queue5 verdicts vs v6: nobook
INCONCLUSIVE 51.4%, pvs REJECT, iir INCONCLUSIVE 52.4%, nmpguard stopped at 406 games
(-9 +/- 28). Nets: kz8c val 0.004700, kz8w restored initial (selection on the plain
loss), kz16 val 0.004659 (best; strata improved in every band) -> 072-kz16 gauntlet
after night4 as the v7.1 candidate.

5 Sep 07:50 PONDERING CLOSED. v7's validation log: "your process is suspended while
your opponent moves, so anything you compute between your own moves does not run".
The rules page's "keeps its core" is not what the platform does. v7 reverted to v6's
engine (6f6398d, PONDER off; zip byte-identical to 4df72cf's build) and handed back
for upload. Platform facts from the log: init 35.7-38.6 s of a 90 s budget (their box
~1.8x slower than ours), slowest smoke move 8.2-8.6 s, "compiled board: on".
night4 gates abandoned; queue7 (16-zone net vs v6) proceeds as the v7.1 candidate.

5 Sep 10:20 v7.1 = v6 engine + 16-zone net (kz16: 4 months, val 0.004659, strata
improved in every band; 072 gauntlet +30 +/- 25 over 479 vs v6, still running; endgame
suite 7.4 vs 7.0). Net promoted in the tree (weights/net.npz, float16 W1). Upload build
submission-v71diag.zip = v7.1 + PONDER + PONDER_DIAG (rated match logs capture stderr:
one line per move with the ponder gap and node count answers whether the platform runs
us between moves). Quick 120 s check 4 games clean. v8 switches (HISTORY2, TT_KEEP,
QS_CAP 14, SAFE_BITS, BOOK_VERIFY) queued vs 072-kz16 (queue8). Round 17 draw: book
moves 7-10 so the clock probe never fired; the v2 probe keys on searched moves.

5 Sep 10:10 094-bookverify REJECT -94 +/- 53 over 76 games on the platform openings
(vs 072-kz16). Verifying book moves with a search costs clock and the low-depth search
disagrees with the book too often; the book stays as is. 090-history2's first run
failed its crash gate on init timeouts (3 jobs sharing the machine); rerun queued.
Desktop worker live (E:/dev/aichessathon-starter): 091/092/093 vs v7.1.

5 Sep 10:30 PONDERING CLOSED DEFINITIVELY: round-18 match log ponder-diag lines show
6-11k ponder nodes for every gap up to 9.6 s -> the process is frozen while the
opponent thinks. Init with the ponder engine + 16-zone net was 50.1 s of 90 on their
box (v6: 36-39 s). Upload for 12:00 = v7.1 clean (16-zone net, all switches as v6).

5 Sep 11:00 (loop iter 1) ROOT_ORDER implemented (backlog 3): from depth 2 the root
moves are sorted by the previous iteration's scores (stable; unreached moves keep
their tail order; book/hash move still leads) instead of re-running order_moves each
depth. Off in the tree; ruff, mypy, check_fastsearch 70/70 exact all PASS. Measured
through the real choose() loop with a depth cap: nodes to depth 8 over 20 bench
middlegame/tactics positions +3.5% overall (per position 0.63x-3.06x, same best move
20/20) -- weakly negative on nodes, SPRT will judge. Queued as 098-rootorder in
overnight/laptop/tasks.json (laptop queue was empty; desktop holds six). Laptop
worker confirmed up; it waits for the 090-history2 rerun gauntlet to clear. No new
verdicts this iteration: desktop mid-run on 091-ttkeep (46 games at 10:30 heartbeat).

5 Sep 11:08 091-ttkeep (TT age handicap) stopped at 108 games, -32 +/- 52, recorded
INCONCLUSIVE leaning reject, to start the self-play labelling pilot on the desktop
(gen-001, gen-002: 3000 games each, 40 ms/move, Stockfish 5000 nodes).

5 Sep 11:45 (loop iter 2) TT_BUCKETS implemented (was backlog 3): the table as
pairs of slots -- the even slot keeps the deeper entry (replaced on key match,
age, or equal-or-greater depth), the odd slot always takes the store, probes
check both. Off in the tree; ruff, mypy, check_fastsearch 70/70 exact all PASS.
testing.bench depth 8 over 40 positions: 1,605,439 nodes vs 1,605,437 off --
exactly neutral, as expected: 1.6M nodes against a 4M-slot table has no eviction
pressure, so only long searches (saturated table, later game phases) can show
the gain; the second probe costs ~5% knps (197 -> 187, part noise). Queued as
099-ttbuckets after 098-rootorder in overnight/laptop/tasks.json. No verdicts
landed: desktop mid-run on 091-ttkeep (108 games at the 11:01 heartbeat, llr
-1.21, trending REJECT -- consistent with ttkeep and ttbuckets attacking the
same replacement question from opposite ends); laptop still on the 090 rerun.

5 Sep 11:35 090-history2 rerun stopped at ~330 games (+12 +/- 31, inconclusive) so the
laptop can run 100-v8all next: v7.1 + HISTORY2 + ROOT_ORDER + TT_BUCKETS + QS_CAP 14 +
SAFE_BITS + ASPIRATION_WINDOW 15 + SEE_MAIN, one SPRT vs v7.1 to see the overall
effect before the single-switch verdicts finish.

5 Sep 12:20 (loop iter 3) LMR_AGGRESSIVE finished (was backlog 0, found half-done
and uncommitted in the tree with 101-lmraggr already queued): reduce plain quiets
from the SECOND searched move with the steeper log(d)*log(m)/1.8 + 0.5 table,
adjusted by butterfly history (-1 ply above +8000, +1 below -8000, clamped at 0,
reduced depth never below 1), PVS forced on inside the same switch. Off in the
tree; ruff, mypy, check_fastsearch 70/70 exact all PASS. Bench: 1,471,277 vs
1,605,437 nodes to depth 8 over 40 positions (0.92x); depth 10 over 12 positions
0.95x -- a real but modest cut, not the hoped <0.6x: at bench depths the steeper
table only adds ~1 ply and the PVS re-searches take part of it back. Queued
challenger dir verified byte-identical to the tree + flip; SPRT will judge when
the worker reaches 101-lmraggr. Verdicts recorded: 091-ttkeep INCONCLUSIVE
leaning reject (stopped at 108 games, -32 +/- 52) moved to the closed list;
gen-001 done on the desktop (281,853 positions), gen-002 running; 100-v8all at
167 games llr +1.01 (+34 +/- 43), trending pass but undecided.

5 Sep 12:55 (loop iter 4) TIME_V5 implemented (was backlog 3): expected-moves
floor 26 -> 18 in _budget_v2, paired with a refund in the iteration stop rule --
after two consecutive completed iterations that kept the same best move with no
score drop, the next iteration is allowed 1.0 soft budgets instead of 1.5. Off
in the tree; ruff, mypy, check_fastsearch 70/70 exact all PASS (the kernel is
untouched). Offline schedule model at 120 s + 0.5 s inc: the floor alone only
binds after move 50 (worst case near-identical to the champion), the refund is
the real effect -- with 60% of moves stable the clock holds 35.6 s at move 60
vs 25.6 s today and move 80 gets 1.06 s vs 0.84 s; no variant dips below the
~15 s LOW_CLOCK equilibrium. 8 s play is byte-identical (the floor never binds
below LOW_CLOCK), so the gate is 102-timev5-clock (clocktest, 1.5x charge) +
102-timev5-120s (40 games at 120 s, platform openings), both appended to the
laptop queue (4 pending there vs 7 on the desktop); the fixed-movetime endgame
suite cannot see a budget change and is waived. No verdicts landed: desktop on
gen-002 (300/500 at the 12:38 heartbeat), laptop on 100-v8all, now 314 games at
llr -0.00 (+10 +/- 31) -- the early +1.01 has washed out, heading inconclusive.

5 Sep 13:45 (loop iter 6) LAZY_ACC implemented (was backlog 3): the accumulator
update moves out of make into the first evaluate on the line. make under the
switch is make_light plus one undo column (U_MOVER, the moving piece's code);
sync_acc replays pending moves from the undo stack -- saving each ply's
snapshot to astack as it goes, so unmake can still restore -- before any
static eval; king moves that cross a zone boundary stay eager (the rebuild
needs its own ply's board), so a pending stretch never spans a zone change;
after unmake_null the acc-ply label is clamped back (the null child shares the
parent's board). Off in the tree; ruff, mypy, check_fastsearch 70/70 exact all
PASS, and a scratch harness held lazy==eager to identical score/nodes/best on
40 random positions (depth 4 no-TT, depth 6 TT). Bench: node counts byte-equal
(1,605,437 at depth 8; 2,085,202 at depth 10), speed 251->256 knps at depth 8
(+2%) and 262->275 at depth 10 (+5%) -- the win comes from nodes that never
evaluate (TT cutoffs, repetitions, null-move prunes), so it grows with depth.
Queued as 103-lazyacc on the desktop (shorter queue). Verdicts recorded:
gen-002 done (287,103 positions; the pilot pair now totals ~570k), v8-clocktest
PASS (0/6, lowest clock 10.0 s); 100-v8all still open at 453 games, llr -0.45.

5 Sep 14:15 (loop iter 7) Verdicts folded: v8-120s scored 67.5% (+18 =18 -4)
over 40 games at 120 s on platform openings -- formally INCONCLUSIVE (40 games
cannot close SPRT[-50,50]) but strong evidence the v8 switch group is positive
at long TC; b1-kz16 closed as a reject on val loss (0.004977 vs the champion
b8-kz16's 0.004659, GPU freed 11:44). Bad news: the 100-v8all gauntlet died
silently at ~453 games (llr -0.45); the worker saw no verdict line, discarded
the log and restarted the task at 13:49 -- the rerun is healthy at ~50 games
but the evidence is lost. A process check confirmed only one worker chain is
alive (the second worker.sh in the list is its own subshell); the 11:43 rm/cp
errors came from run_task rebuilding the challenger dir before its busy-CPU
wait during overlapping restarts. Backlog 3 started as overnight/month5.sh
(detached, idempotent): discovered fishnet-evals ends at 2025_03, so the fifth
month is 2024_11 (6.49 GB downloading) -> pack on 4 workers (the gauntlet
keeps its cores) -> retrain kz16r on five 145M shards at lr 1e-4 from the
b8-kz16 checkpoint -> export float16 + check_nnue + endgame suite into
challengers/104-kz16r. Gauntlet queueing is left to the iteration that reads
the suite. Self-play parquets deliberately excluded from this retrain (equal
shard rotation would overweight 570k positions ~250x).

5 Sep 14:55 (loop iter 8) Book rebuild started (was backlog 4). No new verdicts
this pass: desktop mid-092-qscap14 (62 games, llr -0.68, leaning negative),
laptop 100-v8all rerun healthy at 202 games llr +1.09, month5 pack running
since 14:15. Work: (1) worker.sh gained a `book` task field mirroring `net` --
a task can now swap weights/book.bin in the challenger dir, and if the named
book file has not landed in the tree yet the worker logs it and retries in
120 s rather than silently running the champion book; (2) launched
training.build_book detached on 2 workers (the gauntlet and pack own the CPU):
--min-count 20 --max-drop 10 over 60 row groups of 2025_01 ->
overnight/books/book-mc20-md10.bin, log overnight/eval/book2-build.log;
(3) queued 105-bookprune on the desktop (shorter queue, 5 pending): champion +
the new book, platform openings, SPRT[0,20] at 8 s -- the earlier
--max-drop 30 prune tested exactly 50.0% on DEFAULT openings where the book
barely fires, hence the platform pool this time; (4) added
overnight/book_coverage.py and took the champion baseline on the 80-FEN
platform pool: 610,028 entries, coverage 28/80, 2.6 moves per covered
position, mean 1.25 in-book plies from a pool start. Next iteration commits
the built book (the desktop task waits on it) and records its coverage.

5 Sep 14:55 v8 PROMOTED in the tree (seven switches flipped: HISTORY2, ROOT_ORDER,
TT_BUCKETS, QS_CAP 14, SAFE_BITS, ASPIRATION_WINDOW 15, SEE_MAIN) on the desktop's
120 s evidence (67.5% over 40 platform-opening games, clocktest PASS 0/6) with the
8 s SPRT flat; ruff/mypy/check_fastsearch 70/70 PASS with the flags on. The 100-v8all
rerun was stopped by hand at the human's request (INCONCLUSIVE line appended to its log
so the worker records it instead of retrying) so 101-lmraggr runs next vs v8 -- v9 =
v8 + LMR_AGGRESSIVE is the next target. Moot single-switch tasks removed from both
queues; 092-qscap14 (desktop, started vs v7.1) is void once the desktop pulls this.
submission-v8.zip built from the tested challenger (21.6 MB, 27.9 MB unpacked, import 31 s).

5 Sep 15:15 (loop iter 9) Book line closed. The detached rebuild finished at
15:06 (7,753,103 distinct pairs over 60/599 row groups of 2025_01; kept 31,200
moves played >=20 times, pruned 16,089 more than 10 cp below their best
sibling; 0.50 MB), but the coverage compare killed it: 7/80 platform-pool
positions covered, 1.4 moves per covered position, 0.26 mean in-book plies,
against the champion book's 28/80, 2.6 and 1.38. Scanning only 10% of the
month with min-count 20 starved the counts -- the challenger would open
book-less in ~91% of platform games, so the queued 600-game SPRT could not
resolve anything and 105-bookprune was removed from the desktop queue without
committing the book (closed on coverage, like b1-kz16 was closed on val loss).
A full-month rescan stays in the backlog at LOW priority; every book
experiment to date has been flat or worse. No new verdicts otherwise: desktop
still finishing the void 092-qscap14 (heartbeat 14:53), the laptop's
110-v85all bundle probe was in its crash gate at 15:08, and the month5 pack
was healthy at group 270/509 (15:09). Process check: one worker chain, one
month5 chain -- the laptop is at its ~12-busy budget, so no new CPU work was
started.

5 Sep 17:40 v8.5 PROMOTED in the tree and uploaded by the human (18:00 slot): 110-v85all
+36 over 477 games at 8 s, clocktest PASS; the 120 s gate (v85-120s-b) still queued on the
desktop behind 111-singular (170 games, -18 +/- 42). check_fastsearch 70/70 exact with the
five switches on. Laptop worker queue emptied (073-kz16w stopped at 14 games; the TIME_V5-only
tests dropped since TIME_V6 absorbs them). Research pass consolidated in overnight/eval/V10_PLAN.md.
Round 25 (17:00) lost as white vs the roooookkk: slow slide -55 (m21) -> -213 (m63) -> -826
(m99) while spending 1-3 s/move with 50 s on the clock; the over-banking games.md describes.

5 Sep 18:20 (loop) v9 bundle parts built: ADJUDICATION, HISTORY2_FIX, KILLER_CLEAR
as switches, off in the tree (593053f). ADJUDICATION plays the referee, not just
chess: the ply counter is pinned to match plies at the first request (round 18's
counter ran 13 ahead), the behind-side draw score ramps +20 -> +320 cp as match
ply -> 300 (a repetition near the cap is a half point, not noise), and when a
fifty-move draw is reachable before the cap the kernel's draw threshold C_HMC_DRAW
drops to hmc+16 so a horizon of non-zeroing plies scores as the draw. A smoke test
against the sed-flipped challenger verified pinning (301 on the round-18 final
position), the ramp (+320), arming (56 in the reachable case) and the
ahead/unreachable cases (100). HISTORY2_FIX zeroes quiets[ply, searched] for
non-quiet moves (the malus was punishing stale entries); KILLER_CLEAR clears
killers[ply+2] on node entry and the table between root moves. Bench at depth 8 vs
the champion's 1,491,095 nodes: adj 1.00x, h2fix 1.02x, kclear 0.97x; ruff, mypy,
check_fastsearch 70/70 exact all PASS. Queued 130-v9all on the laptop: the full
five-switch v9 bundle (tamed TIME_V6 + QS_EVAL_CACHE + these three), 500 games at
8 s vs v8.5. A parallel session tamed TIME_V6 (f8286b8) and restarted its
clocktest; if that fails, the next iteration pulls 130-v9all and re-queues it
without the TIME_V6 flip. Desktop untouched (111-singular, then v85-120s-b).

5 Sep 18:10 (loop iter 13) TIME_V6 is dead: the TAMED clocktest (f8286b8) also FAILED
-- flags 0/6 but lowest clock 2.0 s against the 5 s floor over 6 games at 120 s x1.5
charge (overnight/eval/clocktest-timev6c.log; the untamed cut drained to 1.6 s). Two
fails = closed per the don't-chase rule; TIME_V5 stays the shipped time manager and
any future time-management idea must pass a solo clocktest BEFORE entering a bundle.
Per NOTES' standing instruction the v9 bundle was re-queued without the flip as
131-v9all (QS_EVAL_CACHE + ADJUDICATION + HISTORY2_FIX + KILLER_CLEAR, 500 games at
8 s vs v8.5): crash gate already clean 24/24, SPRT running. Messy bit: the old worker
had grabbed the stale, already-pulled 120-timev6 task at 17:44 (a pre-rewrite
tasks.json read) and sat waiting on the busy slot, then launched its gauntlet the
moment the clocktest freed it -- killed the worker and both gauntlet processes,
reaped 7 orphaned pool workers left over from 073-kz16w (parent 47268 long dead;
the worker's own reap missed them) plus 3 more after the kills, and restarted the
worker clean; a third clocktest attempt (timev6d, header only) was also stopped --
TIME_V6 is closed, no more clock tests for it. Desktop untouched (111-singular
+3 +/- 39 at 222, then v85-120s-b), and v9-clocktest + v9-120s queued behind them
with the four-switch sed. Also folded: month5.sh COMPLETED 16:47 and the kz16r
retrain on 2024_11 is a null result -- initial_val == best_val 0.0046589, nine epochs
never improved on the champion checkpoint, early-stop restored the initial weights,
so the exported 104-kz16r net IS the champion net and is closed WITHOUT a gauntlet.
Consequence written into the backlog: more same-style data is exhausted; NET_V10 must
change architecture (mirrored king buckets, rebalanced output buckets, 16-out head).
Laptop now: worker (131-v9all) only. CONT_HIST build not started this iteration (the
worker rescue ate the time); it is next-step #2.

5 Sep 18:50 (loop) CONT_HIST built -- the biggest open search item (V10_PLAN #2),
one iteration after the exts groundwork. A 768x768 int32 table (2.3 MB) indexed by
(previous move's piece*64+to, this quiet's piece*64+to): added to quiet ordering
scores after fb.score_moves (a post-pass in fastsearch, so fastboard is untouched
and the killer/counter bands stay above it), folded into the LMR history term as a
continuous hist//6000 clamped +/-2 (replacing the +/-8000 one-ply step) and into
the prune2 history test, gravity-updated on cutoffs with the butterfly formula
(skipped in the singular excluded-move search), halved under HYGIENE, inert after
a null move. One new kernel argument (conthist1), C_CONT_HIST=38; the quiets[]
bookkeeping now also runs when only CONT_HIST is on. Bench vs champion: depth 8
0.890x nodes (1,327,419 vs 1,491,095) at -3.5% knps, depth 10 0.900x at -4.6% --
both inside the spec's <=0.90x target, so the ordering is genuinely working.
ruff/mypy/check_fastsearch 70/70 exact PASS. Queued 133-conthist (600 games at 8 s)
behind 132-v9core. Context folded in: the parallel session split the v9 gate at
18:28 (132-v9core = four core switches at 8 s; 131-v9all stopped at 98 games +28)
and revived TIME_V6 with final constants and a local clock replay PASS (5.8 s
lowest at 1.5x) -- its desktop v9-clocktest + v9-120s decide it, third strike
closes it for good. Renamed the desktop's duplicate trailing v9-clocktest/v9-120s
tasks to v9core-clocktest/v9core-120s (same-name tasks are skipped once a result
file exists, so the four-switch clocktest -- mandatory for shipping v9core --
would never have run). Desktop otherwise untouched: 111-singular at 272 games
(-6 +/- 36), then v85-120s-b.

5 Sep 19:30 (loop iter 3) INIT_FOLD built and verified -- the init-insurance item the
INIT GUARD rule demands before CONT_HIST can ship (platform init was 63.2 s on v8.5
against the 90 s budget). agent.INIT_FOLD (False in the tree) is the switch; fastsearch
scans agent.py's `NAME: Final = True|False` lines at import and, when the flag is on,
compiles the 18 settled switch slots as constants -- every folded read is a
`_F_X if _FOLD else ctrl[C_X] != 0` ternary, and a four-shape numba experiment confirmed
dead-arm pruning happens before typing for exactly these forms, so with the flag off the
kernel is today's byte-for-byte and with it on the settled branches never reach LLVM.
In-flight slots (QS_CACHE, HIST2_FIX, KILLER_CLEAR, CONT_HIST) and every value/state slot
stay live ctrl reads so challenger seds keep working; C_PVS folds as PVS-or-LMR_AGGRESSIVE
mirroring agent's own write; prepare() raises on any fold/ctrl mismatch as build-time
insurance. Measured back-to-back under gauntlet load: import 43.2 s tree -> 38.4 s folded
(-4.8 s, ~-8 s platform-scaled), bench depth 8 exactly 1,491,095 nodes -- bit-identical,
so it ships inside v9.1 with no gauntlet of its own. ruff/mypy/check_fastsearch 70/70 all
PASS. Meanwhile 132-v9core hit its 200-game checkpoint at +23 Elo -> PROMOTE early: v9
now waits only on the mandatory four-switch clocktest, so v9core-clocktest-l was inserted
at the front of the laptop queue (the desktop copy is hours back behind 111-singular at
352 games and v85-120s-b). Next iteration ships v9 if that clocktest passes.

**5 Sep 2026, ~19:55 (iteration: ship v9).** 132-v9core PROMOTED at checkpoint 200
(+23 Elo, +70 =73 -57, 53.2%) and v9core-clocktest-l PASSED (0/6, lowest 11.3 s), so
v9 shipped: QS_EVAL_CACHE + ADJUDICATION + HISTORY2_FIX + KILLER_CLEAR flipped True in
the tree, ruff/mypy/exact 70/70 PASS, zip built from the tested challenger
(overnight/challengers/132-v9core) -> Downloads/aichessathon-v9.zip (27.9 MB unpacked,
cold import 33.6 s under gauntlet load), CANDIDATE.md written, notify emailed. Bonus:
v9-clocktest-l (with TIME_V6) also PASSED (lowest 5.7 s) -- TIME_V6's fate now rests on
v9-120s-l (laptop, running: v9+TIME_V6 vs v8.5) and the desktop v9-clocktest/v9-120s
pair (their seds now flip only TIME_V6 against the v9 tree, which is exactly the right
test). Queue hygiene after the flip: v85-120s-b, v9core-clocktest and v9core-120s
removed from the desktop queue (seds no-op against the v9 tree -> self-play noise), and
111-singular declared VOID for the same reason (its -4.3 +/- 29 at 404 games is
consistent with champion-vs-champion; do not fold it as SINGULAR attribution). Next:
133-conthist decides the v9.1 anchor; INIT_FOLD rides along; NET_V10 work after 22:00.

5 Sep 21:55 v9.1 SHIPPED (email sent): v9 + TIME_V6, zip from the tested challenger
v9-120s-l (55.0%/40 at 120 s vs v9, clocktest PASS 5.8 s floor), TIME_V6 flipped on in the
tree, exact 70/70. 133-conthist REJECT -> CONT_HIST closed. Stockfish training data:
training/binpack_decode.py written and validated (435k sample entries exact; scale 0.45
cp/unit vs SF 17.1 depth 12, r=0.90; VALUE_NONE 32002 filtered; int16 wrap fixed; null /
marker moves 0xFFFF end a chain), decoding data/sf/test80-2024-02 (6.9 GB zst) to 580M
RECORD positions for a warm-started retrain tonight. Loop paused 21:20-22:20 on the usage
limit; desktop is off, all gates on the laptop.

5 Sep 23:45 (loop iter 9) CAPTURE_ORDER (V10_PLAN #7) built, off in the tree
(C_CAPTURE_ORDER=42, CTRL_SIZE 43). A rescore pass after score_moves computes SEE once
per non-promotion capture: SEE-losing captures drop below every quiet (ordering band
-(1<<21) + see*16, under the quiets' +/-2*HISTORY_MAX), winning/equal captures keep the
MVV-LVA band, and both get a capture-history tiebreak; the capture history (gravity bonus
on capture cutoffs, no malus in v1, >>=1 per-move decay under HYGIENE) lives in the first
4608 entries of the conthist1 buffer, indexed (attacker*64+to)*6 + victim%6 -- CONT_HIST
is rejected/closed so the buffer is free, agent raises if both switches are on, and the
kernel signature stays unchanged (no new numba specialisation). ruff/mypy PASS, exactness
70/70 + table-on 40/40 PASS. Bench depth 8: 1,574,873 vs 1,445,087 nodes (1.090x) at 378
vs 391 knps -- delayed losing captures cost ~9% nodes at fixed depth, so the 8 s SPRT
(144-caporder, 600 games, queued with caporder-clocktest-l behind the v9.2 gates; the sed
flips only CAPTURE_ORDER so a v9.2 tree flip cannot void it) decides whether ordering
pays it back. Also committed: the previous iteration's uncommitted NMP_V2 build (kernel
+ agent together, per the 22:40 rule; the exact check ran on the combined tree). State:
decode done (581M), the interactive session's v9.1 endgame-suite baseline at 300/400,
the worker holding 142-v92prune until it exits; desktop off.

5 Sep 23:30 Stockfish data: 581M positions decoded (45.6 min); the champion net's loss on the
SF validation set is 0.006879 vs 0.00465 on Lichess (endgame strata 2-3x worse) -- the
distribution mismatch is real and large. Warm-started retrain on the SF shards running
(overnight/sf_train.sh); baseline suite under v9.1 search 10.8 cp. 141-v92prune failed its
gate 19/24 on init timeouts under the decode; worker now waits for decode/suite; 142 queued.

6 Sep 00:55 (loop iter 10) v9.2 pruning pair rejected and split; NMP_V2 tests alone;
QS_TT + ASP_WIDE finished. 142-v92prune REJECT at 00:13 (40.0%, +33 =46 -61 over 140,
llr -2.18) -- so the queued trio 143-v92nmp (NMP_V2 stacked on that same pair) was dead
on arrival; killed it ~20 min in, requeued as 143-nmp (NMP_V2 alone, 600 games at 8 s +
clocktest) and the worker picked it up at 00:30 (challenger verified champion+NMP_V2
only). Split-by-bench of the rejected pair (champion 1,445,087 nodes d8): IMPROVING
alone 731,904 nodes (0.506x) with large eval swings -- it over-prunes and is closed as
the -50 Elo culprit without spending a gauntlet; CUTNODE alone 1,438,246 (0.995x,
benign) takes the one allowed requeue as 146-cutnode at the queue tail. The previous
iteration's half-done QS_TT + ASP_WIDE build was finished and committed (1472326):
ruff/mypy PASS, exact 70/70 flags-off + 40/40 table-on (the quiesce signature grew
tt_key/tt_data, so the flags-off identity mattered); bench d8 QS_TT 1,435,882 (0.994x),
ASP_WIDE node-identical to the champion (no aspiration fails on the bench suite at d8)
-- 145-v93fill decides them. Caveat recorded in NOTES: INIT_FOLD's FOLDED map predates
CTRL_SIZE 44; re-verify before it rides in the v9.2 zip. Queue: 143-nmp, 144-caporder,
145-v93fill, 146-cutnode, each with its clocktest; v9.2 ships from the union of passes.
Interactive session's SF chain is at check_nnue on 150-sfnet; GPU untouched by the loop.

6 Sep 00:50 Stockfish-data net (150-sfnet): best epoch 15, SF-val 0.003917 vs champion 0.006891;
Lichess-val 0.011299 vs 0.004633 (forgot the human distribution); suite 10.8 = champion 10.8
(better at 9-12 pieces, worse at 13-16). Gauntlet queued next. Mixed-data net 151-mixnet
training (SF + Lichess shards interleaved, warm start from the champion).

6 Sep 01:25 (loop iter 11) INIT_FOLD caveat cleared; SEE_QUIET built and queued.
The v9.2 blocker from iter 10 is closed: the initfold scratch was rebuilt from the
current tree and benched d8 at exactly 1,445,087 nodes (bit-identical to the champion),
and fold + NMP_V2 matched no-fold + NMP_V2 node-for-node at d6 (279,188) -- the zip
scenario works because FOLDED holds only the 18 settled slots, so growing CTRL_SIZE
cannot break it; prepare()'s check passed in every run. Then the next-step build item:
SEE_QUIET (C_SEE_QUIET=44, CTRL_SIZE 45, off in the tree) skips a late quiet at depth
<= 6 when fb.see on the destination square loses more than 30*depth^2 (fb.see already
handles quiets -- victim 0, both sides free to stop -- so no fastboard change; guards:
not in check, searched > 0, alpha away from mate). Bench d8 1,098,731 vs 1,445,087
nodes (0.760x) at similar knps under load -- a healthy cut, nothing like IMPROVING's
0.506x; one bench score swings (pos 39 +534 -> +214, same best move), the rest hold
within ~35 cp. ruff/mypy PASS, exact 70/70 + table-on 40/40 PASS, committed kernel +
agent together. Queued 147-seequiet (600 games, 8 s) + seequiet-clocktest-l at the
laptop tail; after the fold-verification rebuild the tree copy in
overnight/challengers/initfold is current. 143-nmp was at +50 +/- 49 over 98 games at
01:00 -- NMP_V2 looks like the v9.2 anchor; its checkpoint lands ~02:00. Desktop off;
GPU with the interactive session (151-mixnet training).

6 Sep 02:20 (loop iter 14) finished and committed the eager fastboard signatures
(speed.md leftover, the previous iteration's uncommitted build): 14 leaf helpers
(lsb, msb, popcount, bit, rook/bishop_attacks, attackers_to, is_attacked,
occupancy, _add, _add_promotions, feature, _acc_row, _acc_row_one) now carry
explicit numba signatures so each compiles once instead of up to nine
specialisations per caller type mix. Source-only and exact by construction:
bench d8 is 1,445,087 nodes, bit-identical to the champion; exact 70/70 +
table-on 40/40, ruff/mypy PASS. One under-load import sample 46.0 s vs the
tree's earlier 43.2 s -- gauntlet load noise swamps the ~-3 s idle gain, so the
clean-unzip check at v9.2 zip time is the real measure. Also committed the
binpack_decode ruff annotations from the same interrupted session. State:
150-sfnet at 104 games +33.5 (checkpoint ~03:00), nmp-clocktest-l next (the
v9.2 gate), then 144-caporder / 145-v93fill / 146-cutnode / 147-seequiet.
Everything buildable is built; next iteration ships v9.2 on a clocktest PASS.

6 Sep 02:45 150-sfnet's PROMOTE was VOID: worker.sh rebuilt the challenger dir from the tree
before copying the task's net (which lived inside that dir), so v9.1 played v9.1 (+19 at 200 =
noise; the identical md5 proved it). Worker fixed (nets staged under overnight/nets/, a net equal
to the tree's aborts the task); SF net re-exported to overnight/nets/152-sfnet.npz and queued
first as 152-sfnet. 143-nmp PROMOTE +26/201 stands (real switch test).

6 Sep 03:12 152-sfnet REJECT (-76 +/- 48 at 116 games, llr -2.99): the pure Stockfish-data net is
worse in games although its loss on SF positions is 43% lower -- it forgot human positions
(Lichess-val 0.0113 vs 0.0046). Mixed net 151-mixnet (epoch 21/24) is next.

6 Sep 03:25 Scale bug: the SF-trained net's evals are 1.72x too large on Lichess positions
(slope 1.717 vs 1.014) -- the 0.45 cp/unit binpack scale was wrong; 0.262 is right. Shards
rescaled x0.582 in place; mixed net retraining as 153-mixnet2 with correct targets; the
first mixed run (polluted targets) was stopped at epoch 22.

6 Sep 03:30 v9.2 SHIPPED by the session (email sent): v9.1 + NMP_V2 from the tested 143-nmp
challenger (+26 at the 200 checkpoint, 53.5%); import 33.7 s clean. Loop iteration 15 had
timed out mid-ship. Corrected-scale mixed net (153-mixnet2) training: initial SF-val 0.005318
-> 0.002939 after one epoch.

6 Sep 04:35 (loop iter 16) finished and committed NMP_V2B, the half-done build iter 15
left in the tree (agent.py switch + kernel code were uncommitted): on a null-move cutoff
at depth >= 10 a reduced-depth real search at the same node must confirm the fail-high
before it is trusted, with null pruning disabled below ply + 3*null_depth//4 inside the
verification subtree (Stockfish's nmpMinPly zugzwang guard; C_NMP_V2B=45, C_NMP_MIN_PLY=46
as a state slot, CTRL_SIZE 47). ruff/mypy PASS, exact 70/70 + table-on 40/40 PASS with the
switch off. Benches vs the v9.2 tree: d8 1,385,489 and d10 4,950,623 bit-identical (the
guard only fires at non-root depth >= 10); d11 9,402,271 vs 9,401,418 -- +0.009% nodes,
essentially free, the verification almost always confirms. Too small to resolve in a solo
600-game SPRT at 8 s, so per the small-items rule it gets NO solo gauntlet and rides in
the v9.3 bundle SPRT (its weight is at 120 s depths anyway). New champion bench baselines
recorded (NMP_V2 on): d8 1,385,489 / d10 4,950,623 / d11 9,401,418. State: 144-caporder
at 188 games -5.5 +/- 37 (checkpoint will extend), then 145-v93fill, 146-cutnode,
147-seequiet with clocktests interleaved. Desktop off; GPU with the interactive session
(153-mixnet2). Next: fold the four verdicts, build the v9.3 bundle challenger (passes +
NMP_V2B + INIT_FOLD + eager signatures), one confirming SPRT + clocktest, ship.

6 Sep 05:25 (loop iter 17) research iteration: nothing to fold (144-caporder mid-run,
everything buildable built), so two opus agents ran in parallel. (1) rounds25-29.md:
the three newest platform games analyzed -- none is a clock loss (54/78/17 s in hand
at each turning point), which retro-confirms closing TIME_V6; round 25 shows 387 cp
mean static error at 6-9 pieces, round 29 the worst endgame eval on record (674 cp at
11-16 pieces, ply 108 static +775 vs reference -709), and round 27 is a NEW failure
mode: +141 at 27 pieces drifted to a dead draw with 53% of the clock spent on 0.00
shuffle moves ("failed to convert"). (2) endgame_shrink.md: ENDGAME_SHRINK scoped to
implementation-ready -- blend inside fastsearch.evaluate (call-site blending would
double-blend via QS_EVAL_CACHE and the TT's stored eval), pure-material baseline from
agent._MATERIAL, w 256->179 linear over 17->6 pieces with a +/-300 cp clamp and no
eval-size gate, calibrated in ~2 min against endgame_suite.json's 400 labelled
positions before any engine time is spent; verdict build-the-switch-skip-the-solo-slot
(rides in a bundle). One correction recorded in NOTES: the report cites 150-sfnet's
+19 as evidence, but that verdict is VOID (worker self-play bug). Backlog updated:
ENDGAME_SHRINK is item 4 with the rounds25-29 P1 refinements (ramp to 6, OCB damping)
folded into its calibration; drawn-position budget cap and a peak_eval postmortem
counter recorded as fillers. State: 144-caporder recovered to +5.5 +/- 28.6 at 380
(llr -0.43); its 400 checkpoint will extend to 600. Then 145-v93fill, 146-cutnode,
147-seequiet. Desktop off; GPU with the interactive session (153-mixnet2).

6 Sep 06:05 (loop iter 18) ENDGAME_SHRINK built and calibrated (backlog item 4, from
endgame_shrink.md): C_EG_SHRINK/C_EG_WMIN/C_EG_CAP (CTRL_SIZE 47->50), the blend inside
fastsearch.evaluate (which now takes bb+ctrl at all seven call sites), a simple_eval
material leaf kernel, and the mirror in FastEngine.evaluate for root contempt; off in
the tree, ruff/mypy/exact 70/70 + 40/40 PASS. The 2-minute calibration (testing.eg_calib,
log + reusable npz under overnight/eval/v10/) built the per-band static-error instrument
games.md asked for: the net is off by 673.7 cp mean at 5-8 pieces (pure material is
BETTER there: 444.8), 228.4 at 9-12, 137.3 at 13-16. The report's acceptance (>=25% fall
in 9-12 at CAP 300) was missed (-18%), but every band improves at every swept setting and
gains are monotone in aggressiveness, so defaults were set at the strongest capped point
WMIN 128 / CAP 600: 532.1 / 176.8 / 132.5 (-21% / -23% / -3.5%); uncapped is better on
mean but moves single positions up to 2682 cp (fortress risk) -- rejected. Bench d8 with
the switch ON: 1,451,077 vs 1,385,489 nodes (1.047x, the predicted pruning-margin
interaction). Remaining gate: the 17-min endgame suite on overnight/challengers/egshrink
vs 10.8/17.0/12.0/5.0 when the laptop is quiet (kz16w veto: any band >1.5 cp worse);
then it rides in the v9.3/v9.4 bundle, never a solo slot. State: 144-caporder near its
600-game end (elo ~-1.5 at 476, llr -1.26 -- likely INCONCLUSIVE-negative), then
145-v93fill, 146-cutnode, 147-seequiet. Desktop off; GPU with the interactive session.

6 Sep 06:00 153-mixnet2 trained (best epoch 21): SF-val 0.002702 vs 0.005318, Lichess-val
0.007959 vs 0.004633, suite 13.8 vs 10.8 (better below 9 pieces, worse 9-16). Gauntlet next.

6 Sep 06:55 (loop iter 20) housekeeping plus the last unbuilt filler. Found iter 19's
egshrink endgame-suite gate DEAD at 50/400 -- its background child was killed when the
iteration's session ended -- and its NOTES.md update uncommitted; committed the baton
(7ce5b19) and relaunched the suite fully detached via PowerShell Start-Process (PID
54076, log overnight/eval/v10/egshrink_suite.log; lesson recorded in NOTES: side jobs
longer than an iteration must be launched detached). Built DRAW_BUDGET (rounds25-29
P2), off in the tree: when our last six root scores all sit within +/-25 cp, the
halfmove clock is past 20, ten or fewer pieces remain and the clock is healthy, the
soft deadline is capped at max(0.25 s, 0.8x the observed increment) -- round 27 spent
53% of its clock shuffling through 61 reference-0 moves. FastEngine now persists
root_score (last completed iteration); no kernel change, so bench is identical by
construction. ruff/mypy/exact 70/70 + table-on 40/40 PASS; smoke test proved the cap
engages only with every condition true and resets on a new game. drawcap-clocktest-l
queued at the queue tail (TM ideas need a solo clocktest); Elo value +0..+6 at 120 s,
bundle filler only. State: 153-mixnet2 at 160 games +19.6 +/- 47.7 (checkpoint ~07:10),
suite at 200/400, fills 145/146/147 land this afternoon. Desktop off; GPU with the
interactive session.

6 Sep 07:15 v9.3 SHIPPED (email sent): v9.2 + 153-mixnet2 (mixed SF+Lichess net, PROMOTE +19 at
200, md5 verified different from the old net). Net promoted into the tree (weights/net.npz).

6 Sep 07:45 (loop iter 21) ENDGAME_SHRINK closed on evidence, and the v9.4 bundle queued.
The 17-min endgame suite that iter 20 relaunched detached finished at 07:02: mean 9.2 cp
against the 10.8 baseline, but all of the gain sat in one band -- 5-8 pieces 3.1 vs 17.0,
while 9-12 went 12.0 -> 17.1 and 13-16 5.0 -> 6.6, both past the 1.5 cp veto. Rather than
ship or bin it blindly I re-ran testing.eg_calib against the NEW champion net (v9.3's mixed
Stockfish/Lichess net) to see whether a narrower ramp -- an EG_ON=9 early-out keeping only
the band that won -- was worth a second suite slot. It is not, and the reason is the good
news of the day: the new net has already fixed that band. Static error at 5-8 pieces is
331.9 cp against the old net's 673.7 (-51%), and 252.8 vs 320.9 overall (-21%); pure
material scores 444.8 / 275.1 / 243.9, so it is now WORSE than the net in every band,
where under the old net it beat the net at 5-8 (444.8 vs 673.7). The blend has nothing
left to blend toward, so ENDGAME_SHRINK is CLOSED -- the switch stays in the tree, off and
harmless, and no further suite or gauntlet slot goes to it. A methodological note went into
NOTES alongside it: for 153-mixnet2 the endgame suite said WORSE (13.8 vs 10.8), the static
instrument says 21% BETTER and the 8 s SPRT said +19 Elo, so the 400-position/2.5 s suite
is a weak proxy and should be used as a veto for gross regressions only, never as a
promotion gate. Then queued v9.4 as ONE bundle, 148-v94all + v94all-clocktest-l:
CAPTURE_ORDER (600-game INCONCLUSIVE +0.6, a pass under the human's rule) + QS_TT +
ASP_WIDE + NMP_V2B, which absorbs and replaces 145-v93fill, v93fill-clocktest-l and the
now-moot caporder-clocktest-l -- four switches, one gauntlet slot instead of three. Bundle
bench d8 under gauntlet load 1,512,004 nodes at 238 knps: the ~1.09x is CAPTURE_ORDER's
known ordering cost and nothing else is pathological. INIT_FOLD and the fastboard eager
signatures ride in the v9.4 zip (exact, no gauntlet). Laptop queue: 155-mixnet2s (running,
the interactive session's net) -> 148-v94all -> v94all-clocktest-l -> drawcap-clocktest-l
-> 147-seequiet -> seequiet-clocktest-l -> 146-cutnode -> cutnode-clocktest-l. Desktop off;
GPU with the interactive session. v9.2 and v9.3 are emailed and awaiting the human's
uploads, so v9.4 is the day's third and last slot and can wait for a clean verdict.

6 Sep 07:55 (loop iter 21, correction) Benched the v9.3 champion on the SAME net to
baseline the bundle properly: d8 1,511,432 nodes at 264 knps, against the bundle's
1,512,004 at 238 knps -- 1.0004x, node-neutral. CAPTURE_ORDER's recorded 1.090x node cost
was an artifact of comparing a new-net challenger against the old-net baseline 1,385,489;
the whole +9% is the mixnet2 net changing the search tree, not the capture rescore. The
v9.4 bundle therefore carries no measured node cost at all. New champion bench baseline
recorded in NOTES (1,511,432 at d8); any switch benched before the 07:15 net promotion
needs re-baselining before its ratio is trusted.

6 Sep 08:00 (loop iter 22) Nothing to start, so I built the next filler. The laptop queue
is roughly ten hours deep -- 155-mixnet2s running at 146 games (+23.8 +/- 46.1, checkpoint
at 200 around 08:20), then 148-v94all and its clocktest, drawcap-clocktest-l, 147-seequiet,
146-cutnode and their clocktests -- the GPU is the interactive session's (156-mixnet3
training) and the desktop is off, so the only useful move was to have the v9.5 bundle's
parts ready before its slot comes round. Built ROOT_NODES, the root-move half of
V10_PLAN #12, off in the tree and touching agent.py only: from the second iteration the
root moves after the front move are ordered by the number of nodes their subtree cost on
the previous iteration, most first, with the previous score kept as the tiebreak. The
reason is that the current key is degenerate. After the first root move every move is
searched with a null window and fails low, so its score is only a loose upper bound and
most of them come back at nearly the same value; the node count carries the information
the score threw away, because a move that burned a lot of nodes is one that forced the
full-window re-search and nearly raised alpha, while a move refuted in a few hundred nodes
is junk. The counts were already being collected for TIME_V6's effort factor, so the whole
cost is carrying one dict of at most n ints from one iteration to the next. ruff, mypy and
check_fastsearch (70/70 exact, 40/40 table-on) all pass, and with the switch off the sort
order is unchanged, so the tree is bit-identical to the champion.

One caveat is worth more than the switch itself: testing.bench calls engine.root_search
directly and never runs the iterative-deepening root loop, so ROOT_NODES -- like ROOT_ORDER
and ASP_WIDE before it -- cannot move the bench number at all. Its d8 bench is 1,511,432
nodes, exactly the champion baseline. That confirms the kernel is untouched; it is not
evidence that the switch does nothing, and a future iteration should not read it that way.
I recorded the caveat in NOTES beside the switch. Since the bench is blind here I smoke
tested through get_move at 60 s on four positions (opening, a rook-and-pawn endgame, two
middlegames): sensible moves, no crash, no overrun. ROOT_NODES is a bundle filler for v9.5
alongside DRAW_BUDGET and never gets a gauntlet slot of its own, and v9.5 cannot be queued
until 148-v94all's verdict lands, because the champion moves under it.

6 Sep 08:45 (loop iter 23) The laptop is still saturated -- 155-mixnet2s at 288 games and
drifting down to -7.2 +/- 33.8, having been +69.5 +/- 60 at 76, so the slope-rescaled net
looks like it will land INCONCLUSIVE-negative at 600 around 09:35; behind it sit
148-v94all, its clocktest, drawcap-clocktest-l, 147-seequiet, 146-cutnode and their
clocktests. The GPU is the interactive session's and the desktop is off, so as in iter 22
the only useful move was to have the next v9.5 part ready before its slot comes round.
Built SINGULAR_EXT2, search.md #10 (+3..8 Elo at 120 s), the follow-up to the SINGULAR
that shipped inside v8.5.

The idea is that we currently read the singular verification search as a yes/no. If the
hash move is the only move that reaches sbeta we extend it a ply; otherwise we do nothing.
Both halves of that throw information away. A move that beats every alternative by a wide
margin is more forced than one that squeaks past, and a move that is not singular at all
but whose table score already fails high is one where the cutoff is coming anyway, so the
ply we spend on it is a ply not spent where the tree is still open. So: at a non-PV node,
a hash move whose alternatives all fall more than 25 cp below sbeta is extended TWO plies,
and a non-singular hash move with tt_score >= beta is searched one ply SHALLOWER. Both
arms live entirely inside the existing singular block, so the entry guards are unchanged,
and the double arm additionally requires two spare slots under SINGULAR_EXT_CAP, so no
line can extend further than it can today. I deliberately did not build the third arm
Stockfish has in the same block -- the multi-cut return when sbeta >= beta -- because
multi-cut is on V10_PLAN's closed list and stays there.

ruff, mypy and check_fastsearch all pass (70/70 exact, 40/40 table-on); with the switch
off extend_hash can only be 0 or 1, so the tree is bit-identical to the champion. The
bench is worth recording carefully because it nearly misled me. At depth 8 the switch is
1,503,594 nodes against the champion's 1,511,432, i.e. 0.995x, which reads as "does
nothing" -- but SINGULAR_MIN_DEPTH is 7, so at d8 almost no node has both the depth and a
deep enough hash entry to enter the block at all. At depth 10 it is 5,323,757 against
5,051,285, or 1.054x. That is the honest number, and it is the expected shape: extensions
buy accuracy by spending nodes at fixed depth and are judged at fixed time. For scale,
SINGULAR itself benched 1.55x and promoted inside v8.5, so 5% is cheap. The general
lesson, which I put in NOTES beside the ROOT_NODES bench caveat: never read the cost of a
depth-gated switch off a d8 bench. Smoke-tested through get_move at 60 s on four positions
(two book, Rd1 in a rook endgame, Bxh7+ in a middlegame): sensible, no crash, no overrun.
SINGULAR_EXT2 is a v9.5 bundle filler alongside DRAW_BUDGET and ROOT_NODES and never gets
a gauntlet slot of its own; v9.5 cannot be queued until 148-v94all's verdict lands.

6 Sep 09:25 (loop iter 24) The laptop was saturated again, but this time the thing worth
fixing was not what to build -- it was the queue itself. The human's standing instruction
since 08:55 is that v9.4 is the search bundle plus the WDL net in ONE gauntlet and that he
wants it expedited. The interactive session had already written queue_v94.py to insert
149-v94wdl at index 0 of the laptop queue so that nothing queued in the meantime could get
in front of the release. That closes the wrong half of the hole. Inserting at the front
only wins the race against tasks that have not started; the worker runs one gauntlet at a
time and will not preempt, so whatever it happens to have picked up when the net lands owns
the machine until it finishes. Sitting directly in front of 149-v94wdl were 147-seequiet and
146-cutnode, 600 games each, about three hours each.

So I bounded them rather than removing them, because idling the machine to protect a release
is its own kind of waste. The pending order is now 147-seequiet, then the two cheap
clocktests, then 146-cutnode, and 147-seequiet is capped at games: 200. It stops at its
first checkpoint whatever it reads, so the worst case block on v9.4 is one checkpoint of
roughly seventy to eighty-five minutes rather than one full SPRT. At 200 games the
checkpoint rule can PROMOTE at +10 or better and can return INCONCLUSIVE; it cannot REJECT,
which needs 400. That is enough for what SEE_QUIET actually is here -- a v9.5 bundle filler,
where INCONCLUSIVE with a positive point estimate is already a pass. If a later iteration
wants the full 600-game read it can re-queue it once v9.4 has shipped.

The window is real and I measured its shape rather than guessing. wdl_decode.sh started at
09:03, having waited for the drawcap clocktest, and binpack_decode counts in the worker's
busy_gauntlets probe, so the worker is parked until the decode ends. Then merge and twelve
training epochs sharing the GPU with 156-mixnet3, then export, check_nnue and the endgame
suite, which parks the worker a second time. 149-v94wdl realistically queues around 11:00,
and the free window is the training stretch in the middle -- which is exactly what a capped
147-seequiet fills. I put the gauntlet in that stretch rather than the clocktests on purpose:
clocktests measure time management and are the load-sensitive thing, so they belong after
the release, not underneath a training run.

Two results to fold. drawcap-clocktest-l came back PASS, flags 0/6, errors 0, lowest clock
5.7 s, longest move 11.8 s, so DRAW_BUDGET has cleared its gate and needs nothing further
before it goes into v9.5 alongside ROOT_NODES and SINGULAR_EXT2. And I verified the tree is
green rather than assuming iter 23 left it that way: ruff clean, mypy clean on agent.py and
fastsearch.py, check_fastsearch 70/70 exact at depth 4 and 40/40 best-move agreement at
depth 6 with the table on, node ratio median 1.00.

I deliberately did not build a switch. The three unbuilt search.md items left -- razoring at
depth <= 3, ProbCut, root PVS/LMR -- are all kernel edits, and the twenty minutes left after
the queue work is not enough to finish one plus ruff, mypy, the exactness check and a bench.
NOTES has carried its own rule since 5 Sep 22:40 that the tree must never be left with a
half-done build at the end of an iteration, and that rule outranks filling the slot. What I
did instead was spend those minutes scoping RAZOR against the live source so the next
iteration can write it in one pass: the exact insertion point between the reverse-futility
return and `futile = False`, the guards, the fact that `standing` may still be -INFINITY
there so it must reuse the futility block's fill-in ladder rather than adding a third eval
path, the margin table, C_RAZOR = 51 with CTRL_SIZE going 51 -> 52, and the warning that a
new in-flight slot must stay out of fastsearch.FOLDED or INIT_FOLD will silently break
challenger seds. One caveat recorded with it: unlike SINGULAR_EXT2, razoring is not gated
above depth 3, so a depth-8 bench is a fair read of it and the node count should fall -- if
it does not move at all, the guards are wrong.

Finally, round 31 landed at 08:21 this morning, a draw as Black against abhi-s-chess-demon
and the first platform game since v9.2/v9.3 went live. I delegated its post-mortem to an
opus agent writing overnight/eval/v10/round31.md, with the usual brief: read ARCHITECTURE.md
first, respect the closed list, quantify where the half point went, check the clock profile
against the now-live TIME_V6, test whether the errors again cluster below 16 pieces, and say
plainly if the game shows no new failure mode rather than inventing work. It was still
running when this iteration ended; the report is on disk for the next one to fold.

6 Sep 09:35 (loop iter 24, addendum) The round 31 post-mortem landed before I stopped, so I
folded it rather than leaving it for the next iteration. The verdict is the useful kind of
negative: no new failure mode, and the move that actually cost the half point is already
fixed by the net we shipped. The agent re-probed the position with the current tree at the
same 33.2 s clock the game had, and v9.3 plays the reference move g6g5 in 1.38 s where v9.1
needed a ten second replay. That is in-game evidence for 153-mixnet2 that is independent of
its gauntlet, which is worth more than the gauntlet on its own.

The finding I did not expect is that our headline number has moved and several Elo estimates
are now standing on a figure that is out of date. The errors still cluster below 16 pieces --
six of eight flagged moves, 480 of 781 cp, on 24% of our moves -- but the mean static error
in that band is now 136 cp, against the 475 in games.md and the 674 in rounds25-29. The shape
of the problem is intact; the size of it has collapsed. Anything in the backlog justified by
"the net's static error below 16 pieces is 475 cp" needs re-reading before it gets a slot.

Two smaller things. TIME_V6 is live but tamed -- the tree carries reserve 0.06 and LOW_CLOCK
12.0, not the 0.04 and 9 the plan specified -- and it recovered about 2.7 s of the twelve
second bank, all of which went on 106 moves the reference scores at exactly zero. No error in
the game correlates with a short think, and the hard cap at the one move that mattered was
23% tighter than TIME_V5's would have been. And the ply-300 material adjudication did not
fire: the game reached ply 323 un-adjudicated with White up K+B+P vs K+B at ply 300, which
contradicts the premise round 18 gave for the ADJ_BEHIND_LATE bias we shipped in v9. That is
a rules re-read, not a build, but if the cap is not real then the bias is paying for nothing
and should be measured before v9.5 freezes.

One trap recorded with the actionable item. DRAW_BUDGET's guards would have been inert in
this game -- pieces <= 10 and clock > 12 s overlap on about three of the 106 drawn shuffle
moves -- and widening them to pieces <= 14 and clock > 8 banks thirty to thirty-five seconds.
But DRAW_BUDGET passed drawcap-clocktest-l this morning with the NARROW guards. Widening
makes it fire far more often, so a widened DRAW_BUDGET cannot inherit that PASS and needs its
clocktest re-run before it ships. I wrote that next to the item so a later iteration does not
carry the tick across a change that invalidates it.

6 Sep 10:10 (loop iter 25) Two things were wrong when I started and both were quiet failures
rather than loud ones. The laptop worker had finished the withdrawn 149-v94wdl at 09:40 and
then had nothing at all to run: every entry in tasks.json already carried a result file and
the two live items had been moved into deferred.json to keep them from blocking v9.4. So the
machine the human is waiting on sat idle. And 147-seequiet's REJECT, which the queue had
recorded twenty minutes earlier, is not an engine verdict: it failed the crash gate 7/24 on
init timeouts while the 8-worker WDL binpack decode had the CPU, the same infra failure
NOTES already records for 140/141-v92prune. SEE_QUIET is the most interesting untested
switch left at 0.76x nodes and its clocktest had already passed; closing it on that would
have been a real loss. I re-queued cutnode-clocktest-l (ten minutes, started 09:48) and
147b-seequiet behind it, both sized to finish before the WDL net can possibly queue, with an
explicit instruction to kill 147b if 149-v94wdl is ever waiting on it.

The root cause is one regex. worker.sh's busy_gauntlets guard waits for testing.gauntlet,
testing.clocktest, binpack_decode and endgame_suite, but not for train.py or merge_mix, so a
gauntlet will happily start into a GPU training run that is still holding the CPU. Adding
those two is right and I deliberately did not do it: it would have idled the worker for the
next hour instead of running the two tasks above. It is written into "Next step" as the
first thing to do once v9.4 is out.

Then I built RAZOR, which iteration 24 had scoped down to the line. The site, the shape, the
reuse of reverse futility's standing eval and its fill-in ladder, and the guards were all
exactly as scoped and went in first time -- ruff, mypy and check_fastsearch 70/70 exact plus
40/40 best-move agreement all pass with the switch off. What the scoping got wrong was the
one thing it could not know without measuring: the margins. At the specified 240/300/400 the
depth-8 bench goes to 1,572,671 nodes against the champion's 1,511,432 -- razoring made the
tree four percent BIGGER. Widening to 500/700/900 gives 1,489,958, a 1.4% saving, and
widening further to 700/1000/1400 goes back to 1,516,189, slightly worse than not having it.
I settled the tree at 500/700/900. Node counts at fixed depth are deterministic so those
three are exact comparisons; the knps figures next to them are worthless today because the
WDL training was on the GPU throughout, and I have said so in NOTES rather than let a later
iteration quote them.

The non-monotone shape is the interesting part, because it says the cost is not just the
wasted verification qsearch. If it were, a very wide margin would fire rarely and converge
to the champion from above; instead 700/1000/1400 stays measurably worse. The reason is that
the razor return is taken before the node's TT store, so a razor that fires and then fails
verification also throws away a depth-1..3 entry the parent would have reused. Storing that
fail-low before returning is the one untried lever and it is genuine kernel surgery, not a
one-liner, because the main store at the end of the function is inline. I recorded it as the
thing to try next and recorded equally plainly that re-tuning the margins is not: that curve
has now been measured at three points and 500/700/900 is the top of it.

So RAZOR is a weak switch on the evidence I have -- 1.4% fewer nodes at fixed depth is close
to nothing -- and I have said so rather than dressing it up. It rides in the v9.5 bundle with
ROOT_NODES, SINGULAR_EXT2 and DRAW_BUDGET, it does not get a gauntlet of its own, and it is
named as the second switch to drop if that bundle fails.

6 Sep 10:45 (loop iter 26) The queue said the machine was quiet and the process list said it
was not. The session had stopped 147b-seequiet at 10:18 and moved it into deferred.json, but
seven of its multiprocessing pool workers were still alive with a dead parent, still playing
games, spawned at 10:00. Free RAM was 8.8 GB of 31.4 and the WDL trainer -- the one thing the
v9.4 release the human is waiting on actually depends on -- had gone from 115 s an epoch to
734, then 595, then 242. I ran reap_orphans; it reaped seven and the python count fell from
thirteen to six. The lesson is worth carrying: killing a gauntlet's parent does not stop its
pool, so reap_orphans belongs after every stop, and the process list is the truth rather than
tasks.json or the heartbeat file.

I then left the laptop deliberately idle, which is not the usual instinct. Every entry in
tasks.json already has a result file so the worker has nothing pending, and at roughly 115 s
an epoch unloaded the WDL run has about twelve to fifteen minutes left, which puts 149-v94wdl
in the queue near 11:10. A ten minute clocktest started now would take perhaps five minutes
off the release path for no gain, because the v9.5 bundle cannot be queued until v9.4's
verdict moves the champion anyway. Idle is the correct state until that gauntlet is running.
Two trainers are sharing the GPU -- the WDL fine-tune and, since 10:17, the twelve-bucket
NET_V10 pilot -- both session-owned; I noted them so nobody later mistakes the second for a
stray and kills it.

The build this iteration was the one concrete item round 31's post-mortem produced: widening
DRAW_BUDGET's guards from ten pieces and twelve seconds to fourteen pieces and eight seconds.
Round 31 measured the narrow version inert, overlapping on about three of the hundred and six
drawn shuffle moves, so it was shipping a switch that never fired. The change is agent.py
only, no kernel, and I moved the clock test off LOW_CLOCK_V6 onto its own _DRAW_MIN_CLOCK
rather than coupling it to a TIME_V6 constant. ruff, mypy and check_fastsearch all pass --
70/70 exact and 40/40 best-move agreement. I checked the guards by calling the function
directly instead of trusting the arithmetic: a thirteen-piece shuffle with ten seconds left
now caps the soft budget to 0.40 s where the narrow guards refused it on both counts, a
nine-piece position still caps, and twenty pieces still does not.

What matters more than the widening is that it cannot inherit its own tick. drawcap-clocktest-l
passed this morning against the narrow guards, and widening makes the cap fire far more often,
so that PASS is not evidence about the switch that now sits in the tree. I parked
drawcap2-clocktest-l in deferred.json and wrote into NOTES that DRAW_BUDGET does not ship in
v9.5 without it. The narrow version is dead either way; there is no configuration left that
the old PASS describes.

The last thing I did was chase round 31's loose end, which cost no machine time at all and
turned out to be the most important thing in the iteration. Round 31's post-mortem had
flagged that the game reached 323 plies un-adjudicated and said plainly: verify the platform
rules, do not build. I fetched the canonical source that harness/rules.py names in its first
line, twice with different questions, and got the same answer both times -- a game still
running at 600 plies is drawn, the opening position counts toward those 600, and material is
never considered. Our local harness/rules.py says PLY_CAP = 300 with the game awarded on raw
material. The copy is stale, and round 18 versus round 31 shows the platform changed under
us: round 18 really was adjudicated at exactly 300, round 31 ran to 323 and ended on
insufficient material, which cannot happen under a 300-ply cap.

That makes the premise of a feature we shipped in v9 false. ADJ_BEHIND_LATE adds up to three
hundred centipawns to the behind-side draw score on a ramp that reaches full strength at ply
300, bought entirely with the argument that being behind on material at the cap is a loss. At
600 it is a draw whatever the material, and the correct ramp value at the longest game we
have ever played is 0.077 rather than 1.0. The fifty-move plan that drops the kernel's draw
threshold arms between plies 220 and 300; under the real rule its window is 520 to 600, so it
has never legitimately fired. The ahead-side contempt has the same error, and the real rule
has a consequence nobody has modelled at all: at 600 plies a won position becomes a draw, so
the urgency belongs near 600, not near 300.

The part that makes this more than a bug report is that testing/referee.py imports PLY_CAP
from harness.rules, so our own gauntlet has been playing the 300-ply material-adjudication
game all along. Every verdict we have taken, including the v9 bundle's +23 that carried
ADJUDICATION in with it, was measured by a referee that shared the mistake. A corrected
ADJ_V2 will therefore look worse in our own SPRT while being right on the platform, and I
have written that into NOTES as loudly as I can, because it is exactly the shape of finding
that gets closed for the wrong reason. The fix has an order: give testing/gauntlet.py a
--ply-cap argument first -- testing/ is ours, harness/ is not, and referee.py already takes
the cap as a parameter -- and only then build and judge ADJ_V2. I did not start that build.
It is a contained agent.py job but it is not a fifteen-minute one, and the standing rule
against leaving a half-done build in the tree at the end of an iteration is the right rule.
