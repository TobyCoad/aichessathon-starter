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
