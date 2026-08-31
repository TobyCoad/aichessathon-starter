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
