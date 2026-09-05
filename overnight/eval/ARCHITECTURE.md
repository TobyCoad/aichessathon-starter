# The engine and the evaluation framework -- briefing for research agents

Read this before proposing anything. It says what exists, how it is measured, what has
already been tried, and the constraints a proposal must respect. Companion documents:
`overnight/eval/V10_PLAN.md` (ranked ideas, the closed list), `overnight/eval/v10/*.md`
(the research reports: search, network, games, timeman, speed), `overnight/continuous/NOTES.md`
(live state), `overnight/JOURNAL.md` (dated history of every verdict).

## 1. Competition constraints (aichessathon.com)
- One `agent.py` plus helper modules, pure Python; numba is allowed; no compiled
  extensions we build ourselves, no chess engines or engine binaries shipped.
- 1 CPU core, 2 GB RAM, unpacked submission < 50 MB (ours is 27.9 MB).
- Clock: 120 s per side + 0.5 s per move. Init budget 90 s (all our numba compilation
  happens then: v8.5 imports in ~41 s on the laptop; the platform box is ~1.8x slower).
- The platform SUSPENDS our process between moves: pondering is impossible (proved with a
  diagnostic build). The referee claims threefold and fifty-move draws for us automatically
  and adjudicates on raw material at match ply 300 (draw only on an exact tie).
- Hourly rated ladder games 08:00-22:00 UK; up to 10 uploads a day; uploads close
  11 Sep 2026 11:00, then a 13-round Swiss on the locked build. Record before v8.5: 10-6-6.

## 2. The engine (v8.5, the champion in the tree, live since 5 Sep 18:00)
Files: `agent.py` (entry point, NNUE inference kernels, root search, time management,
book, tablebases), `fastboard.py` (numba board), `fastsearch.py` (numba search kernel),
`weights/net.npz` (13.6 MB, float16 first layer), `weights/book.bin` (Polyglot, 9.8 MB),
`weights/syzygy/` (3-4 men, 4.4 MB; `TB_MEN = 4`).

Board (`fastboard.py`): bitboards + a 64-square piece array, make/unmake with an undo
stack, incremental NNUE accumulator maintained in make (one-sided rebuild when the king
crosses a zone), static exchange evaluation `see` (swap algorithm; heap-allocates a
32-array per call -- a known cost), `score_moves`/`pick_move` (hash move, MVV-LVA,
killers, counter move, butterfly history), `zone_of` (king zones), `pawn_index`.

Search (`fastsearch.py`, compiled negamax + quiescence; all switches live in an
`int64[40]` `ctrl` array so one kernel serves every variant):
- Iterative deepening from the Python root loop (`FastEngine.choose`), aspiration
  window 15 cp (widening x4 per fail, full window after 3), root moves ordered by the
  previous iteration's scores (ROOT_ORDER), PVS null-window re-searches at the root.
- Transposition table 2^22 entries, packed into two uint64 arrays, 2-slot buckets
  (depth-preferred + always-replace), age, cached static eval.
- Pruning/reductions: mate-distance pruning, reverse futility (80 cp/ply to depth 6),
  null move (R = 2 + depth/6, not in check, non-pawn material), futility depth <= 2
  (150/300), PRUNE_V2: futility 100/ply to depth 4 + history cut below -1500*depth for
  quiets after the first move, SEE pruning of losing captures at depth <= 5, LMR from
  the second quiet with a log table steepened (LMR_TABLE_AGGR) and adjusted +/-1 by
  butterfly history above +8000 / below -8000, never into quiescence; check extension
  (uncapped); SINGULAR extensions at depth >= 7 (excluded-move re-search at half depth,
  window stored - 2*depth).
- Move ordering knowledge: hash move, captures by MVV-LVA, two killers per ply, one
  counter move per previous (from,to), butterfly history with gravity updates and a
  malus for the quiets searched before a cutoff (HISTORY2). No continuation history,
  no capture history.
- Quiescence: captures + promotions, stand pat, delta pruning, SEE pruning, cap of 14
  plies (QS_CAP), no transposition table, no checks.
- Repetition: twofold inside the search counts as a draw (REPETITION_TWOFOLD; the
  referee's automatic threefold claim made this necessary); draw score carries a contempt
  ramp toward the ply-300 adjudication when ahead only.
- Lazy accumulator (LAZY_ACC): the NNUE update is deferred to the first evaluate on the
  line; exact.
- Speed: ~195 knps single core at depth 8 on the 40-position bench; mean depth ~12 at
  4 s/move at the platform control. Node time (old profile, 256-wide net): evaluate 29%,
  accumulator 15%, board push/pop 14%, movegen 13%. Our calibration: one ply ~2.1x nodes;
  a node-rate doubling ~ +65 Elo at 8 s, ~ +32 at 120 s; lossy pruning realises ~27% of
  its node saving as Elo (LMR: predicted +175, measured +47).

Evaluation (NNUE in `agent.py`, trained by `training/`):
- (768 -> 512)x2 -> 32 -> 1, clipped ReLU, float32 inference in numba (blocked head).
  768 = 12 piece types x 64 squares per perspective (own/opponent), first-layer weights
  selected by the perspective's own KING ZONE (16 zones); 8 OUTPUT BUCKETS by piece count.
- Trained on Lichess fishnet evaluation dumps (2024_11 .. 2025_03 exist; 2025_04+ are
  404): human-game positions labelled by Stockfish at depth ~20 (1.5M nodes). 4 months
  (~580M positions) in the current net; a fifth month is being packed/trained (104-kz16r).
  Loss: MSE on sigmoid(cp/400); packed with a 0.5 quiet fraction, cp clamp 2000.
  Validation loss 0.00465; validation loss is a weak predictor of Elo (see network.md).
- Known weakness: <= 16-piece positions -- static error 475 cp vs 70 cp in the
  middlegame on flagged moves; every platform loss and draw reached that band.

Time management (`agent.py`, TIME_V2..V5 on; TIME_V6 built, off, under test):
soft budget remaining/expected-moves (floor 18) + 0.25, hard = min(12% of clock, 3 soft,
clock - 10% reserve); below 15 s the budget is remaining/30. The next iteration is
predicted (2.5x the last) and not started if it would overrun 1.5 soft budgets (2.5 when
the best move changed, 1.0 after two stable iterations). Known defect: an absorbing floor
at ~13 s that banks ~12 s per game (games.md). TIME_V6 replaces this with an observed
increment, 4% reserve, 9 s low-clock and Ethereal/Stash-style stop factors.

Opening book: Polyglot built from Lichess games; mean 1.1 in-book plies per platform game
(the platform starts from curated openings, so the book rarely matters).
Tablebases: 3-4 men probed at the root and in search.

## 3. The evaluation framework (how a change gets judged)
- Every engine change is a SWITCH in `agent.py` (`NAME: Final = False`), wired to a
  `ctrl` slot in `prepare()`, OFF by default in the tree. The tree is always the champion.
- Exactness gate: `python -m testing.check_fastsearch --depth 4 --random 30` proves the
  compiled kernel with all flags off is bit-identical to the Python reference search
  (70 positions) and agrees on best moves with the table on. Plus `ruff check` and
  `mypy agent.py fastsearch.py`. Required before any commit touching the engine files.
- Bench: `python -m testing.bench --agent <dir> --depth 8` (40 positions; deterministic
  node counts, knps). A pruning change should cut nodes at fixed depth; an exact speed
  change should raise knps at equal nodes.
- Endgame suite: `testing/endgame_suite.py`, 400 positions with 5-16 pieces labelled by
  Stockfish depth 18; mean cp loss at 2.5 s/move. Baseline 7.0 cp under v5.5/v6 search;
  NO baseline yet for the v8 search -- run it before comparing nets.
- Gauntlet: `testing/gauntlet.py`, SPRT[0, 20] at 8 s + 0.08 s, pairs with colours swapped,
  a 24-game crash gate vs a random mover first; verdict PROMOTE / REJECT / INCONCLUSIVE
  (games cap). It resolves about +/-12 Elo at 8 s; smaller effects come back
  INCONCLUSIVE and are shipped in bundles on bench evidence. 8 s and 120 s can disagree
  (v8 was +3 at 8 s and 67.5% at 120 s): long-search switches need the 120 s test.
- Platform-control tests: `testing/clocktest.py` (6 games at 120 s + 0.5 s with a 1.5x
  time charge; reports flags, lowest clock, longest move -- must be 0 flags) and 40 games
  at 120 s on `testing/platform_openings.txt` (80 curated FENs), run on the desktop.
- Machines: laptop (this one: editing, GPU training, one gauntlet at a time via
  `overnight/worker.sh laptop`) and a 16-core desktop (`overnight/desktop_worker.sh`),
  both driven by `overnight/<role>/tasks.json` -> `results/<task>.txt` through git.
  A task = `{name, sed (switch flips), kind (switch|clocktest|generate), games, workers,
  base_ms, openings, elo0, elo1, net}`.
- Post-mortems: `testing/fetch_games.py` downloads every platform game; `testing/postmortem.py`
  labels each of our moves against a reference search (delta, cause: search / horizon /
  evaluation / time) into `overnight/postmortem/*.json` + REPORT.md.
- Ship gate for a bundle: 8 s SPRT vs champion + clocktest PASS + 40 games at 120 s
  non-negative + cold import < 45 s here + unpacked zip < 50 MB. The human uploads.

## 4. Measured history (so nobody re-proposes it)
PROMOTED (Elo at 8 s vs the champion of the time): compiled search +67, LMR +47,
aspiration +41, SEE +25, twofold repetition +66, 16 king zones +31, v8 bundle (+10 at 8 s,
67.5%/40 at 120 s), v8.5 bundle +36 (LMR_AGGRESSIVE+PVS, LAZY_ACC, TIME_V5, PRUNE_V2,
SINGULAR).
REJECTED / flat / closed: PVS alone (x3), LMP (-40), correction history (x2), TT_EVAL,
RFP by phase, IIR, NMP guard, book-off, book-verify (-94), QS eval cache (exact, +2%),
check-extension cap, TT keep-deeper (-32), TT buckets alone (neutral), root order alone
(+3.5% nodes), 32 king zones, no-bucket net, endgame-weighted fine-tune (suite 9.1 vs 7.0),
1024-wide net (15% better loss, ~0 Elo), QS evasions (-18), staged movegen (x2),
pondering (platform freezes us), tighter book (coverage collapsed), int8/int16 inference
(slower twice), self-play labelling at scale (too slow for the deadline; 570k labelled
positions exist on the desktop but are not in the repo).

## 5. What a useful proposal looks like
State the mechanism, the source engine's measured gain and how it scales to our depth
(~12) and speed (~200 knps), the implementation site (file/function), the switch name,
the bench you expect (nodes at depth 8, knps), the gate that can see it (8 s SPRT, 120 s,
suite), the cost in hours, and the risk. Say honestly when an idea is unlikely to reach
+20 Elo; those go into a bundle, not a gauntlet slot. Respect the closed list unless you
bring new evidence.
