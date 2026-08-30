# Experiment backlog

The overnight loop works down this file. Edit it freely — it is the steering wheel.
Take the topmost item whose `status` is `todo`, work only on that, and record the
outcome. One experiment per run.

Priorities come from measured evidence, not enthusiasm. Elo figures cited are from
published SPRT results in comparable engines and are estimates until our own SPRT
says otherwise.

---

## P0 — blocked on a human, do not start

- [ ] **Ask the organisers about `rust-chess` and `numba`.** Worth more than
  everything below combined: a pip-installable compiled move generator measured
  2.2x end-to-end and a full extra ply (~150 Elo) over python-chess, and a
  numba-JIT'd movegen benchmarks ~40x on the generator alone. Both hinge on
  whether the rules' "compiled dependencies come from PyPI" covers them.
  **Toby must email hello@aichessathon.com.** Do not build on either until answered.

---

## P1 — search, highest value per hour

- [ ] `status: todo` — **Killer moves and history heuristic.** Two quiet moves per
  ply plus a from/to history table, inserted after captures in ordering. Published
  at +56 and +19 Elo, but a Python-specific measurement found only 1.1x wall-time
  gain against 1.4x node gain, because the sort key costs what it saves. Genuinely
  uncertain here — which is exactly what the SPRT is for. Keep the sort key cheap.
- [ ] `status: todo` — **Reverse futility pruning.** Best measured Elo-per-token of
  any feature in a comparable competition, and it fires at `depth < 7`, which is
  where we live. Start with margin `70 * depth`, prune when `eval - margin >= beta`
  at shallow depth with no check.
- [ ] `status: todo` — **Null-move pruning, fixed R=2.** Published +116 Elo. Skip the
  depth-scaling variant, which measured only +14. Guard against zugzwang: require
  non-pawn material for the side to move, and never in check.
- [ ] `status: todo` — **Check extension.** One ply when the move gives check.
  Measured +57 Elo in an engine of roughly our strength and ~1 Elo in Stockfish —
  a shallow-depth feature, so it should be worth more to us than to them.
- [ ] `status: todo` — **Principal variation search.** Only after ordering is solid.
  Measured +55 Elo *after* good ordering and **−33 before it**, so if the killer and
  RFP experiments above have not landed, skip this and come back.
- [ ] `status: todo` — **Time management sweep.** The `_budget` constants in
  `agent.py` were reasoned, not measured. Try expected-moves in {22, 26, 30} and
  hard-limit fractions in {0.25, 0.35, 0.45}. Cheap, and flagging is the most
  common self-inflicted loss.

## P2 — the learned evaluation, the single biggest jump

The rules require that a learned model materially drives move selection, so this is
mandatory, not optional. It is also where a bad job is worse than no job: the net
costs about a ply (~150 Elo) and must clear that to break even. Published results
range from +305 down to +20, and the difference was training data quality.

- [ ] `status: todo` — **Download `Lichess/fishnet-evals`.** CC0 Parquet on
  HuggingFace, schema `fen, cp, mate, move`, ~7 GB for one month. No PGN parsing.
  Store to `data/` (gitignored). Filter on eval depth; keep at least half the
  positions in the +/-100cp band. Target 20-50M positions.
- [ ] `status: todo` — **Training script.** 768 inputs (6 piece types x 2 colours x
  64 squares), both perspectives, `768 -> 256x2 -> 32 -> 1`, SCReLU. Loss is MSE in
  win-draw-loss space, `wdl = sigmoid(cp / 400)`, mates clamped to +/-10000cp.
  Float32 — int16 measured *slower* in numpy. Watch for the data-loader bottleneck:
  a naive `__getitem__` that builds a python-chess Board caps at 5-20k positions/sec
  and idles the GPU.
- [ ] `status: todo` — **Incremental accumulator in `agent.py`.** Maintain
  `acc_white`/`acc_black`, push the old accumulator on a stack so `pop` is a restore
  not a recompute. Verify against a full rebuild on several thousand random plies
  before trusting a single game — a silent accumulator bug looks like weak play.
- [ ] `status: todo` — **SPRT the net against the PST champion.** The decisive
  experiment of the whole project.

## P3 — bounded, no statistical validation needed

- [ ] `status: todo` — **Syzygy 3-4-man tablebase.** Exactly 4,346,080 bytes for all
  70 WDL+DTZ files from `tablebase.lichess.ovh`. Probe WDL at leaves once material
  is down to four men (33-59 us, affordable), DTZ at the root only (up to 514 us).
  Use `get_wdl`/`get_dtz`, never `probe_*`: a missing table raises and a crash is a
  loss. Dominates hand-written mate drivers — it solves KBNvK for free.
- [ ] `status: todo` — **Opening book, 20 plies.** Polyglot `.bin`, not a pickled
  dict: measured on identical data a pickle is not smaller and costs full-file load
  plus heap residency, where `.bin` is memory-mapped. Two traps: the weight field is
  `uint16` and saturates at 65535, so scale raw game counts; and castling must be
  encoded king-takes-rook (e1h1, not e1g1). Build from human master games.

## P4 — robustness, run whenever the queue is empty

- [ ] `status: recurring` — **Crash hunt.** A few hundred games against the random
  baseline at a fast control. Random play reaches promotion, en passant and
  stalemate corners far faster than a real opponent. Any failure termination is a
  P0 bug: fix it before anything else in this file.

---

## Do not build

Measured or published as not worth it at our depth, recorded here so no run
rediscovers them: singular extensions (needs depth >= 6), internal iterative
deepening (depth >= 7), razoring (~1 Elo), aspiration windows (+23 against −25 in
two separate tests), MCTS (50k UCT iterations plays like a 2-ply minimax once you
have a real evaluation), batched leaf evaluation for alpha-beta (gives up the
ordering win to gain less), int8/int16 quantisation in numpy (slower than float32),
ONNX Runtime at batch 1 (4x slower than numpy), and rewriting move generation in
pure Python (python-chess already uses precomputed bitboards; every hand-rolled
alternative benchmarks worse).
