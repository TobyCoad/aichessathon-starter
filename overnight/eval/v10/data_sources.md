# Training data sources for the eval net -- survey, 6 Sep 2026

Research pass. Nothing was trained, queued or gauntleted. No large file was downloaded: every
candidate was verified by dataset card, HF tree API, HTTP HEAD, and a **3 MB HTTP range probe**
decoded locally with `training/binpack_decode.py`. Probe scripts are in
`overnight/eval/v10/probe/` (`probe.py`, `pair.py`, `calib.py`); the `.head` blobs they read were
deleted after use and the exact `curl` commands to regenerate them are in each script's docstring
and below.

---

## 1. Ten-line summary

1. **Our labels are not Stockfish's.** `test80` is Leela Chess Zero **training run T80**
   (`storage.lczero.org/files/training_data/test80/`), converted by `linrock/lc0-data-converter`.
   The stored score is Leela's own MCTS evaluation; Stockfish only ever ran at **depth 6 MultiPV 2
   as a discard filter**. There is no Stockfish depth-12 pass anywhere in the T80 pipeline.
2. **`2tb7p` is not a size or a depth** -- it is *2 TB of 7-piece Syzygy mounted during rescoring*.
   `16tb7p` = 16 TB (near-complete 7-man), `69gb6p` = 69 GB of 6-piece. Our file is the *thinnest*
   tablebase variant linrock published.
3. **Deeper-labelled bulk data exists and is one file.** `vondele/rescored` is the same
   T80 month (2023-06) relabelled by Stockfish at **3k / 5k / 10k / 20k / 40k / 60k / 80k nodes**,
   min-v2 binpack, 15.8-17.2 GB each.
4. **Measured locally, 114,969 paired positions, 0 EPD mismatches** (Leela original vs SF n20000
   of the identical file): median |score| **92 -> 151** (SF is 1.64x *louder* in quiet positions)
   but on |score| > 1145 units the mean is **16,679 -> 10,730** (SF is 0.64x, far *quieter* in
   winning positions). Pearson r 0.762, sign agreement 89.1%.
5. That is exactly the two errors we measured, both moving the right way: our quiet-position
   error is **+80 cp** (net under-confident -> Leela's compressed quiet labels) and our attacking
   error is **-120 cp** (net over-confident -> Leela's blown-out winning labels).
6. **Local calibration against our own Stockfish at depth 12, n=160 each, same method:**
   our current file r = **0.861** (scale 0.200); SF-n20000 r = **0.971** (scale 0.240);
   `wrongIsRight` r = 0.975 (0.324); `UHO` r = 0.983 (0.362). The 0.861 control is consistent with
   the production 0.262 / r=0.90 figure, so the relative ordering is trustworthy.
7. **Scale will change and it is the known landmine.** Our production `--scale 0.262` is calibrated
   to *Leela* units. Same-method ratio puts SF-n20000 at ~**0.315**; this MUST be re-derived with
   `--sample` before training, exactly as the 0.45 -> 0.262 correction was.
8. **Newer is not better.** T91 (Leela 2026 run, `adamtwiss/t91-binpacks-filtered`) is published
   with its own matched-control measurement: **~100 Elo weaker than equal-sized T80**, and its
   filtering costs a further ~20 Elo. `linrock` has published nothing new since Oct 2024.
9. **BT4 relabelling is what Stockfish master now trains on, and it would do almost nothing for
   us.** Paired against our exact file (98,178 positions, 0 mismatches):
   **r = 0.977, median ratio 1.01**. Same positions, same scale, nearly the same numbers.
10. **Weak-engine game archives exist (CCRL, ~387 MB for the 1500-2200 band) but do not fit this
    week**: ~3.75 M positions after extraction, needing ~11 CPU-hours of Stockfish labelling on the
    only gauntlet machine, for 0.6% of the corpus. Rejected on cost, not on principle.

---

## 2. Candidate datasets

Sizes are exact `Content-Length` / HF tree bytes. "Decodes" = verified locally, first ~3 MB pulled
by HTTP range and run through `training/binpack_decode.py::iter_chunk`.

| # | Dataset / file | Size | Format | Labels: engine + budget | Positions | Decodes | URL |
|---|---|---|---|---|---|---|---|
| 1 | `vondele/rescored` -- `test80-2023-06-jun-2tb7p.min-v2-rescore_SF_n20000.binpack` | **16.32 GB** (plain) | binpack min-v2 | **Stockfish, 20,000 nodes/move** | ~4.5 B (0.2765 entries/byte, measured) | **yes**, 869,654 entries, 0 bad boards, 0 illegal moves | https://huggingface.co/datasets/vondele/rescored |
| 1b | same repo: `...n5000` 16.99 / `n10000` 16.66 / `n40000` 16.03 / `n60000` 15.90 / `n80000` 15.81 GB; `...orig_Leela` 15.85 GB | as above | binpack min-v2 | SF at that node count / Leela original | same positions in all 8 | yes (n5000, n20000, orig probed) | same |
| 2 | `official-stockfish/master-binpacks` -- `wrongIsRight_nodes5000pv2.binpack` | **7.32 GB** | binpack (minimized) | **Stockfish gensfen, 5000 nodes, MultiPV 2** | ~1.63 B | **yes**, 702,648 entries, 0 errors | https://huggingface.co/datasets/official-stockfish/master-binpacks |
| 3 | same repo -- `nodes5000pv2_UHO.binpack` | 40.29 GB | binpack | SF gensfen 5000 nodes, MultiPV 2, from the **UHO unbalanced-openings book** | ~9.8 B | **yes**, 764,511 entries, 0 errors | same |
| 4 | same repo -- `dfrc_n5000.binpack` | 37.48 GB | binpack | SF gensfen 5000 nodes, **Double Fischer Random** starts | ~8.7 B | **yes**, 730,604 entries, 0 errors | same |
| 5 | same repo -- `fishpack32.binpack` 5.58 GB, `wrongNNUE_02_d9.binpack` 5.78 GB, `T60T70wIsRightFarseer.binpack` 32.98 GB, `farseerT74/75/76`, `training_data_pylon`, `multinet_pv-2_diff-100_nodes-5000` 27.64 GB, `test80-2022-08-aug-16tb7p.v6-dd.min.binpack` 10.81 GB | 5.6-47 GB | binpack | mixed: `d9` = SF depth 9; Farseer/T60/T70 = Leela runs | -- | not individually probed | same |
| 6 | `xushawn/test80-bt4-relabel` -- `test80-2024-02-feb-2tb7p.min-v2.v6.relabel.binpack` | 8.36 GB / **7.15 GB .zst** | binpack min-v2.v6 | **Leela BT4-tf13tune static net eval** on *our exact positions* | same as ours | **yes**, 789,715 entries, 0 errors, 18.3% skip-marked | https://huggingface.co/datasets/xushawn/test80-bt4-relabel |
| 7 | `vondele/linrock_relabel_2` (T80 2023, 12 months) / `linrock_relabel_1` / `from_kaggle_1_relabel` / `from_kaggle_2_relabel` / `master-binpacks_relabel` | 203-565 GB per repo; individual files 5.86-47.9 GB | binpack min-v2 | BT4-tf13tune relabel | -- | not probed (same family as #6) | https://huggingface.co/datasets/vondele/master-binpacks_relabel |
| 8 | `adamtwiss/t91-binpacks-filtered` -- 5 monthly files | **1.20-5.58 GB** | binpack **min-v2.v6** | **Leela T91 self-play (2026)**, SF depth-6 MultiPV-2 filter | 4.44 B total, **3.08 B trainable** (published, per-file md5s) | not probed | https://huggingface.co/datasets/adamtwiss/t91-binpacks-filtered |
| 9 | `xushawn/t80-binpacks-raw` -- `test80-2025-01-jan-69gb6p.binpack.zst` (smallest month) | 6.72 GB .zst / 12.82 GB raw | binpack, **un-minimized** (~18 B/pos) | Leela T80, Oct 2024 - Aug 2025 | ~712 M in that month | not probed | https://huggingface.co/datasets/xushawn/t80-binpacks-raw |
| 10 | `linrock/test80-2023` -- `test80-2023-02-feb-16tb7p.v6-dd.min.binpack.zst` | 9.08 GB .zst | binpack `.min` (v1 minimizer) | Leela T80, **16 TB 7-piece TB rescoring** | -- | not probed; `.min` vs `.min-v2` compat unverified | https://huggingface.co/datasets/linrock/test80-2023 |
| 11 | `linrock/bullet-training-data` | 306 GB | **bulletformat `.bullet.bin`** -- our decoder cannot read it | -- | -- | n/a | https://huggingface.co/datasets/linrock/bullet-training-data |
| 12 | CCRL per-engine PGN, engines rated 1500-2200 (40/15 + Blitz + 40/2 Archive) | **~387 MB** (7z) | PGN, **unlabelled** | none -- we would label | ~1.47 M engine-games -> ~3.75 M positions | n/a | https://computerchess.org.uk/ccrl/4040/games.html |
| 13 | CEGT `cegtallblitz.rar` + 166 updates | ~1.0 GB | RAR/PGN, unlabelled | none | 4.19 M games to 2025-06 | n/a | http://www.cegt.net/downloads.htm |
| 14 | TCEC S29 full | 76.28 MB/season; `TCEC-events-full.7z` 507 MB | PGN | 3600+ Elo engines -- wrong direction | ~60 k games | n/a | https://github.com/TCEC-Chess/tcecgames |
| 15 | Lichess monthly dump 2026-07, filtered on `[WhiteTitle "BOT"]` | 29.05 GB .zst (streamable) | PGN, unlabelled, **CC0** | none | 89.3 M games in the month; BOT share unmeasured | n/a | https://database.lichess.org/ |
| 16 | Lichess Elite DB | 80.3 MB/month, **stale: last file 2025-11** | PGN | 2500+/2300+ humans -- wrong direction | -- | n/a | https://database.nikonoel.fr/ |
| 17 | `Lichess/chess-position-evaluations` (`lichess_db_eval.jsonl.zst`) | 21.68 GB | JSONL, CC0 | Stockfish, fishnet 1.5 M nodes | 394,669,566 positions | n/a | https://huggingface.co/datasets/Lichess/chess-position-evaluations |
| 18 | `ezipe/lichess_elo_binned` -- the `1500` bin | 36.66 GB (1000 shards) | PGN .zst, unlabelled | none | 2021-01 - 2023-08 | n/a | https://huggingface.co/datasets/ezipe/lichess_elo_binned |

**Measured distribution of the probed files** (first ~3 MB, 700k-900k entries each; `probe.py`):

| file | median \|score\| units | frac >1145 units (~300 cp) | median pieces | frac <=16 pieces | frac <=10 pieces |
|---|---|---|---|---|---|
| our corpus family (`orig_Leela`, T80) | 92 | 0.124 | 19 | 0.419 | 0.219 |
| `rescore_SF_n20000` (same positions) | 151 | 0.119 | 19 | 0.419 | 0.219 |
| `rescore_SF_n5000` (same positions) | 159 | 0.112 | 19 | 0.419 | 0.218 |
| our local `feb24` file | 81 | 0.135 | 20 | 0.400 | 0.199 |
| `wrongIsRight_nodes5000pv2` | 92 | 0.098 | **11** | **0.792** | **0.485** |
| `nodes5000pv2_UHO` | 291 | 0.145 | 13 | 0.629 | 0.366 |
| `dfrc_n5000` | 273 | **0.218** | 14 | 0.590 | 0.365 |

The Stockfish-gensfen files (`wrongIsRight`, `UHO`, `dfrc`) carry **1.5-1.9x more <=16-piece
positions** than the Leela T80 family, because gensfen plays games out while Leela self-play
adjudicates early. That is the band where our static error is 475 cp.

---

## 3. What our current file actually is

`data/sf/test80-2024-02-feb-2tb7p.min-v2.v6.binpack.zst`, 6,908,056,676 bytes.

| token | meaning |
|---|---|
| `test80` | **Leela Chess Zero training run T80.** Not fishtest, not Stockfish self-play. Raw source `storage.lczero.org/files/training_data/test80/`, converted with `linrock/lc0-data-converter` (rescore -> `.plain` -> filter -> `.binpack`). |
| `2024-02-feb` | the month of the Leela run the games come from |
| `2tb7p` | **2 TB of 7-piece Syzygy** mounted during the lc0 rescore pass (full 7-man is ~17-18 TB). Only the ~9 common 7-man material configurations listed in `lc0-data-converter` are covered. `16tb7p` = near-complete; `69gb6p` = 6-piece only. |
| `min-v2` | `transform minimize_binpack`, second-generation minimizer: chain/delta encoding, ~18 bytes/position down to ~2.4-2.8. This is what our decoder's (move, score-delta) chain reader consumes. |
| `v6` | `csv_filter_v6.py`, filter version 6. It **marks** unusable positions with score 32002 (`VALUE_NONE`) rather than deleting them: in-check, capture/promotion best moves, start positions, ply <= 28, and positions failing a depth-6 MultiPV-2 "only one good move" test. Our decoder already drops these (`abs(score) >= VALUE_NONE`, line 309). |
| other tokens seen in the wild | `-dd` = v6 + de-duplication; `sk16`/`sk20`/`sk28` = early-ply skip threshold; `unmin` = un-minimized (32002 rows stripped, full positions restored); `no-db` = lc0 rescorer's de-blunder pass NOT applied; `d9` = **Stockfish** gensfen at depth 9; `nodes5000` / `n5000` / `pv-2 diff-100` = Stockfish gensfen 5000 nodes MultiPV 2, discard if the top-2 gap > 100; `dfrc` = Double Fischer Random starts; `.tar.zst` = raw lc0 tars, **not** binpack; `.q.binpack` = a parallel variant of every BT4 relabel, purpose unverified. |

**The label.** `binpack_to_csv.sh` runs `transform rescore filter_depth 6 filter_multipv 2`, and
`csv_filter_v6.py` writes `score = bestmove_score`, which is the **lc0 rescorer's** score, i.e.
Leela's MCTS evaluation at T80 self-play visit counts, with exact tablebase values substituted
inside the mounted 3-6 man and partial 7-man sets. The Stockfish depth-6 MultiPV-2 numbers are used
only to decide what to discard. So "our binpack labels are ~depth 12 equivalent" is right about the
*strength* (r = 0.86-0.90 against SF d12) but wrong about the *kind*: they are a Leela network's
search value, not a Stockfish search score. Two consequences we can see in the data: the Leela
scores saturate at |score| = 26,624 where the SF-rescored versions of the same positions run to
31,999, and the Leela scores are systematically compressed in quiet positions and inflated in
winning ones.

---

## 4. Recommendation -- one download

**Take the first 5.0 GB of `test80-2023-06-jun-2tb7p.min-v2-rescore_SF_n20000.binpack`
from `vondele/rescored`.**

```bash
mkdir -p data/sf
curl -L --range 0-4999999999 \
  -o data/sf/t80-2023-06-sf20k.binpack \
  https://huggingface.co/datasets/vondele/rescored/resolve/main/test80-2023-06-jun-2tb7p.min-v2-rescore_SF_n20000.binpack
```

Then, **before anything else**, re-derive the scale -- this is the step that cost us a night last
time:

```bash
.venv/Scripts/python.exe -m training.binpack_decode data/sf/t80-2023-06-sf20k.binpack \
    --sample 5000 > overnight/eval/v10/sf20k_sample.tsv
# regress column 2 (internal) against Stockfish depth 12 exactly as sample.tsv was used for 0.262
```

then decode:

```bash
.venv/Scripts/python.exe -m training.binpack_decode data/sf/t80-2023-06-sf20k.binpack \
    --out data/sfd/jun23sf20k --target 580000000 --workers 8 --scale <measured>
```

**Why this file.**

- It is the *only* published data that answers "deeper labels": Stockfish at **20,000 nodes/move**
  against our current few-hundred-visit Leela MCTS. Everything else on offer is either the same
  label depth or a different *network's* opinion.
- It is a **controlled swap**. `vondele/rescored` is one T80 month published in eight versions -- the
  Leela original plus seven Stockfish node budgets -- so the positions are identical and only the
  label changes. Keep the Lichess half of the 50/50 mix fixed, swap the binpack half, and the
  experiment has one variable.
- The paired measurement predicts the direction: on 114,969 identical positions the SF label is
  **1.64x louder in quiet positions** and **0.64x quieter in winning positions** than the Leela
  label. Our quiet error is +80 cp (under-confident) and our attacking error is -120 cp
  (over-confident). Both move toward zero.
- Against our own Stockfish d12 the SF-n20000 labels correlate at **r = 0.971** where our current
  file's correlate at **r = 0.861** (same script, same n, same depth). Less target noise.
- Our decoder reads it unmodified -- verified on real bytes, 869,654 entries, 0 invalid boards,
  0 illegal moves.
- 5.0 GB is **smaller than the 6.9 GB we already downloaded**, and at 0.2765 entries/byte yields
  ~1.38 B raw positions, more than the 1.03 B we decoded from feb-2024.

**Why the range request is safe.** Binpack is a sequence of `BINP` + uint32-size chunks (~1 MB
each, verified). `training/binpack_decode.py::chunks()` already returns cleanly on a short final
payload (`if len(payload) < size: return`), so a truncated file simply ends early. HuggingFace
returns `206 Partial Content` with `accept-ranges: bytes` on all these files -- verified. Download
the whole 16.32 GB instead if the line is fast; the range is only there to keep it inside budget.

**Expected cost.** Download 5.0 GB. Calibration ~15 min. Decode ~45-50 min at 8 workers (the
feb-2024 decode of 1.03 B positions took 39 min across 8 shards). Then 2 GPU hours and one 8 s
SPRT slot. Total ~1.5 h of wall clock before the GPU, and it does **not** touch the gauntlet
machine's CPU.

**What to expect it to change.** The static-eval error table should compress toward zero on both
ends -- quiet error down from +80, attacking error up from -120 -- because the target now comes
from the same engine family as the reference. **Be honest about the circularity**: we measure
error against Stockfish d16, and this trains on Stockfish labels, so *some* of that improvement is
guaranteed by construction and is not Elo. The only verdict that counts is the SPRT. If the SPRT
is flat, the reading is that our eval error metric was measuring label provenance, not strength,
and we should stop spending nights on data.

**Fallback if the SPRT is flat and a slot remains:** `wrongIsRight_nodes5000pv2.binpack`
(7.32 GB, whole file, no range needed) as a *third* mix component. Measured 79.2% of positions at
<=16 pieces and 48.5% at <=10, against 41.9% / 21.9% in the T80 family -- a 1.9x enrichment in the
band where our static error is 475 cp. Its scale calibrated at 0.324 in the same probe. This is the
cheapest attack on the endgame problem that does not cost gauntlet CPU.

### Explicitly not recommended

- **`xushawn/test80-bt4-relabel`** (our exact month, BT4-relabelled, 7.15 GB). Paired against our
  file: **r = 0.977, median ratio 1.01, slope 0.895**. It is what Stockfish master trains on, it is
  the safest download on the list, and it would move almost nothing. Not worth the night.
- **T91 / T90 (any repo).** Newer, and measured ~100 Elo *worse* than equal-sized T80 by the person
  who published it, with the filtering costing a further ~20 Elo.
- **CCRL / CEGT / Lichess-bot PGNs.** Correct distribution, wrong economics this week: ~3.75 M
  positions is 0.6% of the corpus and needs ~11 CPU-hours of Stockfish labelling on the one machine
  that must run gauntlets between now and the 10th. Worth doing after the freeze, not before.
- **`dfrc_n5000`.** The most imbalanced distribution on the list (21.8% of positions past 300 cp)
  but the starts are Fischer-random, so king-safety and rook-placement priors are learned from
  positions that never occur in standard chess. Too risky to introduce five days out.
- **`linrock/bullet-training-data`** (306 GB) -- bulletformat, our decoder cannot read it.
- **Any `.tar.zst`** in the linrock repos -- raw lc0 tars, they need the lc0 rescorer first.

---

## 5. Rules

`docs/IDEAS.md` states "Whatever you train on, the model has to be yours", matching the live rules
("Training data unrestricted, including positions annotated by an existing engine"; "any network
you ship is one you trained yourself"). Nothing above crosses that line:

- Every recommendation is **positions annotated by an engine**, which is named as permitted, and is
  how our current corpus was already built.
- **No pretrained NNUE file is recommended for download or fine-tuning.** `nn-*.nnue`,
  `BT4-tf13tune.pb.gz` and everything in `linrock/dual-nnue` are out of scope and were not
  considered as weights.
- One line worth naming even though it stays on the right side: the **BT4-relabelled** corpora
  (`vondele/*_relabel`, `xushawn/test80-bt4-relabel`) carry labels that are the *static output of a
  published Leela network*, i.e. pure distillation rather than search annotation. That is still
  "positions annotated by an existing engine", and our current Leela-MCTS labels are already the
  same category, so it is not a new exposure -- but the recommended file avoids the question
  entirely, because its labels come from a **Stockfish search at 20,000 nodes**, not from reading a
  network's weights.
- Licences: `vondele/rescored` and `official-stockfish/master-binpacks` carry no explicit dataset
  licence (Stockfish tooling is GPL/LGPL); Lichess dumps are CC0; T91 repos are ODbL; **CCRL and
  CEGT state no licence at all** -- no grant and no prohibition.

---

## 6. What I could not verify

1. **The exact T80 self-play visit count.** Confirmed the labels are Leela's, not confirmed how many
   MCTS visits per move produced them. lczero.org does not publish it per run.
2. **`.q.binpack`.** Every BT4 relabel ships a parallel `.q` variant. Most plausibly the raw Q/WDL
   label against the eval-space one, but nothing states it. Stockfish's own `threats.yaml` uses the
   non-`.q` files.
3. **Whether `.min` (v1 minimizer) files decode with our reader.** Only `.min-v2` and un-minimized
   files were probed. `linrock/test80-2023`'s 16tb7p months are `.min`, so treat #10 as unverified
   until probed.
4. **How the Stockfish-rescore-vs-BT4-relabel ablation actually turned out.** Only `tb5dtm.binpack`
   from `vondele/rescored` reached the shipped `threats.yaml`, which hints the SF rescore lost to
   BT4 for *Stockfish's* purposes -- but the `vondele/nettest` PRs (#360, #367, #403) have empty
   bodies. Stockfish's purposes are also not ours: they are not trying to match a d16 reference with
   a 512-wide net.
5. **Position counts for #1-#5.** Derived from bytes-per-entry measured on a 3 MB head, not from a
   published count. Expect +/-10%.
6. **Distribution representativeness of the 3 MB probes.** The piece-count and score figures come
   from the first ~0.04% of each file, in generation order. The gensfen-vs-Leela endgame gap is
   almost certainly structural, but the exact fractions are not.
7. **`tests.stockfishchess.org/nns`** -- Cloudflare bot-gated, could not be read.
8. **CCRL's 1500-2200 band totals** (~387 MB, ~1.47 M engine-games) come from joining their rating
   lists to their per-engine download tables; the joins were done by an agent from live pages and
   the arithmetic was not independently re-run here. Also: **CCRL Elo is not Lichess Elo**, so the
   band is a knob, not a calibrated match to our opponents.
9. **Whether the fallback (`wrongIsRight`) endgame enrichment actually helps.** It is an argument
   from a distribution statistic to an Elo outcome, with no measurement in between.
10. **Our production 0.262 was not re-derived here.** The control run (`local_feb24`, n=160,
    |score| <= 4000, SF d12) gave 0.200 by the same script that gave 0.240 for SF-n20000; the two
    are comparable to each other but the absolute numbers are a smaller, more filtered sample than
    the production calibration. Use them for the *ratio* (~1.20x, i.e. ~0.315), then re-derive
    properly with `--sample`.
