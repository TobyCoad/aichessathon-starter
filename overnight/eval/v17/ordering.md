# Move ordering / history / quiescence / SEE: Alexandria 9.0 vs ours (v16f + S1-lean)

Written 9 Sep from static reading only (no benches, no gauntlets). Reference paths are
`engines/alexandria/src/<file>:<line>` (A:) and ours `fastsearch.py` (FS:), `fastboard.py`
(FB:), `agent.py` (AG:). Ideas and constants only; no code text was or should be copied.

Champion assumed: v16f net + SEE_QUIET + HISTORY_V2 (s1lean-vs-v16f +116 +/- 62 at 112 games),
with CAPTURE_ORDER, HISTORY2(+FIX), KILLER_CLEAR, SEE_MAIN, QS_TT, QS_EVAL_CACHE, SINGULAR on;
CONT_HIST, KILLER_SHIFT, LMP, IMPROVING, CUTNODE, CORRECTION off/closed.

## 0. Bottom line (read this if nothing else)

- Nothing in this area is a clean single +20 at 120 s. NOTES "SEARCH AXIS IS CLOSED" is right
  about ordering-as-cutoff-rate (85.4% first-move cutoffs); what still pays is history used as a
  PRUNING/REDUCTION signal, which is exactly what S1-lean (HISTORY_V2 + SEE_QUIET) just proved.
- Recommended use of the ~5-6 remaining slots on this axis: ONE bundle slot ("ORDER2", items
  1-6 below, each bench-selected first) and at most ONE solo slot for CONT_HIST_V2 (item 7),
  taken only if its d10 bench beats 0.85x nodes at <= -4% knps. Everything else here is a filler
  that rides in ORDER2 or is not worth building before 11 Sep.
- Two genuine defects found by reading, both cheap: (a) `SEE_VALUE` prices knight 320 / bishop
  330, so BxN-defended is SEE -10: pruned in QS and ranked BELOW EVERY QUIET by CAPTURE_ORDER,
  while NxB-defended is +10 and ranked above killers (item 1). (b) The capture history is on the
  dead d^2/16384 scale that HISTORY_V2 fixed for the butterfly, and is then divided by 16 -- it is
  a +/-100-point tiebreak that never orders anything (item 2).

## 1. MOVE ORDERING

| Mechanism | Alexandria | Ours | Gap |
|---|---|---|---|
| Staging | 9-stage picker: TT, good noisy, killer, counter, quiets, bad noisy (A:movepicker.cpp:61-187) | one gen_legal + score_moves + pick_move best-first (FS:859-927, FB:991-1051) | closed x2 (staged movegen); same effective order except below |
| TT move | first, pseudo-legality checked (A:movepicker.cpp:84-99); tried in QS too if tactical | 1<<30 score (FB:1011); QS passes hash_move=0 (FS:824) | QS does not use the hash move (item 5) |
| Killers | ONE per ply (ss->searchKiller), cleared for ply+1 on node entry (A:search.cpp:488-489) and per aspiration pass (A:330-343); skipped if == ttMove | two per ply, KILLER_CLEAR ply+2 on entry + whole table per move (FS:968-970, AG:2781-2789) | none material; ours is a superset |
| Counter move | one per FromTo(prev) (A:movepicker.cpp:45), placed after the killer | same table, COUNTER_MOVE band under killers (FS:875-879, FB:1022) | none |
| Captures | victim SEEValue*16 + capthist (NO LVA) (A:movepicker.cpp:19); then SEE(move, -score/32 + 236) splits good/bad; bad go after quiets (A:107-128, 170-180) | CAPTURE_ORDER: SEE<0 -> band -(1<<21)+see*16+caphist//16; else CAPTURE_BONUS + MVV*16 - LVA + caphist//16 (FS:907-926) | capthist weight (theirs up to +/-16384 vs victim 1600..16240 -- it can promote/demote a capture a whole victim class; ours +/-~100) |
| Quiet score | HH(side,from-to) + CH1 + CH2 + CH4 (piece-to) + 4*rootHist at root (A:history.cpp:191-197) | butterfly(side,from-to) only (FB:1025) | continuation histories (item 7) |
| Bad-capture search | after quiets, at full depth but LMR applies to noisy moves too (A:search.cpp:785, 817-832, lmrNoisyBase -0.36) | after quiets (CAPTURE_ORDER) at FULL depth; LMR is `plain` only (FS:1417-1419); SEE_MAIN prunes only depth<=5 | at depth 6-12 a losing capture costs a full-depth subtree (item 3) |
| Root | same picker + rootHist x4 | previous-iteration scores, hint first (AG:2862-2888) | ours is better for iterative deepening; no action |
| skipQuiets | one history-prune failure sets skipQuiets and jumps to bad captures (A:719-722, movepicker.cpp:63-70) | `continue` per move; pick_move stays O(n) each (FS:1388-1389) | equivalent decisions (quiets are picked in descending butterfly order); ~0 speed |

## 2. HISTORY TABLES

Alexandria tables (A:history.h:11-17, history.cpp): HH `searchHistory[side][from-to]` max 8192;
rootHistory same shape; contHist `[piece-to of prev][piece-to]` at offsets 1, 2 AND 4 plies
(A:history.cpp:68-72, 141-149), max 16384; captHist `[piece-to][captured]` max 16384;
corrHist pawn / whiteNonPawn / blackNonPawn keyed (32768 each) + contCorr
`[pieceTypeTo(ss-1)][pieceTypeTo(ss-2)]`, max 1024, grain 256. No decay between moves (tables
keep across searches); gravity `entry += bonus - entry*|bonus|/MAX` everywhere
(A:history.cpp:47-92).

Bonus/malus (A:tune.h:145-178), all `min(mul*depth + off, cap)`, depth passed as
`depth + (eval <= alpha)` (A:search.cpp:902):
- HH: bonus 333d+159 cap 2910; malus 398d+139 cap **452**.
- capthist: bonus 349d-76 cap 2296; malus 310d-88 cap 1306 -- malus applied to every
  noisy move searched before the cutoff EVEN WHEN THE CUTOFF MOVE IS QUIET (A:history.cpp:124-128).
- conthist: bonus 159d-135 cap 2538; malus 401d+114 cap 806; extra TT-cutoff malus
  min(155*depth, 385) on the parent's move when a TT cutoff happens with a quiet parent move and
  parent moveCount < 4 (A:search.cpp:476-480); LMR re-search bonus/malus (A:852-855).
- roothist: bonus 225d+165 cap 1780; malus 402d+75 cap 892.
- Opponent HH bonus from the static-eval swing after a quiet move: clamp(-10*(evalPrev+evalNow),
  -1830, 1427) + 624 (A:search.cpp:519-525, their comment: ~6 Elo).

Ours (FS:1536-1590; AG:176-199): butterfly int32[8192] side-indexed, HISTORY_MAX 16384,
HISTORY_V2 bonus 300d-250 cap 2400 (FS:301-303), MALUS = SAME BONUS applied to quiets searched
before the cutoff (FS:1551-1555), halved every move under HYGIENE (AG:2826-2829). Capture history
in conthist1[:4608] `(attacker*64+to)*6+victim`, bonus d^2 cap 1200 on the 16384 range, no
malus, then //16 in the score (FS:921-926, 1579-1589). Consumers: ordering, LMR step +/-1 at
|hist|>8000 (FS:1437-1440), PRUNE_V2 history cut at -1000*depth (FS:1385-1389).

Gaps, ranked (see section 6 for the table with Elo/cost/gate):
- Capture history is inert on two counts (scale, //16) and has no malus -> item 2.
- Malus asymmetry: theirs caps the quiet malus at 452 (1/6 of the bonus); ours pushes every
  earlier quiet down by up to 2400 per cutoff. With gravity that drives the losers to -16k
  quickly (which is what the -1000*depth prune reads) -- direction unknown; bench-tunable -> item 4.
- Eval-swing bonus to the opponent's move: no equivalent -> item 6.
- Continuation history: closed as built; see item 7 for whether Alexandria's differs.
- TT-cutoff malus, re-search bonus, root history, 2/4-ply: all ride on conthist; not separately
  worth a slot.

Correction history (closed x2, 048/048b): ours was PAWN-KEY ONLY, weight min(d+1,16)/256,
cap 400 then 100 cp, running-average style (AG:1335-1348, 1484-1495). Alexandria (A:history.cpp:
160-189) keys on pawn key + white-non-pawn key + black-non-pawn key + a 2-move continuation
pair, weights 29/34/34/26 over grain 256, entries capped +/-1024 with gravity, bonus
clamp(diff*depth/8, +/-256), updated only when the bound is consistent with the direction
(no fail-low above static, no fail-high below static), best move not tactical, not in check
(A:search.cpp:929-934); applied to pruning eval AND the QS stand-pat. Verdict: materially
different KEYING (non-pawn material hashes target exactly the <=16-piece systematic optimism
the eval audit keeps finding) and a different update rule; the pawn-key idea should stay closed.
A non-pawn-keyed CORRECTION_V2 is a legitimate new proposal but it is an eval-side gamble with two
rejects behind it -- not for a pre-freeze slot (section 6, item 9).

## 3. QUIESCENCE

| | Alexandria (A:search.cpp:943-1103) | Ours (FS:762-855) |
|---|---|---|
| TT | probe; cut on bound at any depth in non-PV (983-986); TT score replaces stand-pat when bound allows (1003-1005); store bound + best move + raw eval, depth 0, never exact (1098-1100) | QS_TT: cut on exact/lower/upper (784-793); store move 0, NO_EVAL, exact flag when alpha rose (852-854); no stand-pat improvement from the TT score |
| Stand pat | static (with corrhist); fail-high returns (best+beta)/2 (1016-1019) | static via QS_EVAL_CACHE; returns standing (810-813) |
| Delta / futility | futilityBase = static + 268; if <= alpha skip any capture with SEE < 1 (1045-1053) | per-victim: standing + MVV[victim] + 200 < alpha skip (829); BIG_DELTA 975 whole-node (814) |
| SEE in QS | via the picker: only captures with SEE >= -score/32 + 236 are searched at all (movepicker.cpp:112-117 with QSEARCH skip). With capthist 0 that is SEE >= +186 for a pawn victim, >= +25 for a minor -- an equal minor trade is NOT searched in QS unless its capthist is positive | SEE < 0 pruned (831-833); equal trades searched |
| Checks / evasions | generates all moves in check (1028, MOVEGEN_ALL when in check); no quiet checks | none (QS_EVASIONS -18, closed) |
| Depth cap | none (ply cap only) | QS_CAP 14 |
| Ordering | TT move, then victim*16 + capthist | MVV-LVA only (824); no hash move, no caphist (quiesce has no conthist1 argument) |

Gaps worth anything: QS hash move (item 5, tiny); capthist-gated SEE threshold (their most
distinctive QS idea -- it is a tuned SPSA artefact whose transfer is unknown; a mild version
"skip SEE-equal captures whose caphist is negative" is item 2c, bench-gated). TT-score-as-stand-pat
is +0..2 and needs the eval field stored from QS; skip.

## 4. SEE

Values: theirs 100/422/422/642/1015 (N == B); ours 100/320/330/500/900/20000 (FB:1054).
Algorithm: theirs threshold form `SEE(move, thr)` -> bool with early exits before any attack
generation (A:search.cpp:125-157), pinned pieces excluded unless moving along the pin ray
(A:170-179), king-capture guard (A:203-204). Ours: full swap-list value with a heap-allocated
gain[32] per call (FB:1071), correct negamax back-up and the standard early stop
(FB:1107-1109), no pin awareness, king handled by the 20000 value. Consulted: ordering
(CAPTURE_ORDER, every non-promotion capture at every node), SEE_MAIN -20*d^2 at d<=5,
SEE_QUIET -30*d^2 at d<=6, QS < 0, PROBCUT threshold. Theirs: quiet margin -98*lmrDepth
(linear, uses the post-LMR depth), noisy -27*lmrDepth^2, no depth cap (A:724-727).

Gaps:
- N/B asymmetry (item 1): with N=320/B=330 every "bishop takes defended knight" is SEE -10
  (QS-pruned; demoted below all quiets in the main search) and "knight takes defended bishop"
  is +10 (good capture, above killers). Set N == B (320/320 or their 422/422 with R 642 / Q 1015).
  Bench nodes will move; exactness with SEE off is untouched.
- Pins: a pinned defender counted as a defender makes SEE too pessimistic on a real capture;
  needs a pin mask we do not maintain -> not worth it (+0..2, 2 h).
- Threshold form: every consumer except CAPTURE_ORDER's losing band is a boolean test; a
  `see_ge(move, thr)` with the two early exits skips attackers_to on most calls. Exact by
  construction if CAPTURE_ORDER keeps the value call. speed.md put the allocation at the noise
  floor, so expect +1..3% knps, +0..2 Elo -> filler (item 8).
- SEE_QUIET margin shape: ours prunes a quiet that hangs >30 cp at depth 1 (theirs >98) and is
  gentler than theirs at depth 5-6. It just promoted; leave it. A linear variant is a 0.5 h bench
  filler at most.

## 5. Things we have NO equivalent of

capthist malus on captures searched before a quiet cutoff; conthist at 1/2/4 plies with its own
bonus/malus scale; conthist TT-cutoff malus; LMR re-search history update; eval-swing bonus to the
opponent's move; LMR on noisy moves (with a noisy history divisor); lmrDepth-based futility and SEE
margins (FUTILITY_LMR exists, off); capthist-dependent SEE threshold in the picker; root history
(unneeded: ROOT_ORDER); ttPv/former-PV flag in the TT gating reductions; pin-aware SEE;
TT score as improved stand-pat in QS; depth+(eval<=alpha) as the history-update depth.

## 6. Ranked proposals (expected value per gauntlet slot)

Elo is at 120 s (8 s reads ~2x); "bench" = nodes at depth 8/10 on testing.bench, knps.
Numbers are judgement from reading, not measurements; the uncertainty is the whole point.

| # | Switch | Mechanism (site) | Elo @120 s | Cost | Risk | Gate |
|---|---|---|---|---|---|---|
| 1 | SEE_VALUES | `SEE_VALUE`/`MVV` N == B (FB:1054, FS:52, FB:933); keep king 20000 | +1..5 | 0.3 h | none; nodes change, exactness (SEE off) holds | bench, then ORDER2 bundle |
| 2 | CAPHIST_V2 | (a) bonus on the HISTORY_V2 scale (300d-250 cap 2400) for the cutting capture (FS:1585-1589); (b) malus to captures searched earlier -- record ALL searched moves in `quiets[ply, :]` and split at cutoff time by `sq[to] >= 0` (board is restored there), which retires the HIST2_FIX zeroing; apply the malus also when the cutoff move is quiet; (c) full weight in the score (drop the //16, clamp so promotions/hash stay above; FS:921-926); optional (d) pass conthist1 into `quiesce` for QS ordering (signature change at ~5 sites, skip unless cheap) | +2..7 | 2 h | table saturation (HYGIENE halving already covers conthist1) | bench d8 nodes; ORDER2 |
| 3 | LMR_BADCAP | in the LMR block (FS:1417-1424) also reduce a non-promotion capture whose CAPTURE_ORDER score is in the losing band (score < 0 in `sc[i]`), same table, `searched >= 1`, not hash; re-search on `score > alpha` already exists | +3..8 | 1 h | tactical misses on a bad-capture sacrifice; PVS re-search limits it | bench (expect a node cut at d8/d10); ORDER2, split out if the bundle fails |
| 4 | HIST_MALUS_CAP | separate malus cap (try 600 and 1200 vs the current 2400) in FS:1551-1555 | -3..+4 | 0.5 h | none | bench only (d8+d10 nodes, root-move agreement); include in ORDER2 only if nodes fall |
| 5 | QS_HASH_MOVE | pass the probed entry's move into `fb.score_moves` in `quiesce` (FS:824; the probe at 784 already unpacks data) and store the best move in `qs_tt_store` (FS:758) | +0..3 | 0.5 h | none | bench; ORDER2 filler |
| 6 | HIST_EVALDIFF | after the static eval is known at a non-check node whose parent move was a plain quiet, add clamp(-10*(evalParent+evalHere), -1830, 1427)+624 (the numbers already fit our 16384 range) to the OPPONENT's butterfly entry of that move, gravity form. Needs the per-ply eval lane `exts[MAX_PLY+ply]` written whenever an eval is computed (today only under IMPROVING/HINDSIGHT, FS:1051-1068) | +1..5 | 1.5 h | double-counting with the cutoff malus; one extra evaluate at nodes that evaluate nothing today (rare: RFP/NMP already evaluate most non-check nodes) | bench; ORDER2 |
| 7 | CONT_HIST_V2 | reopen the built CONT_HIST (FS:883-906, 1385-1387, 1428-1436, 1561-1578) with: its own bonus 160d-135 cap 2500 / malus 400d+115 cap 800 (not d^2); prune2 tests butterfly+conthist against -1000*depth (already the sum); LMR term = the +/-8000 step on butterfly PLUS a separate +/-1 step on conthist at +/-6000 (do NOT use the shared //6000 clamp: AG:196-198 explains it measures the butterfly); TT-cutoff malus optional. Keep 1-ply only; add 2-ply only if 1-ply promotes | +5..20 | 3 h (code exists) | the one item that could be +20; also the one that was rejected (133-conthist 46.1%/280 at 8 s, 0.89x nodes at -3.5% knps) | bench d10 must show <= 0.85x nodes at >= -4% knps, else do not spend the slot; 8 s SPRT, then 40 x 120 s (history tables train better with more nodes, 8 s under-reads it) |
| 8 | SEE_GE | boolean threshold SEE with the two pre-attack early exits for SEE_MAIN / SEE_QUIET / QS / PROBCUT; CAPTURE_ORDER keeps the value call | +0..2 | 1.5 h | must be proved node-identical on the bench | bench knps only; never a slot |
| 9 | CORRECTION_V2 | non-pawn-material-keyed correction (two 32k tables keyed by each colour's non-pawn Zobrist subset) + gravity update, bound-consistency guard, cap ~+/-64 cp, applied to pruning eval and stand-pat | -20..+15 | 3 h | closed x2 as pawn-key; eval-side; the net changes weekly | NOT before 11 Sep unless a slot is otherwise idle; endgame suite + 8 s SPRT + 120 s |

Verdict on the closed items asked about:
- CONT_HIST: Alexandria's is the same index scheme (piece-to x piece-to) and the same idea; what
  differs is the scale (live bonus/malus with a malus larger than the bonus) and the consumers
  (continuous LMR on the summed history, lmrDepth, history pruning on the sum, TT-cutoff malus,
  re-search update, 2- and 4-ply). What we tested had the ORDERING part working (0.89x nodes) and
  the LMR/prune parts on a scale where they could not fire (AG:176-199 documents the dead table).
  So it is materially different in the parts that carry the Elo in Alexandria. Reopen only as
  item 7 under its bench precondition; otherwise it stays closed.
- Correction history: differs materially in keying; stays closed as pawn-key; item 9 is a new,
  low-priority proposal.
- Staged movegen, LMP, QS evasions, IIR, TT_EVAL: Alexandria's versions are the same ideas;
  nothing here reopens them.

## 7. Suggested slot plan

1. ORDER2 bundle (items 1, 2, 3, 5, 6; 4 only if the bench says so): ~5 h to build, each item
   benched at d8/d10 for nodes and knps before inclusion (drop any that raises nodes or costs
   >2% knps). One 8 s SPRT vs the champion. Expected +6..+20 combined at 8 s; likely
   INCONCLUSIVE-positive, ship on bench + non-negative sign per the standing bundle rule.
2. CONT_HIST_V2 solo (item 7) only if the d10 bench precondition holds; one 8 s SPRT; if the
   point estimate is positive but INCONCLUSIVE, decide on 40 x 120 s, not on the 8 s number.
3. Do not spend slots on 8 or 9.
