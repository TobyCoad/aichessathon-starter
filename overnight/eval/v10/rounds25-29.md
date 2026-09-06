# Rounds 25 / 27 / 29 -- the three newest platform post-mortems

Sources: `overnight/postmortem/round-{25,27,29}-*.json`, `overnight/pgn/platform/*.pgn`
(clocks), `overnight/JOURNAL.md` (ship times), `agent.py` (`_budget_v2`, `_budget_v6`).
Piece counts recomputed from the PGNs. Read-only. Companion: `games.md` (rounds 1-24).
Causes per `testing/postmortem.py`: **horizon** = our engine plays the reference move given
5 s; **evaluation** = it still prefers its own move at 5 s *and* its static is far from the
reference; **search** = it still prefers its own move but the evals agree.

## Which build played which game

Hourly slots, 5 Sep 2026. v8.5 uploaded in the **18:00** slot (JOURNAL 5 Sep 17:40), v9
shipped **19:55**, v9.1 (v9 + TIME_V6) shipped **21:55**. So: round 25 (17:00) = **v8** --
the 17:40 journal entry logs it as already played, before the upload; round 27 (19:00) =
**v8.5**; round 29 (21:00) = **v9**, still on TIME_V5. All three floor at 13.0-14.8 s, the
`RESERVE_FRACTION 0.10` + `remaining/30` signature of TIME_V5. **None ran TIME_V6.**

## Round 25 -- loss, White vs THE ROOOOOKKK (v8, checkmate, 156 plies, 78 our moves)

ACPL by band 7.5 (27+, n=10) / 19.3 (17-26, n=6) / **109.1 (<=16, n=62)**. Causes: search
11, horizon 3, evaluation 3. **Zero `time` flags.** Min clock 14.8 s. Shape: level to ply 21
(-55, 26 pieces), a dip to -155 at ply 25, then we **recover to -25 by ply 63** in a 7-piece
ending and lose it from there: -25 (63) -> -289 (67) -> -484 (75) -> -751 (93) -> -826 (99)
-> mate score from ply 103.

**Turning point: ply 63, White's move 38 `Bd1`, 7 pieces, reference -25 -> -213, our static
-4, clock 54.1 s, spent 2.3 s, cause `search`.** The position is **opposite-coloured
bishops**: our K + light bishop + h3 pawn against K + dark bishop + connected e/f passers.
The reference calls it near-level, we read it level too, it is losing, and we never see it
turn. **43 of our 78 moves were played at 6-9 pieces.** Static error over the 15 flagged
moves at <= 10 pieces: **387 cp mean** -- ply 101 static -370 / ref -1143; ply 109 -631 /
-1244; ply 129 -424 / -1049.

Clock is not the story. The turning point had **54 s in hand (45% of the clock)**, the game
ended with 19.3 s unspent, and we finished with *more* clock than the opponent (19.3 vs
14.7) -- the reverse of the games.md losses. Spend through the losing stretch was a flat
1.1-2.6 s/move, consistent with `_budget_v2`.

Root cause: **endgame evaluation**, in a band games.md under-weights. Failure mode 1 in kind
but **not in range**: games.md measures 475 cp at 11-16 and only 141 cp at <= 10, with
`time` dominant there (9 of 20 flags). Here <= 10 is 387 cp with zero time flags.
**NEW in degree and location.**

## Round 27 -- draw, Black vs slopfish (v8.5, fifty moves, 171 plies, 86 our moves)

ACPL 13.9 / 6.5 / **1.7**. Only **3 flagged moves in the whole game** (search 2, horizon 1);
zero blunders. Min clock 13.0 s. Cleanest game in the post-mortem set. We were **better**:
reference +141 at ply 23 (27 pieces), holding +96..+141 across plies 23-49 as the position
simplified from 27 to 15 pieces. Then:

- ply 45 `Bxe3`, 17 pieces, -37, `search`, clock 81.8 s, spent 1.5 s (+137 -> +100)
- **ply 49 `Rc7`, 15 pieces, -85, `horizon`, clock 78.5 s, spent 1.5 s (+89 -> +4)**
- ply 51 onward: reference **0 at every one of our remaining 61 moves**.

Ply 49 is the whole game. `horizon` means our own engine at 5 s plays the reference move
(`h5h4`) -- and we had **78.5 s on the clock and spent 1.5 s**. The static (+140 vs +89) was
fine; this is depth, not the net. Meanwhile our clock at ply 51 was 76.5 s and we finished
on 13.0 s, so **63.5 s -- 53% of the game clock -- went on 61 moves whose reference
evaluation was exactly 0**, in a dead-drawn 7-piece rook ending. games.md saw these shuffles
only as an ACPL artefact ("no move loses anything"); it never priced the clock they consume.

Under TIME_V5 the ply-49 soft budget was `78.5/33.6 + 0.25 = 2.59 s` and the stable-score
refund (`allowance = 1.0`) cut the search at 1.5 s. Under **v9.1's TIME_V6 it would be
smaller**: `expected = max(30, 56 - 0.4*fullmove) = 43.6` gives `78.5/43.6 + 0.35 = 2.15 s`,
scaled by a stability/node-effort factor that clamps to 0.4 on a settled position. TIME_V6
trades middlegame budget for tail budget; round 27 is a game where the tail was worth
nothing and the middlegame was worth half a point.

Root cause: **search depth at 15-17 pieces**, aggravated by allocation. Opponent strength is
not a factor (slopfish ended on 6.0 s and never held an advantage). **NEW as a named mode**
-- games.md has no "failed to convert" entry; every draw it analyses is from a level or
worse position, and its clock analysis targets only the 13 s floor, which cost nothing here.

## Round 29 -- loss, White vs Emile Andrieu (v9, checkmate, 127 plies, 63 our moves)

ACPL 5.4 (27+) / **24.4 (17-26, n=27)** / **37.1 (<=16, n=24)**. Causes: search 14,
evaluation 6, horizon 1, book 1. 22 of 63 moves flagged. Min clock 13.0 s. No blunder until
it was over. This is round 8's accuracy slide one band higher: level at ply
34 (-8, 24 pieces), then -33 (46) -> -127 (58) -> -283 (66) -> -380 (78) -> -453 (86) ->
-642 (94) -> -1042 (106) -> mate from ply 112. The decline runs **plies 54-106, 22 pieces
down to 11**, in a queen-and-rooks middlegame, 20 flagged moves of -30 to -116 each. The
opponent never blundered; it pushed a queenside pawn mass, gave the exchange for it and
queened. **This opponent is simply strong.**

Static error is the worst yet recorded: **160 cp at 17-26 (n=10), 674 cp at 11-16 (n=11)**
against games.md's 155 / 475. Individual readings are extreme:

| ply | move | pcs | our static | reference | error | clock |
|---|---|---|---|---|---|---|
| 108 | Kf3 | 11 | **+775** | -709 | 1484 | 17.0 s |
| 110 | Ke3 | 11 | +330 | -928 | 1258 | 15.5 s |
| 106 | Kf2 | 11 | +187 | -1042 | 1229 | 18.7 s |
| 84 | Rf5 | 16 | +125 | -365 | 490 | 37.7 s |
| 94 | Rxf2 | 14 | -1125 | -642 | 483 (pessimistic) | 29.4 s |

The clock again is not the cause: the slide is complete (-1042) before we reach 18.7 s, and
every move from ply 54 to 100 was played with 23-70 s in hand at 1.4-4.4 s each. Root cause:
**late-middlegame/endgame evaluation + opponent strength**. games.md failure mode 1,
magnitude worse and onset earlier (22-24 pieces, not 16).

## Summary

**The known endgame-eval weakness explains rounds 25 and 29 outright, and harder than
games.md's numbers suggest.** Round 27 is the one genuinely new thing. What is *not* new, and
should not consume a gauntlet slot: none of these is a clock loss.
The three turning points were played with 54.1 s, 78.5 s and 17.0 s in hand; two of the
three games ended with us holding more clock than the opponent; there are zero `time` flags
in rounds 25 and 27 and none in round 29's decisive stretch. **TIME_V6 (v9.1) would have
changed no result in these three games**, and in round 27 it makes the critical move's
budget ~17% smaller. That is real negative evidence against V10_PLAN #1's +10..+25 headline:
TIME_V6 is insurance, not Elo, and should not be re-tuned on this data.

### Backlog proposals

**P1 -- extend the endgame shrink down to 6 pieces, and instrument the <= 10 band.**
V10_PLAN #2a and `v10/endgame_shrink.md` scope the shrink to "below 17 pieces". Round 25 says
the damage runs to 6: 387 cp mean static error over 15 flagged moves at <= 10 pieces, in the
5-10 gap between `TB_MEN = 4` and the net's trained range, where 43 of our 78 moves lived.
Keep that doc's cap-on-(net - material) form but let the cap tighten monotonically to 6
pieces rather than flattening below 11, and add an opposite-coloured-bishop damping term so a
level-material OCB position with connected enemy passers is not read as 0. *Before any of it*:
extend the
`testing/endgame_suite.py` static-error mode to report a separate `<= 10` band -- these
games predict ~70 / 160 / 550 / 390 cp for 27-32 / 17-26 / 11-16 / <= 10, and the <= 10
target is new. **Elo at 120 s: +8..+20** (a subset of #2's range, but it targets ~43
moves/game in the games we lose). **Gate:** the instrument must move the <= 10 band below
200 cp with 11-16 no worse, *then* 8 s SPRT + 40 games at 120 s on `platform_openings.txt`.
Cheap: the existing shrink with a wider ramp.

**P2 -- do not spend half the clock on a proven-drawn position.**
Round 27 spent 63.5 s (53% of the clock) on 61 moves at reference 0. Concretely: when the
root score has stayed within +/-20 cp of the draw score for our last 6 moves, the halfmove
clock is above 20, and pieces <= 10, cap the soft budget at `0.8 * observed increment`.
Honest caveat: **this banks time it cannot spend backwards**, so it is worth nothing in
round 27 itself; its value is only in games that go dead-level and then come alive (the
rounds 16 / 21 / 27 shape -- 3 of 26 post-mortemed games). **Elo at 120 s: +0..+6.**
**Gate:** ~15 lines, one switch, `agent.py` only (kernel untouched, `check_fastsearch` stays
exact); ship in a bundle on a clocktest whose lowest clock must *rise*, plus a 40-game 120 s
non-negative. **No gauntlet slot of its own** -- below the 8 s SPRT's +/-12 Elo resolution.

**P3 -- record, do not build: a conversion counter in the post-mortem.**
Round 27's failure is invisible to every current metric (ACPL 1.7 at <= 16, 3 flags, 0
blunders). Add a per-game `peak_eval_ours` and the ply at which it decayed below half, so
"we were +141 and drew" appears in REPORT.md without a human reading the trajectory.
Minutes of work, no Elo; it stops this class of game being mis-filed as "endgame eval".

**Not proposed:** a 5-man syzygy subset still changes nothing -- round 25 was already lost at
8 pieces, and 6-7 men (where its damage was) is unshippable. The book is again a non-factor
(round 29's one `book` flag is -30 cp at move 2, 32 pieces; rounds 25 and 27 have none). No
new time-management item beyond P2.
