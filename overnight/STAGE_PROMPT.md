# Stage review

One stage of an unattended night has just finished on the AI Chessathon engine in
`C:\dev\aichessathon\starter`. Write a short, blunt review of that stage only.

Stage: **@STAGE@**. Its evidence:
- `@STATE@/night.log` -- the orchestrator's log (read the tail).
- `@STATE@/SUMMARY.md` -- one line per stage so far.
- `@STATE@/@STAGE@.gauntlet.log` if it exists -- crash gate and SPRT.
- For the `contempt` stage: `overnight/eval/contempt.champion-vs-weiss-d6.log` and
  `overnight/eval/contempt.041-vs-weiss-d6.log`, 60 games each against a weaker engine.
- For the `kingzones` stage: `overnight/eval/kingzones-chain.log`,
  `overnight/eval/039-kz4-fast.gauntlet.log`, `overnight/eval/clock.03*.x1.5.log`,
  `overnight/eval/match.final-kingzones.120s.log`.
- `git log --oneline -5` and `git show --stat HEAD` for what, if anything, was promoted.
- The switch itself is a block in `agent.py` behind a constant of the same name
  (FUTILITY, TT_AGE, PVS, CONTEMPT); read it if the result needs explaining.

Write at most 250 words with these headings: **What was tested**, **Result** (the
numbers: games, score, Elo and error bar, failures, terminations), **Trustworthy?**
(failures mean bug not result; under 25 pairs is not a verdict; repeated openings
make error bars optimistic), **Promoted?**, **For the morning** (one or two lines: a
concern, or "nothing"). No praise, no padding. Do not edit, run matches, or commit.
