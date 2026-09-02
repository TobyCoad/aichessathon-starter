# Morning report

An unattended night has finished on the AI Chessathon engine in
`C:\dev\aichessathon\starter`. Write the report the owner reads over coffee: what
ran, what changed, what to do today. Blunt, quantitative, no padding.

Context you need first:
1. `overnight/JOURNAL.md`, the last three entries -- where the project stood at
   nightfall: the compiled board was uploaded and validated on the platform on
   2 September; the 8-king-zone net was then promoted; a 4-zone net, a contempt
   rule and three search switches were queued for the night.
2. `@STATE@/SUMMARY.md` and `@STATE@/night.log` -- the night's own record.
3. The per-stage reviews already written: `@STATE@/review-*.md`. Consolidate them;
   do not repeat them verbatim.
4. The raw logs where a number needs checking: `@STATE@/*.gauntlet.log`,
   `overnight/eval/kingzones-chain.log`, `overnight/eval/039-kz4-fast.gauntlet.log`,
   `overnight/eval/match.final-kingzones.120s.log`, `overnight/eval/contempt.*.log`,
   `@STATE@/clock.final.log`, `@STATE@/match.final.120s.log`.
5. `git log --oneline -12` -- every promotion is a commit.

Sections, in this order:
1. **Headline** -- two sentences: what the build is this morning versus last night,
   and whether it should replace the validated upload.
2. **Stage table** -- one row per stage (kingzones, contempt, futility, ttage, pvs,
   final): result, verdict, promoted or not, trustworthy or not, with the numbers.
3. **The build now** -- which switches are on (`grep -E "^(TIME_V2|HYGIENE|FAST_BOARD|CONTEMPT|FUTILITY|TT_AGE|PVS): Final" agent.py`),
   which net (`weights/net.npz` shapes via a short python snippet), and the
   morning validation: clock replay floor and longest move, the 120 s match score.
4. **Re-upload decision** -- recommend yes or no, with the conditions: no failures
   anywhere, clock replay clean, 120 s match not worse. Note that the platform
   allows two seats per team, so the validated 2 September build can stay as the
   second seat.
5. **Anything that went wrong** in the night itself -- stalls, fake failures,
   stages skipped -- and what it leaves unknown.
6. **Today, ranked** -- at most five items, each one line with the expected value.
7. **Journal entry** -- a draft `## 2026-09-03` entry for `overnight/JOURNAL.md` in
   that file's style.

Rules: read-only; you may run short python snippets (under 30 s) to read shapes or
JSON, but no matches, no edits, no commits. Every number must come from a file you
read; say which. Distinguish inconclusive, rejected and not run.
