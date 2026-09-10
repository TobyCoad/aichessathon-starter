#!/bin/bash
# Conversion switches (WIN_FOCUS + CONVERT_BUDGET) on the shipped ADJ_V2 config. Invisible at
# 8 s, so: clocktest at the real 120 s control first, then a 40-game 120 s match vs the
# shipped config. Both write to overnight/eval; night4.log gets one line per step.
cd /c/dev/aichessathon/starter
L=overnight/eval/night4.log
echo "$(date +%H:%M) conv chain: clocktest conv-adjv2 (6 x 120 s, x1.5 charge) starting" >> $L
./.venv/Scripts/python.exe -m testing.clocktest --agent overnight/challengers/conv-adjv2 --games 6 --workers 4 > overnight/eval/conv-adjv2.clocktest.log 2>&1
V=$(grep -E "PASS|FAIL" overnight/eval/conv-adjv2.clocktest.log | tail -1)
echo "$(date +%H:%M) conv chain: clocktest $V" >> $L
if echo "$V" | grep -q PASS; then
  echo "$(date +%H:%M) conv chain: 40 x 120 s conv-adjv2 vs adjv2-shipped starting (workers 4, no crash gate needed: clocktest passed)" >> $L
  ./.venv/Scripts/python.exe -m testing.gauntlet --challenger overnight/challengers/conv-adjv2 --champion overnight/challengers/adjv2-shipped --games 40 --base-ms 120000 --increment-ms 500 --workers 4 --checkpoint 0 > overnight/eval/conv-vs-adjv2-120s.gauntlet.log 2>&1
  echo "$(date +%H:%M) conv chain: 120 s match done: $(grep -E '^(PROMOTE|REJECT|INCONCLUSIVE)' overnight/eval/conv-vs-adjv2-120s.gauntlet.log | tail -1) | $(grep -E 'games' overnight/eval/conv-vs-adjv2-120s.gauntlet.log | tail -1)" >> $L
else
  echo "$(date +%H:%M) conv chain: clocktest did not pass, 120 s match skipped" >> $L
fi
echo "conv chain done" >> $L
