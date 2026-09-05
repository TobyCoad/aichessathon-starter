#!/bin/bash
# v7 assembly and gates: v6 + PONDER (+ the 16-zone net if its gauntlet passed).
# Pondering thinks on the opponent's time, so every match here runs 6 workers and the
# clock replay (both sides in one process) is replaced by the 120 s games' flag count.
cd /c/dev/aichessathon/starter || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
REPORT=overnight/eval/V7_REPORT.md
say() { echo "$(date '+%H:%M') $*" >> "$LOG"; }
verdict() { grep -E "^(PROMOTE|REJECT|INCONCLUSIVE)" "$1" 2>/dev/null | tail -n 1; }
passed() { grep -qE "^PROMOTE" "$1" 2>/dev/null; }
until grep -q "queue5 done" "$LOG" 2>/dev/null; do sleep 120; done
say "night4: assembling v7 (pondering only; nets follow as v7.1)"

BASE=overnight/challengers/060-v6
V7=overnight/challengers/080-v7
rm -rf "$V7"; cp -r "$BASE" "$V7"
sed -i 's/^PONDER: Final = False$/PONDER: Final = True/' "$V7/agent.py"
ON="PONDER"
if passed overnight/eval/072-kz16.gauntlet.log; then
    cp overnight/challengers/072-kz16/weights/net.npz "$V7/weights/net.npz"; ON="$ON NET16"
elif passed overnight/eval/070-kz8c.gauntlet.log; then
    cp overnight/challengers/070-kz8c/weights/net.npz "$V7/weights/net.npz"; ON="$ON NET8C"
fi
say "v7 on: $ON"

$PY -u -m testing.gauntlet --challenger "$V7" --champion "$BASE" --elo0 0 --elo1 20 --games 600 --workers 6 \
    > overnight/eval/080-v7.gauntlet.log 2>&1
say "080-v7 vs v6: $(verdict overnight/eval/080-v7.gauntlet.log)"
$PY -u -m testing.gauntlet --challenger "$V7" --champion "$BASE" --elo0 900 --elo1 950 --games 200 --workers 6 \
    > overnight/eval/080-v7.crash.log 2>&1
say "crash hunt: $(grep -E 'terminations|score' overnight/eval/080-v7.crash.log | tail -n 2 | tr '\n' ' ')"
$PY -u -m testing.gauntlet --challenger "$V7" --champion "$BASE" --elo0 -1000 --elo1 1000 --games 40 \
    --base-ms 120000 --increment-ms 500 --workers 4 > overnight/eval/080-v7.120s.log 2>&1
say "120s vs v6: $(grep -E 'score|terminations' overnight/eval/080-v7.120s.log | tail -n 2 | tr '\n' ' ')"
{
    echo "# v7 candidate -- $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "On top of v6: $ON"
    echo
    for n in 061-ponder 065-nobook 063-pvs 066-iir 062-nmpguard 072-kz16 070-kz8c 080-v7; do
        echo "- $n: $(verdict overnight/eval/$n.gauntlet.log)"
    done
    echo
    echo "Crash hunt (200, 6 workers): $(grep -E 'terminations' overnight/eval/080-v7.crash.log | tail -n 1)"
    echo "120 s vs v6 (40 games, 4 workers): $(grep -E 'score' overnight/eval/080-v7.120s.log | tail -n 1) / $(grep -E 'terminations' overnight/eval/080-v7.120s.log | tail -n 1)"
    echo
    echo "Endgame suite: $(for n in kz8c kz8w kz16 kz32b; do echo -n "$n=$(grep -E 'positions at' overnight/eval/suite-$n.log 2>/dev/null | sed 's/.*mean loss //; s/,.*//') "; done)"
} > "$REPORT"
say "night4 done -> $REPORT"
