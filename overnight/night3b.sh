#!/bin/bash
# Night 3b: once every verdict is in, assemble v6 from what passed and gate it.
# Waits for the last queued verdict (057-twofold) in night3.log, then:
#   1. switches on every PROMOTEd search switch; takes the 32-zone net if it passed
#   2. SPRT of the bundle vs 050 (the v5 build) at 8 s
#   3. 500-game crash hunt (no early stop), clock replay at 120 s x1.5
#   4. 40 games at 120 s + 0.5 s vs 050
#   5. zip + REPORT for the morning; nothing is uploaded by this script
cd /c/dev/aichessathon/starter || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
REPORT=overnight/eval/V6_REPORT.md
say() { echo "$(date '+%H:%M') $*" >> "$LOG"; }
verdict() { grep -E "^(PROMOTE|REJECT|INCONCLUSIVE)" "$1" 2>/dev/null | tail -n 1; }
passed() { grep -qE "^PROMOTE" "$1" 2>/dev/null; }

until grep -q "057-twofold:" "$LOG" 2>/dev/null && grep -q "059-kz32b:" "$LOG" 2>/dev/null; do sleep 120; done
say "night3b: assembling v6"

BASE=overnight/challengers/050-compiled-search
V6=overnight/challengers/060-v6
rm -rf "$V6"; mkdir -p "$V6/weights"
cp agent.py fastboard.py fastsearch.py "$V6/"
sed -i 's/^COMPILED_SEARCH: Final = False$/COMPILED_SEARCH: Final = True/' "$V6/agent.py"
cp weights/book.bin "$V6/weights/"; cp -r weights/syzygy "$V6/weights/"
ON=""
turn_on() {  # switch, gauntlet log
    if passed "overnight/eval/$2.gauntlet.log"; then
        sed -i "s/^$1: Final = False\$/$1: Final = True/" "$V6/agent.py"
        ON="$ON $1"
    fi
}
# LMR: the fixed variant if its run passed, else the original promoted run
if passed overnight/eval/052b-lmr.gauntlet.log || passed overnight/eval/052-lmr.gauntlet.log; then
    sed -i 's/^LMR: Final = False$/LMR: Final = True/' "$V6/agent.py"; ON="$ON LMR"
fi
turn_on PVS 051-pvs
turn_on LMP 053-lmp
turn_on ASPIRATION 054-aspiration
turn_on SEE 055-see
turn_on REPETITION_TWOFOLD 057-twofold
# 056 was the untrained tiled net on the v5.5 engine (void as a net test); 059 is
# the retrained 32-zone net measured against the v5.5 engine with the old net.
if passed overnight/eval/059-kz32b.gauntlet.log; then
    cp overnight/challengers/059-kz32b/weights/net.npz "$V6/weights/net.npz"; ON="$ON NET32"
else
    cp weights/net.npz "$V6/weights/"
fi
say "v6 switches:$ON"
grep -E "^(COMPILED_SEARCH|LMR|LMP|PVS|ASPIRATION|SEE|REPETITION_TWOFOLD): Final" "$V6/agent.py" > overnight/eval/v6-switches.txt

# 2. bundle vs 050
$PY -u -m testing.gauntlet --challenger "$V6" --champion "$BASE" --elo0 0 --elo1 20 --games 600 \
    > overnight/eval/060-v6.gauntlet.log 2>&1
say "060-v6 vs 050: $(verdict overnight/eval/060-v6.gauntlet.log)"

# 3. crash hunt (SPRT bounds that cannot resolve, so all 500 games play) + clock replay
$PY -u -m testing.gauntlet --challenger "$V6" --champion "$BASE" --elo0 900 --elo1 950 --games 500 \
    > overnight/eval/060-v6.crash.log 2>&1
say "crash hunt: $(grep -E 'terminations|score' overnight/eval/060-v6.crash.log | tail -n 2 | tr '\n' ' ')"
$PY -u -m testing.clocktest --agent "$V6" --workers 4 > overnight/eval/060-v6.clocktest.log 2>&1
say "clocktest: $(tail -n 1 overnight/eval/060-v6.clocktest.log) $(grep -E '^flags' overnight/eval/060-v6.clocktest.log)"

# 4. 40 games at the tournament control vs 050
$PY -u -m testing.gauntlet --challenger "$V6" --champion "$BASE" --elo0 -1000 --elo1 1000 --games 40 \
    --base-ms 120000 --increment-ms 500 --workers 8 > overnight/eval/060-v6.120s.log 2>&1
say "120s vs 050: $(grep -E 'score' overnight/eval/060-v6.120s.log | tail -n 1)"

# 5. report; the zip is built by hand in the morning after reading it
{
    echo "# v6 candidate -- $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "Switches on:$ON"
    echo
    for n in 052-lmr 051-pvs 053-lmp 054-aspiration 055-see 056-kz32 059-kz32b 052b-lmr 057-twofold 060-v6; do
        echo "- $n: $(verdict overnight/eval/$n.gauntlet.log)"
    done
    echo
    echo "Crash hunt: $(grep -E 'terminations' overnight/eval/060-v6.crash.log | tail -n 1)"
    echo "Clock replay: $(grep -E '^flags' overnight/eval/060-v6.clocktest.log) -> $(tail -n 1 overnight/eval/060-v6.clocktest.log)"
    echo "120 s vs 050: $(grep -E 'score' overnight/eval/060-v6.120s.log | tail -n 1)"
    echo
    echo "Training (kz32b): $(grep -E 'restored|wrote.*json' overnight/eval/train-kz32b.log | tail -n 2 | tr '\n' ' ')"
    echo "NNUE check: $(tail -n 1 overnight/eval/check_nnue-kz32b.log 2>/dev/null)"
} > "$REPORT"
say "night3b done -> $REPORT"
