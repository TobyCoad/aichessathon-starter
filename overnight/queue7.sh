#!/bin/bash
# Net gauntlets one at a time after queue5: kz16 (best validation) then kz8c.
cd /c/dev/aichessathon/starter || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') $*" >> "$LOG"; }
verdict() { grep -E "^(PROMOTE|REJECT|INCONCLUSIVE)" "$1" 2>/dev/null | tail -n 1; }
BASE=overnight/challengers/060-v6
until grep -q "queue5 done" "$LOG" 2>/dev/null; do sleep 120; done
for pair in "kz16:072" "kz8c:070"; do
    name=${pair%%:*}; num=${pair#*:}; d="overnight/challengers/$num-$name"
    # the 8-zone control only matters if the 16-zone net failed
    if [ "$name" = "kz8c" ] && grep -qE "^PROMOTE" overnight/eval/072-kz16.gauntlet.log 2>/dev/null; then continue; fi
    if [ ! -f "$d/weights/net.npz" ]; then
        rm -rf "$d"; cp -r "$BASE" "$d"
        $PY -u -m training.export --checkpoint "training/checkpoints/net_w512-b8-$name.pt" --out "$d/weights/net.npz" --half > "overnight/eval/export-$name.log" 2>&1
        $PY -u -m training.check_nnue --agent "$d" --checkpoint "training/checkpoints/net_w512-b8-$name.pt" > "overnight/eval/check_nnue-$name.log" 2>&1 || { say "CHECK_NNUE $name FAILED"; continue; }
    fi
    if [ ! -f "overnight/eval/suite-$name.log" ]; then
        $PY -u -m testing.endgame_suite run --agent "$d" --seconds 2.5 > "overnight/eval/suite-$name.log" 2>&1
        say "suite $name: $(grep 'mean loss' overnight/eval/suite-$name.log | tail -n 1)"
    fi
    $PY -u -m testing.gauntlet --challenger "$d" --champion "$BASE" --elo0 0 --elo1 20 --games 600 > "overnight/eval/$num-$name.gauntlet.log" 2>&1
    say "$num-$name: $(verdict overnight/eval/$num-$name.gauntlet.log)"
done
say "queue7 done"
