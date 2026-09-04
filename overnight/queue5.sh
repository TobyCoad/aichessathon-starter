#!/bin/bash
# v7 search candidates, one switch each on top of v6, after night3b has finished
# its gates (the CPU is free then). Each is judged against 060-v6.
cd /c/dev/aichessathon/starter || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') $*" >> "$LOG"; }
verdict() { grep -E "^(PROMOTE|REJECT|INCONCLUSIVE)" "$1" 2>/dev/null | tail -n 1; }
until grep -q "night3b done" "$LOG" 2>/dev/null; do sleep 120; done
BASE=overnight/challengers/060-v6
if [ ! -f "$BASE/agent.py" ]; then say "queue5: no 060-v6, using 058-v5.5"; BASE=overnight/challengers/058-v5.5; fi
say "queue5 start (base $BASE)"

build() {  # name switch
    local d="overnight/challengers/$1"
    rm -rf "$d"; cp -r "$BASE" "$d"
    sed -i "s/^$2: Final = False\$/$2: Final = True/; s/^$2: Final = True\$/$2: Final = True/" "$d/agent.py"
}
run() {  # name workers [env]
    $PY -u -m testing.gauntlet --challenger "overnight/challengers/$1" --champion "$BASE" \
        --elo0 0 --elo1 20 --games 600 --workers "$2" > "overnight/eval/$1.gauntlet.log" 2>&1
    say "$1: $(verdict overnight/eval/$1.gauntlet.log)"
}

# pondering: the challenger's processes think on the opponent's time, so fewer workers
build 061-ponder PONDER;        run 061-ponder 6
build 064-rfpphase RFP_PHASE;   run 064-rfpphase 12
# the book, judged on the platform's own start positions
d=overnight/challengers/065-nobook; rm -rf "$d"; cp -r "$BASE" "$d"
sed -i 's/^BOOK_ENABLED: Final = True$/BOOK_ENABLED: Final = False/' "$d/agent.py"
GAUNTLET_OPENINGS=platform $PY -u -m testing.gauntlet --challenger "$d" --champion "$BASE" \
    --elo0 0 --elo1 20 --games 600 --workers 12 > overnight/eval/065-nobook.gauntlet.log 2>&1
say "065-nobook (platform openings): $(verdict overnight/eval/065-nobook.gauntlet.log)"
build 063-pvs PVS;              run 063-pvs 12
build 066-iir IIR;              run 066-iir 12
build 062-nmpguard NMP_GUARD;   run 062-nmpguard 12
say "queue5 done"
