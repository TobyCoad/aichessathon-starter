#!/bin/bash
# v8 search candidates, one switch each, against the current champion build, after
# the 16-zone net's verdict. Base = 072-kz16 if it passed, else 060-v6.
cd /c/dev/aichessathon/starter || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') $*" >> "$LOG"; }
verdict() { grep -E "^(PROMOTE|REJECT|INCONCLUSIVE)" "$1" 2>/dev/null | tail -n 1; }
until grep -q "queue7 done" "$LOG" 2>/dev/null; do sleep 120; done
BASE=overnight/challengers/060-v6
if grep -qE "^PROMOTE" overnight/eval/072-kz16.gauntlet.log 2>/dev/null; then BASE=overnight/challengers/072-kz16; fi
say "queue8 start (base $BASE)"
build() {  # name, sed expression
    local d="overnight/challengers/$1"
    rm -rf "$d"; mkdir -p "$d/weights"
    cp agent.py fastboard.py fastsearch.py "$d/"
    cp "$BASE/weights/net.npz" "$d/weights/"; cp weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
    sed -i "$2" "$d/agent.py"
}
run() {  # name [env...]
    env "${@:2}" $PY -u -m testing.gauntlet --challenger "overnight/challengers/$1" --champion "$BASE" \
        --elo0 0 --elo1 20 --games 600 > "overnight/eval/$1.gauntlet.log" 2>&1
    say "$1: $(verdict overnight/eval/$1.gauntlet.log)"
}
build 090-history2 's/^HISTORY2: Final = False$/HISTORY2: Final = True/';   run 090-history2
build 091-ttkeep 's/^TT_KEEP: Final = False$/TT_KEEP: Final = True/';       run 091-ttkeep
build 092-qscap14 's/^QS_CAP: Final = 8$/QS_CAP: Final = 14/';               run 092-qscap14
build 093-safe 's/^SAFE_BITS: Final = False$/SAFE_BITS: Final = True/';     run 093-safe
build 094-bookverify 's/^BOOK_VERIFY: Final = False$/BOOK_VERIFY: Final = True/'
run 094-bookverify GAUNTLET_OPENINGS=platform
say "queue8 done"
