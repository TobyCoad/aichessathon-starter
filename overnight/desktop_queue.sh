#!/bin/bash
# The desktop's share of the v8 gauntlets. Champion = this checkout (v7.1: the v6
# engine with the 16-zone net, every switch off). Each challenger is the checkout
# with one switch on. Run from the repo root in Git Bash:
#   bash overnight/desktop_queue.sh
# Results: overnight/eval/*.gauntlet.log and one line per verdict in overnight/eval/desktop.log.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
[ -x "$PY" ] || PY=python
LOG=overnight/eval/desktop.log
mkdir -p overnight/eval overnight/challengers
say() { echo "$(date '+%H:%M') $*" | tee -a "$LOG"; }
verdict() { grep -E "^(PROMOTE|REJECT|INCONCLUSIVE)" "$1" 2>/dev/null | tail -n 1; }
build() {  # name, sed expression
    local d="overnight/challengers/$1"
    rm -rf "$d"; mkdir -p "$d/weights"
    cp agent.py fastboard.py fastsearch.py "$d/"
    cp weights/net.npz weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
    sed -i "$2" "$d/agent.py"
    grep -q "Final = True$\|Final = 14$" "$d/agent.py" || { say "switch not applied for $1"; return 1; }
}
run() {  # name
    $PY -u -m testing.gauntlet --challenger "overnight/challengers/$1" --champion . \
        --elo0 0 --elo1 20 --games 600 > "overnight/eval/$1.gauntlet.log" 2>&1
    say "$1: $(verdict overnight/eval/$1.gauntlet.log)"
}
say "desktop queue start"
build 091-ttkeep 's/^TT_KEEP: Final = False$/TT_KEEP: Final = True/' && run 091-ttkeep
build 092-qscap14 's/^QS_CAP: Final = 8$/QS_CAP: Final = 14/' && run 092-qscap14
build 093-safe 's/^SAFE_BITS: Final = False$/SAFE_BITS: Final = True/' && run 093-safe
say "desktop queue done"
