#!/bin/bash
# The desktop's worker. Loops forever: pull main, read overnight/desktop/tasks.json,
# run the first task without a result, commit the result to main, push, repeat.
# The laptop adds tasks by editing tasks.json on main; results are files under
# overnight/desktop/results/, so the two machines never edit the same file.
#
#   bash overnight/desktop_worker.sh        # run from the repo root on the desktop
#
# The desktop never edits code: every pull is a hard reset to origin/main plus the
# results it has committed (rebased on top). Challenger builds live in
# overnight/challengers/ (ignored by git).
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
RESULTS=overnight/desktop/results
# Provision the environment if it is missing or incomplete (same pins as the laptop).
if ! [ -x "$PY" ] || ! "$PY" -c "import chess, numba, numpy" 2>/dev/null; then
    echo "$(date '+%H:%M') setting up .venv"
    python -m venv .venv || py -3.12 -m venv .venv || exit 1
    "$PY" -m pip install -q "numpy==2.5.2" "numba==0.67.0" "chess==1.11.2" || exit 1
fi
"$PY" -c "import chess, numba, numpy; print('env ok', numba.__version__)" || exit 1
mkdir -p "$RESULTS" overnight/challengers overnight/eval
say() { echo "$(date '+%H:%M') $*"; }

sync_down() {
    git fetch origin main >/dev/null 2>&1 || { say "fetch failed"; return 1; }
    git pull --rebase origin main >/dev/null 2>&1 || { say "pull --rebase failed; resetting"; git rebase --abort 2>/dev/null; git reset --hard origin/main; }
}
push_up() {
    for attempt in 1 2 3 4 5; do
        git push origin main >/dev/null 2>&1 && return 0
        git pull --rebase origin main >/dev/null 2>&1 || git rebase --abort 2>/dev/null
        sleep $((attempt * 20))
    done
    say "push failed after retries"; return 1
}
next_task() {  # prints the JSON of the first task without a result, or nothing
    $PY - <<'EOF'
import json, pathlib
tasks = json.loads(pathlib.Path("overnight/desktop/tasks.json").read_text(encoding="utf-8"))
done = {p.stem for p in pathlib.Path("overnight/desktop/results").glob("*.txt")}
for task in tasks:
    if task["name"] not in done:
        print(json.dumps(task)); break
EOF
}
run_task() {  # $1 = task json
    local name kind sed_expr champion games openings workers base elo0 elo1 tc
    name=$(echo "$1" | $PY -c "import json,sys; print(json.load(sys.stdin)['name'])")
    kind=$(echo "$1" | $PY -c "import json,sys; print(json.load(sys.stdin).get('kind','switch'))")
    sed_expr=$(echo "$1" | $PY -c "import json,sys; print(json.load(sys.stdin).get('sed',''))")
    champion=$(echo "$1" | $PY -c "import json,sys; print(json.load(sys.stdin).get('champion','.'))")
    games=$(echo "$1" | $PY -c "import json,sys; print(json.load(sys.stdin).get('games',600))")
    openings=$(echo "$1" | $PY -c "import json,sys; print(json.load(sys.stdin).get('openings','default'))")
    workers=$(echo "$1" | $PY -c "import json,sys; print(json.load(sys.stdin).get('workers',0))")
    elo0=$(echo "$1" | $PY -c "import json,sys; print(json.load(sys.stdin).get('elo0',0))")
    elo1=$(echo "$1" | $PY -c "import json,sys; print(json.load(sys.stdin).get('elo1',20))")
    tc=$(echo "$1" | $PY -c "import json,sys; print(json.load(sys.stdin).get('base_ms',8000))")
    say "task $name ($kind)"
    local d="overnight/challengers/$name"
    rm -rf "$d"; mkdir -p "$d/weights"
    cp agent.py fastboard.py fastsearch.py "$d/"
    cp weights/net.npz weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
    [ -n "$sed_expr" ] && sed -i "$sed_expr" "$d/agent.py"
    local log="$RESULTS/$name.gauntlet.log"
    if [ "$kind" = "clocktest" ]; then
        $PY -u -m testing.clocktest --agent "$d" --workers 4 > "$log" 2>&1
        { echo "name $name"; echo "kind clocktest"; echo "host $(hostname)"; grep -E "^flags|PASS|FAIL" "$log"; } > "$RESULTS/$name.txt"
    else
        local inc=$(( tc / 200 ))  # 8000 -> 40? keep the harness defaults: 8 s + 80 ms, 120 s + 500 ms
        [ "$tc" -ge 100000 ] && inc=500 || inc=80
        GAUNTLET_OPENINGS=$([ "$openings" = platform ] && echo platform || echo "") \
        $PY -u -m testing.gauntlet --challenger "$d" --champion "$champion" --elo0 "$elo0" --elo1 "$elo1" \
            --games "$games" --workers "$workers" --base-ms "$tc" --increment-ms "$inc" > "$log" 2>&1
        { echo "name $name"; echo "kind $kind"; echo "host $(hostname)"; echo "base_ms $tc openings $openings games $games";
          grep -E "^(PROMOTE|REJECT|INCONCLUSIVE)" "$log" | tail -n 1; grep -E "score|terminations" "$log" | tail -n 2; } > "$RESULTS/$name.txt"
    fi
    if ! grep -qE "^(PROMOTE|REJECT|INCONCLUSIVE|flags)" "$RESULTS/$name.txt"; then
        # No verdict: the run failed. Keep the log, drop the result so the task retries later.
        rm -f "$RESULTS/$name.txt"
        say "task $name produced no verdict; see $log"; tail -n 5 "$log"
        sleep 300
        return 1
    fi
    cat "$RESULTS/$name.txt"
    git add "$RESULTS/$name.txt" "$log"
    git -c user.name=desktop -c user.email=desktop@local commit -q -m "desktop: $name" && push_up
}

say "worker start on $(hostname)"
while true; do
    sync_down
    task=$(next_task)
    if [ -z "$task" ]; then
        sleep 120
        continue
    fi
    run_task "$task"
done
