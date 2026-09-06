#!/bin/bash
# Gauntlet worker, one per machine:  bash overnight/worker.sh laptop|desktop
# Loops: pull main, take the first task in overnight/<role>/tasks.json without a
# result in overnight/<role>/results/, wait for a free CPU, run it, commit the
# result to main, push, repeat. Tasks and results are separate files, so the
# machines and the improvement loop never edit the same file.
ROLE="${1:-desktop}"
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
TASKS="overnight/$ROLE/tasks.json"
RESULTS="overnight/$ROLE/results"
if ! [ -x "$PY" ] || ! "$PY" -c "import chess, numba, numpy" 2>/dev/null; then
    echo "$(date '+%H:%M') setting up .venv"
    python -m venv .venv || py -3.12 -m venv .venv || exit 1
    "$PY" -m pip install -q "numpy==2.5.2" "numba==0.67.0" "chess==1.11.2" "pyarrow==21.0.0" || exit 1
fi
"$PY" -c "import chess, numba, numpy; print('env ok', numba.__version__)" || exit 1
mkdir -p "$RESULTS" overnight/challengers overnight/eval
say() { echo "$(date '+%H:%M') [$ROLE] $*"; }

sync_down() {
    git fetch origin main >/dev/null 2>&1 || { say "fetch failed"; return 1; }
    git stash push -q -m worker-pull >/dev/null 2>&1
    if ! git pull --rebase origin main >/dev/null 2>&1; then
        git rebase --abort 2>/dev/null
        if [ "$ROLE" = "desktop" ]; then say "pull --rebase failed; resetting"; git reset --hard origin/main
        else say "pull --rebase failed; the laptop tree is never reset -- retrying later"; fi
    fi
    git stash pop -q >/dev/null 2>&1
}
push_up() {
    for attempt in 1 2 3 4 5; do
        git push origin main >/dev/null 2>&1 && return 0
        git pull --rebase origin main >/dev/null 2>&1 || git rebase --abort 2>/dev/null
        sleep $((attempt * 20))
    done
    say "push failed after retries"; return 1
}
reap_orphans() {  # python pool workers whose parent died (a Ctrl-C'd gauntlet) keep playing games
    powershell -NoProfile -Command "\$all = Get-CimInstance Win32_Process; \$ids = @{}; foreach (\$p in \$all) { \$ids[\$p.ProcessId] = 1 }; \$o = \$all | Where-Object { \$_.Name -match 'python' -and -not \$ids.ContainsKey(\$_.ParentProcessId) -and \$_.CommandLine -match 'multiprocessing|harness' }; foreach (\$x in \$o) { Stop-Process -Id \$x.ProcessId -Force -ErrorAction SilentlyContinue }; \"reaped \$(\$o.Count) orphans\"" 2>/dev/null | tr -d '
'
}
busy_gauntlets() {  # other gauntlets or clock tests running on this machine
    # python processes only: the query's own PowerShell command line would match itself
    powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -match 'python' -and \$_.CommandLine -match 'testing.gauntlet|testing.clocktest|binpack_decode|endgame_suite' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | tail -n 1
}
heartbeat() {  # task, log, pid: notice the end within 30 s, commit progress every 10 min
    local waited=0
    while kill -0 "$3" 2>/dev/null; do
        sleep 30; waited=$((waited + 30))
        kill -0 "$3" 2>/dev/null || break
        [ $waited -lt 600 ] && continue
        waited=0
        { echo "host $(hostname)"; echo "time $(date '+%Y-%m-%d %H:%M')"; echo "task $1"; echo "progress $(grep -E 'games' "$2" | tail -n 1)"; } > "overnight/$ROLE/heartbeat.txt"
        git add "overnight/$ROLE/heartbeat.txt"
        git -c user.name="$ROLE" -c user.email="$ROLE@local" commit -q -m "$ROLE heartbeat: $1" && push_up
    done
}
next_task() {
    TASKS="$TASKS" RESULTS="$RESULTS" $PY - <<'EOF'
import json, os, pathlib
tasks = json.loads(pathlib.Path(os.environ["TASKS"]).read_text(encoding="utf-8"))
done = {p.stem for p in pathlib.Path(os.environ["RESULTS"]).glob("*.txt")}
for task in tasks:
    if task["name"] not in done:
        print(json.dumps(task)); break
EOF
}
field() { echo "$1" | $PY -c "import json,sys; t=json.load(sys.stdin); print(t.get('$2', '$3'))"; }
run_task() {
    local task="$1"
    local name kind sed_expr champion games openings workers elo0 elo1 tc inc
    name=$(field "$task" name ""); kind=$(field "$task" kind switch); sed_expr=$(field "$task" sed "")
    champion=$(field "$task" champion .); games=$(field "$task" games 600); openings=$(field "$task" openings default)
    workers=$(field "$task" workers 0); elo0=$(field "$task" elo0 0); elo1=$(field "$task" elo1 20); tc=$(field "$task" base_ms 8000)
    say "task $name ($kind)"
    local d="overnight/challengers/$name"
    local net; net=$(field "$task" net "")
    # A net task may point INSIDE the challenger dir (a training chain exported it there):
    # stage it first, because the dir is rebuilt from the tree below. (150-sfnet was tested
    # against itself because of exactly this.)
    if [ -n "$net" ]; then
        mkdir -p overnight/nets
        if [ "$net" != "overnight/nets/$name.npz" ]; then
            cp "$net" "overnight/nets/$name.npz" || { say "net $net missing"; return 1; }
        fi
        net="overnight/nets/$name.npz"
    fi
    rm -rf "$d"; mkdir -p "$d/weights"
    cp agent.py fastboard.py fastsearch.py "$d/"
    cp weights/net.npz weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
    if [ -n "$net" ]; then
        cp "$net" "$d/weights/net.npz"   # a task may test a different net
        cmp -s "$net" weights/net.npz && { say "net task $name: the net equals the tree net -- nothing to test"; return 1; }
    fi
    local book; book=$(field "$task" book "")
    if [ -n "$book" ] && ! cp "$book" "$d/weights/book.bin"; then
        say "task $name: book $book missing -- waiting for it to land"; sleep 120; return 1
    fi
    [ -n "$sed_expr" ] && sed -i "$sed_expr" "$d/agent.py"
    local log="$RESULTS/$name.gauntlet.log"
    reap_orphans
    while [ "$(busy_gauntlets)" != "0" ] && [ -n "$(busy_gauntlets)" ]; do sleep 60; done
    if [ "$kind" = "generate" ]; then
        # self-play positions labelled by Stockfish; the Parquet is committed as a result
        local games nodes movetime
        games=$(field "$task" games 2000); nodes=$(field "$task" nodes 5000); movetime=$(field "$task" movetime_ms 40)
        $PY -m pip install -q "pyarrow==21.0.0" >/dev/null 2>&1
        mkdir -p "$RESULTS/data"
        [ "$workers" = "0" ] && workers=$(( $(nproc 2>/dev/null || echo 8) - 2 ))
        $PY -u -m training.generate --games "$games" --nodes "$nodes" --movetime-ms "$movetime"             --workers "$workers" --out "$RESULTS/data/$name.parquet" > "$log" 2>&1 &
        heartbeat "$name" "$log" $!; wait
        { echo "name $name"; echo "kind generate"; echo "host $(hostname)"; grep -E "^wrote" "$log" | tail -n 1 | sed 's/^/flags /'; } > "$RESULTS/$name.txt"
        git add "$RESULTS/data/$name.parquet" 2>/dev/null
    elif [ "$kind" = "clocktest" ]; then
        $PY -u -m testing.clocktest --agent "$d" --workers 4 > "$log" 2>&1 &
        heartbeat "$name" "$log" $!; wait
        { echo "name $name"; echo "kind clocktest"; echo "host $(hostname)"; grep -E "^flags|PASS|FAIL" "$log"; } > "$RESULTS/$name.txt"
    else
        [ "$tc" -ge 100000 ] && inc=500 || inc=80
        GAUNTLET_OPENINGS=$([ "$openings" = platform ] && echo platform || echo "") \
        $PY -u -m testing.gauntlet --challenger "$d" --champion "$champion" --elo0 "$elo0" --elo1 "$elo1" \
            --games "$games" --workers "$workers" --base-ms "$tc" --increment-ms "$inc" > "$log" 2>&1 &
        heartbeat "$name" "$log" $!; wait
        { echo "name $name"; echo "kind $kind"; echo "host $(hostname)"; echo "base_ms $tc openings $openings games $games";
          grep -E "^(PROMOTE|REJECT|INCONCLUSIVE)" "$log" | tail -n 1; grep -E "score|terminations" "$log" | tail -n 2; } > "$RESULTS/$name.txt"
    fi
    if ! grep -qE "^(PROMOTE|REJECT|INCONCLUSIVE|flags)" "$RESULTS/$name.txt"; then
        rm -f "$RESULTS/$name.txt"
        say "task $name produced no verdict; see $log"; tail -n 5 "$log"
        sleep 60
        return 1
    fi
    cat "$RESULTS/$name.txt"
    git add "$RESULTS/$name.txt" "$log"
    git -c user.name="$ROLE" -c user.email="$ROLE@local" commit -q -m "$ROLE: $name" && push_up
}

say "worker start on $(hostname)"
while true; do
    sync_down
    task=$(next_task)
    if [ -z "$task" ]; then sleep 120; continue; fi
    run_task "$task"
done
