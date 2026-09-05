#!/bin/bash
# Continuous improvement loop (after github.com/AnandChowdhary/continuous-claude):
# every ITERATION_MINUTES, pull the repo, fetch the platform's games, run one
# Claude Code iteration on overnight/continuous/PROMPT.md with the shared notes as
# its memory, then push. Gauntlets run on the two workers; the loop only edits,
# measures, queues and folds results.
#
#   bash overnight/continuous/loop.sh            # runs until stopped (Ctrl+C)
#   MAX_RUNS=5 bash overnight/continuous/loop.sh
cd "$(dirname "$0")/../.." || exit 1
PY=./.venv/Scripts/python.exe
CLAUDE="${CLAUDE:-$HOME/.local/bin/claude.exe}"
LOG=overnight/continuous/loop.log
ITERATION_MINUTES="${ITERATION_MINUTES:-25}"
MAX_RUNS="${MAX_RUNS:-0}"
say() { echo "$(date '+%Y-%m-%d %H:%M') $*" | tee -a "$LOG"; }
[ -x "$CLAUDE" ] || { say "no claude CLI at $CLAUDE"; exit 1; }
say "loop start (every $ITERATION_MINUTES min)"
run=0
errors=0
while true; do
    run=$((run + 1))
    say "iteration $run"
    git stash push -q -m loop-pull >/dev/null 2>&1; git pull --rebase origin main >/dev/null 2>&1 || say "pull failed"; git stash pop -q >/dev/null 2>&1
    $PY -m testing.fetch_games --team "make_no_mistakes" > overnight/continuous/fetch.log 2>&1 || true
    timeout 45m "$CLAUDE" --print --model fable --dangerously-skip-permissions --max-turns 80 \
        "$(cat overnight/continuous/PROMPT.md)" > "overnight/continuous/iter-$run.log" 2>&1
    code=$?
    if [ $code -ne 0 ]; then
        errors=$((errors + 1)); say "iteration $run exited $code (errors in a row: $errors)"
        [ $errors -ge 3 ] && { say "three failures in a row; pausing 1 h"; sleep 3600; errors=0; }
    else
        errors=0
        say "iteration $run done: $(tail -c 300 "overnight/continuous/iter-$run.log" | tr '\n' ' ')"
    fi
    git push origin main >/dev/null 2>&1 || { git pull --rebase origin main >/dev/null 2>&1; git push origin main >/dev/null 2>&1 || say "push failed"; }
    [ "$MAX_RUNS" -gt 0 ] && [ $run -ge "$MAX_RUNS" ] && { say "max runs reached"; break; }
    sleep $((ITERATION_MINUTES * 60))
done
