#!/usr/bin/env bash
# Second night: three search switches, each a gated SPRT against the current
# champion, then the contempt decision from its Weiss games, then the morning
# validation, a fresh zip, and the Fable review. Waits for whatever is running.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
# GATED: the shipped files only. The repo-wide sweep is permanently red -- ~99 ruff and
# ~136 mypy errors, all of them in testing/ and training/ scratch audit scripts -- so
# gating on it meant the zip could never be built. These four are clean on both, and are
# the only things the platform ever imports.
# Plus the files whose breakage would corrupt a SHIPPED artifact rather than a scratch
# report: export.py writes weights/net.npz, check_nnue.py is the only guard that a bad
# export is caught, and the testing/ harnesses produce the PROMOTE/REJECT exit code that
# decides what gets copied into agent.py. The ~99 dirty files are all audit_* scratch.
GATED="agent.py fastboard.py fastsearch.py harness training/export.py training/check_nnue.py training/features.py testing/gauntlet.py testing/arena.py testing/sprt.py testing/referee.py testing/openings.py testing/clocktest.py testing/check_fastboard.py testing/check_fastsearch.py testing/mistakes.py testing/replay_eval.py testing/audit_mirrorplay.py testing/check_bundle.py"
STATE=overnight/night2
mkdir -p "$STATE"
MAIN="$STATE/night.log"
SUMMARY="$STATE/SUMMARY.md"
[ -f "$SUMMARY" ] || printf '# Night 2 summary\n\n' > "$SUMMARY"
say()  { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$MAIN"; }
note() { echo "- $(date -u +%H:%M)Z $*" >> "$SUMMARY"; }
finished() { [ -f "$STATE/$1.done" ]; }
mark() { date -u +%FT%TZ > "$STATE/$1.done"; }

challenger() {  # name switch
    local d="overnight/challengers/$1"
    rm -rf "$d"; mkdir -p "$d/weights"
    # fastsearch too: _scan_agent_flags() reads the agent.py next to FASTSEARCH, so a dir
    # without its own copy silently takes HISTORY_V2 (and INIT_FOLD) from the tree --
    # which makes an SPRT that seds those switches measure nothing at all.
    cp agent.py fastboard.py fastsearch.py "$d/"
    sed -i "s/^$2: Final = False\$/$2: Final = True/" "$d/agent.py"
    grep -q "^$2: Final = True\$" "$d/agent.py" || { say "switch $2 not found"; return 1; }
    cp weights/net.npz weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
}
gauntlet() {  # name elo0 elo1 games
    say "gauntlet $1: SPRT[$2, $3], up to $4 games"
    $PY -u -m testing.gauntlet --challenger "overnight/challengers/$1" --elo0 "$2" --elo1 "$3" \
        --games "$4" > "$STATE/$1.gauntlet.log" 2>&1
    local code=$?; tail -n 8 "$STATE/$1.gauntlet.log" >> "$MAIN"; return $code
}
verdict_line() { grep -E "^(PROMOTE|REJECT|INCONCLUSIVE)" "$STATE/$1.gauntlet.log" | tail -n 1; }
promote() {  # name what
    local d="overnight/challengers/$1"
    local b="overnight/champion_backup_$(date -u +%Y%m%dT%H%M%SZ)"
    # Kernels too -- a backup without them is not a runnable engine.
    mkdir -p "$b/weights"; cp agent.py fastboard.py fastsearch.py "$b/"; cp weights/net.npz "$b/weights/"
    cp "$d/agent.py" agent.py
    [ -f "$d/fastboard.py" ] && cp "$d/fastboard.py" fastboard.py
    [ -f "$d/fastsearch.py" ] && cp "$d/fastsearch.py" fastsearch.py
    if $PY -m ruff check $GATED >> "$MAIN" 2>&1 && $PY -m mypy $GATED >> "$MAIN" 2>&1; then
        git add agent.py && git -c core.safecrlf=false commit -q -m "promote $1: $2" \
            -m "Unattended, by overnight/night2.sh. Backup in $b." && say "promoted $1, committed"
        note "**PROMOTED $1** -- $2"
    else
        say "GATE RED after promoting $1 -- restored"
        cp "$b/agent.py" agent.py; cp "$b/fastboard.py" fastboard.py; cp "$b/fastsearch.py" fastsearch.py
        note "$1 passed but the gate went red; NOT promoted"
    fi
}
CLAUDE="$HOME/.local/bin/claude.exe"
review_stage() {  # stage
    if [ ! -x "$CLAUDE" ]; then say "no claude CLI; review of $1 skipped"; return 0; fi
    say "review of stage $1 by Claude Fable"
    sed -e "s#@STATE@#$STATE#g" -e "s#@STAGE@#$1#g" overnight/STAGE_PROMPT.md         | "$CLAUDE" --print --model fable --permission-mode dontAsk         > "$STATE/review-$1.md" 2> "$STATE/review-$1.err"
    say "review-$1.md: $(wc -c < "$STATE/review-$1.md") bytes"
}
switch_stage() {  # stage name switch what
    if finished "$1"; then return 0; fi
    say "=== $2: $3 ==="
    challenger "$2" "$3" || return 1
    gauntlet "$2" 0 20 600; local code=$?
    note "$2 SPRT[0, 20]: $(verdict_line "$2")"
    [ $code -eq 0 ] && promote "$2" "$4"
    # The stage log carries the name the reviewer looks for.
    cp "$STATE/$2.gauntlet.log" "$STATE/$1.gauntlet.log" 2>/dev/null
    mark "$1"
    review_stage "$1"
}

say "=== night 2 starts ==="
say "waiting for the contempt test to finish"
until grep -q "games saved" overnight/eval/contempt.041-vs-weiss-d6.log 2>/dev/null; do sleep 30; done
say "machine free"
cp agent.py "$STATE/agent.start.py"

# The king-zone chain ran before this script; fold its outcome into the record.
if ! finished kingzones; then
    note "king zones: $(grep -E 'committed|failed|promote' overnight/eval/kingzones-chain.log | tr '
' ';')"
    note "4-zone vs 8-zone: $(grep -E '^(PROMOTE|REJECT|INCONCLUSIVE)' overnight/eval/039-kz4-fast.gauntlet.log | tail -n 1)"
    note "120 s, king-zone champion vs pre-king-zone build: $(grep -E 'score' overnight/eval/match.final-kingzones.120s.log | tail -n 1)"
    mark kingzones
    review_stage kingzones
fi

# Contempt: promote if fewer repetition draws and the score against Weiss d6 did
# not fall by more than two points.
if ! finished contempt; then
    C_SCORE=$(grep -oE "score [0-9.]+%" overnight/eval/contempt.champion-vs-weiss-d6.log | grep -oE "[0-9.]+" | head -n 1)
    T_SCORE=$(grep -oE "score [0-9.]+%" overnight/eval/contempt.041-vs-weiss-d6.log | grep -oE "[0-9.]+" | head -n 1)
    C_REP=$(grep -oE "threefold_repetition [0-9]+" overnight/eval/contempt.champion-vs-weiss-d6.log | grep -oE "[0-9]+" | head -n 1); C_REP=${C_REP:-0}
    T_REP=$(grep -oE "threefold_repetition [0-9]+" overnight/eval/contempt.041-vs-weiss-d6.log | grep -oE "[0-9]+" | head -n 1); T_REP=${T_REP:-0}
    say "contempt vs weiss-d6: champion ${C_SCORE}% (${C_REP} repetition draws), challenger ${T_SCORE}% (${T_REP})"
    note "contempt vs weiss-d6: champion ${C_SCORE}% / ${C_REP} repetition draws; CONTEMPT ${T_SCORE}% / ${T_REP}"
    if [ -n "$C_SCORE" ] && [ -n "$T_SCORE" ] && [ "$T_REP" -lt "$C_REP" ] && \
       [ "$($PY -c "print(int(float('$T_SCORE') >= float('$C_SCORE') - 2.0))")" = 1 ]; then
        # Rebuild the challenger from the current champion so nothing else rides along.
        challenger 041-contempt CONTEMPT && promote 041-contempt "repetition contempt: fewer draws vs weiss-d6 at no cost in score"
    else
        note "CONTEMPT not promoted"
    fi
    mark contempt
    review_stage contempt
fi

switch_stage futility 042-futility FUTILITY "futility pruning at depth 1-2"
switch_stage ttage 043-tt-age TT_AGE "transposition table replacement by age and depth"
switch_stage pvs 044-pvs PVS "principal variation search"

if ! finished final; then
    say "=== final: clock replay, 120 s match vs the night's starting build, zip ==="
    $PY -u -m testing.clocktest --agent . --games 6 --factor 1.5 > "$STATE/clock.final.log" 2>&1
    C=$?; tail -n 2 "$STATE/clock.final.log" | tee -a "$MAIN"; note "final clock replay x1.5: $(tail -n 2 "$STATE/clock.final.log" | head -n 1) (exit $C)"
    if ! cmp -s agent.py "$STATE/agent.start.py"; then
        PRE="$STATE/start-build"; rm -rf "$PRE"; mkdir -p "$PRE/weights"
        cp "$STATE/agent.start.py" "$PRE/agent.py"; cp fastboard.py fastsearch.py "$PRE/"; cp weights/net.npz weights/book.bin "$PRE/weights/"; cp -r weights/syzygy "$PRE/weights/"
        $PY -u -m testing.arena --agent . --opponent "$PRE" --games 40 --base-ms 120000 --increment-ms 500 > "$STATE/match.final.120s.log" 2>&1
        tail -n 5 "$STATE/match.final.120s.log" | tee -a "$MAIN"; note "120 s match, morning build vs night start: $(grep -E 'score' "$STATE/match.final.120s.log" | tail -n 1)"
    else
        note "nothing promoted tonight; no 120 s match needed"
    fi
    # No pipefail in this script, so a pipeline reports tee's status: package.py can refuse
    # (missing kernel, missing net), delete the partial zip, exit 1 -- and the loop would
    # log a blank line and mark the night finished with no submission at all.
    if $PY -m harness.package > "$STATE/package.out" 2>&1; then
        head -n 1 "$STATE/package.out" | tee -a "$MAIN"
        # Presence is not importability -- cold-import the zip as the platform will.
        if ! $PY -m testing.check_bundle submission.zip >> "$MAIN" 2>&1; then
            say "BUNDLE FAILED ITS COLD IMPORT -- do not upload"
            note "**submission.zip built but fails a cold import; do not upload**"
            packaged=0
        fi
    else
        cat "$STATE/package.out" >> "$MAIN"; say "PACKAGE FAILED -- no new submission.zip"
        note "**package failed -- submission.zip is stale or absent, do not upload blind**"
        packaged=0
    fi
    { echo; echo "## Switches now on"; grep -E "^(TIME_V2|HYGIENE|FAST_BOARD|CONTEMPT|FUTILITY|TT_AGE|PVS): Final" agent.py; echo; echo "## Commits"; git log --oneline -8; } >> "$SUMMARY"
    # Not marked done unless a zip was actually built: final.done is what a resume skips on.
    [ "${packaged:-1}" = 1 ] && mark final
fi

if ! finished review; then
    say "=== morning report by Claude Fable ==="
    if [ -x "$CLAUDE" ]; then
        sed "s#@STATE@#$STATE#g" overnight/MORNING_PROMPT.md | "$CLAUDE" --print --model fable --permission-mode dontAsk > "$STATE/REPORT.md" 2> "$STATE/report.err"
        say "report exit $?, $(wc -c < "$STATE/REPORT.md") bytes in $STATE/REPORT.md"
    fi
    mark review
fi
say "=== night 2 done ==="
