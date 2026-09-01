#!/usr/bin/env bash
# One unattended night, stage by stage, every stage gated and every promotion a
# commit. Rerunning resumes: a finished stage leaves a `.done` marker and is
# skipped. Stages run strictly one at a time -- a match and a training run that
# share the machine produced 46 timeout flags and one invalidated result already.
#
#   1  clock     TIME_V2      clock-safety replay at 120s+0.5s, then a not-worse SPRT
#   2  shard     pack a second 145M-position shard from row groups the first never read
#   3  train     continue the 512-wide net on both shards, export, verify, SPRT
#   4  evasions  QS_EVASIONS   SPRT[0, 20]
#   5  staged    STAGED_MOVEGEN SPRT[0, 20]
#   6  hygiene   HYGIENE       not-worse SPRT
#   7  final     ruff, mypy, submission.zip, summary
#
# NIGHT_DRY=1 runs the identical chain with tiny parameters and promotes nothing.
set -u
cd "$(dirname "$0")/.." || exit 1

PY=./.venv/Scripts/python.exe
DRY=${NIGHT_DRY:-0}
STATE=overnight/night${NIGHT_TAG:-}
mkdir -p "$STATE"
MAIN="$STATE/night.log"
SUMMARY="$STATE/SUMMARY.md"
[ -f "$SUMMARY" ] || printf '# Night summary\n\n' > "$SUMMARY"

say()  { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$MAIN"; }
note() { echo "- $(date -u +%H:%M)Z $*" >> "$SUMMARY"; }
finished() { [ -f "$STATE/$1.done" ]; }
mark() { date -u +%FT%TZ > "$STATE/$1.done"; }

if [ "$DRY" = 1 ]; then
    CT_GAMES=2; CT_BASE=8000; CT_INC=100; CT_MIN=300
    NW_GAMES=6; SPRT_GAMES=6; NET_GAMES=6
    GAUNTLET_EXTRA="--base-ms 1500 --increment-ms 20"
    PACK_TARGET=300000; PACK_VAL=20000
    SHARD=data/dry_shard.npy; SHARD_VAL=data/dry_val.npy
    TRAIN_EXTRA="--limit 100000 --epochs 2"
    CKPT_OUT=$STATE/net.pt
else
    CT_GAMES=6; CT_BASE=120000; CT_INC=500; CT_MIN=5000
    NW_GAMES=300; SPRT_GAMES=400; NET_GAMES=600
    GAUNTLET_EXTRA=""
    PACK_TARGET=150000000; PACK_VAL=200000
    SHARD=data/positions_w512-150m-b.npy; SHARD_VAL=data/validation_w512-150m-b.npy
    TRAIN_EXTRA="--epochs 24"
    CKPT_OUT=training/checkpoints/net_w512-300m.pt
fi
SHARD_A=data/positions_w512-150m.npy
VAL=data/validation_w512-150m.npy
RESUME=training/checkpoints/net_w512-150m.pt

# ---------------------------------------------------------------- helpers ----

# Build a challenger from the current champion with one switch turned on.
challenger() {  # name switch
    local d="overnight/challengers/$1"
    rm -rf "$d"; mkdir -p "$d/weights"
    cp agent.py "$d/agent.py"
    if [ -n "$2" ]; then
        sed -i "s/^$2: Final = False\$/$2: Final = True/" "$d/agent.py"
        if ! grep -q "^$2: Final = True\$" "$d/agent.py"; then
            say "switch $2 not found in agent.py"; return 1
        fi
    fi
    cp weights/net.npz weights/book.bin "$d/weights/"
    cp -r weights/syzygy "$d/weights/"
}

gauntlet() {  # name elo0 elo1 games
    say "gauntlet $1: SPRT[$2, $3], up to $4 games"
    # shellcheck disable=SC2086
    $PY -u -m testing.gauntlet --challenger "overnight/challengers/$1" \
        --elo0 "$2" --elo1 "$3" --games "$4" $GAUNTLET_EXTRA > "$STATE/$1.gauntlet.log" 2>&1
    local code=$?
    tail -n 12 "$STATE/$1.gauntlet.log" >> "$MAIN"
    return $code
}

verdict_line() { grep -E "^(PROMOTE|REJECT|INCONCLUSIVE)" "$STATE/$1.gauntlet.log" | tail -n 1; }

promote() {  # name what [net]
    local d="overnight/challengers/$1"
    if [ "$DRY" = 1 ]; then say "DRY: would promote $1"; return 0; fi
    local b="overnight/champion_backup_$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$b/weights"
    cp agent.py "$b/agent.py"; cp weights/net.npz "$b/weights/net.npz"
    cp "$d/agent.py" agent.py
    [ "${3:-}" = net ] && cp "$d/weights/net.npz" weights/net.npz
    if $PY -m ruff check . >> "$MAIN" 2>&1 && $PY -m mypy >> "$MAIN" 2>&1; then
        git add agent.py weights/net.npz
        git commit -q -m "promote $1: $2" -m "Unattended, by overnight/night.sh. Backup in $b." \
            && say "promoted $1, committed"
        note "**PROMOTED $1** -- $2"
    else
        say "GATE RED after promoting $1 -- restored the backup"
        cp "$b/agent.py" agent.py; cp "$b/weights/net.npz" weights/net.npz
        note "$1 passed its SPRT but the ruff/mypy gate went red; NOT promoted"
        return 1
    fi
}

# --------------------------------------------------------------- preflight ----

say "=== night starts (dry=$DRY) ==="
if ! git diff --quiet || ! git diff --cached --quiet; then
    say "ABORT: tracked files are modified; commit or stash first"
    git status --short | tee -a "$MAIN"
    exit 1
fi
if [ "$(tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -c python.exe)" -gt 1 ]; then
    say "ABORT: python is already running; refusing to share the machine"
    exit 1
fi
for f in agent.py weights/net.npz weights/book.bin "$SHARD_A" "$VAL" "$RESUME"; do
    [ -e "$f" ] || { say "ABORT: missing $f"; exit 1; }
done
say "preflight clean"

# ------------------------------------------------------------ 1. clock -------

if ! finished clock; then
    say "=== 1/7 clock: TIME_V2 ==="
    challenger 026-time-v2 TIME_V2 || exit 1
    say "clock replay, champion, charged x1.5 (expected to show the problem)"
    $PY -u -m testing.clocktest --agent . --games "$CT_GAMES" --base-ms "$CT_BASE" \
        --increment-ms "$CT_INC" --factor 1.5 --min-clock-ms "$CT_MIN" \
        > "$STATE/clock.champion.x1.5.log" 2>&1
    say "champion x1.5: $(tail -n 2 "$STATE/clock.champion.x1.5.log" | head -n 1)"
    say "clock replay, challenger, charged x1.0 (profile)"
    $PY -u -m testing.clocktest --agent overnight/challengers/026-time-v2 --games "$CT_GAMES" \
        --base-ms "$CT_BASE" --increment-ms "$CT_INC" --factor 1.0 --min-clock-ms "$CT_MIN" \
        > "$STATE/clock.challenger.x1.0.log" 2>&1
    say "challenger x1.0: $(tail -n 2 "$STATE/clock.challenger.x1.0.log" | head -n 1)"
    say "clock replay, challenger, charged x1.5 (must pass)"
    $PY -u -m testing.clocktest --agent overnight/challengers/026-time-v2 --games "$CT_GAMES" \
        --base-ms "$CT_BASE" --increment-ms "$CT_INC" --factor 1.5 --min-clock-ms "$CT_MIN" \
        > "$STATE/clock.challenger.x1.5.log" 2>&1
    SAFE=$?
    say "challenger x1.5: $(tail -n 2 "$STATE/clock.challenger.x1.5.log" | head -n 1) (exit $SAFE)"
    note "clock replay x1.5 -- champion: $(tail -n 2 "$STATE/clock.champion.x1.5.log" | head -n 1); TIME_V2: $(tail -n 2 "$STATE/clock.challenger.x1.5.log" | head -n 1)"

    gauntlet 026-time-v2 -25 0 "$NW_GAMES"; CODE=$?
    note "026-time-v2 not-worse SPRT: $(verdict_line 026-time-v2)"
    if [ $SAFE -eq 0 ] && [ $CODE -eq 0 ]; then
        promote 026-time-v2 "iteration-cost prediction, 12% hard cap, 10% reserve"
    elif [ $SAFE -eq 0 ] && [ $CODE -eq 2 ]; then
        note "026-time-v2 is clock-safe but the SPRT was inconclusive -- Toby to decide"
    else
        note "026-time-v2 NOT promoted (safety exit $SAFE, gauntlet exit $CODE)"
    fi
    mark clock
fi

# ------------------------------------------------------------ 2. shard -------

if ! finished shard; then
    say "=== 2/7 shard: pack $PACK_TARGET positions from row groups after the first shard ==="
    if [ -f "$SHARD" ]; then
        say "shard exists, skipping pack"
    else
        $PY -u -m training.pack --target "$PACK_TARGET" --val-target "$PACK_VAL" \
            --min-ply 16 --quiet-fraction 0 --skip-groups 278 \
            --out "$SHARD" --val-out "$SHARD_VAL" > "$STATE/pack.log" 2>&1 \
            || { say "PACK FAILED"; note "shard pack FAILED, training skipped"; mark shard; mark train; }
    fi
    if [ -f "$SHARD" ]; then
        $PY -u -m training.check_pack --file "$SHARD" > "$STATE/check_pack.log" 2>&1 \
            || { say "CHECK_PACK FAILED"; note "check_pack FAILED on the new shard, training skipped"; mark train; }
        tail -n 3 "$STATE/check_pack.log" >> "$MAIN"
        note "shard: $(grep -E '^train:' "$STATE/pack.log" | tail -n 1)"
    fi
    mark shard
fi

# ------------------------------------------------------------ 3. train -------

if ! finished train; then
    say "=== 3/7 train: continue the 512 net on both shards ==="
    # shellcheck disable=SC2086
    $PY -u -m training.train --data "$SHARD_A" "$SHARD" --val "$VAL" \
        --resume "$RESUME" --accumulator 512 --lr 3e-4 --patience 6 --skip-sanity \
        --out "$CKPT_OUT" $TRAIN_EXTRA > "$STATE/train.log" 2>&1
    TCODE=$?
    grep -E "epoch|initial|restored|wrote" "$STATE/train.log" | tail -n 30 >> "$MAIN"
    JSON="${CKPT_OUT%.pt}.json"
    if [ $TCODE -ne 0 ] || [ ! -f "$JSON" ]; then
        say "TRAIN FAILED (exit $TCODE)"; note "training FAILED (exit $TCODE), see train.log"
    else
        IMPROVED=$($PY -c "import json,sys; d=json.load(open('$JSON')); print(int(d['best_val'] < d['initial_val'] - 1e-6))")
        note "training: $(cat "$JSON" | tr -d '\n ')"
        if [ "$IMPROVED" = 1 ]; then
            challenger 027-net-300m "" || exit 1
            $PY -u -m training.export --checkpoint "$CKPT_OUT" \
                --out overnight/challengers/027-net-300m/weights/net.npz > "$STATE/export.log" 2>&1 \
                || { say "EXPORT FAILED"; note "export FAILED"; }
            $PY -u -m training.check_nnue --agent overnight/challengers/027-net-300m \
                --checkpoint "$CKPT_OUT" > "$STATE/check_nnue.log" 2>&1
            NCODE=$?
            tail -n 5 "$STATE/check_nnue.log" >> "$MAIN"
            if [ $NCODE -eq 0 ]; then
                gauntlet 027-net-300m 0 20 "$NET_GAMES"; CODE=$?
                note "027-net-300m SPRT: $(verdict_line 027-net-300m)"
                [ $CODE -eq 0 ] && promote 027-net-300m "512 net continued on 290M positions" net
            else
                say "CHECK_NNUE FAILED"; note "check_nnue FAILED on the new net -- not tested"
            fi
        else
            say "continuation did not beat the checkpoint on validation; nothing to test"
        fi
    fi
    mark train
fi

# ------------------------------------------------ 4-6. search switches -------

switch_stage() {  # stage name switch elo0 elo1 games what
    if finished "$1"; then return 0; fi
    say "=== $2: $3 ==="
    challenger "$2" "$3" || return 1
    gauntlet "$2" "$4" "$5" "$6"; local code=$?
    note "$2 SPRT[$4, $5]: $(verdict_line "$2")"
    [ $code -eq 0 ] && promote "$2" "$7"
    mark "$1"
}

switch_stage evasions 028-qs-evasions QS_EVASIONS 0 20 "$SPRT_GAMES" "check evasions in quiescence"
switch_stage staged 029-staged-movegen STAGED_MOVEGEN 0 20 "$SPRT_GAMES" "hash move before move generation"
switch_stage hygiene 030-hygiene HYGIENE -25 0 "$NW_GAMES" "history decay, post-move repetition keys, mate-bound RFP guard"

# ------------------------------------------------------------ 7. final -------

if ! finished final; then
    say "=== 7/7 final ==="
    if $PY -m ruff check . >> "$MAIN" 2>&1 && $PY -m mypy >> "$MAIN" 2>&1; then
        say "gate green"
        if [ "$DRY" != 1 ]; then
            $PY -m harness.package >> "$MAIN" 2>&1 && say "submission.zip built"
        fi
    else
        say "GATE RED at the end of the night"; note "**gate red at the end -- look before uploading**"
    fi
    {
        echo
        echo "## Switches now on in agent.py"
        grep -E "^(TIME_V2|QS_EVASIONS|STAGED_MOVEGEN|HYGIENE): Final" agent.py
        echo
        echo "## Commits tonight"
        git log --oneline -8
    } >> "$SUMMARY"
    mark final
fi

# ----------------------------------------------------------- 8. review -------
# Claude Fable reads the night and writes a critical evaluation. Read-only: the
# permission mode denies anything not pre-granted in .claude/settings.json.

if ! finished review; then
    say "=== 8/8 review by Claude Fable ==="
    CLAUDE="$HOME/.local/bin/claude.exe"
    if [ -x "$CLAUDE" ]; then
        sed "s#@STATE@#$STATE#g" overnight/REVIEW_PROMPT.md \
            | "$CLAUDE" --print --model fable --permission-mode dontAsk \
            > "$STATE/REVIEW.md" 2> "$STATE/review.err"
        RCODE=$?
        say "review exit $RCODE, $(wc -c < "$STATE/REVIEW.md") bytes in $STATE/REVIEW.md"
        note "review: exit $RCODE, see REVIEW.md"
    else
        say "claude CLI not found at $CLAUDE; review skipped"
        note "review skipped: claude CLI not found"
    fi
    mark review
fi
say "=== night done ==="
