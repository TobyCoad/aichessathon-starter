#!/bin/bash
# v7 net candidates on the GPU after the kz32b run: an 8-zone control on three
# shards, the same with endgame loss weighting, then 16 zones. Each net is
# exported float16, checked against torch, scored on the endgame suite, and
# gauntleted against 060-v6 once queue5 has released the CPU.
cd /c/dev/aichessathon/starter || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') $*" >> "$LOG"; }
verdict() { grep -E "^(PROMOTE|REJECT|INCONCLUSIVE)" "$1" 2>/dev/null | tail -n 1; }
DATA="data/positions_w512-150m.npy data/positions_w512-150m-b.npy data/positions_2025_02.npy"
VAL=data/validation_w512-150m.npy

until grep -qE "wrote training.checkpoints.net_w512-b8-kz32b.json" overnight/eval/train-kz32b.log 2>/dev/null; do sleep 120; done
say "queue6 start"

train_net() {  # name zones extra-args...
    local name=$1 zones=$2; shift 2
    [ -f "training/checkpoints/net_w512-b8-$name.json" ] && return 0
    $PY -u training/train.py --data $DATA --val $VAL \
        --resume training/checkpoints/net_w512-b8-kz8.pt --accumulator 512 --buckets 8 \
        --king-zones "$zones" --lr 1.5e-4 --epochs 18 --patience 8 --warmup-epochs 2 --skip-sanity \
        --out "training/checkpoints/net_w512-b8-$name.pt" "$@" > "overnight/eval/train-$name.log" 2>&1
    say "train $name: $(grep -E 'restored|wrote.*json' overnight/eval/train-$name.log | tail -n 2 | tr '\n' ' ')"
}
gate_net() {  # name challenger-number
    local name=$1 d="overnight/challengers/$2-$1"
    local base=overnight/challengers/060-v6
    [ -f "$base/agent.py" ] || base=overnight/challengers/058-v5.5
    rm -rf "$d"; cp -r "$base" "$d"
    $PY -u -m training.export --checkpoint "training/checkpoints/net_w512-b8-$name.pt" \
        --out "$d/weights/net.npz" --half > "overnight/eval/export-$name.log" 2>&1 || { say "EXPORT $name FAILED"; return 1; }
    if ! $PY -u -m training.check_nnue --agent "$d" --checkpoint "training/checkpoints/net_w512-b8-$name.pt" \
        > "overnight/eval/check_nnue-$name.log" 2>&1; then say "CHECK_NNUE $name FAILED"; return 1; fi
    say "check_nnue $name: $(tail -n 1 overnight/eval/check_nnue-$name.log)"
    # the endgame suite is the instrument for the net; then the gauntlet once the CPU is free
    $PY -u -m testing.endgame_suite run --agent "$d" --seconds 2.5 > "overnight/eval/suite-$name.log" 2>&1
    say "suite $name: $(grep -E 'mean loss' overnight/eval/suite-$name.log | tail -n 1)"
    until grep -q "queue5 done" "$LOG" 2>/dev/null; do sleep 120; done
    $PY -u -m testing.gauntlet --challenger "$d" --champion "$base" --elo0 0 --elo1 20 --games 600 \
        > "overnight/eval/$2-$name.gauntlet.log" 2>&1
    say "$2-$name: $(verdict overnight/eval/$2-$name.gauntlet.log)"
}

train_net kz8c 8
gate_net kz8c 070 &
train_net kz8w 8 --weight-endgame
gate_net kz8w 071 &
train_net kz16 16
gate_net kz16 072 &
wait
say "queue6 done"
