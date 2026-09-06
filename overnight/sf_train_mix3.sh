#!/bin/bash
# mixnet3: the mix2 recipe with ONE change -- model selection validates on a 50/50
# Lichess+SF validation set (data/mixval.npy) instead of SF alone. mix2 early-stopped on
# sf_val only, which is why its Lichess-val came out 72% worse than the champion's and its
# static error regressed at 9-12 (228.4 -> 262.6) and 13-16 (137.3 -> 184.4) while halving
# 5-8 (673.7 -> 331.9). Same shards, same warm start, same lr/epochs: only the criterion moves.
#   nohup bash overnight/sf_train_mix3.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') mixnet3 $*" >> "$LOG"; }
say "start (mix2 shards + warm start; validation = 50/50 Lichess+SF data/mixval.npy)"

SHARDS="data/sf/feb24_00.npy data/positions_w512-150m.npy data/sf/feb24_01.npy data/positions_2025_02.npy \
data/sf/feb24_02.npy data/positions_w512-150m-b.npy data/sf/feb24_03.npy data/positions_2025_03.npy \
data/sf/feb24_04.npy data/positions_w512-150m.npy data/sf/feb24_05.npy data/positions_2025_02.npy \
data/sf/feb24_06.npy data/positions_w512-150m-b.npy data/sf/feb24_07.npy data/positions_2025_03.npy"

if [ ! -f training/checkpoints/net_w512-b8-kz16-mix3.json ]; then
    $PY -u training/train.py \
        --data $SHARDS \
        --val data/mixval.npy \
        --resume training/checkpoints/net_w512-b8-kz16.pt \
        --accumulator 512 --buckets 8 --king-zones 16 \
        --lr 1e-4 --epochs 24 --patience 8 --warmup-epochs 1 --skip-sanity \
        --out training/checkpoints/net_w512-b8-kz16-mix3.pt \
        > overnight/eval/train-mix3.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
fi
say "train done: $(grep -E 'restored|wrote.*json' overnight/eval/train-mix3.log | tail -n 2 | tr '\n' ' ')"

d=overnight/challengers/156-mixnet3
if [ ! -f overnight/eval/suite-156-mixnet3.log ]; then
    rm -rf "$d"; mkdir -p "$d/weights"
    cp agent.py fastboard.py fastsearch.py "$d/"
    cp weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
    $PY -u -m training.export --checkpoint training/checkpoints/net_w512-b8-kz16-mix3.pt \
        --out "$d/weights/net.npz" --half > overnight/eval/export-mix3.log 2>&1 \
        || { say "EXPORT FAILED"; exit 1; }
    if ! $PY -u -m training.check_nnue --agent "$d" \
        --checkpoint training/checkpoints/net_w512-b8-kz16-mix3.pt \
        > overnight/eval/check_nnue-mix3.log 2>&1; then
        say "CHECK_NNUE FAILED"; exit 1
    fi
    say "check_nnue: $(tail -n 1 overnight/eval/check_nnue-mix3.log)"
    $PY -u -m testing.endgame_suite run --agent "$d" --seconds 2.5 \
        > overnight/eval/suite-156-mixnet3.log 2>&1
fi
say "suite 156-mixnet3: $(grep -E 'mean loss' overnight/eval/suite-156-mixnet3.log | tail -n 1)"

# cross-validation + slope on Lichess targets (slope ~1.0 is the calibration check)
$PY -u - > overnight/eval/xval-mix3.log 2>&1 <<'EOF'
import sys, torch
sys.path.insert(0, ".")
from pathlib import Path
from training.train import load_checkpoint, Batches, evaluate_loss, _records
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sets = {"lichess_val": Path("data/validation_w512-150m.npy"), "sf_val": Path("data/sf/feb24_val.npy")}
net = load_checkpoint(Path("training/checkpoints/net_w512-b8-kz16-mix3.pt"), 8, 16).to(device)
for label, path in sets.items():
    recs = _records(path, 300_000)
    g = torch.Generator(); g.manual_seed(1)
    print(f"mix3-trained {label:12s} {evaluate_loss(net, Batches(recs, 16384, device), g):.6f}", flush=True)
EOF
say "xval: $(tr '\n' ' ' < overnight/eval/xval-mix3.log)"

# stage the net OUTSIDE the challenger dir (the worker rebuilds that dir from the tree first)
mkdir -p overnight/nets
cp "$d/weights/net.npz" overnight/nets/156-mixnet3.npz
if [ "$(md5sum < overnight/nets/156-mixnet3.npz)" = "$(md5sum < weights/net.npz)" ]; then
    say "ABORT -- 156-mixnet3 is byte-identical to the tree net; not queued"
    exit 1
fi

$PY - <<'EOF'
import json
p = "overnight/laptop/tasks.json"; t = json.load(open(p))
if not any(x["name"] == "156-mixnet3" for x in t):
    t.append({"name": "156-mixnet3", "net": "overnight/nets/156-mixnet3.npz", "sed": "", "games": 600})
    json.dump(t, open(p, "w"), indent=1)
EOF
git add overnight/laptop/tasks.json overnight/eval/suite-156-mixnet3.log overnight/eval/xval-mix3.log \
    overnight/eval/train-mix3.log overnight/sf_train_mix3.sh 2>/dev/null
git -c user.name=mixnet3 -c user.email=mixnet@local commit -q -m "mixnet3: combined-validation mixed net trained, suite + xval logged, 156-mixnet3 queued" && \
    (git pull -q --rebase --autostash origin main >/dev/null 2>&1; git push -q origin main >/dev/null 2>&1)
say "done -- 156-mixnet3 queued for the laptop worker"
