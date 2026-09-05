#!/bin/bash
# Mixed-data net: Stockfish self-play shards interleaved with the Lichess shards (one shard
# per epoch, alternating), warm-started from the champion, so the net keeps the human
# distribution the pure Stockfish retrain forgot (Lichess-val 0.0113 vs 0.0046).
#   nohup bash overnight/sf_train_mix.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') mixnet $*" >> "$LOG"; }
say "start (interleaved SF + Lichess shards, warm start from b8-kz16)"

SHARDS="data/sf/feb24_00.npy data/positions_w512-150m.npy data/sf/feb24_01.npy data/positions_2025_02.npy \
data/sf/feb24_02.npy data/positions_w512-150m-b.npy data/sf/feb24_03.npy data/positions_2025_03.npy \
data/sf/feb24_04.npy data/positions_w512-150m.npy data/sf/feb24_05.npy data/positions_2025_02.npy \
data/sf/feb24_06.npy data/positions_w512-150m-b.npy data/sf/feb24_07.npy data/positions_2025_03.npy"

if [ ! -f training/checkpoints/net_w512-b8-kz16-mix.json ]; then
    $PY -u training/train.py \
        --data $SHARDS \
        --val data/sf/feb24_val.npy \
        --resume training/checkpoints/net_w512-b8-kz16.pt \
        --accumulator 512 --buckets 8 --king-zones 16 \
        --lr 1e-4 --epochs 24 --patience 8 --warmup-epochs 1 --skip-sanity \
        --out training/checkpoints/net_w512-b8-kz16-mix.pt \
        > overnight/eval/train-mix.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
fi
say "train done: $(grep -E 'restored|wrote.*json' overnight/eval/train-mix.log | tail -n 2 | tr '\n' ' ')"

d=overnight/challengers/151-mixnet
if [ ! -f overnight/eval/suite-151-mixnet.log ]; then
    rm -rf "$d"; mkdir -p "$d/weights"
    cp agent.py fastboard.py fastsearch.py "$d/"
    cp weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
    $PY -u -m training.export --checkpoint training/checkpoints/net_w512-b8-kz16-mix.pt \
        --out "$d/weights/net.npz" --half > overnight/eval/export-mix.log 2>&1 \
        || { say "EXPORT FAILED"; exit 1; }
    if ! $PY -u -m training.check_nnue --agent "$d" \
        --checkpoint training/checkpoints/net_w512-b8-kz16-mix.pt \
        > overnight/eval/check_nnue-mix.log 2>&1; then
        say "CHECK_NNUE FAILED"; exit 1
    fi
    say "check_nnue: $(tail -n 1 overnight/eval/check_nnue-mix.log)"
    $PY -u -m testing.endgame_suite run --agent "$d" --seconds 2.5 \
        > overnight/eval/suite-151-mixnet.log 2>&1
fi
say "suite 151-mixnet: $(grep -E 'mean loss' overnight/eval/suite-151-mixnet.log | tail -n 1)"

$PY -u - > overnight/eval/xval-mix.log 2>&1 <<'EOF'
import sys, torch
sys.path.insert(0, ".")
from pathlib import Path
from training.train import load_checkpoint, Batches, evaluate_loss, _records
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sets = {"lichess_val": Path("data/validation_w512-150m.npy"), "sf_val": Path("data/sf/feb24_val.npy")}
net = load_checkpoint(Path("training/checkpoints/net_w512-b8-kz16-mix.pt"), 8, 16).to(device)
for label, path in sets.items():
    recs = _records(path, 300_000)
    g = torch.Generator(); g.manual_seed(1)
    print(f"mix-trained {label:12s} {evaluate_loss(net, Batches(recs, 16384, device), g):.6f}", flush=True)
EOF
say "xval: $(tr '\n' ' ' < overnight/eval/xval-mix.log)"

$PY - <<'EOF'
import json
p = "overnight/laptop/tasks.json"; t = json.load(open(p))
if not any(x["name"] == "151-mixnet" for x in t):
    t.append({"name": "151-mixnet", "net": "overnight/challengers/151-mixnet/weights/net.npz", "sed": "", "games": 600})
    json.dump(t, open(p, "w"), indent=1)
EOF
git add overnight/laptop/tasks.json overnight/eval/suite-151-mixnet.log overnight/eval/xval-mix.log overnight/eval/train-mix.log 2>/dev/null
git -c user.name=mixnet -c user.email=mixnet@local commit -q -m "mixnet: mixed SF+Lichess net trained, suite + xval logged, 151-mixnet queued" && \
    (git pull -q --rebase --autostash origin main >/dev/null 2>&1; git push -q origin main >/dev/null 2>&1)
say "done -- 151-mixnet queued for the laptop worker"
