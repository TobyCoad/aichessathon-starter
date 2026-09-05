#!/bin/bash
# Stockfish-data net, hands-off chain (5-6 Sep night):
#   wait for the decoded shards -> baseline endgame suite of the champion net under the
#   current search -> warm-started retrain on the Stockfish shards -> export --half ->
#   check_nnue -> endgame suite -> queue the net task on the laptop worker (SPRT vs the
#   champion) and push. Idempotent: every stage is skipped when its output exists.
#   nohup bash overnight/sf_train.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') sfnet $*" >> "$LOG"; }
say "start (wait for data/sf/feb24_val.npy -> baseline suite -> train -> export -> check -> suite -> queue)"

while [ ! -f data/sf/feb24_val.npy ]; do sleep 120; done
SHARDS=$(ls data/sf/feb24_[0-9][0-9].npy | tr '\n' ' ')
say "shards ready: $SHARDS"

# Baseline: the champion net under the v9.1 search (every older suite number was measured
# under v5.5/v6 search and cannot be compared with anything trained tonight).
if [ ! -f overnight/eval/suite-v91-champion.log ]; then
    $PY -u -m testing.endgame_suite run --agent . --seconds 2.5 \
        > overnight/eval/suite-v91-champion.log 2>&1
fi
say "baseline suite (champion net, v9.1 search): $(grep -E 'mean loss' overnight/eval/suite-v91-champion.log | tail -n 1)"

if [ ! -f training/checkpoints/net_w512-b8-kz16-sf.json ]; then
    $PY -u training/train.py \
        --data $SHARDS \
        --val data/sf/feb24_val.npy \
        --resume training/checkpoints/net_w512-b8-kz16.pt \
        --accumulator 512 --buckets 8 --king-zones 16 \
        --lr 1e-4 --epochs 24 --patience 6 --warmup-epochs 1 --skip-sanity \
        --out training/checkpoints/net_w512-b8-kz16-sf.pt \
        > overnight/eval/train-sf.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
fi
say "train done: $(grep -E 'restored|wrote.*json' overnight/eval/train-sf.log | tail -n 2 | tr '\n' ' ')"

d=overnight/challengers/150-sfnet
if [ ! -f overnight/eval/suite-150-sfnet.log ]; then
    rm -rf "$d"; mkdir -p "$d/weights"
    cp agent.py fastboard.py fastsearch.py "$d/"
    cp weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
    $PY -u -m training.export --checkpoint training/checkpoints/net_w512-b8-kz16-sf.pt \
        --out "$d/weights/net.npz" --half > overnight/eval/export-sf.log 2>&1 \
        || { say "EXPORT FAILED"; exit 1; }
    if ! $PY -u -m training.check_nnue --agent "$d" \
        --checkpoint training/checkpoints/net_w512-b8-kz16-sf.pt \
        > overnight/eval/check_nnue-sf.log 2>&1; then
        say "CHECK_NNUE FAILED"; exit 1
    fi
    say "check_nnue: $(tail -n 1 overnight/eval/check_nnue-sf.log)"
    $PY -u -m testing.endgame_suite run --agent "$d" --seconds 2.5 \
        > overnight/eval/suite-150-sfnet.log 2>&1
fi
say "suite 150-sfnet: $(grep -E 'mean loss' overnight/eval/suite-150-sfnet.log | tail -n 1)"

# Cross-validation: each net's loss on both validation sets (Lichess and Stockfish data).
$PY -u - > overnight/eval/xval-sf.log 2>&1 <<'EOF'
import sys, torch, numpy as np
sys.path.insert(0, ".")
from pathlib import Path
from training.train import load_checkpoint, Batches, evaluate_loss, _records
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sets = {"lichess_val": Path("data/validation_w512-150m.npy"), "sf_val": Path("data/sf/feb24_val.npy")}
for name, ckpt in (("champion b8-kz16", "training/checkpoints/net_w512-b8-kz16.pt"), ("sf-trained", "training/checkpoints/net_w512-b8-kz16-sf.pt")):
    net = load_checkpoint(Path(ckpt), 8, 16).to(device)
    for label, path in sets.items():
        recs = _records(path, 300_000)
        g = torch.Generator(); g.manual_seed(1)
        print(f"{name:18s} {label:12s} {evaluate_loss(net, Batches(recs, 16384, device), g):.6f}", flush=True)
EOF
say "xval: $(tr '\n' ' ' < overnight/eval/xval-sf.log)"

# Queue the gauntlet on the laptop worker (net task: the challenger is the tree + this net).
$PY - <<'EOF'
import json
p = "overnight/laptop/tasks.json"; t = json.load(open(p))
if not any(x["name"] == "150-sfnet" for x in t):
    t.append({"name": "150-sfnet", "net": "overnight/challengers/150-sfnet/weights/net.npz", "sed": "", "games": 600})
    json.dump(t, open(p, "w"), indent=1)
EOF
git add overnight/laptop/tasks.json overnight/eval/suite-v91-champion.log overnight/eval/suite-150-sfnet.log overnight/eval/xval-sf.log overnight/eval/train-sf.log 2>/dev/null
git -c user.name=sfnet -c user.email=sfnet@local commit -q -m "sfnet: Stockfish-data net trained, suite + xval logged, 150-sfnet queued" && \
    (git pull -q --rebase --autostash origin main >/dev/null 2>&1; git push -q origin main >/dev/null 2>&1)
say "done -- 150-sfnet queued for the laptop worker"
