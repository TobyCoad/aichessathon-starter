#!/bin/bash
# v16f, take 3: 32 epochs from scratch -- v12's epoch count, pairwise + dual + endgame buckets + factoriser,
# on v12's shard list capped at 40M. One continuous run with ONE cosine schedule -- the
# chunking in v16_long.sh existed only because train.py saved at the end and a kill lost
# everything. train.py now writes the best state to <out>.partial.pt on every new best,
# so the run is interruptible: stop it whenever and gate the partial.
#
# Matched to v12 on all three axes now: 32 epochs, the full 145M-row human shards
# (v12's own log shows no --limit), and therefore the same ~2.64B total positions. The
# ONLY difference left is the architecture and that v12 was warm-started from v11 while
# this starts from noise -- so this is an equal-BUDGET comparison, not an equal-learning
# one. lr is 1e-3 not v12's 1e-4 for exactly that reason: 1e-4 is a continuation rate.
#
# The first attempt lost 35 epochs (8.75 h) to a reboot at 07:50 with nothing on disk.
#   nohup bash overnight/v16_run.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night4.log
CKPT=training/checkpoints/net_v16f.pt
say() { echo "$(date '+%H:%M') v16f2 $*" >> "$LOG"; }
say "start: 32 epochs (v12's count), FULL shards (v12's data), single cosine"

SF=(data/sf80kf/sf80kf_00.npy data/sf80kf/sf80kf_05.npy data/sf80kf/sf80kf_10.npy data/sf80kf/sf80kf_15.npy \
    data/sf80kf/sf80kf_20.npy data/sf80kf/sf80kf_25.npy data/sf80kf/sf80kf_30.npy data/sf80kf/sf80kf_35.npy)
HU=(data/positions_w512-150m.npy data/positions_2025_02.npy data/positions_w512-150m-b.npy data/positions_2025_03.npy)
SHARDS=""
for i in 0 1 2 3 4 5 6 7; do SHARDS="$SHARDS ${SF[$i]} ${HU[$((i % 4))]}"; done

$PY -u training/train.py \
    --data $SHARDS \
    --val data/mixed_val.npy \
    --mirror --pairwise --dual --endgame-buckets --factoriser \
    --accumulator 1024 --buckets 8 --king-zones 16 \
    --lr 1e-3 --epochs 32 --patience 40 --warmup-epochs 2 --skip-sanity \
    --out "$CKPT" > overnight/eval/train-v16f2.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
say "train done: $(grep -E 'restored' overnight/eval/train-v16f2.log | tail -n 1)"
