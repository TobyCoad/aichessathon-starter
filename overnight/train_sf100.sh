#!/bin/bash
# 100% engine data with Stockfish n20000 labels -- the net the whole day's diagnosis points at.
#
# THE CHAIN OF EVIDENCE. Our net overvalues attacking positions: measured -452 cp for the
# Lichess-trained net, -209 for the shipped 50/50 mix, -120 for pure engine data, against a
# Stockfish d16 reference on 258 positions from our own games. The cause is selection bias in
# HUMAN positions: on 2000 real game positions the TRUE median attack reward is +5 cp, the
# Lichess corpus teaches +440, the engine corpora teach +6 and +10. The corpus lies by 435 cp
# and the net inherited 452 cp of it -- which is the Bxf4 blunder and the blown wins.
# This drops human data entirely and uses labels that correlate 0.978 with our reference
# (against 0.861 for the labels the shipped net learned from).
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') sf100 $*" >> "$LOG"; }
say "start: 321M positions, Stockfish n20000 labels, warm start from the v9.4 net"

if [ ! -f training/checkpoints/net_w512-b8-kz16-sf100.json ]; then
    $PY -u training/train.py \
        --data data/sf20k/sf20k_[0-9][0-9].npy \
        --val data/sf20k/sf20k_val.npy \
        --resume training/checkpoints/net_w512-b8-kz16-wdl.pt \
        --accumulator 512 --buckets 8 --king-zones 16 \
        --lr 1e-4 --epochs 14 --patience 4 --warmup-epochs 1 --skip-sanity \
        --out training/checkpoints/net_w512-b8-kz16-sf100.pt \
        > overnight/eval/train-sf100.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
fi
say "train: $(grep -E 'restored|wrote.*json' overnight/eval/train-sf100.log | tail -n 2 | tr '\n' ' ')"

# THE FIRST DECISION POINT -- seconds, not an hour. If the attack bias has not fallen, the
# mechanism is wrong and nothing downstream is worth running.
$PY -u -m training.attack_bias --checkpoint training/checkpoints/net_w512-b8-kz16-sf100.pt \
    > overnight/eval/v10/bias-sf100.log 2>&1
say "BIAS: $(grep -E 'attacking|quiet|weighted' overnight/eval/v10/bias-sf100.log | tr '\n' ' ')"

d=overnight/challengers/180-sf100
rm -rf "$d"; mkdir -p "$d/weights"
cp agent.py fastboard.py fastsearch.py "$d/"
cp weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
$PY -u -m training.export --checkpoint training/checkpoints/net_w512-b8-kz16-sf100.pt \
    --out "$d/weights/net.npz" --half > overnight/eval/export-sf100.log 2>&1 \
    || { say "EXPORT FAILED"; exit 1; }
if ! $PY -u -m training.check_nnue --agent "$d" \
    --checkpoint training/checkpoints/net_w512-b8-kz16-sf100.pt \
    > overnight/eval/check_nnue-sf100.log 2>&1; then
    say "CHECK_NNUE FAILED"; exit 1
fi
say "check_nnue: $(tail -n 1 overnight/eval/check_nnue-sf100.log)"

$PY -u -m testing.endgame_suite run --agent "$d" --seconds 2.5 \
    > overnight/eval/suite-180-sf100.log 2>&1
say "suite: $(grep -E 'mean loss' overnight/eval/suite-180-sf100.log | tail -n 1)"

mkdir -p overnight/nets
cp "$d/weights/net.npz" overnight/nets/180-sf100.npz
if cmp -s overnight/nets/180-sf100.npz weights/net.npz; then
    say "ABORT -- identical to the tree net"; exit 1
fi
say "STAGED overnight/nets/180-sf100.npz -- read bias-sf100.log before queueing any gauntlet"
