#!/bin/bash
# 161-sfshare67 -- the one net knob v9.4 left untested.
# v9.3 was 33.3% Stockfish by position count (an accident of shard sizes). v9.4 fixed that to
# 50% AND added WDL targets in the same run, so we cannot say which did the work. The human's
# thesis is that the field is bots, which argues for MORE engine data: this trains the same
# recipe at 67% Stockfish, warm-started from the v9.4 net. No new decode -- it re-merges the
# WDL shards we already have, so it costs GPU time only.
#
# It does NOT queue a gauntlet. One gauntlet per version is the rule, and v9.5 is the loop's
# search bundle; this net is staged for whoever decides what v9.5/v9.6 carries.
#
# Waits for v94-120s: the loop has train.py in worker.sh's busy check, so starting a trainer
# any earlier would block the validation gauntlet of the build the human just uploaded.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') sfshare $*" >> "$LOG"; }

while [ ! -f overnight/laptop/results/v94-120s.txt ]; do sleep 60; done
say "v94-120s reported; starting (67% Stockfish by position count, warm start from the v9.4 net)"

if [ ! -f data/mix67/mixw_00.npy ]; then
    $PY -u -m training.merge_mix --sf-share 0.667 --out data/mix67 --val data/mixval67.npy \
        > overnight/eval/merge-67.log 2>&1 || { say "MERGE FAILED"; exit 1; }
fi
say "merge: $(tail -n 2 overnight/eval/merge-67.log | tr '\n' ' ')"

if [ ! -f training/checkpoints/net_w512-b8-kz16-sf67.json ]; then
    $PY -u training/train.py \
        --data data/mix67/mixw_*.npy \
        --val data/mixval67.npy \
        --resume training/checkpoints/net_w512-b8-kz16-wdl.pt \
        --accumulator 512 --buckets 8 --king-zones 16 \
        --lr 1e-4 --epochs 10 --patience 4 --warmup-epochs 1 --skip-sanity \
        --out training/checkpoints/net_w512-b8-kz16-sf67.pt \
        > overnight/eval/train-sf67.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
fi
say "train: $(grep -E 'restored|wrote.*json' overnight/eval/train-sf67.log | tail -n 2 | tr '\n' ' ')"

d=overnight/challengers/161-sfshare67
rm -rf "$d"; mkdir -p "$d/weights"
cp agent.py fastboard.py fastsearch.py "$d/"
cp weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
$PY -u -m training.export --checkpoint training/checkpoints/net_w512-b8-kz16-sf67.pt \
    --out "$d/weights/net.npz" --half > overnight/eval/export-sf67.log 2>&1 \
    || { say "EXPORT FAILED"; exit 1; }
if ! $PY -u -m training.check_nnue --agent "$d" \
    --checkpoint training/checkpoints/net_w512-b8-kz16-sf67.pt \
    > overnight/eval/check_nnue-sf67.log 2>&1; then
    say "CHECK_NNUE FAILED"; exit 1
fi
say "check_nnue: $(tail -n 1 overnight/eval/check_nnue-sf67.log)"

$PY -u -m testing.endgame_suite run --agent "$d" --seconds 2.5 \
    > overnight/eval/suite-161-sfshare67.log 2>&1
say "suite: $(grep -E 'mean loss' overnight/eval/suite-161-sfshare67.log | tail -n 1)"

mkdir -p overnight/nets
cp "$d/weights/net.npz" overnight/nets/161-sfshare67.npz
if cmp -s overnight/nets/161-sfshare67.npz weights/net.npz; then
    say "identical to the tree net -- nothing learned"; exit 1
fi
say "STAGED at overnight/nets/161-sfshare67.npz -- NOT queued. Compare its suite with the"
say "v9.4 net's 11.4 cp (5-8 4.2 / 9-12 21.5 / 13-16 7.7) and decide whether v9.5 carries it."
git add overnight/eval/ training/merge_mix.py overnight/sf_share.sh 2>/dev/null
git -c user.name=sfshare -c user.email=sf@local commit -q -m "161-sfshare67: 67% Stockfish net trained and staged (not queued)" && \
    (git pull -q --rebase --autostash origin main >/dev/null 2>&1; git push -q origin main >/dev/null 2>&1)
