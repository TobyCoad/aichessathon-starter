#!/bin/bash
# v10: a BRAND NEW network trained FROM SCRATCH on n80000 data.
#
# Everything we ship descends in an unbroken line from the original Lichess-trained net, so it
# still carries weights shaped by the human corpus that taught it an attack is worth +440 cp
# when the truth is +5. v9.5 overwrote that bias by fine-tuning (-158 -> +11), but a net that
# never saw human data has no bias to overwrite and no starting point built around one.
#
# Data: n80000 -- the same T80 positions as v9.5's corpus but labelled by Stockfish searching
# 80,000 nodes instead of 20,000. Calibrated locally: scale 0.2479, r 0.981 vs our Stockfish
# d12 (n20000 measured 0.2584 / 0.978).
#
# NOTE the byte ranges: a mid-file range does NOT decode (the binpack is BINP+size chunks and a
# range starting at byte 5e9 lands mid-chunk). The three downloads are contiguous, so they are
# concatenated back into the original file first. This was verified by trying to decode a range
# chunk standalone -- it crashes.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') v10 $*" >> "$LOG"; }

# 1. Only the CLOCKTEST must not be disturbed -- it is timing-sensitive and load invalidates
# it. The decode is front-loaded ahead of the release gauntlet deliberately, so the night is
# spent training rather than decoding. The worker will not START a gauntlet while a decode
# runs, so the gauntlet picks up the moment the decode ends.
while [ ! -f overnight/laptop/results/v95net-clocktest-l.txt ]; do sleep 30; done
say "clocktest done; decoding now (this holds the gauntlet for ~70 min by design)"

# 2. wait for the downloads, then rebuild the original file
while pgrep -f "t80-sf80k" > /dev/null 2>&1 || \
      [ "$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*t80-sf80k*' -and \$_.CommandLine -notlike '*CimInstance*' }).Count" 2>/dev/null | tr -d '\r')" != "0" ]; do
    sleep 60
done
if [ ! -f data/sf/t80-sf80k-full.binpack ]; then
    say "concatenating the three ranges back into one file"
    cat data/sf/t80-2023-06-sf80k.binpack data/sf/t80-sf80k-b.binpack data/sf/t80-sf80k-c.binpack \
        > data/sf/t80-sf80k-full.binpack || { say "CONCAT FAILED"; exit 1; }
fi
say "source: $(du -m data/sf/t80-sf80k-full.binpack | cut -f1) MB"

# 3. decode a from-scratch-sized corpus
if [ ! -f data/sf80kf/sf80kf_00.npy ]; then
    mkdir -p data/sf80kf
    $PY -u -m training.binpack_decode data/sf/t80-sf80k-full.binpack \
        --out data/sf80kf/sf80kf --target 800000000 --shard 20000000 \
        --workers 8 --wdl-lambda 0.75 --scale 0.2479 \
        > overnight/eval/decode-sf80kf.log 2>&1 || { say "DECODE FAILED"; exit 1; }
fi
say "decode: $(grep '^done:' overnight/eval/decode-sf80kf.log)"

# Let the worker claim the gauntlet first: once the decode exits its busy check clears, it
# starts 181-v95-vs-v94 within a minute. Training does NOT stop a gauntlet already running --
# it only prevents a new one starting -- so a short pause buys us both at once.
sleep 300
say "paused 5 min so the release gauntlet could claim the machine; starting training"

# 4. FROM SCRATCH -- no --resume. Higher lr and many more passes than a fine-tune needs.
if [ ! -f training/checkpoints/net_v10-scratch.json ]; then
    $PY -u training/train.py \
        --data data/sf80kf/sf80kf_[0-9][0-9].npy \
        --val data/sf80kf/sf80kf_val.npy \
        --accumulator 512 --buckets 8 --king-zones 16 \
        --lr 1e-3 --epochs 200 --patience 12 --warmup-epochs 3 --skip-sanity \
        --out training/checkpoints/net_v10-scratch.pt \
        > overnight/eval/train-v10.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
fi
say "train: $(grep -E 'restored|wrote.*json' overnight/eval/train-v10.log | tail -n 2 | tr '\n' ' ')"

# 5. the instrument first -- seconds, and it decides whether anything downstream is worth running
$PY -u -m training.attack_bias --checkpoint training/checkpoints/net_v10-scratch.pt \
    > overnight/eval/v10/bias-v10.log 2>&1
say "BIAS: $(grep -E 'attacking|quiet|weighted' overnight/eval/v10/bias-v10.log | tr '\n' ' ')"

d=overnight/challengers/190-v10
rm -rf "$d"; mkdir -p "$d/weights"
cp agent.py fastboard.py fastsearch.py "$d/"
cp weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
$PY -u -m training.export --checkpoint training/checkpoints/net_v10-scratch.pt \
    --out "$d/weights/net.npz" --half > overnight/eval/export-v10.log 2>&1 \
    || { say "EXPORT FAILED"; exit 1; }
if ! $PY -u -m training.check_nnue --agent "$d" \
    --checkpoint training/checkpoints/net_v10-scratch.pt > overnight/eval/check_nnue-v10.log 2>&1; then
    say "CHECK_NNUE FAILED"; exit 1
fi
say "check_nnue: $(tail -n 1 overnight/eval/check_nnue-v10.log)"

$PY -u -m testing.endgame_suite run --agent "$d" --seconds 2.5 > overnight/eval/suite-190-v10.log 2>&1
say "suite: $(grep -E 'mean loss' overnight/eval/suite-190-v10.log | tail -n 1)  [v9.5 scored 7.5]"

mkdir -p overnight/nets
cp "$d/weights/net.npz" overnight/nets/190-v10.npz
say "STAGED overnight/nets/190-v10.npz -- compare bias-v10.log and the suite against v9.5 before queueing"
