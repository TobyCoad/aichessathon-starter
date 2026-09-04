#!/bin/bash
# Night 3 (4-5 Sep): finish the stage-3 queue, gauntlet aspiration and SEE, pack the
# second Lichess month, train the 32-zone net on three shards, gate the net.
# CPU discipline: one gauntlet at a time; packing runs alone; training takes the GPU
# and one core, so gauntlets beside it use 10 workers.
cd /c/dev/aichessathon/starter || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') $*" >> "$LOG"; }
verdict() { grep -E "^(PROMOTE|REJECT|INCONCLUSIVE)" "$1" 2>/dev/null | tail -n 1; }
say "night3 start"

# 1. the stage-3 queue (052-lmr, 051-pvs, 053-lmp) owns the CPU until its last verdict
until grep -qE "^(PROMOTE|REJECT|INCONCLUSIVE)" overnight/eval/053-lmp.gauntlet.log 2>/dev/null; do sleep 120; done
say "stage 3: $(verdict overnight/eval/052-lmr.gauntlet.log) | $(verdict overnight/eval/051-pvs.gauntlet.log) | $(verdict overnight/eval/053-lmp.gauntlet.log)"

# 2. the download resumes and returns once the file is whole
$PY -u training/fetch.py --month 2025_02 >> overnight/eval/fetch-2025_02.log 2>&1
say "download: $(tail -n 1 overnight/eval/fetch-2025_02.log)"

# 3. pack month 2 with every core, nothing else running
if [ ! -f data/positions_2025_02.npy ]; then
    $PY -u -m training.pack --source data/standard_rated_2025_02.parquet \
        --target 150000000 --val-target 200000 --min-ply 16 --quiet-fraction 0 \
        --out data/positions_2025_02.npy --val-out data/validation_2025_02.npy \
        > overnight/eval/pack-2025_02.log 2>&1 || say "PACK FAILED"
fi
say "pack: $(grep -E '^train:' overnight/eval/pack-2025_02.log | tail -n 1)"

# 4. train 32 king zones from the 8-zone champion on three shards (GPU, background)
TRAIN_PID=""
if [ -f data/positions_2025_02.npy ]; then
    $PY -u training/train.py \
        --data data/positions_w512-150m.npy data/positions_w512-150m-b.npy data/positions_2025_02.npy \
        --val data/validation_w512-150m.npy \
        --resume training/checkpoints/net_w512-b8-kz8.pt \
        --accumulator 512 --buckets 8 --king-zones 32 --lr 3e-4 --epochs 24 --patience 6 \
        --skip-sanity --out training/checkpoints/net_w512-b8-kz32.pt \
        > overnight/eval/train-kz32.log 2>&1 &
    TRAIN_PID=$!
    say "training started (pid $TRAIN_PID)"
fi

# 5. aspiration and SEE, each vs the compiled-search build, beside the training
for n in 054-aspiration 055-see; do
    $PY -u -m testing.gauntlet --challenger overnight/challengers/$n \
        --champion overnight/challengers/050-compiled-search \
        --elo0 0 --elo1 20 --games 600 --workers 10 > overnight/eval/$n.gauntlet.log 2>&1
    say "$n: $(verdict overnight/eval/$n.gauntlet.log)"
done

# 6. the net: export as float16, check against torch, gauntlet vs 050
if [ -n "$TRAIN_PID" ]; then
    wait "$TRAIN_PID"
    say "training: $(grep -E 'restored|wrote.*json' overnight/eval/train-kz32.log | tail -n 2 | tr '\n' ' ')"
    d=overnight/challengers/056-kz32
    rm -rf "$d"; mkdir -p "$d/weights"
    cp agent.py fastboard.py fastsearch.py "$d/"
    sed -i 's/^COMPILED_SEARCH: Final = False$/COMPILED_SEARCH: Final = True/' "$d/agent.py"
    cp weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
    $PY -u -m training.export --checkpoint training/checkpoints/net_w512-b8-kz32.pt \
        --out "$d/weights/net.npz" --half > overnight/eval/export-kz32.log 2>&1 || say "EXPORT FAILED"
    if $PY -u -m training.check_nnue --agent "$d" --checkpoint training/checkpoints/net_w512-b8-kz32.pt \
        > overnight/eval/check_nnue-kz32.log 2>&1; then
        say "check_nnue: $(tail -n 1 overnight/eval/check_nnue-kz32.log)"
        $PY -u -m testing.gauntlet --challenger "$d" \
            --champion overnight/challengers/050-compiled-search \
            --elo0 0 --elo1 20 --games 600 > overnight/eval/056-kz32.gauntlet.log 2>&1
        say "056-kz32: $(verdict overnight/eval/056-kz32.gauntlet.log)"
    else
        say "CHECK_NNUE FAILED: $(tail -n 2 overnight/eval/check_nnue-kz32.log | tr '\n' ' ')"
    fi
fi
say "night3 done"
