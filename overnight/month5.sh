#!/bin/bash
# Backlog 3 (5 Sep): fifth month of data, then the 16-zone retrain.
#   fetch 2024_11 -> pack it (4 workers: the laptop gauntlet keeps its cores)
#   -> train kz16r on five months at lr 1e-4 from the champion kz16 checkpoint
#   -> export float16 + check_nnue + endgame suite into challengers/104-kz16r.
# Idempotent: every step is skipped when its output already exists, so the
# script can be relaunched after a crash. It does NOT queue the gauntlet --
# the next loop iteration reads the suite numbers and queues 104-kz16r itself.
# The desktop self-play parquets (gen-001/002, ~570k positions) are left out
# on purpose: as a sixth rotating shard they would get ~250x the per-position
# weight of the 145M-position shards; a self-play mix is its own experiment.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') $*" >> "$LOG"; }

say "month5 start (fetch 2024_11 -> pack -> train kz16r -> suite)"

# 1. fetch the fifth month (resumable, ~7.5 GB)
if [ ! -f data/standard_rated_2024_11.parquet ] || \
   ! grep -q COMPLETE overnight/eval/fetch-2024_11.log 2>/dev/null; then
    $PY -u -m training.fetch --month 2024_11 --attempts 40 \
        > overnight/eval/fetch-2024_11.log 2>&1 || { say "month5 FETCH FAILED"; exit 1; }
fi
say "month5 fetch done: $(tail -n 1 overnight/eval/fetch-2024_11.log)"

# 2. pack it like months 2 and 3, but on 4 workers: the gauntlet owns the CPU
if [ ! -f data/positions_2024_11.npy ]; then
    $PY -u -m training.pack --source data/standard_rated_2024_11.parquet \
        --target 150000000 --val-target 200000 --min-ply 16 --quiet-fraction 0 \
        --workers 4 \
        --out data/positions_2024_11.npy --val-out data/validation_2024_11.npy \
        > overnight/eval/pack-2024_11.log 2>&1 || { say "month5 PACK FAILED"; exit 1; }
fi
say "month5 pack done: $(grep -E '^wrote' overnight/eval/pack-2024_11.log | tail -n 1)"

# 3. retrain 16 zones on all five shards, lr 1e-4, from the champion checkpoint
if [ ! -f training/checkpoints/net_w512-b8-kz16r.json ]; then
    $PY -u training/train.py \
        --data data/positions_w512-150m.npy data/positions_w512-150m-b.npy \
               data/positions_2025_02.npy data/positions_2025_03.npy \
               data/positions_2024_11.npy \
        --val data/validation_w512-150m.npy \
        --resume training/checkpoints/net_w512-b8-kz16.pt \
        --accumulator 512 --buckets 8 --king-zones 16 \
        --lr 1e-4 --epochs 20 --patience 8 --warmup-epochs 1 --skip-sanity \
        --out training/checkpoints/net_w512-b8-kz16r.pt \
        > overnight/eval/train-kz16r.log 2>&1 || { say "month5 TRAIN FAILED"; exit 1; }
fi
say "month5 train done: $(grep -E 'restored|wrote.*json' overnight/eval/train-kz16r.log | tail -n 2 | tr '\n' ' ')"

# 4. challenger 104-kz16r: tree + the new net, checked and suite-scored
d=overnight/challengers/104-kz16r
if [ ! -f overnight/eval/suite-104-kz16r.log ]; then
    rm -rf "$d"; mkdir -p "$d/weights"
    cp agent.py fastboard.py fastsearch.py "$d/"
    cp weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
    $PY -u -m training.export --checkpoint training/checkpoints/net_w512-b8-kz16r.pt \
        --out "$d/weights/net.npz" --half > overnight/eval/export-kz16r.log 2>&1 \
        || { say "month5 EXPORT FAILED"; exit 1; }
    if ! $PY -u -m training.check_nnue --agent "$d" \
        --checkpoint training/checkpoints/net_w512-b8-kz16r.pt \
        > overnight/eval/check_nnue-kz16r.log 2>&1; then
        say "month5 CHECK_NNUE FAILED"; exit 1
    fi
    say "check_nnue kz16r: $(tail -n 1 overnight/eval/check_nnue-kz16r.log)"
    $PY -u -m testing.endgame_suite run --agent "$d" --seconds 2.5 \
        > overnight/eval/suite-104-kz16r.log 2>&1
fi
say "suite 104-kz16r: $(grep -E 'mean loss' overnight/eval/suite-104-kz16r.log | tail -n 1)"
say "month5 done -- next iteration: queue 104-kz16r (net task) if the suite looks sane"
