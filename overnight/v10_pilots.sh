#!/bin/bash
# NET_V10 pilots. Three short runs from the SAME start (the v9.3 champion checkpoint),
# the same shards and the same budget, so the only difference is the architecture change:
#   control  8 heads, unmirrored  -- what continuing to train alone buys
#   heads12  12 endgame-dense heads (BUCKET_MAP_12)
#   mirror   mirrored king zones
# Judged on the STRATIFIED val loss below 17 pieces, which is where every measured
# move-quality loss sits; val loss alone is a weak instrument (network.md point 1).
# A pilot only earns the full run + a gauntlet slot if it beats the control there.
# Waits for the mixnet3 training to finish: one GPU, one job.
#   bash.exe -lc "exec bash overnight/v10_pilots.sh"
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') v10pilot $*" >> "$LOG"; }

while ! grep -qE "mixnet3 (done|TRAIN FAILED|EXPORT FAILED|CHECK_NNUE FAILED)" "$LOG"; do
    sleep 120
done
say "start (mixnet3 finished; GPU free)"

START=training/checkpoints/net_w512-b8-kz16-mix2.pt   # the v9.3 champion net
SHARDS="data/sf/feb24_00.npy data/positions_w512-150m.npy data/sf/feb24_01.npy \
data/positions_2025_02.npy data/sf/feb24_02.npy data/positions_w512-150m-b.npy"
COMMON="--val data/mixval.npy --resume $START --accumulator 512 --king-zones 16 \
--lr 1e-4 --epochs 6 --patience 6 --warmup-epochs 1 --skip-sanity --limit 40000000"

run() {  # name, extra args
    local name=$1; shift
    if [ -f "overnight/eval/pilot-$name.log" ]; then say "$name already done"; return; fi
    $PY -u training/train.py --data $SHARDS $COMMON "$@" \
        --out "training/checkpoints/pilot-$name.pt" \
        > "overnight/eval/pilot-$name.log" 2>&1 || { say "$name FAILED"; return; }
    say "$name: $(grep -E 'restored|best' "overnight/eval/pilot-$name.log" | tail -n 1)"
    say "$name strata: $(grep 'strata' "overnight/eval/pilot-$name.log" | tail -n 1)"
}

run control --buckets 8
run heads12 --buckets 12
run mirror  --buckets 8 --mirror

say "pilots done -- compare the last strata line of each; the 2-8/9-12/13-16 cells decide"
git add overnight/eval/pilot-*.log overnight/v10_pilots.sh 2>/dev/null
git -c user.name=v10pilot -c user.email=v10@local commit -q -m "v10 pilots: control / 12 heads / mirrored, same start and budget" && \
    (git pull -q --rebase --autostash origin main >/dev/null 2>&1; git push -q origin main >/dev/null 2>&1)
