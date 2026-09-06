#!/bin/bash
# Re-run the WDL decode. The first attempt (4 workers, one 72.5M shard) hung after task 0
# with its pool workers gone and, because nothing is written until a shard fills, had
# nothing to show for 20 minutes. This one uses 20M shards so progress lands on disk and a
# stall is visible, and 8 workers now that the gauntlet is not competing.
# Waits for the running clocktest first: clocktests measure time management and any heavy
# load invalidates them.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') wdldecode $*" >> "$LOG"; }
while [ ! -f overnight/laptop/results/drawcap-clocktest-l.txt ]; do sleep 20; done
say "clocktest finished; decoding with 8 workers, 20M shards"
$PY -u -m training.binpack_decode \
    data/sf/test80-2024-02-feb-2tb7p.min-v2.v6.binpack.zst \
    --out data/sfw/feb24w --target 145000000 --shard 20000000 \
    --workers 8 --wdl-lambda 0.75 > overnight/eval/sf-decode-wdl.log 2>&1
say "decode: $(grep '^done:' overnight/eval/sf-decode-wdl.log || echo FAILED)"
