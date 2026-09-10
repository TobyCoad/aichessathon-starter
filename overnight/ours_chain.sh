#!/bin/bash
# OURS chain (9 Sep night): our-distribution data -> warm-start fine-tune of v16f -> gate.
#   1. wait for training.gen_ours (11 workers x 6 h, overnight/eval/gen-ours.log) to finish
#   2. training.mix_ours: raw rows -> corpus cp (uci_table) + lambda 0.75 blend, mixed shards
#   3. train: resume net_v16f.pt, lr 1e-4, 16 epochs, patience 6, val = ours + old
#   4. gate: export -> check_nnue -> challenger v17o (adjv2-shipped search + new net)
#      -> gauntlet vs adjv2-shipped at 8 s, checkpoint 200. Packaging is a morning decision.
#   nohup bash overnight/ours_chain.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night4.log
NAME=v17o
CKPT=training/checkpoints/net_$NAME.pt
say() { echo "$(date '+%H:%M') OURS $*" >> "$LOG"; }

while ! grep -q "^done:" overnight/eval/gen-ours.log; do
    sleep 120
    if ! ps -ef | grep -q "[g]en_ours"; then
        sleep 30
        grep -q "^done:" overnight/eval/gen-ours.log || { say "GENERATOR GONE without done -- using what is on disk"; break; }
    fi
done
say "generation: $(grep -E '^done:' overnight/eval/gen-ours.log || echo 'partial') ($(ls data/ours/ours_*.npy | wc -l) files)"
[ -f data/ours/uci_table.txt ] || { say "NO uci_table.txt -- chain stopped"; exit 1; }

$PY -u -m training.mix_ours --raw data/ours --table data/ours/uci_table.txt --out data/mix_ours \
    > overnight/eval/mix-ours.log 2>&1 || { say "MIX FAILED"; exit 1; }
say "mix: $(grep -E '^ours:' overnight/eval/mix-ours.log) | $(grep -E '^val:' overnight/eval/mix-ours.log)"

SHARDS=$(ls data/mix_ours/mix_*.npy | sort | tr '\n' ' ')
say "train start: resume v16f, lr 1e-4, 16 epochs, shards: $SHARDS"
$PY -u training/train.py --data $SHARDS --val data/mix_ours/val.npy \
    --resume training/checkpoints/net_v16f.pt --mirror \
    --accumulator 1024 --buckets 8 --king-zones 16 \
    --lr 1e-4 --epochs 16 --patience 6 --warmup-epochs 1 --skip-sanity \
    --out "$CKPT" > overnight/eval/train-$NAME.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
say "train done: $(grep -E 'restored|initial' overnight/eval/train-$NAME.log | tr '\n' ' ')"
if grep -q "epochs': 0" "${CKPT%.pt}.json" 2>/dev/null; then say "no epoch beat the parent -- nothing to gate"; exit 0; fi
INIT=$($PY -c "import json; print(json.load(open('${CKPT%.pt}.json'))['initial_val'])")
BEST=$($PY -c "import json; print(json.load(open('${CKPT%.pt}.json'))['best_val'])")
say "val: parent $INIT -> best $BEST"
if $PY -c "import sys; sys.exit(0 if float('$BEST') < float('$INIT') - 1e-7 else 1)"; then :; else say "best == parent: the fine-tune never improved the mixed val -- nothing to gate"; exit 0; fi

d=overnight/challengers/$NAME
rm -rf "$d"; mkdir -p "$d/weights"
cp overnight/challengers/adjv2-shipped/agent.py overnight/challengers/adjv2-shipped/fastboard.py overnight/challengers/adjv2-shipped/fastsearch.py "$d/"
cp overnight/challengers/adjv2-shipped/weights/book.bin "$d/weights/"; cp -r overnight/challengers/adjv2-shipped/weights/syzygy "$d/weights/"
$PY -u -m training.export --checkpoint "$CKPT" --out "$d/weights/net.npz" --half --mirror \
    > "overnight/eval/export-$NAME.log" 2>&1 || { say "EXPORT FAILED"; exit 1; }
say "export: $(head -n 1 overnight/eval/export-$NAME.log)"
$PY -u -m training.check_nnue --agent "$d" --checkpoint "$CKPT" \
    > "overnight/eval/check_nnue-$NAME.log" 2>&1 || { say "CHECK_NNUE FAILED"; exit 1; }
say "check_nnue: $(tail -n 1 overnight/eval/check_nnue-$NAME.log)"
say "gauntlet $NAME vs adjv2-shipped, 8 s, checkpoint 200 starting"
$PY -u -m testing.gauntlet --challenger "$d" --champion overnight/challengers/adjv2-shipped \
    --games 400 --workers 4 > "overnight/eval/$NAME-vs-adjv2.gauntlet.log" 2>&1
say "gauntlet: $(grep -E '^(PROMOTE|REJECT|INCONCLUSIVE)|checkpoint' overnight/eval/$NAME-vs-adjv2.gauntlet.log | tr '\n' ' ') | $(grep -E 'games' overnight/eval/$NAME-vs-adjv2.gauntlet.log | tail -n 1)"
say "chain done"
