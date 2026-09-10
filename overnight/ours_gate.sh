#!/bin/bash
# Gate only (the chain's json parse failed on 10 Sep; training had improved 0.007391 -> 0.007019).
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night4.log
NAME=v17o
CKPT=training/checkpoints/net_$NAME.pt
say() { echo "$(date '+%H:%M') OURS $*" >> "$LOG"; }
say "gate start (chain json parse bug: real val parent 0.007391 -> best 0.007019 at epoch 8)"
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
