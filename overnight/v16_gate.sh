#!/bin/bash
# Gate one v16 checkpoint: export -> check_nnue -> eval_rank -> Alexandria comparison
# -> mistake replay at both eval scales -> 400 games against v12 as shipped.
#
# Split out of v16_long.sh so a checkpoint can be judged without committing to the rest
# of the training chain. Waits for the checkpoint's .json report, which train.py writes
# AFTER the .pt, so the file being there means the save finished.
#
#   nohup bash overnight/v16_gate.sh training/checkpoints/net_v16f_c0.pt 253-v16f > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night4.log
CKPT="${1:?usage: v16_gate.sh <checkpoint.pt> <name>}"
NAME="${2:?usage: v16_gate.sh <checkpoint.pt> <name>}"
CHAMP=200-mixmirror-s400          # v12 exactly as shipped
say() { echo "$(date '+%H:%M') gate $*" >> "$LOG"; }

# Wait for training to land the checkpoint. No timeout: the trainer either finishes or
# it does not, and a gate that gave up early would be worse than one that waited.
if [ ! -f "${CKPT%.pt}.json" ]; then
    say "waiting for $CKPT"
    while [ ! -f "${CKPT%.pt}.json" ]; do
        sleep 60
        if ! ps -ef | grep -q "[t]rain\.py"; then
            sleep 30   # it may be mid-save
            [ -f "${CKPT%.pt}.json" ] || { say "TRAINER GONE and no checkpoint -- nothing to gate"; exit 1; }
        fi
    done
fi
say "$NAME start: $(grep -E 'restored' overnight/eval/train-v16f-c0.log | tail -n 1)"

d=overnight/challengers/$NAME
rm -rf "$d" "$d-s300"; mkdir -p "$d/weights"
cp agent.py fastboard.py fastsearch.py "$d/"
cp weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
$PY -u -m training.export --checkpoint "$CKPT" --out "$d/weights/net.npz" --half --mirror \
    > "overnight/eval/export-$NAME.log" 2>&1 || { say "$NAME EXPORT FAILED"; exit 1; }
say "$NAME export: $(head -n 1 overnight/eval/export-$NAME.log)"

$PY -u -m training.check_nnue --agent "$d" --checkpoint "$CKPT" \
    > "overnight/eval/check_nnue-$NAME.log" 2>&1 || { say "$NAME CHECK_NNUE FAILED"; exit 1; }
say "$NAME check_nnue: $(tail -n 1 overnight/eval/check_nnue-$NAME.log)"

$PY -u -m testing.eval_rank --agent "$d" --depth 12 --out "overnight/eval/rank-$NAME.jsonl" \
    > "overnight/eval/rank-$NAME.log" 2>&1
say "$NAME eval_rank: $(grep -E '^  all' overnight/eval/rank-$NAME.log | tail -n 1)  (v12: all 60 +50 145 202 0.35 +0.40)"

# Alexandria 9.0 is a REFERENCE ONLY -- never shipped, never loaded by the engine.
$PY -u engines/alexandria/compare.py --agent "$d" --label "$NAME" \
    > "overnight/eval/alexcmp-$NAME.log" 2>&1
say "$NAME vs alexandria: $(grep -E 'gain-fitted|^top-loss:' overnight/eval/alexcmp-$NAME.log | tr '\n' ' ')"

# The tree ships EVAL_SCALE False, so the SCALED variant is the one to build.
cp -r "$d" "$d-s300"
sed -i 's/^EVAL_SCALE: Final = False$/EVAL_SCALE: Final = True/' "$d-s300/agent.py"
grep -q '^EVAL_SCALE: Final = True$' "$d-s300/agent.py" || { say "SCALE VARIANT NOT APPLIED"; exit 1; }
for v in "$NAME" "$NAME-s300"; do
    $PY -u -m testing.mistakes retest --agent "overnight/challengers/$v" --set overnight/eval/mistakes50.json \
        --dump "overnight/eval/retest-$v.jsonl" > "overnight/eval/retest-$v.log" 2>&1
    say "$v replay: $(grep -E 'FIXED \(|WORSE by|total cp' overnight/eval/retest-$v.log | sed 's/  */ /g' | tr '\n' ';')"
done

a=$(grep -oE "now [0-9,]+" "overnight/eval/retest-$NAME.log" | tr -d ',' | awk '{print $2}')
b=$(grep -oE "now [0-9,]+" "overnight/eval/retest-$NAME-s300.log" | tr -d ',' | awk '{print $2}')
pick=$NAME; [ "${b:-99999}" -lt "${a:-99999}" ] && pick=$NAME-s300
[ -d "overnight/challengers/$pick" ] || { say "PICK $pick MISSING -- no gauntlet"; exit 1; }
say "gauntlet: $pick vs $CHAMP (v12 as shipped), 8 s, checkpoint 200"
$PY -u -m testing.gauntlet --challenger "overnight/challengers/$pick" \
    --champion "overnight/challengers/$CHAMP" --games 400 --workers 4 \
    > "overnight/eval/$NAME-vs-v12.gauntlet.log" 2>&1
say "gauntlet result: $(grep -E '^(PROMOTE|REJECT|INCONCLUSIVE)|score' overnight/eval/$NAME-vs-v12.gauntlet.log | tail -n 2 | tr '\n' ' ')"
say "$NAME done"
