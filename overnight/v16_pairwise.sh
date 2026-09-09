#!/bin/bash
# v16 = v12's data and v12's weights, with three architecture changes taken from
# Alexandria 9.0 (its net is NOT used and cannot be: rules.md requires every shipped
# network to be one we trained ourselves, and calls starting from a published one
# shipping it. Architecture is not weights; this net is trained from v12 on our data):
#
#   pairwise   the accumulator's halves are multiplied rather than each lane squared
#   dual       each hidden neuron emits relu(z) and clamp(z,0,1)**2
#   buckets    the endgame-dense (quadratic) 8-head map instead of equal-width bands
#
# The accumulator doubles to 1024 so the head still sees 1024 inputs, which costs
# about 13% of node rate (accumulator updates are 15.4% of search time and double;
# the forward pass is unchanged) -- the architecture has to beat that to be worth it.
#
# WARM START, not from scratch. v15 went from scratch on more data than this and lost
# to v12 on both gates. training/warm_start.py makes the change an identity at
# initialisation instead: measured 0.00639845 against v12's 0.00637566 on mixed_val,
# +0.36%, which is the symmetry-breaking jitter plus the two heads that share a seed.
#
# Data is v12's exactly, and "exactly" was checked against v12's own training log
# rather than against its script: overnight/mixmirror_train.sh's header talks about
# --limit 40M, but the command as committed has no such flag and train-mixmirror.log
# shows v12 loading the FULL 145M-row human shards. So no --limit here either. That
# is ~10 GB resident for one shard (Batches copies the index columns out of the
# memmap), against 16 GB free, so it fits with room but nothing else large should run
# beside it. A first run with 40M was killed and restarted for this reason: capping
# the human shards would have confounded the architecture against the data volume,
# which is the one thing this run exists to separate.
#
#   nohup bash overnight/v16_pairwise.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night4.log
NAME=250-v16                       # ONE name: v15's chain built 210-v13 and then
CHAMP=200-mixmirror-s400           # gauntleted 230-v15, which did not exist, and the
CKPT=training/checkpoints/net_v16.pt   # 400-game verdict never ran at all.
INIT=training/checkpoints/net_v16_init.pt
say() { echo "$(date '+%H:%M') v16 $*" >> "$LOG"; }
say "start (pairwise + dual + endgame buckets, warm-started from v12, v12's shards)"

SF=(data/sf80kf/sf80kf_00.npy data/sf80kf/sf80kf_05.npy data/sf80kf/sf80kf_10.npy data/sf80kf/sf80kf_15.npy \
    data/sf80kf/sf80kf_20.npy data/sf80kf/sf80kf_25.npy data/sf80kf/sf80kf_30.npy data/sf80kf/sf80kf_35.npy)
HU=(data/positions_w512-150m.npy data/positions_2025_02.npy data/positions_w512-150m-b.npy data/positions_2025_03.npy)
SHARDS=""
for i in 0 1 2 3 4 5 6 7; do SHARDS="$SHARDS ${SF[$i]} ${HU[$((i % 4))]}"; done

if [ ! -f "$INIT" ]; then
    $PY -u -m training.warm_start --parent training/checkpoints/net_v12-mixmirror.pt \
        --out "$INIT" --verify data/mixed_val.npy > overnight/eval/warm-v16.log 2>&1 \
        || { say "WARM START FAILED"; exit 1; }
fi
say "warm start: $(grep -E 'validation|difference' overnight/eval/warm-v16.log | tr '\n' ' ')"

# lr 1e-4, measured rather than argued. Probe of 8 epochs over two shard pairs at
# --limit 4M, all warm-started from the same init (best val, lower is better):
#     1e-4  0.006254   <- also beats v12's own 0.006376
#     3e-4  0.006384
#     1e-3  0.006398   <- never improved on the warm start at all
# So the rate that failed to improve v12's ARCHITECTURE (v13, v14) is the right one for
# a warm-started change TO that architecture: the new parameters have somewhere to go
# and the old ones are already where they should be.
if [ ! -f "${CKPT%.pt}.json" ]; then
    $PY -u training/train.py \
        --data $SHARDS \
        --val data/mixed_val.npy \
        --resume "$INIT" --mirror \
        --accumulator 1024 --buckets 8 --king-zones 16 \
        --lr 1e-4 --epochs 64 --patience 16 --warmup-epochs 2 --skip-sanity \
        --out "$CKPT" > overnight/eval/train-v16.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
fi
say "train done: $(grep -E 'restored|wrote.*json' overnight/eval/train-v16.log | tail -n 2 | tr '\n' ' ')"

d=overnight/challengers/$NAME
rm -rf "$d"; mkdir -p "$d/weights"
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

# Both scales, as every net gets: the shipped default and EVAL_SCALE off.
cp -r "$d" "$d-s400"
sed -i 's/^EVAL_SCALE: Final = True$/EVAL_SCALE: Final = False/' "$d-s400/agent.py"
for v in "$NAME" "$NAME-s400"; do
    $PY -u -m testing.mistakes retest --agent "overnight/challengers/$v" --set overnight/eval/mistakes50.json \
        --dump "overnight/eval/retest-$v.jsonl" > "overnight/eval/retest-$v.log" 2>&1
    say "$v replay: $(grep -E 'FIXED \(|WORSE by|total cp' overnight/eval/retest-$v.log | sed 's/  */ /g' | tr '\n' ';')"
done

# Pick the better scale by cp given away, exactly as the v13/v15 chains did, but from
# files named after THIS run.
a=$(grep -oE "now [0-9,]+" "overnight/eval/retest-$NAME.log" | tr -d ',' | awk '{print $2}')
b=$(grep -oE "now [0-9,]+" "overnight/eval/retest-$NAME-s400.log" | tr -d ',' | awk '{print $2}')
pick=$NAME; [ "${b:-99999}" -lt "${a:-99999}" ] && pick=$NAME-s400
if [ ! -d "overnight/challengers/$pick" ]; then say "PICK $pick MISSING -- no gauntlet"; exit 1; fi
say "gauntlet: $pick vs $CHAMP (v12 as shipped), 8 s, checkpoint 200"
$PY -u -m testing.gauntlet --challenger "overnight/challengers/$pick" \
    --champion "overnight/challengers/$CHAMP" --games 400 --workers 4 \
    > overnight/eval/v16-vs-v12.gauntlet.log 2>&1
say "gauntlet result: $(grep -E '^(PROMOTE|REJECT|INCONCLUSIVE)|score' overnight/eval/v16-vs-v12.gauntlet.log | tail -n 2 | tr '\n' ' ')"
say "done"
