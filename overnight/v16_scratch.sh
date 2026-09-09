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
# FROM SCRATCH, deliberately. The warm start this script originally used was measured
# and found inert for the change it existed to enable: seeding both accumulator halves
# from the same v12 column leaves them at cosine 0.99920, and 8 epochs at lr 1e-4, 3e-4
# and 1e-3 all left that unchanged to five decimals. d(h)/da = clip(b) and d(h)/db =
# clip(a), so at a == b both halves get the same gradient and the symmetry is a saddle
# that 2% jitter does not escape -- the net stays SCReLU wearing a pairwise coat.
# Random init gives independent halves for free. The cost is v15's precedent: from
# scratch on this data reached better val than v12 (0.006163 vs 0.006376) and still lost
# eval_rank and the replay. So this run is judged on those two gates before any games.
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
NAME=251-v16s                      # ONE name: v15's chain built 210-v13 and then
CHAMP=200-mixmirror-s400           # gauntleted 230-v15, which did not exist, and the
CKPT=training/checkpoints/net_v16s.pt   # 400-game verdict never ran at all.
say() { echo "$(date '+%H:%M') v16s $*" >> "$LOG"; }
say "start (from scratch: pairwise + dual + endgame buckets + factoriser, v12's shards)"

SF=(data/sf80kf/sf80kf_00.npy data/sf80kf/sf80kf_05.npy data/sf80kf/sf80kf_10.npy data/sf80kf/sf80kf_15.npy \
    data/sf80kf/sf80kf_20.npy data/sf80kf/sf80kf_25.npy data/sf80kf/sf80kf_30.npy data/sf80kf/sf80kf_35.npy)
HU=(data/positions_w512-150m.npy data/positions_2025_02.npy data/positions_w512-150m-b.npy data/positions_2025_03.npy)
SHARDS=""
for i in 0 1 2 3 4 5 6 7; do SHARDS="$SHARDS ${SF[$i]} ${HU[$((i % 4))]}"; done

# No warm start: see the header. The architecture flags build the net directly."

# lr 1e-3 is v15's from-scratch rate, not the 1e-4 the warm-start probe chose: that probe
# measured how far to move a net ALREADY at v12's optimum, which is the opposite problem.
#
# FACTORISER on. One 768-row plane shared by all 16 king zones, summed into the accumulator
# during training and folded into every zone at export, so the shipped net is identical in
# shape and cost. Without it each zone learns from ~1/16 of the data and has to rediscover
# what a feature means independently -- which is what separates this run from v15, the only
# other 16-zone net trained from noise, and the one whose failure (better val than v12,
# worse eval_rank and replay) looks exactly like undertrained zones.
# Verified before launch: the exported fold reproduces the torch model over 399 positions
# (check_nnue all checks passed), and after 3 smoke epochs the shared plane holds the
# signal -- factor rms 0.105 against the zoned plane's 0.015, zone spread 0.011 -- so the
# zones are learning residuals rather than the whole function.
#
# 32 epochs, not 56: the second embedding lookup costs ~40% of throughput (0.09M pos/s
# against 0.153M, measured on a full 20.1M shard), and train.py only writes its checkpoint
# at the END, so a run that overshoots and has to be killed loses everything. 32 epochs is
# two full passes over v12's data, ~8.2 h -- and ~2.6B samples against v15's ~9B, so if
# this lands near v12 rather than past it, undertraining is the first thing to suspect and
# a longer run is the follow-up, not a different architecture.
if [ ! -f "${CKPT%.pt}.json" ]; then
    $PY -u training/train.py \
        --data $SHARDS \
        --val data/mixed_val.npy \
        --mirror --pairwise --dual --endgame-buckets --factoriser \
        --accumulator 1024 --buckets 8 --king-zones 16 \
        --lr 1e-3 --epochs 32 --patience 12 --warmup-epochs 2 --skip-sanity \
        --out "$CKPT" > overnight/eval/train-v16s.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
fi
say "train done: $(grep -E 'restored|wrote.*json' overnight/eval/train-v16s.log | tail -n 2 | tr '\n' ' ')"

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
    > overnight/eval/v16s-vs-v12.gauntlet.log 2>&1
say "gauntlet result: $(grep -E '^(PROMOTE|REJECT|INCONCLUSIVE)|score' overnight/eval/v16s-vs-v12.gauntlet.log | tail -n 2 | tr '\n' ' ')"
say "done"
