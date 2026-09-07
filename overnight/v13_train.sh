#!/bin/bash
# Overnight 7-8 Sep: two candidate nets on top of v12, each fully evaluated by morning.
#   v13  = v12 continued on the WIDE mix: all 38 n80000 shards rotated against the five
#          human months (4 Lichess + 2024_11), --limit 40M per human shard, 72 epochs.
#          v12 was still improving at its last epoch on only 8 SF shards + 4 human.
#   v13b = v13 continued 16 epochs with the endgame-heavy Lichess shard (70% <= 16 pieces)
#          alternating with SF shards, aimed at the 13-16-piece hole every net shares.
# Each: export --mirror -> check_nnue -> eval_rank (the ranking gate) -> mistake replays at
# scale 300 and 400 -> then v13 plays a 200-game checkpoint gauntlet vs v12 at 8 s.
# Clock-sensitive steps run late, after the GPU work, when nothing else is on the box.
#   nohup bash overnight/v13_train.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') v13 $*" >> "$LOG"; }
say "start"

HU=(data/positions_w512-150m.npy data/positions_2025_02.npy data/positions_w512-150m-b.npy data/positions_2025_03.npy data/positions_2024_11.npy)
SHARDS=""
i=0
for s in $(seq -w 0 37); do
    SHARDS="$SHARDS data/sf80kf/sf80kf_$s.npy ${HU[$((i % 5))]}"
    i=$((i + 1))
done

evaluate() {  # name checkpoint
    local name=$1 ck=$2 d=overnight/challengers/$1
    rm -rf "$d"; mkdir -p "$d/weights"
    cp agent.py fastboard.py fastsearch.py "$d/"
    cp weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
    $PY -u -m training.export --checkpoint "$ck" --out "$d/weights/net.npz" --half --mirror \
        > "overnight/eval/export-$name.log" 2>&1 || { say "$name EXPORT FAILED"; return 1; }
    $PY -u -m training.check_nnue --agent "$d" --checkpoint "$ck" > "overnight/eval/check_nnue-$name.log" 2>&1 \
        || { say "$name CHECK_NNUE FAILED"; return 1; }
    say "$name check_nnue: $(tail -n 1 overnight/eval/check_nnue-$name.log)"
    $PY -u -m testing.eval_rank --agent "$d" --depth 12 --out "overnight/eval/rank-$name.jsonl" \
        > "overnight/eval/rank-$name.log" 2>&1
    say "$name eval_rank: $(grep -E '^  all' overnight/eval/rank-$name.log | tail -n 1)  (v12: all 60 +55 114 201 0.35 +0.40; v11: +39 133 239 0.27 +0.30)"
    # scale 300 (tree default now) and 400
    cp -r "$d" "$d-s400"; sed -i 's/^EVAL_SCALE: Final = True$/EVAL_SCALE: Final = False/' "$d-s400/agent.py"
    for v in "$name" "$name-s400"; do
        $PY -u -m testing.mistakes retest --agent "overnight/challengers/$v" --set overnight/eval/mistakes50.json \
            --dump "overnight/eval/retest-$v.jsonl" > "overnight/eval/retest-$v.log" 2>&1
        say "$v replay: $(grep -E 'FIXED \(|WORSE by|total cp' overnight/eval/retest-$v.log | sed 's/  */ /g' | tr '\n' ';')"
    done
}

if [ ! -f training/checkpoints/net_v13.json ]; then
    $PY -u training/train.py --data $SHARDS --val data/mixed_val.npy \
        --resume training/checkpoints/net_v12-mixmirror.pt --mirror \
        --accumulator 512 --buckets 8 --king-zones 16 \
        --lr 1e-4 --epochs 72 --patience 16 --warmup-epochs 1 --skip-sanity --limit 40000000 \
        --out training/checkpoints/net_v13.pt > overnight/eval/train-v13.log 2>&1 || { say "v13 TRAIN FAILED"; exit 1; }
fi
say "v13 train done: $(grep -E 'restored|wrote.*json' overnight/eval/train-v13.log | tail -n 2 | tr '\n' ' ')"

EG=""
for s in 00 06 12 18 24 30 36 03; do EG="$EG data/sf80kf/sf80kf_$s.npy data/positions_endgame.npy"; done
if [ ! -f training/checkpoints/net_v13b.json ]; then
    $PY -u training/train.py --data $EG --val data/mixed_val.npy \
        --resume training/checkpoints/net_v13.pt --mirror \
        --accumulator 512 --buckets 8 --king-zones 16 \
        --lr 5e-5 --epochs 16 --patience 8 --warmup-epochs 1 --skip-sanity --limit 40000000 \
        --out training/checkpoints/net_v13b.pt > overnight/eval/train-v13b.log 2>&1 || say "v13b TRAIN FAILED"
fi
say "v13b train done: $(grep -E 'restored|wrote.*json' overnight/eval/train-v13b.log | tail -n 2 | tr '\n' ' ')"

evaluate 210-v13 training/checkpoints/net_v13.pt
[ -f training/checkpoints/net_v13b.pt ] && evaluate 211-v13b training/checkpoints/net_v13b.pt

# The strength verdict: 200-game checkpoint SPRT, v13 (at the better of its two scales) vs the
# v12 build in the tree at the time (overnight/challengers/200-mixmirror = v12 at scale 300).
a=$(grep -oE "now [0-9,]+" overnight/eval/retest-210-v13.log | tr -d ',' | awk '{print $2}')
b=$(grep -oE "now [0-9,]+" overnight/eval/retest-210-v13-s400.log | tr -d ',' | awk '{print $2}')
pick=210-v13; [ "${b:-99999}" -lt "${a:-99999}" ] && pick=210-v13-s400
say "gauntlet: $pick vs 200-mixmirror-s400 (v12 as shipped, scale 400), 8 s, checkpoint 200"
$PY -u -m testing.gauntlet --challenger "overnight/challengers/$pick" --champion overnight/challengers/200-mixmirror-s400 \
    --games 400 --workers 4 > overnight/eval/v13-vs-v12.gauntlet.log 2>&1
say "gauntlet result: $(grep -E '^(PROMOTE|REJECT|INCONCLUSIVE)|score' overnight/eval/v13-vs-v12.gauntlet.log | tail -n 2 | tr '\n' ' ')"
say "done"
