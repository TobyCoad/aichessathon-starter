#!/bin/bash
# v16f: 200 epochs from scratch -- pairwise + dual + endgame buckets + factoriser --
# on v12's shard list, then the full gate chain against v12 AND against Alexandria.
#
# CHUNKED, and that is not cosmetic: training/train.py writes its checkpoint only when
# the whole run ends, so a single 200-epoch run that has to be killed loses everything.
# Four chunks of 50 leave a gateable checkpoint every ~4.6 h. Chunk 1 builds the net from
# the architecture flags; chunks 2-4 --resume, which carries the architecture in the file
# (the flags are IGNORED on resume -- the file wins -- so they are not repeated).
#
# --limit 40M, unlike the 32-epoch run this replaces. 200 epochs of the full 145M-row
# human shards would be ~51 h; capped they are ~18.6 h. The cost is distinct data: 324M
# positions rotated 12.5 times instead of 744M rotated twice. Total samples ~6.0B, against
# v15's ~9B and the 32-epoch run's 2.6B -- so this is the run that actually answers
# whether the architecture is worth having, rather than whether it can be undertrained.
#
# lr steps down per chunk (1e-3, 6e-4, 3e-4, 1.5e-4), each with its own cosine decay.
# One 200-epoch cosine would be better in principle; this approximates it while keeping
# the crash-safety of separate runs.
#
#   nohup bash overnight/v16_long.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night4.log
NAME=253-v16f
CHAMP=200-mixmirror-s400          # v12 exactly as shipped
say() { echo "$(date '+%H:%M') v16f $*" >> "$LOG"; }
say "start: 200 epochs from scratch, 4 x 50, pairwise+dual+endgame+factoriser"

SF=(data/sf80kf/sf80kf_00.npy data/sf80kf/sf80kf_05.npy data/sf80kf/sf80kf_10.npy data/sf80kf/sf80kf_15.npy \
    data/sf80kf/sf80kf_20.npy data/sf80kf/sf80kf_25.npy data/sf80kf/sf80kf_30.npy data/sf80kf/sf80kf_35.npy)
HU=(data/positions_w512-150m.npy data/positions_2025_02.npy data/positions_w512-150m-b.npy data/positions_2025_03.npy)
SHARDS=""
for i in 0 1 2 3 4 5 6 7; do SHARDS="$SHARDS ${SF[$i]} ${HU[$((i % 4))]}"; done

LRS=(1e-3 6e-4 3e-4 1.5e-4)
prev=""
for c in 0 1 2 3; do
    ck="training/checkpoints/net_v16f_c$c.pt"
    if [ ! -f "${ck%.pt}.json" ]; then
        if [ -z "$prev" ]; then
            ARCH="--pairwise --dual --endgame-buckets --factoriser"
            RESUME=""
        else
            ARCH=""
            RESUME="--resume $prev"
        fi
        $PY -u training/train.py \
            --data $SHARDS \
            --val data/mixed_val.npy \
            --mirror $ARCH $RESUME \
            --accumulator 1024 --buckets 8 --king-zones 16 \
            --lr "${LRS[$c]}" --epochs 50 --patience 20 --warmup-epochs 2 --skip-sanity \
            --limit 40000000 \
            --out "$ck" >> "overnight/eval/train-v16f-c$c.log" 2>&1 \
            || { say "chunk $c TRAIN FAILED"; exit 1; }
    fi
    say "chunk $c (lr ${LRS[$c]}): $(grep -E 'restored' overnight/eval/train-v16f-c$c.log | tail -n 1)"
    prev="$ck"
done
CKPT="$prev"

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

# Ranking gate, and the three-way static comparison against v12 and Alexandria 9.0 on the
# same corpus with one shared Stockfish reference. Alexandria is a REFERENCE ONLY -- its
# net is never shipped and never loaded by the engine; rules.md requires every shipped net
# to be one we trained.
$PY -u -m testing.eval_rank --agent "$d" --depth 12 --out "overnight/eval/rank-$NAME.jsonl" \
    > "overnight/eval/rank-$NAME.log" 2>&1
say "$NAME eval_rank: $(grep -E '^  all' overnight/eval/rank-$NAME.log | tail -n 1)  (v12: all 60 +50 145 202 0.35 +0.40)"

$PY -u engines/alexandria/compare.py --agent "$d" --label v16f \
    > "overnight/eval/alexcmp-$NAME.log" 2>&1
say "$NAME vs alexandria: $(grep -E 'top-loss:|gain-fitted' overnight/eval/alexcmp-$NAME.log | tr '\n' ' ')"

# Both eval scales. The tree ships EVAL_SCALE False, so the SCALED variant is the one
# that has to be built -- the previous chain sed'd True->False against a tree that was
# already False, which silently produced two identical directories.
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
    > overnight/eval/v16f-vs-v12.gauntlet.log 2>&1
say "gauntlet result: $(grep -E '^(PROMOTE|REJECT|INCONCLUSIVE)|score' overnight/eval/v16f-vs-v12.gauntlet.log | tail -n 2 | tr '\n' ' ')"
say "done"
