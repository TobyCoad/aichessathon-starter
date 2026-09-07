#!/bin/bash
# v12 candidate net: the v11 mirrored architecture fine-tuned on MIXED data -- the n80000
# Stockfish shards interleaved with the four Lichess (human-game) months, one shard per
# epoch -- warm-started from net_v11-mirror.pt. The three-net ranking audit (7 Sep 20:20)
# showed the pure-Stockfish nets misrank human-position moves; the mixed recipe is the one
# that last won games (v9.3, +19). Gate here is eval_rank, NOT the endgame suite; the
# gauntlet follows only if eval_rank is not worse than the champion.
# --limit 40M: the first run took the machine down while copying a 145M-position (9.8 GB)
# Lichess shard into RAM; 40M keeps the SF:human position ratio at ~1:2 per epoch pair.
#   nohup bash overnight/mixmirror_train.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') mixmirror $*" >> "$LOG"; }
say "start (v11 warm start, sf80kf x Lichess interleaved, --mirror)"

SF=(data/sf80kf/sf80kf_00.npy data/sf80kf/sf80kf_05.npy data/sf80kf/sf80kf_10.npy data/sf80kf/sf80kf_15.npy \
    data/sf80kf/sf80kf_20.npy data/sf80kf/sf80kf_25.npy data/sf80kf/sf80kf_30.npy data/sf80kf/sf80kf_35.npy)
HU=(data/positions_w512-150m.npy data/positions_2025_02.npy data/positions_w512-150m-b.npy data/positions_2025_03.npy)
SHARDS=""
for i in 0 1 2 3 4 5 6 7; do SHARDS="$SHARDS ${SF[$i]} ${HU[$((i % 4))]}"; done

if [ ! -f training/checkpoints/net_v12-mixmirror.json ]; then
    $PY -u training/train.py \
        --data $SHARDS \
        --val data/sf80kf/sf80kf_val.npy \
        --resume training/checkpoints/net_v11-mirror.pt --mirror \
        --accumulator 512 --buckets 8 --king-zones 16 \
        --lr 1e-4 --epochs 32 --patience 10 --warmup-epochs 1 --skip-sanity \
        --out training/checkpoints/net_v12-mixmirror.pt \
        > overnight/eval/train-mixmirror.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
fi
say "train done: $(grep -E 'restored|wrote.*json' overnight/eval/train-mixmirror.log | tail -n 2 | tr '\n' ' ')"

d=overnight/challengers/200-mixmirror
rm -rf "$d"; mkdir -p "$d/weights"
cp agent.py fastboard.py fastsearch.py "$d/"
cp weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
$PY -u -m training.export --checkpoint training/checkpoints/net_v12-mixmirror.pt \
    --out "$d/weights/net.npz" --half > overnight/eval/export-mixmirror.log 2>&1 \
    || { say "EXPORT FAILED: $(tail -n 2 overnight/eval/export-mixmirror.log | tr '\n' ' ')"; exit 1; }
if ! $PY -u -m training.check_nnue --agent "$d" \
    --checkpoint training/checkpoints/net_v12-mixmirror.pt \
    > overnight/eval/check_nnue-mixmirror.log 2>&1; then
    say "CHECK_NNUE FAILED: $(tail -n 2 overnight/eval/check_nnue-mixmirror.log | tr '\n' ' ')"; exit 1
fi
say "check_nnue: $(tail -n 1 overnight/eval/check_nnue-mixmirror.log)"

$PY -u -m testing.eval_rank --agent "$d" --depth 12 --out overnight/eval/rank-mixmirror.jsonl \
    > overnight/eval/rank-mixmirror.log 2>&1
say "eval_rank: $(grep -E '^  all' overnight/eval/rank-mixmirror.log | tail -n 1)  (champion v11: all 60 +39 133 239 0.27 +0.30)"

$PY -u - > overnight/eval/xval-mixmirror.log 2>&1 <<'EOF'
import sys, torch
sys.path.insert(0, ".")
from pathlib import Path
from training.train import load_checkpoint, Batches, evaluate_loss, _records
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sets = {"lichess_val": Path("data/validation_w512-150m.npy"), "sf80kf_val": Path("data/sf80kf/sf80kf_val.npy")}
for name, ck, mirror in (("v11", "training/checkpoints/net_v11-mirror.pt", True), ("v12-mixmirror", "training/checkpoints/net_v12-mixmirror.pt", True)):
    net = load_checkpoint(Path(ck), None, None, mirror=mirror).to(device)
    for label, path in sets.items():
        recs = _records(path, 300_000)
        g = torch.Generator(); g.manual_seed(1)
        print(f"{name:14s} {label:12s} {evaluate_loss(net, Batches(recs, 16384, device), g):.6f}", flush=True)
EOF
say "xval: $(tr '\n' ' ' < overnight/eval/xval-mixmirror.log)"
say "done -- 200-mixmirror built; queue its gauntlet only if eval_rank is not worse than v11"
