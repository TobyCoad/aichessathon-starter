#!/bin/bash
# 157-wdlnet -- the next version's net, one idea tested at full size (no pilots).
# Two changes from the v9.3 recipe, both training-only (no engine file is touched):
#   1. WDL-blended targets on the Stockfish half (lambda 0.75), so the net learns which
#      positions actually got converted. Our platform losses are all "winning but not
#      converted" and the eval's sign disagrees with the game result on 18.6% of
#      decisive positions -- that is the signal we were throwing away.
#   2. A TRUE 50/50 Stockfish:Lichess mix by POSITION COUNT. The v9.3 net alternated
#      whole shards and Lichess shards are exactly 2x the size of Stockfish shards, so
#      the "1:1" mix was really 33.3% Stockfish -- and alternating whole shards made the
#      validation oscillate by whichever distribution came last.
# Warm start: the current champion. Judged by the gauntlet, with eg_calib's per-band
# static error as the offline read (val loss is a weak instrument here).
#   bash.exe -lc "exec bash overnight/wdl_net.sh"
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
say() { echo "$(date '+%H:%M') wdlnet $*" >> "$LOG"; }

# The decoder writes the validation split last, then prints "done:" -- that line is the
# completion gate (Git bash has no pgrep).
while ! grep -q "^done:" overnight/eval/sf-decode-wdl.log 2>/dev/null; do sleep 60; done
say "decode: $(grep '^done:' overnight/eval/sf-decode-wdl.log)"

# --- merged 50/50 shards + a validation set that matches the training distribution ---
if [ ! -f data/mixw/mixw_01.npy ]; then
    mkdir -p data/mixw
    $PY -u - > overnight/eval/wdl-merge.log 2>&1 <<'EOF' || { say "MERGE FAILED"; exit 1; }
import sys; sys.path.insert(0, ".")
import numpy as np
from numpy.lib.format import open_memmap
from training.pack import RECORD
SF = ["data/sfw/feb24w_00.npy", "data/sfw/feb24w_01.npy"]
LIC = ["data/positions_w512-150m.npy", "data/positions_2025_02.npy"]
CHUNK = 4_000_000
for k, (sf_p, lic_p) in enumerate(zip(SF, LIC, strict=True)):
    sf = np.load(sf_p, mmap_mode="r")
    lic = np.load(lic_p, mmap_mode="r")
    half = min(len(sf), len(lic))          # equal counts: this is what makes it 50/50
    out = open_memmap(f"data/mixw/mixw_{k:02d}.npy", mode="w+", dtype=RECORD, shape=(2 * half,))
    for src, base in ((sf, 0), (lic, half)):
        for i in range(0, half, CHUNK):
            j = min(i + CHUNK, half)
            out[base + i : base + j] = src[i:j]
    out.flush(); del out
    print(f"mixw_{k:02d}: {half:,} Stockfish + {half:,} Lichess = {2*half:,}", flush=True)
# Validation must match what we train toward, or early stopping penalises the WDL shift.
lic = np.load("data/validation_w512-150m.npy", mmap_mode="r")
sfv = np.load("data/sfw/feb24w_val.npy", mmap_mode="r")
n = min(500_000, len(lic), len(sfv))
np.save("data/mixvalw.npy", np.concatenate([np.array(lic[:n]), np.array(sfv[:n])]))
print(f"mixvalw: {n:,} Lichess + {n:,} Stockfish-WDL", flush=True)
EOF
fi
say "merge: $(tail -n 3 overnight/eval/wdl-merge.log | tr '\n' ' ')"

# --- train ---
if [ ! -f training/checkpoints/net_w512-b8-kz16-wdl.json ]; then
    $PY -u training/train.py \
        --data data/mixw/mixw_00.npy data/mixw/mixw_01.npy \
        --val data/mixvalw.npy \
        --resume training/checkpoints/net_w512-b8-kz16-mix2.pt \
        --accumulator 512 --buckets 8 --king-zones 16 \
        --lr 1e-4 --epochs 10 --patience 4 --warmup-epochs 1 --skip-sanity \
        --out training/checkpoints/net_w512-b8-kz16-wdl.pt \
        > overnight/eval/train-wdl.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
fi
say "train done: $(grep -E 'restored|wrote.*json' overnight/eval/train-wdl.log | tail -n 2 | tr '\n' ' ')"

d=overnight/challengers/157-wdlnet
if [ ! -f overnight/eval/suite-157-wdlnet.log ]; then
    rm -rf "$d"; mkdir -p "$d/weights"
    cp agent.py fastboard.py fastsearch.py "$d/"
    cp weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
    $PY -u -m training.export --checkpoint training/checkpoints/net_w512-b8-kz16-wdl.pt \
        --out "$d/weights/net.npz" --half > overnight/eval/export-wdl.log 2>&1 \
        || { say "EXPORT FAILED"; exit 1; }
    if ! $PY -u -m training.check_nnue --agent "$d" \
        --checkpoint training/checkpoints/net_w512-b8-kz16-wdl.pt \
        > overnight/eval/check_nnue-wdl.log 2>&1; then
        say "CHECK_NNUE FAILED"; exit 1
    fi
    say "check_nnue: $(tail -n 1 overnight/eval/check_nnue-wdl.log)"
    $PY -u -m testing.endgame_suite run --agent "$d" --seconds 2.5 \
        > overnight/eval/suite-157-wdlnet.log 2>&1
fi
say "suite: $(grep -E 'mean loss' overnight/eval/suite-157-wdlnet.log | tail -n 1)"

# per-band static error -- the instrument that caught the v9.3 net halving the 5-8 band
$PY -u -m testing.eg_calib --agent "$d" > overnight/eval/v10/eg_calib_wdl.log 2>&1 || true
say "bands: $(grep -A 4 'static |eval' overnight/eval/v10/eg_calib_wdl.log | tr '\n' ' ' | cut -c1-200)"

# slope on Lichess targets (diagnostic only -- 155-mixnet2s showed correcting it buys nothing)
$PY -u - > overnight/eval/slope-wdl.log 2>&1 <<'EOF'
import sys, torch; sys.path.insert(0, ".")
from pathlib import Path
from training.train import load_checkpoint, Batches, _records, SCALE
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = load_checkpoint(Path("training/checkpoints/net_w512-b8-kz16-wdl.pt"), 8, 16).to(dev).eval()
for label, p in (("lichess", "data/validation_w512-150m.npy"), ("sf_wdl", "data/sfw/feb24w_val.npy")):
    recs = _records(Path(p), 300_000); g = torch.Generator(); g.manual_seed(1)
    num = den = 0.0
    with torch.no_grad():
        for w, b, m, s, t in Batches(recs, 16384, dev).epoch(g):
            t = t / SCALE; pr = net(w, b, m, s)
            num += float((pr * t).sum()); den += float((t * t).sum())
    print(f"slope on {label}: {num/den:.4f}", flush=True)
EOF
say "slope: $(tr '\n' ' ' < overnight/eval/slope-wdl.log)"

# --- stage OUTSIDE the challenger dir (the 150-sfnet self-play bug) and queue one task ---
mkdir -p overnight/nets
cp "$d/weights/net.npz" overnight/nets/157-wdlnet.npz
if [ "$(md5sum < overnight/nets/157-wdlnet.npz)" = "$(md5sum < weights/net.npz)" ]; then
    say "ABORT -- identical to the tree net; not queued"; exit 1
fi
$PY - <<'EOF'
import json
p = "overnight/laptop/tasks.json"; t = json.load(open(p))
if not any(x["name"] == "157-wdlnet" for x in t):
    t.append({"name": "157-wdlnet", "net": "overnight/nets/157-wdlnet.npz", "sed": "", "games": 600})
    json.dump(t, open(p, "w"), indent=1)
EOF
git add overnight/laptop/tasks.json overnight/eval/*wdl*.log overnight/eval/suite-157-wdlnet.log \
    overnight/eval/v10/eg_calib_wdl.log overnight/wdl_net.sh 2>/dev/null
git -c user.name=wdlnet -c user.email=wdl@local commit -q -m "157-wdlnet: WDL-blended targets + true 50/50 mix, trained and queued" && \
    (git pull -q --rebase --autostash origin main >/dev/null 2>&1; git push -q origin main >/dev/null 2>&1)
say "done -- 157-wdlnet queued for the laptop worker"
