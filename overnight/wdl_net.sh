#!/bin/bash
# v9.4 = the v9.4 search bundle + the WDL net, tested as ONE gauntlet (the human's call,
# 6 Sep 08:55: "fold wdl into the 9.4 release ... only one gauntlet per version").
#
# The net's two changes from v9.3, both training-only (no engine file is touched):
#   1. WDL-blended targets on the Stockfish half (lambda 0.75): the eval mixed with how the
#      game actually ended. On decisive games the eval sign disagrees with the outcome on
#      18.6% of kept positions -- the "winning but not converted" signal our losses are made of.
#   2. A TRUE 50/50 Stockfish:Lichess mix by POSITION COUNT. v9.3 alternated whole shards and
#      Lichess shards are exactly 2x the size, so its "1:1" was really 33.3% Stockfish.
#
# QUALITY GUARD: if the net regresses any per-band static error against v9.3 by more than
# 10%, it is dropped and v9.4 is queued as the search bundle alone -- one gauntlet either way.
#   bash.exe -lc "exec bash overnight/wdl_net.sh"
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd -P)"
PY=./.venv/Scripts/python.exe
LOG=overnight/eval/night3.log
SEDV94='s/^CAPTURE_ORDER: Final = False$/CAPTURE_ORDER: Final = True/; s/^QS_TT: Final = False$/QS_TT: Final = True/; s/^ASP_WIDE: Final = False$/ASP_WIDE: Final = True/; s/^NMP_V2B: Final = False$/NMP_V2B: Final = True/'
say() { echo "$(date '+%H:%M') wdlnet $*" >> "$LOG"; }

while ! grep -q "^done:" overnight/eval/sf-decode-wdl.log 2>/dev/null; do sleep 30; done
say "decode: $(grep '^done:' overnight/eval/sf-decode-wdl.log)"

if [ ! -f data/mixw/mixw_00.npy ]; then
    mkdir -p data/mixw
    $PY -u training/merge_mix.py > overnight/eval/wdl-merge.log 2>&1 || { say "MERGE FAILED"; exit 1; }
fi
say "merge: $(tail -n 2 overnight/eval/wdl-merge.log | tr '\n' ' ')"

if [ ! -f training/checkpoints/net_w512-b8-kz16-wdl.json ]; then
    $PY -u training/train.py \
        --data data/mixw/mixw_*.npy \
        --val data/mixvalw.npy \
        --resume training/checkpoints/net_w512-b8-kz16-mix2.pt \
        --accumulator 512 --buckets 8 --king-zones 16 \
        --lr 1e-4 --epochs 12 --patience 4 --warmup-epochs 1 --skip-sanity \
        --out training/checkpoints/net_w512-b8-kz16-wdl.pt \
        > overnight/eval/train-wdl.log 2>&1 || { say "TRAIN FAILED"; exit 1; }
fi
say "train: $(grep -E 'restored|wrote.*json' overnight/eval/train-wdl.log | tail -n 2 | tr '\n' ' ')"

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

# eg_calib does `import agent`, so it must run INSIDE the challenger dir; PYTHONPATH keeps
# `testing` importable from the repo root.
( cd "$d" && PYTHONPATH="$ROOT" "$ROOT/.venv/Scripts/python.exe" -u -m testing.eg_calib ) \
    > overnight/eval/v10/eg_calib_wdl.log 2>&1 || true
say "bands: $(grep -E 'net +[0-9]' overnight/eval/v10/eg_calib_wdl.log | head -4 | tr '\n' ' ')"

mkdir -p overnight/nets
cp "$d/weights/net.npz" overnight/nets/157-wdlnet.npz
if cmp -s overnight/nets/157-wdlnet.npz weights/net.npz; then
    say "ABORT -- the WDL net is identical to the tree net"; exit 1
fi

VERDICT=$($PY training/queue_v94.py "$SEDV94")
say "$VERDICT"
git add overnight/laptop/tasks.json overnight/eval/ overnight/wdl_net.sh training/merge_mix.py training/queue_v94.py 2>/dev/null
git -c user.name=wdlnet -c user.email=wdl@local commit -q -m "157-wdlnet trained; v9.4 queued as ONE combined gauntlet (search bundle + WDL net)" && \
    (git pull -q --rebase --autostash origin main >/dev/null 2>&1; git push -q origin main >/dev/null 2>&1)
say "done -- 149-v94wdl is the single v9.4 gauntlet; the loop ships on its verdict"
