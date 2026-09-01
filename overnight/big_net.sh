#!/usr/bin/env bash
# Overnight: repack 150M positions without the plies the book covers, train a
# 512-wide accumulator on them, and leave a verified challenger ready to SPRT.
#
# Two changes at once, deliberately. Widening only pays if there is enough data to
# feed it -- the 1024-wide net was rejected on 16M positions, which was far too few
# -- and the ply filter is what frees that budget. Testing them separately would
# repeat the earlier mistake with the width.
#
# Stages run one at a time. Each saturates the machine, and overlapping them is
# what previously cost 46 games to timeout flags and invalidated a match.
set -u
cd "$(dirname "$0")/.." || exit 1

TARGET=${TARGET:-150000000}
MINPLY=${MINPLY:-16}
WIDTH=${WIDTH:-512}
EPOCHS=${EPOCHS:-35}
TAG="w${WIDTH}-150m"
LOG="overnight/logs/${TAG}.log"
PY=./.venv/Scripts/python.exe
mkdir -p overnight/logs
: > "$LOG"

say() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$LOG"; }

# The ladder measures the new net against outside opponents whatever the SPRT
# decided. A challenger that loses a self-play SPRT can still be the more useful
# datapoint here: self-play Elo and absolute strength are different questions, and
# roughly 60% of self-play gain is reckoned to survive against strangers.
ladder() {
    say "=== 8/8 ladder: the new net against both opponent families ==="
    $PY -u -m testing.calibrate --agent "$1" --games 50 >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then say "LADDER FAILED"; else say "=== ladder done ==="; fi
}

# Wait for anything already using the cores, so this does not contend with the
# ladder. Poll rather than assume: the ladder's own runtime varies with the rungs.
say "waiting for the machine to be free"
while [ "$(tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -c python.exe)" -gt 2 ]; do
    sleep 60
done
say "machine free, starting"

say "=== 1/8 pack ${TARGET} positions, dropping ply < ${MINPLY} ==="
$PY -u -m training.pack --target "$TARGET" --val-target 1000000 --min-ply "$MINPLY" \
    --quiet-fraction 0 \
    --out "data/positions_${TAG}.npy" --val-out "data/validation_${TAG}.npy" >> "$LOG" 2>&1
[ $? -ne 0 ] && { say "PACK FAILED"; exit 1; }

say "=== 2/8 verify the packed data decodes to legal chess ==="
$PY -u -m training.check_pack --file "data/positions_${TAG}.npy" >> "$LOG" 2>&1
[ $? -ne 0 ] && { say "CHECK_PACK FAILED -- encoding is wrong, nothing downstream is valid"; exit 1; }

say "=== 3/8 train ${WIDTH}-wide, ${EPOCHS} epochs ==="
$PY -u -m training.train --epochs "$EPOCHS" --accumulator "$WIDTH" \
    --data "data/positions_${TAG}.npy" --val "data/validation_${TAG}.npy" \
    --out "weights/net_${TAG}.pt" >> "$LOG" 2>&1
[ $? -ne 0 ] && { say "TRAIN FAILED"; exit 1; }

say "=== 4/8 export ==="
$PY -u -m training.export --checkpoint "weights/net_${TAG}.pt" \
    --out "weights/net_${TAG}.npz" >> "$LOG" 2>&1
[ $? -ne 0 ] && { say "EXPORT FAILED"; exit 1; }

say "=== 5/8 build challenger and verify numpy inference against torch ==="
C="overnight/challengers/021-${TAG}"
rm -rf "$C"; mkdir -p "$C/weights"
cp agent.py "$C/agent.py"
cp "weights/net_${TAG}.npz" "$C/weights/net.npz"
cp weights/book.bin "$C/weights/book.bin"
cp -r weights/syzygy "$C/weights/"
$PY -u -m training.check_nnue --agent "$C" --checkpoint "weights/net_${TAG}.pt" >> "$LOG" 2>&1
[ $? -ne 0 ] && { say "CHECK_NNUE FAILED -- the engine would load a net that computes something else"; exit 1; }

say "=== 6/7 SPRT against the current champion ==="
$PY -u -m testing.gauntlet --challenger "$C" >> "$LOG" 2>&1
VERDICT=$?

# 0 PROMOTE, 1 REJECT, 2 INCONCLUSIVE. Only a pass touches the submission.
if [ $VERDICT -ne 0 ]; then
    case $VERDICT in
        1) say "=== REJECT -- champion unchanged, challenger kept at ${C} ===" ;;
        2) say "=== INCONCLUSIVE -- champion unchanged, challenger kept at ${C} ===" ;;
        *) say "=== gauntlet exited ${VERDICT} -- champion unchanged ===" ;;
    esac
    ladder "$C"
    exit 0
fi

say "=== 7/7 PROMOTE -- backing the champion up before touching it ==="
# Everything promoted is already in git, but a plain copy costs nothing and does
# not depend on the working tree being clean or on anyone knowing the commit.
BACKUP="overnight/champion_backup_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BACKUP/weights"
cp agent.py "$BACKUP/agent.py"
cp weights/net.npz "$BACKUP/weights/net.npz"
cp weights/book.bin "$BACKUP/weights/book.bin"
cp -r weights/syzygy "$BACKUP/weights/"
say "backup written to ${BACKUP}"

cp "$C/agent.py" agent.py
cp "$C/weights/net.npz" weights/net.npz
say "promoted ${C} -- champion is now the ${WIDTH}-wide net on ${TARGET} positions"

$PY -m ruff check . >> "$LOG" 2>&1 && $PY -m mypy >> "$LOG" 2>&1 && say "gate green after promotion"

ladder "."
say "=== done ==="
