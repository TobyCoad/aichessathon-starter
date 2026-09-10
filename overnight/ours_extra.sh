#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
LOG=overnight/eval/night4.log
while ! grep -q "conv chain done" "$LOG"; do sleep 60; done
END=$(date -d "2026-09-10 03:14" +%s); NOW=$(date +%s); H=$(python -c "print(max(0.2, ($END - $NOW) / 3600))")
echo "$(date '+%H:%M') OURS extra: match finished, 4 more gen workers (files 11-14) for $H h" >> "$LOG"
./.venv/Scripts/python.exe -u -m training.gen_ours --workers 4 --first-index 11 --hours "$H" --seed 911 --out data/ours > overnight/eval/gen-ours-extra.log 2>&1
echo "$(date '+%H:%M') OURS extra: $(grep -E '^done:' overnight/eval/gen-ours-extra.log)" >> "$LOG"
