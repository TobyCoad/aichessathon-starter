#!/bin/bash
# Laptop side: pull the desktop's results and print them.
cd "$(dirname "$0")/.." || exit 1
git pull --rebase origin main >/dev/null 2>&1 || echo "pull failed"
for f in overnight/desktop/results/*.txt; do [ -f "$f" ] && { echo "== $f"; cat "$f"; }; done
