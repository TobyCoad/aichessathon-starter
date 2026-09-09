#!/bin/bash
# Build one challenger dir per switch set (tree engine + v12 net, flag flipped by sed,
# flip VERIFIED) and bench each at depth 8. Node counts at fixed depth are exact.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
OUT=overnight/eval/bench-s1s2.log
: > "$OUT"
build() {  # name flag...
  local d=overnight/challengers/$1; shift
  rm -rf "$d"; mkdir -p "$d/weights"
  cp agent.py fastboard.py fastsearch.py "$d/"
  cp weights/net.npz weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
  for f in "$@"; do
    sed -i "s/^$f: Final = False\$/$f: Final = True/" "$d/agent.py"
    grep -q "^$f: Final = True\$" "$d/agent.py" || { echo "FLIP FAILED $f in $d" >> "$OUT"; exit 1; }
  done
}
build s2-probcut   PROBCUT
build s2-hindsight HINDSIGHT
build s2-futlmr    FUTILITY_LMR
build s2-all       PROBCUT HINDSIGHT FUTILITY_LMR
build s1-all       SEE_QUIET RAZOR HISTORY_V2 CUTNODE
echo "baseline tree: $(grep 'fixed depth' overnight/eval/bench-tree-d8.log)" >> "$OUT"
for n in s2-probcut s2-hindsight s2-futlmr s2-all s1-all; do
  r=$($PY -m testing.bench --depth 8 --agent overnight/challengers/$n 2>&1 | grep -E "fixed depth|Error|error" | tail -1)
  echo "$n: $r" >> "$OUT"
done
echo "done" >> "$OUT"
