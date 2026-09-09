#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
OUT=overnight/eval/bench-s1solo.log
: > "$OUT"
build() { local d=overnight/challengers/$1; shift
  rm -rf "$d"; mkdir -p "$d/weights"; cp agent.py fastboard.py fastsearch.py "$d/"
  cp weights/net.npz weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
  for f in "$@"; do sed -i "s/^$f: Final = False\$/$f: Final = True/" "$d/agent.py"
    grep -q "^$f: Final = True\$" "$d/agent.py" || { echo "FLIP FAILED $f" >> "$OUT"; exit 1; }; done; }
build s1-seequiet SEE_QUIET
build s1-histv2   HISTORY_V2
build s1-noseeq   RAZOR HISTORY_V2 CUTNODE
for n in s1-seequiet s1-histv2 s1-noseeq; do
  echo "$n: $($PY -m testing.bench --depth 8 --agent overnight/challengers/$n 2>&1 | grep -E 'fixed depth|Error' | tail -1)" >> "$OUT"
done
echo done >> "$OUT"
