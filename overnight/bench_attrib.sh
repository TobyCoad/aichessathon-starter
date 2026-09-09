#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
build() { local d=overnight/challengers/$1; shift
  rm -rf "$d"; mkdir -p "$d/weights"; cp agent.py fastboard.py fastsearch.py "$d/"
  cp weights/net.npz weights/book.bin "$d/weights/"; cp -r weights/syzygy "$d/weights/"
  for f in "$@"; do sed -i "s/^$f: Final = False\$/$f: Final = True/" "$d/agent.py"; done; }
build s1-razor RAZOR
build s1-cutnode CUTNODE
for n in s1-razor s1-cutnode s1-histv2 s1-noseeq; do
  $PY -m testing.bench --depth 8 --agent overnight/challengers/$n --json overnight/eval/bench-$n-d8.json > /dev/null 2>&1
done
echo done > overnight/eval/bench-attrib.done
