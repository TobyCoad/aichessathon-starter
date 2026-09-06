import os

os.environ.setdefault("NUMBA_DEVELOPER_MODE", "1")
import agent  # noqa
import fastsearch as fs
import fastboard as fb
for name in ("search", "quiesce", "make_move", "order_node", "gen_legal"):
    d = getattr(fs, name, None) or getattr(fb, name, None)
    if d is None:
        continue
    for _sig, cres in d.overloads.items():
        md = getattr(cres, "metadata", None) or {}
        pt = md.get("pipeline_times") or {}
        inf = low = 0.0
        for _pipe, passes in pt.items():
            for pname, t in passes.items():
                tot = getattr(t, "init", 0) + getattr(t, "run", 0) + getattr(t, "finalize", 0)
                if "type_inference" in pname:
                    inf += tot
                if "lowering" in pname:
                    low += tot
        try:
            ll = cres.library.get_llvm_str()
            nlines = ll.count("\n")
        except Exception:
            nlines = -1
        print(f"{name:12s} inference {inf:6.2f}s  lowering {low:6.2f}s  llvm_lines {nlines}")
        break

# Usage: copy into a challenger dir and run with that dir as cwd, e.g.
#   cp overnight/eval/initprof.py overnight/challengers/<name>/ &&
#   cd overnight/challengers/<name> && ../../../.venv/Scripts/python.exe initprof.py
# Prints, per njit dispatcher, the numba type-inference and lowering seconds of the
# cold compile plus an IR-size proxy (lines of the function's LLVM module). Needs
# NUMBA_DEVELOPER_MODE, which it sets itself.
