"""Does the mirrored/king-zone guard in agent.py fire, and only when it should?

The guard is `MIRRORED and KING_ZONES != 16`. It exists because fastboard.zone_of
hardcodes the folded 16-zone map when mirrored and adds `zones` for the reflected
half, so it returns block indices up to 31 whatever KING_ZONES says. With 8 zones
W1 holds 16 blocks and refresh() reads ~12 MB past the end of it -- and numba
compiles with bounds checking off, so that is a SIGSEGV, not an exception. A kz8
mirrored net otherwise imports clean, prints "compiled board: on" and scores
nonsense, so the guard is the only thing standing between a mis-export and a
process death on the platform.

Three cases, all in subprocesses because two of them are meant to die:
  unmirrored kz16 -> imports, runner announces ready   (the shipping build)
  mirrored   kz16 -> imports, runner announces ready   (the build we are training)
  mirrored   kz8  -> import raises, runner never announces ready
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FEATURES = 768


def build(dest: Path, mirrored: bool, zones: int) -> None:
    """A scratch engine directory whose net.npz carries the flag and zone count asked for."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("agent.py", "fastboard.py", "fastsearch.py"):
        shutil.copy2(ROOT / name, dest / name)
    (dest / "weights").mkdir(exist_ok=True)
    src = np.load(ROOT / "weights" / "net.npz")
    payload = {k: src[k] for k in src.files}
    # KING_ZONES is derived from W1's row count, so the zone count is set by truncating it.
    payload["W1"] = payload["W1"][: FEATURES * zones]
    if mirrored:
        payload["mirrored"] = np.asarray(1, dtype=np.uint8)
    np.savez(dest / "weights" / "net.npz", **payload)


def try_import(directory: Path) -> tuple[int, str]:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import agent; "
        "print('IMPORT-OK', agent.MIRRORED, agent.KING_ZONES)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(directory)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def try_runner(directory: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "harness.runner", str(directory)],
        input="", capture_output=True, text=True, cwd=str(ROOT),
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()[-400:]


def expect_live(label: str, directory: Path) -> bool:
    ok = True
    rc, out = try_import(directory)
    print(f"[{label}] import rc={rc} -> {out.splitlines()[-1] if out else ''}")
    if rc != 0 or "IMPORT-OK" not in out:
        print("  FAIL: the guard breaks a net it should accept")
        ok = False
    rc, stdout, _ = try_runner(directory)
    ready = bool(stdout) and json.loads(stdout.splitlines()[0]).get("ready")
    print(f"[{label}] harness.runner rc={rc} ready={ready}")
    if not ready:
        print("  FAIL: runner no longer starts")
        ok = False
    return ok


def main() -> int:
    scratch = ROOT / "testing" / "_mirrorguard"
    if scratch.exists():
        shutil.rmtree(scratch)
    ok = True

    plain = scratch / "plain"
    build(plain, mirrored=False, zones=16)
    ok &= expect_live("unmirrored kz16", plain)

    mir16 = scratch / "mirror16"
    build(mir16, mirrored=True, zones=16)
    ok &= expect_live("mirrored kz16 ", mir16)

    # The one that must die. Note what it would do otherwise: import clean, then over-read.
    mir8 = scratch / "mirror8"
    build(mir8, mirrored=True, zones=8)
    rc, out = try_import(mir8)
    tail = out.splitlines()[-1] if out else ""
    print(f"[mirrored kz8  ] import rc={rc} -> {tail}")
    if rc == 0 or "mirrored nets require KING_ZONES == 16" not in out:
        print("  FAIL: guard did not fire on a mirrored net with the wrong zone count")
        ok = False

    rc, stdout, _stderr = try_runner(mir8)
    print(f"[mirrored kz8  ] harness.runner rc={rc} stdout={stdout!r}")
    if "ready" in stdout:
        print("  FAIL: runner announced ready with a mirrored kz8 net")
        ok = False

    shutil.rmtree(scratch, ignore_errors=True)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
