"""Queue the single v9.4 gauntlet: the search bundle, plus the WDL net if it earns its place.

The human's rule is one gauntlet per version, so v9.4 is tested as one challenger. The net
rides in it only if it does not regress the per-band static error against v9.3 by more than
10%; otherwise v9.4 is the search bundle alone. Either way exactly one gauntlet is queued.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# The v9.3 net measured by testing.eg_calib (overnight/eval/v10/eg_calib_v93.log).
BASELINE = {"5- 8": 331.9, "9-12": 262.6, "13-16": 184.4}
TOLERANCE = 1.10
NET = "overnight/nets/157-wdlnet.npz"
TASKS = Path("overnight/laptop/tasks.json")
CALIB = Path("overnight/eval/v10/eg_calib_wdl.log")


def bands() -> dict[str, float]:
    try:
        log = CALIB.read_text()
    except OSError:
        return {}
    found = {}
    for band in BASELINE:
        match = re.search(rf"^\s*{re.escape(band)}:.*?net\s+([\d.]+)", log, re.M)
        if match:
            found[band] = float(match.group(1))
    return found


def main() -> None:
    sed = sys.argv[1]
    measured = bands()
    if len(measured) == len(BASELINE):
        worse = [b for b in BASELINE if measured[b] > BASELINE[b] * TOLERANCE]
        keep = not worse
        detail = ", ".join(f"{b} {measured[b]:.1f} vs {BASELINE[b]:.1f}" for b in BASELINE)
        why = detail + ("" if keep else f" -- REGRESSED at {worse}, net dropped")
    else:
        # The instrument failed, not the net. The gauntlet is the real gate.
        keep = True
        why = "eg_calib produced no bands; keeping the net and letting the gauntlet judge"

    task: dict[str, object] = {"name": "149-v94wdl", "sed": sed, "games": 600}
    clock: dict[str, object] = {"name": "v94wdl-clocktest-l", "kind": "clocktest", "sed": sed}
    if keep:
        task["net"] = clock["net"] = NET

    tasks = json.loads(TASKS.read_text())
    names = {t["name"] for t in tasks}
    # Insert at the FRONT: the worker takes the first task without a result, and anything
    # queued while the net trained (156-mixnet3, say) would otherwise take the machine for an
    # hour ahead of the release the human is waiting on.
    for entry in reversed([task, clock]):
        if entry["name"] not in names:
            tasks.insert(0, entry)
    TASKS.write_text(json.dumps(tasks, indent=1))
    print(f"v9.4 queued as 149-v94wdl, net {'IN' if keep else 'DROPPED'}: {why}")


if __name__ == "__main__":
    main()
