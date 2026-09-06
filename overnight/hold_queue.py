"""Keep the gauntlet clear for the v9.4 release.

The worker takes the first pending task and runs it to completion, so any 600-game task
that starts before 149-v94wdl is queued blocks the release for up to three hours. The
mixnet3 training chain queues itself when it finishes, which is exactly that case. This
watcher moves any such task into overnight/laptop/deferred.json until the v9.4 task exists,
then exits and leaves the queue alone.

Nothing is thrown away: deferred.json is the holding area to re-add from.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

TASKS = Path("overnight/laptop/tasks.json")
DEFERRED = Path("overnight/laptop/deferred.json")
RELEASE = "149-v94wdl"
DEADLINE = time.time() + 3 * 3600


def main() -> None:
    while time.time() < DEADLINE:
        try:
            tasks = json.loads(TASKS.read_text())
        except (OSError, json.JSONDecodeError):
            time.sleep(20)
            continue
        if any(t["name"] == RELEASE for t in tasks):
            print(f"{RELEASE} is queued; releasing the hold", flush=True)
            return
        # Only PENDING work can block the machine. A task with a result file is history
        # kept in the queue and must be left alone.
        hold = [
            t
            for t in tasks
            if t.get("games", 0) >= 200
            and not Path(f"overnight/laptop/results/{t['name']}.txt").exists()
        ]
        if hold:
            keep = [t for t in tasks if t not in hold]
            existing = json.loads(DEFERRED.read_text()) if DEFERRED.exists() else []
            names = {t["name"] for t in existing}
            existing += [t for t in hold if t["name"] not in names]
            DEFERRED.write_text(json.dumps(existing, indent=1))
            TASKS.write_text(json.dumps(keep, indent=1))
            print(f"deferred {[t['name'] for t in hold]}", flush=True)
        time.sleep(20)
    print("deadline reached; hold released", flush=True)


if __name__ == "__main__":
    main()
