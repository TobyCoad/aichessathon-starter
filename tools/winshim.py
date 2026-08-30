"""Windows transport shim for harness.sandbox.

harness/sandbox.py reads the agent's pipes with `selectors`, which on Windows only
accepts sockets: every game dies with WinError 10038. This patches only the pipe
reads, on a background thread, and leaves harness/ untouched. The clock still lives
in harness/referee.py and is not affected.

Usage:  python -m tools.winshim play  --white . --black baselines/greedy
        python -m tools.winshim arena --opponent baselines/greedy --games 20
"""

import subprocess
import sys
import threading
import time
from queue import Empty, Queue
from typing import IO, Any

from harness import sandbox
from harness.rules import STDOUT_CAP


def _pump(stream: IO[bytes], queue: "Queue[tuple[str, bytes]]", tag: str) -> None:
    try:
        while True:
            chunk = stream.read(1)
            if not chunk:
                break
            queue.put((tag, chunk))
    except (OSError, ValueError):
        pass
    queue.put((tag, b""))


def _start(self: Any, init_budget_s: float) -> None:
    process = subprocess.Popen(
        self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=0,
    )
    self._process = process
    self._queue = Queue()
    for tag, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        threading.Thread(target=_pump, args=(stream, self._queue, tag), daemon=True).start()
    ready = self._await_line(time.monotonic() + init_budget_s)
    if ready is None:
        raise sandbox.AgentFailure("init" if process.poll() is None else "crash")
    if not sandbox._is_ready(ready):
        raise sandbox.AgentFailure("init")


def _await_line(self: Any, deadline: float) -> bytes | None:
    while b"\n" not in self._buffer:
        if len(self._buffer) >= STDOUT_CAP:
            raise sandbox.AgentFailure("illegal")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            tag, chunk = self._queue.get(timeout=remaining)
        except Empty:
            return None
        if tag == "stderr":
            self._tail += chunk
        elif not chunk:
            raise sandbox.AgentFailure("crash")
        else:
            self._buffer += chunk
    line, _, self._buffer = self._buffer.partition(b"\n")
    return bytes(line)


def _stop(self: Any) -> None:
    if self._process is None:
        return
    self._process.kill()
    self.stderr_tail = self._tail.decode("utf-8", "replace")
    for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
        if stream is not None:
            stream.close()
    self._process.wait()
    self._process = None


def install() -> None:
    sandbox.Agent.start = _start  # type: ignore[method-assign]
    sandbox.Agent._await_line = _await_line  # type: ignore[method-assign]
    sandbox.Agent.stop = _stop  # type: ignore[method-assign]


if __name__ == "__main__":
    install()
    target = sys.argv.pop(1)
    sys.argv[0] = f"harness.{target}"
    if target == "play":
        from harness.play import main
    elif target == "arena":
        from harness.arena import main
    else:
        raise SystemExit("usage: python -m tools.winshim {play|arena} [args...]")
    main()
