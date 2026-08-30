"""Download Lichess engine-evaluation data.

Source: https://huggingface.co/datasets/Lichess/fishnet-evals -- CC0-1.0, Parquet,
schema `fen, cp, mate, move`. Roughly 7 GB per monthly file.

This dataset rather than `lichess_db_eval.jsonl.zst` for two reasons: it is already
flat and columnar, so there is no PGN parsing anywhere in the pipeline (a full
monthly PGN dump is 29 GB and takes about 36 hours to parse single-threaded); and it
carries a human policy target (`move`) alongside the engine value target.

Training on engine-annotated positions is explicitly permitted. From the event's
rules.md: "Training data: unrestricted, including positions annotated by an existing
engine; the ban covers only what ships and runs inside the submission."

Downloads resume. A 7 GB transfer that has to restart from zero because a laptop
slept is exactly the kind of thing that eats an unattended night.
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/Lichess/fishnet-evals/resolve/main/"
DEFAULT_MONTH = "2025_01"
CHUNK = 1 << 20


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "python"})
    with urllib.request.urlopen(request, timeout=60) as response:
        # HuggingFace serves an LFS redirect; x-linked-size is the real file size.
        size = response.headers.get("x-linked-size") or response.headers.get("Content-Length")
    return int(size or 0)


def download(month: str, destination: Path) -> Path:
    name = f"standard_rated_{month}.parquet"
    url = BASE + name
    target = destination / name
    destination.mkdir(parents=True, exist_ok=True)

    total = remote_size(url)
    have = target.stat().st_size if target.exists() else 0
    if total and have == total:
        print(f"{name}: already complete ({total / 1e9:.2f} GB)")
        return target
    if have > total > 0:
        print(f"{name}: local file larger than remote, restarting")
        have = 0

    headers = {"User-Agent": "python"}
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"{name}: resuming at {have / 1e9:.2f} / {total / 1e9:.2f} GB")
    else:
        print(f"{name}: downloading {total / 1e9:.2f} GB")

    request = urllib.request.Request(url, headers=headers)
    started = time.monotonic()
    written = have
    last_report = 0.0
    with urllib.request.urlopen(request, timeout=120) as response:
        mode = "ab" if have else "wb"
        with target.open(mode) as handle:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                now = time.monotonic()
                if now - last_report > 30.0:
                    last_report = now
                    rate = (written - have) / max(now - started, 1e-9) / 1e6
                    percent = 100.0 * written / total if total else 0.0
                    print(
                        f"  {written / 1e9:6.2f} / {total / 1e9:.2f} GB "
                        f"({percent:5.1f}%)  {rate:5.1f} MB/s",
                        flush=True,
                    )

    print(f"{name}: done, {written / 1e9:.2f} GB in {(time.monotonic() - started) / 60:.1f} min")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Lichess fishnet-evals Parquet.")
    parser.add_argument("--month", default=DEFAULT_MONTH, help="e.g. 2025_01")
    parser.add_argument("--out", type=Path, default=Path("data"))
    arguments = parser.parse_args()

    try:
        download(arguments.month, arguments.out)
    except KeyboardInterrupt:
        print("\ninterrupted; rerun to resume", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
