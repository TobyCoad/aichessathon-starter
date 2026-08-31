"""Download the 3-4-man Syzygy tablebases into the submission.

Explicitly permitted. From the event's rules.md: "Books and tablebases: permitted
as shipped data within the 200 MB cap; `chess.polyglot` and `chess.syzygy` are in
the base image." And the data itself is uncopyrightable -- from the generator's own
notice, the files "may be freely redistributed... free of copyright at least under
US law (Feist v. Rural Telephone) and under EU law (Football Dataco v. Yahoo)."

Only 3 and 4 men, which is all 70 files at about 4.1 MB. Five men would be 378 MB
of WDL alone -- nearly twice the whole size cap -- and the Chess Programming Wiki
puts 5-man at roughly +2 Elo even for Stockfish, because the 5-man tables that
actually arise are the pawn endings, which are the expensive ones.

This exists because the network cannot convert won endgames. It scores four very
different KQvK positions at +1260, +1175, +1144 and +1241, so the search has no
gradient to follow and shuffles until the referee claims a draw. Exact data
replaces a class of losses that no amount of evaluation training would fix.
"""

import argparse
import re
import urllib.request
from pathlib import Path

BASE = "https://tablebase.lichess.ovh/tables/standard/"
DIRECTORIES = ("3-4-5-wdl", "3-4-5-dtz")
MAX_MEN = 4


def men(name: str) -> int:
    """Piece count from a Syzygy filename: 'KQvKR.rtbw' -> 4."""
    return len(name.split(".")[0].replace("v", ""))


def listing(directory: str) -> list[str]:
    url = BASE + directory + "/"
    request = urllib.request.Request(url, headers={"User-Agent": "python"})
    with urllib.request.urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8", "replace")
    return re.findall(r'href="([^"]+\.rtb[wz])"', html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch 3-4-man Syzygy tablebases.")
    parser.add_argument("--out", type=Path, default=Path("syzygy"))
    parser.add_argument("--max-men", type=int, default=MAX_MEN)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)

    wanted: list[tuple[str, str]] = []
    for directory in DIRECTORIES:
        for name in listing(directory):
            if men(name) <= arguments.max_men:
                wanted.append((directory, name))

    print(f"{len(wanted)} files with at most {arguments.max_men} men")
    total = 0
    for directory, name in sorted(wanted):
        target = arguments.out / name
        if target.exists():
            total += target.stat().st_size
            continue
        url = BASE + directory + "/" + name
        request = urllib.request.Request(url, headers={"User-Agent": "python"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        target.write_bytes(payload)
        total += len(payload)
        print(f"  {name:<16} {len(payload):>10,} bytes", flush=True)

    print(f"\n{len(wanted)} files, {total:,} bytes ({total / 1e6:.2f} MB) in {arguments.out}")


if __name__ == "__main__":
    main()
