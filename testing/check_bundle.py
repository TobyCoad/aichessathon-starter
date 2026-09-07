r"""Unzip the submission and cold-import it the way the platform will.

`harness/package.py` checks that files are PRESENT. Nothing checked that the result
IMPORTS -- and the engine has at least one deliberately fatal, uncaught, module-level
`raise` (agent.py refuses a mirrored net whose KING_ZONES is not 16). On the platform a
module-level raise is an import failure, which forfeits every game in the event, and the
packager would ship such a bundle without a word.

The check has to be a cold subprocess with the extraction directory as the ONLY entry on
sys.path. Importing in-process, or from the repo, proves nothing: the tree's own
fastboard.py/fastsearch.py shadow-fill anything the zip is missing, so a bundle with no
kernels imports cleanly here and plays the pure-python engine at ~4x fewer nodes there.
That exact bundle passed every local harness for weeks.

What it asserts:
  * the archive opens and every member passes a CRC check;
  * a cold import succeeds inside the init budget, with only the bundle on sys.path;
  * the compiled board really came up ("compiled board: on"), so a silent kernel
    fallback is a failure here rather than a surprise on the platform;
  * the engine answers the platform's own entry point with a legal move from the
    opening position and from a middlegame position;
  * agent.MIRRORED agrees with fastboard._F_MIRRORED and with the shipped net.npz.

Run:  .venv\Scripts\python.exe -m testing.check_bundle submission.zip
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import textwrap
import time
import zipfile
from pathlib import Path

import chess

# The measured platform budget, not harness.rules.INIT_BUDGET_S's shipped 60. Imports of
# 74.1 s and 88.1 s both played their games; only >90 s forfeited. This box is roughly
# 1.8-2.1x faster than theirs, so a cold import here should land far under it -- treat
# anything past WARN_S as a red flag even though it passes.
BUDGET_S = 90.0
WARN_S = 45.0

MIDDLEGAME = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"

PROBE = textwrap.dedent(
    """
    import json, sys, time
    from pathlib import Path
    started = time.monotonic()
    # The bundle leads sys.path, exactly as harness/runner.py arranges it, and the repo
    # root is scrubbed from whatever the interpreter started with. Never re-add it: the
    # whole point is to fail when the zip is incomplete, and the tree's own kernels
    # would shadow-fill the gap and hide it. Clearing sys.path outright is wrong the
    # other way -- it takes the stdlib with it.
    banned = sys.argv[3]
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != Path(banned).resolve()]
    sys.path.insert(0, sys.argv[1])
    import agent
    imported = time.monotonic() - started
    import chess
    report = {"import_s": imported}
    report["mirrored"] = bool(getattr(agent, "MIRRORED", False))
    report["king_zones"] = int(getattr(agent, "KING_ZONES", 0))
    report["fast_ok"] = bool(getattr(agent, "_FAST_OK", False))
    try:
        import fastboard
        report["fb_mirrored"] = bool(getattr(fastboard, "_F_MIRRORED", False))
        report["fb_file"] = fastboard.__file__
    except Exception as exc:
        report["fb_mirrored"] = None
        report["fb_error"] = f"{type(exc).__name__}: {exc}"
    moves = []
    for fen in json.loads(sys.argv[2]):
        board = chess.Board(fen)
        uci = agent.get_move(fen, 10_000)
        moves.append({"fen": fen, "uci": uci,
                      "legal": chess.Move.from_uci(uci) in board.legal_moves})
    report["moves"] = moves
    print("BUNDLE-REPORT " + json.dumps(report))
    """
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, nargs="?", default=Path("submission.zip"))
    parser.add_argument("--keep", action="store_true", help="leave the extraction on disk")
    arguments = parser.parse_args()

    if not arguments.bundle.is_file():
        raise SystemExit(f"no such bundle: {arguments.bundle}")
    failures = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"{name:<28} : {'OK  ' if ok else 'FAIL'}  {detail}")

    size = arguments.bundle.stat().st_size
    with zipfile.ZipFile(arguments.bundle) as archive:
        names = archive.namelist()
        bad = archive.testzip()
        check("archive", bad is None, f"{len(names)} members, {size:,} bytes, crc {bad or 'clean'}")
        root = Path(tempfile.mkdtemp(prefix="bundle-"))
        archive.extractall(root)

    for needed in ("agent.py", "fastboard.py", "fastsearch.py", "weights/net.npz"):
        check(f"member {needed}", (root / needed).is_file(), str(needed))

    # A fresh interpreter, no inherited sys.path, no PYTHONPATH. `-E` also drops
    # PYTHONPATH from the environment, which four of the audit scripts set.
    started = time.monotonic()
    completed = subprocess.run(
        [
            sys.executable, "-E", "-c", PROBE, str(root),
            json.dumps([chess.STARTING_FEN, MIDDLEGAME]), str(Path.cwd()),
        ],
        capture_output=True,
        text=True,
        timeout=BUDGET_S * 2,
        cwd=str(root),
    )
    elapsed = time.monotonic() - started
    output = completed.stdout + completed.stderr
    line = next((x for x in output.splitlines() if x.startswith("BUNDLE-REPORT ")), None)
    if line is None:
        check("cold import", False, f"exit {completed.returncode}; no report")
        print(textwrap.indent(output.strip()[-2000:], "    "))
        raise SystemExit(1)

    report = json.loads(line[len("BUNDLE-REPORT ") :])
    check(
        "cold import",
        report["import_s"] < BUDGET_S,
        f"{report['import_s']:.1f}s in-process, {elapsed:.1f}s wall, budget {BUDGET_S:.0f}s"
        + ("  <-- over the 45 s release gate" if report["import_s"] > WARN_S else ""),
    )
    check(
        "compiled board",
        "compiled board: on" in output,
        "on" if "compiled board: on" in output else "OFF -- the bundle is missing a kernel "
        "or a kernel failed to build; it plays legal chess at ~4x fewer nodes",
    )
    check("compiled search", bool(report["fast_ok"]), f"_FAST_OK {report['fast_ok']}")
    check(
        "mirroring agrees",
        report["fb_mirrored"] == report["mirrored"],
        f"agent {report['mirrored']}, fastboard {report['fb_mirrored']}, "
        f"KING_ZONES {report['king_zones']}",
    )
    check(
        "kernel came from the zip",
        str(root) in str(report.get("fb_file", "")),
        str(report.get("fb_file", report.get("fb_error", "?"))),
    )
    for move in report["moves"]:
        check(
            f"move from {move['fen'][:18]}",
            bool(move["legal"]),
            f"{move['uci']}" + ("" if move["legal"] else "  ILLEGAL"),
        )

    if arguments.keep:
        print(f"extraction kept at {root}")
    print("RESULT: " + ("bundle is shippable" if not failures else f"{failures} CHECK(S) FAILED"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
