import argparse
import zipfile
from collections.abc import Iterator
from pathlib import Path

DEFAULT_INCLUDES = ("agent.py", "fastboard.py", "fastsearch.py", "requirements.txt", "weights")
# Absent from the zip, each of these degrades silently rather than failing: agent.py falls
# back to the pure-python engine, prints "compiled board: off" and plays legal chess at ~4x
# fewer nodes. No local harness can see it either -- every local run has the repo root on
# sys.path, so the tree's copy shadow-fills the gap and the bundle passes. Only the platform,
# where the zip is the whole sys.path, plays the crippled build. So: fail here instead.
REQUIRED = ("agent.py", "fastboard.py", "fastsearch.py", "weights/net.npz")
SKIP = {"__pycache__", ".DS_Store"}


def members(root: Path, includes: tuple[str, ...]) -> Iterator[tuple[Path, str]]:
    for name in includes:
        source = root / name
        if source.is_file():
            yield source, name
        elif source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_file() and not SKIP & set(path.parts):
                    yield path, str(path.relative_to(root))


def build(root: Path, destination: Path, includes: tuple[str, ...]) -> list[str]:
    """Write the bundle, or leave whatever was already there untouched.

    Built to a temporary file and moved into place only once it is known to be complete.
    Opening `destination` directly with mode "w" truncates it the instant the call
    starts, so a build that fails partway -- an unreadable file, a full disk, a Ctrl-C --
    used to destroy the last good submission AND leave behind a zip that is structurally
    valid (zipfile still writes a central directory on close) but missing the kernels.
    That bundle plays legal chess at a quarter of the nodes, so nothing downstream
    notices. Verified: an exception on the third member left a readable archive holding
    only agent.py and fastboard.py, and the completeness check below never ran.
    """
    temporary = destination.with_name(destination.name + ".tmp")
    written: list[str] = []
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for source, name in members(root, includes):
                archive.write(source, name)
                written.append(name)
        # `written` carries os-native separators on Windows; REQUIRED is spelled with "/".
        have = {name.replace("\\", "/") for name in written}
        missing = [name for name in REQUIRED if name not in have]
        if missing:
            raise SystemExit(
                "refusing to ship a bundle missing "
                + ", ".join(missing)
                + f" (looked under {root}); {destination} is unchanged"
            )
        # `with`, because the replace() below is a rename and Windows refuses a rename
        # whose source still has an open handle. Refcounting happens to close it in time
        # on CPython; that is not something to ship a submission on.
        with zipfile.ZipFile(temporary) as check:
            bad = check.testzip()
        if bad is not None:
            raise SystemExit(f"refusing to ship a bundle with a corrupt member: {bad}")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a submission zip.")
    parser.add_argument("--out", type=Path, default=Path("submission.zip"))
    parser.add_argument("--include", action="append", default=[])
    arguments = parser.parse_args()

    includes = DEFAULT_INCLUDES + tuple(arguments.include)
    written = build(Path.cwd(), arguments.out, includes)
    size = arguments.out.stat().st_size
    print(f"{arguments.out} ({size:,} bytes)")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
