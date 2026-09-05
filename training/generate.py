"""Engine-distribution training data: self-play positions labelled by Stockfish.

The net is trained on human Lichess positions, but the positions our engine
reaches -- and loses in -- are engine-vs-engine positions, especially long
endings. This plays fast self-play games with the current engine (a few tens of
milliseconds a move, random openings for diversity), keeps the positions, and
labels each with Stockfish at a fixed node count, the recipe every strong NNUE
engine uses. The output is a Parquet file with the fishnet columns (fen, cp,
mate, move), so training/pack.py packs it like any other month.

  .venv\\Scripts\\python.exe -m training.generate --games 2000 --movetime-ms 40 --nodes 5000 \\
      --workers 14 --out data/selfplay/batch-001.parquet

Stockfish is fetched into engines/stockfish/ if it is missing (the official
Windows AVX2 build). One worker = one engine process + one Stockfish process.
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import chess
import chess.engine
import chess.pgn

STOCKFISH_DIR = Path("engines/stockfish")
STOCKFISH_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/"
    "sf_17.1/stockfish-windows-x86-64-avx2.zip"
)
MATE_CP = 2000
SKIP_PLIES = 8  # opening moves are book territory and over-represented


def stockfish_path() -> Path:
    found = sorted(STOCKFISH_DIR.glob("stockfish*.exe")) + sorted(STOCKFISH_DIR.glob("stockfish*"))
    candidates = [c for c in found if c.is_file() and c.suffix in (".exe", "")]
    if candidates:
        return candidates[0]
    STOCKFISH_DIR.mkdir(parents=True, exist_ok=True)
    print("fetching Stockfish ...", flush=True)
    data = urllib.request.urlopen(STOCKFISH_URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.endswith(".exe"):
                target = STOCKFISH_DIR / Path(name).name
                target.write_bytes(z.read(name))
                return target
    raise SystemExit("no executable in the Stockfish archive")


def openings() -> list[str]:
    fens: list[str] = []
    platform = Path("testing/platform_openings.txt")
    if platform.exists():
        fens += [x.strip() for x in platform.read_text(encoding="utf-8").splitlines() if x.strip()]
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from testing import openings as book_openings

        fens += list(book_openings.opening_fens())
    except Exception:
        pass
    return fens or [chess.STARTING_FEN]


def worker(job: tuple[int, int, int, int, str, int]) -> str:
    """Play `games` games, label their positions, write one Parquet part."""
    index, games, movetime_ms, nodes, out, seed = job
    import pyarrow as pa
    import pyarrow.parquet as pq

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import agent
    import fastboard

    rng = random.Random(seed)
    starts = openings()
    sf = chess.engine.SimpleEngine.popen_uci(str(stockfish_path().resolve()))
    sf.configure({"Threads": 1, "Hash": 32})
    engine = agent.FastEngine()
    rows: dict[str, list[object]] = {"fen": [], "cp": [], "mate": [], "move": []}
    played = 0
    started = time.time()
    while played < games:
        board = chess.Board(rng.choice(starts))
        # two random plies for diversity, then the engine on both sides
        for _ in range(2):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        positions: list[str] = []
        ply = 0
        while not board.is_game_over(claim_draw=True) and ply < 240:
            if ply >= SKIP_PLIES and not board.is_check():
                positions.append(board.fen())
            engine.prepare(board, 0)
            now = time.monotonic()
            engine.deadline = now + movetime_ms / 1000.0 * 1.5
            try:
                move = engine.choose(now + movetime_ms / 1000.0, now + movetime_ms / 1000.0 * 1.5)
                mv = chess.Move.from_uci(fastboard.move_to_uci(move))
                if mv not in board.legal_moves:
                    mv = rng.choice(list(board.legal_moves))
            except Exception:
                mv = rng.choice(list(board.legal_moves))
            board.push(mv)
            ply += 1
        for fen in positions:
            b = chess.Board(fen)
            info = sf.analyse(b, chess.engine.Limit(nodes=nodes))
            score = info["score"].white()
            best = info["pv"][0].uci() if info.get("pv") else ""
            rows["fen"].append(fen)
            rows["cp"].append(None if score.is_mate() else int(score.score() or 0))
            rows["mate"].append(int(score.mate() or 0) if score.is_mate() else None)
            rows["move"].append(best)
        played += 1
        if played % 25 == 0:
            rate = len(rows["fen"]) / max(time.time() - started, 1)
            print(
                f"  worker {index}: {played}/{games} games, {len(rows['fen'])} positions, "
                f"{rate:.0f} pos/s",
                flush=True,
            )
    sf.quit()
    part = Path(out).with_suffix(f".part{index}.parquet")
    table = pa.table({
        "fen": pa.array(rows["fen"], pa.string()),
        "cp": pa.array(rows["cp"], pa.int32()),
        "mate": pa.array(rows["mate"], pa.int32()),
        "move": pa.array(rows["move"], pa.string()),
    })
    pq.write_table(table, part, compression="zstd")
    return str(part)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play positions labelled by Stockfish.")
    parser.add_argument("--games", type=int, default=1000, help="total games")
    parser.add_argument("--movetime-ms", type=int, default=40)
    parser.add_argument("--nodes", type=int, default=5000, help="Stockfish nodes per label")
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) - 2))
    parser.add_argument("--out", type=Path, default=Path("data/selfplay/batch.parquet"))
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    stockfish_path()
    per = -(-arguments.games // arguments.workers)
    out = str(arguments.out)
    jobs = [
        (i, per, arguments.movetime_ms, arguments.nodes, out, arguments.seed * 1000 + i)
        for i in range(arguments.workers)
    ]
    started = time.time()
    with mp.Pool(arguments.workers) as pool:
        parts = pool.map(worker, jobs)
    import pyarrow.parquet as pq

    tables = [pq.read_table(p) for p in parts]
    import pyarrow as pa

    table = pa.concat_tables(tables)
    # small row groups: training/pack.py holds out whole row groups for validation
    pq.write_table(table, arguments.out, compression="zstd", row_group_size=50_000)
    for p in parts:
        Path(p).unlink(missing_ok=True)
    print(
        f"wrote {arguments.out}: {table.num_rows:,} positions from {arguments.games} games "
        f"in {(time.time() - started) / 60:.1f} min ({arguments.out.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
