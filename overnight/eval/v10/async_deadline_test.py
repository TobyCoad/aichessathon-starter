"""Exercise INIT_ASYNC's deadline path: the branch the clocktest can never reach.

Locally the kernel compiles in ~30 s, well inside INIT_READY_S=72, so `import agent`
still blocks and the async code is a no-op. The platform is the case where it does NOT
fit. We reproduce that by importing a copy whose INIT_READY_S is 3 s, then asking for a
move: import must return early, and the first get_move must join the compile, charge
itself the wait and still return a legal move.
"""
import re, shutil, subprocess, sys, pathlib, tempfile

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(tempfile.mkdtemp(prefix="asyncdl-"))
shutil.copytree(src, dst / "e", dirs_exist_ok=True)
p = dst / "e" / "agent.py"
t = p.read_text(encoding="utf-8")
assert "INIT_ASYNC: Final = True" in t, "challenger must have INIT_ASYNC on"
t2 = re.sub(r"^INIT_READY_S: Final = .*$", "INIT_READY_S: Final = 3.0", t, flags=re.M)
assert t2 != t
p.write_text(t2, encoding="utf-8")

probe = r'''
import time, sys
t0 = time.monotonic()
import agent
t_import = time.monotonic() - t0
import chess
b = chess.Board()
t1 = time.monotonic()
mv = agent.get_move(b.fen(), 120000)
t_move = time.monotonic() - t1
assert chess.Move.from_uci(mv) in b.legal_moves, "illegal move %r" % mv
print("IMPORT_S %.1f" % t_import)
print("FIRSTMOVE_S %.1f" % t_move)
print("MOVE %s" % mv)
t2 = time.monotonic()
b.push_uci(mv)
mv2 = agent.get_move(b.fen(), 118000)
print("SECONDMOVE_S %.1f" % (time.monotonic() - t2))
print("MOVE2 %s" % mv2)
'''
r = subprocess.run([sys.executable, "-c", probe], cwd=dst / "e",
                   capture_output=True, text=True, timeout=900)
print(r.stdout)
print(r.stderr[-2000:], file=sys.stderr)
sys.exit(r.returncode)
