# Draft email to the organisers

To: hello@aichessathon.com
Subject: Rules clarification: compiled dependencies from PyPI, and base image versions

---

Hi,

Three questions on the agent contract, all about what is permitted inside the
submission rather than about training. Happy for any of these to be answered with a
one-line yes or no.

**1. Is a compiled *move generator* from PyPI permitted?**

The docs say native binaries in the zip are rejected but that compiled dependencies
come from PyPI via `requirements.txt`. I would like to list `rust-chess`, which is a
Rust legal-move generator with Python bindings — board representation and move
generation only, with no search, no evaluation and no opening book. Functionally it
is the same category of library as the `python-chess` already in the base image,
just faster.

I am asking because the prohibition names "wrappers around any existing engine", and
I would rather be told no now than be disqualified retroactively. To be explicit:
all search and evaluation would be my own code, and move selection would be driven
by a network I trained myself.

**2. Is `numba` permitted?**

Same question in a different form. `numba` is a PyPI package with Linux wheels that
JIT-compiles Python at import time. Nothing precompiled ships in the zip — the
source is plain Python that a judge can read, and compilation happens inside the 60
second init budget. Is that within the rules?

**3. What are the pinned versions in the base image?**

The docs say torch, numpy, python-chess and onnxruntime are pinned but do not give
versions. The specific thing I need to know is the `python-chess` version, because
`Board._transposition_key()` — which your own `docs/IDEAS.md` recommends as a
transposition table key — is a private API that could change between releases. If
you would rather not publish the whole list, just the python-chess version would
help.

Thanks, and thanks for putting the event on.

Toby Coad

---

## Why each of these matters, for our own reference — do not send this part

1. **`rust-chess`** measured 16x faster move generation and **2.2x end-to-end**, which
   is a full extra ply. At the depth this engine searches, a ply is worth roughly
   150 Elo — comparable to the entire neural network, for one line in
   `requirements.txt`. It cross-validated against python-chess on 47,218 positions
   with zero disagreements and passes the standard perft suite exactly.
2. **`numba`** is the fallback if the answer to (1) is no: a JIT'd move generator
   benchmarks ~40x on the generator alone and compiles in 2-4 seconds, comfortably
   inside the 60 second budget.
3. **Version pinning** de-risks the transposition table, which is the second largest
   search feature in the engine after move ordering.

If both (1) and (2) come back no, nothing is lost: the engine is being built on
python-chess regardless, and the ~100k nps ceiling is simply real.
