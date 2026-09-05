# v10 speed review: init time and node rate

All numbers measured 2026-09-05 on this laptop (Windows, `.venv` python 3.12.10, numba 0.67.0,
CPU `arrowlake`) with a gauntlet and a data pack running, so timings are noisy; node counts are
exact. Bench = `python -m testing.bench --depth 8` (40 positions). Scratch variants live in
`%TEMP%/claude/.../scratchpad/speed/`. No tracked file was modified. Note: `agent.py` in the
working tree was rewritten by the continuous pipeline at 17:31 during this session, so all
node-rate variants below were cut from one frozen snapshot (`speed/cpuv3/`, md5 `57b579e7…`,
bench = 1,491,095 nodes) and are directly comparable to each other.

## Summary (10 lines)

1. Init is **31.7 s** on this box in the current snapshot: `fs.warm_up` 23.3 s, `fb.warm_up` 7.1 s,
   agent's five eval kernels 1.1 s, weights 0.3 s. One function, `fastsearch.search`, is 73% of it.
2. `cache=True` **segfaults** when numba rebuilds `fastsearch.search` from `.nbc`, on every path I
   tried (including with `objmode` removed). Caching everything *except* `search`/`quiesce` works
   and gives a warm init of **17.8 s**, verified at a completely different absolute path (18.5 s).
3. The numba cache index key is `(signature, (llvm-triple, host-CPU-name, feature-string),
   sha256(bytecode))`; the path is **not** in the key, but the freshness stamp is the source file's
   exact `(st_mtime, st_size)`, which a zip extraction destroys unless `agent.py` re-`utime`s it.
4. Shipping a cache also needs `NUMBA_CPU_NAME=x86-64-v3` (else the key contains `arrowlake`), a
   Linux build host (the triple is in the key; the platform is Linux), an exact numba-version match,
   and a `/tmp` copy because the agent directory is read-only. Portable codegen costs **-8% knps**.
5. **Rule risk, flagged not decided:** AGENTS.md says "Native binaries in the zip are rejected. Ship
   source." `.nbc` files are serialised machine code, and `numba.pycc` AOT produces an actual
   `.so`. Both look like they violate that line. Option (d) is out; option (a) needs a ruling.
6. The cheapest *source-only* init win is **constant-folding the settled `ctrl` switches**:
   measured `fs.warm_up` 23.28 s -> 18.99 s (-4.3 s, -18%), total init 31.7 -> 27.4 s.
7. Second source-only win: `fastboard` compiles **61 redundant specialisations** (`attackers_to`,
   `is_attacked`, `rook_attacks`, `bishop_attacks` nine times each) because every argument is `Any`.
8. Node rate: `evaluate` is **~1.7 us of the 3.1 us node** (measured by a node-identical double-eval
   build), the 32x1024 float32 head alone is 866 ns. Everything else is small.
9. **int16/int8 quantisation is a loss in numba**, decisively: int16 head 1675 ns and int8 1662 ns
   against float32's 866 ns. LLVM emits no `VPMADDWD`; float32+FMA+fastmath is already ~62 GFLOP/s.
   A naive sparse head is worse still (2534 ns) despite 62.7% of clipped-ReLU outputs being zero.
10. Free and exact: `QS_EVAL_CACHE` is currently `False`; turning it on measured **+4.2% knps** with
    an identical node count. The objmode clock poll, SEE's `np.zeros(32)` and `boundscheck` are all
    at or below the noise floor and are not worth touching.

## Measured import breakdown

Instrumented copy (`speed/timed/`), older snapshot, heavier load, total **41.0 s**:

| stage | s | note |
|---|---:|---|
| weights load + python defs | 0.25 | `np.load`, `W2T` transpose |
| `agent` eager njit kernels | 0.80 | `_eval_kernel` .31, `_eval_bucket_kernel` .22, `_push` .15, `_refresh` .08, `_pop` .06 |
| agent kernel warm calls | 0.01 | eager signatures, so already compiled |
| `import fastboard` | 0.01 | |
| **`fb.warm_up()`** | **9.08** | `gen_legal` 5.00, `make_full` 1.99, `refresh` 0.76, `unmake_full` 0.33, `in_check` 0.22, `order_moves` 0.20, `make/unmake_null` 0.16, `score_moves` 0.13, `pick_move` 0.08, rest 0.21 |
| `import fastsearch` | 0.03 | |
| **`fs.warm_up()`** | **30.80** | one `search(depth=2)` call = the whole `search` + `quiesce` compile; every sub-step around it is 0.00 |

Frozen snapshot, lighter load, same call sequence timed directly:

| stage | s |
|---|---:|
| `fb.warm_up()` | 7.05 |
| `fs.warm_up()` | 23.28 |
| agent kernels + weights | ~1.35 |
| **total** | **~31.7** |

Load inflates this by ~1.3x (hence the 40.8 s figure quoted for the v8.5 bundle under 8 workers).
The platform box is ~1.8x slower: **~57 s projected against a 60 s budget**, or ~74 s if the
platform box is also contended. Init is a live risk and `search` is the whole of it.

### Decorators and flags in use

* `fastboard.py`: 35 x `@njit(cache=False)`, no signatures, no `fastmath`, no `nogil`.
* `fastsearch.py`: 18 x `@njit(cache=False)`; `evaluate` adds `fastmath=True`; `quiesce` and
  `search` add `nogil=True`; `timed_out` uses `objmode` for `time.monotonic()`.
* `agent.py`: 5 kernels with **eager explicit signatures** + `cache=False, fastmath=True` (so they
  compile at decoration, i.e. at import, which is why their warm calls cost 0.006 s).
* `NUMBA_BOUNDSCHECK` is unset -> `config.BOUNDSCHECK is None` -> bounds checks already off.
* `error_model` is nowhere set, so it is `'python'`. Every division in the hot path is by a literal
  (`// 6`, `// 32`, `(depth - 1) // 2`), so LLVM folds the zero-check away. Nothing to win.
* Compilation is triggered by `agent.py`'s module-level `try:` block: eager kernels at decoration,
  then `_fb.warm_up()` (calls every fastboard kernel once on a real position), then
  `_fs.warm_up(...)` (one `search()` at depth 2). Any exception sets `_FAST_OK = False` and the
  engine silently falls back to the python-chess `Engine` -- see the caching warning below.

## Option (a): a prebuilt numba cache in the zip

### How numba's cache actually keys entries (`.venv/Lib/site-packages/numba/core/caching.py`)

* `Cache._index_key` (line 780) returns
  `(sig, codegen.magic_tuple(), (sha256(py_func.__code__.co_code), sha256(closure)))`.
  **The source path is not in the key.**
* `CPUCodegen.magic_tuple()` returns `(llvm_module.triple, _get_host_cpu_name(), _tm_features)`.
  Measured here: `('x86_64-pc-windows-msvc', 'arrowlake', '+64bit,+adx,+aes,...')` -- a ~1 KB
  feature string. This is the portability killer.
* Freshness is separate from the key: `IndexDataCacheFile._load_index` compares a stored
  `source_stamp` against `_SourceFileBackedLocatorMixin.get_source_stamp()` =
  `(os.stat(py_file).st_mtime, st_size)`, as an exact float compare. Verified: the shipped `.nbi`
  held `(1788625664.6823893, 40293)` and matched the on-disk stat exactly.
* Locator order (`CacheImpl._locator_classes`): `UserProvidedCacheLocator` (only if
  `NUMBA_CACHE_DIR` is set; subpath = `basename(dir) + '_' + sha1(abspath(dir))` -> **path
  dependent**), then `InTreeCacheLocator` (`<dir of source>/__pycache__` -> **path independent**),
  then `UserWideCacheLocator` (appdirs, again `sha1(abspath(dir))`), IPython, Zip.
  `numba 0.67` also adds `InTreeCacheLocatorFsAgnostic` (floors the mtime) but it is **not** in the
  default list; `NUMBA_CACHE_LOCATOR_CLASSES` can select it.
* Filename base = `<module>.<qualname>-<co_firstlineno>.py312`, so **moving a function by one line
  invalidates its cache file name**.

### What I measured

| build | cold import | warm import | bench nodes |
|---|---:|---:|---:|
| current (`cache=False`) | 41.0 / 31.7 s | n/a | 1,491,095 |
| all `cache=True` | 48.2 s | **SEGFAULT** | n/a |
| all `cache=True`, `objmode` removed from `timed_out` | 44.1 s | **SEGFAULT** | n/a |
| `cache=True` except `search`/`quiesce` | 37.1 s | **17.79 / 17.97 s** | 1,491,095 |
| same, copied to a different absolute path (`cp -rp`) | - | **18.53 s** | - |

* The segfault is reproducible and lands **after** `[cache] data loaded from
  fastsearch.search-483.py312.1.nbc`, i.e. while numba rebuilds that `CompileResult`. A minimal
  repro (recursive + `nogil` + cached, calling a cached helper) works fine, so it is specific to
  this function -- most likely its size or its mutual recursion with the equally large `quiesce`.
  numba emitted **no** "cannot cache" warning; it happily wrote a `.nbc` that kills the process.
* Even the working configuration failed once: loading `agent.py`'s cached eval kernels under
  `testing.bench`'s `importlib.module_from_spec('bench_agent', ...)` raised inside the `try:`,
  giving `_COMPILED = False`, `_FAST_OK = False`, "compiled board: off". **That is the worst
  possible failure mode on the platform**: no crash, no log, just an engine playing at
  python-chess speed for the whole game.
* Portable codegen: with `NUMBA_CPU_NAME=x86-64-v3` and `NUMBA_CPU_FEATURES=""` the magic tuple
  becomes `('x86_64-pc-windows-msvc', 'x86-64-v3', '')` -- deterministic. Bench: **5.0 s vs 4.6 s
  = -8% knps**, node count identical (1,491,095), so results stay bit-identical.

### What shipping one would require

1. Build on **Linux x86_64** (AGENTS.md: "only packages with a Linux wheel on PyPI"), python 3.12,
   with the **exact** numba version the platform resolves (`numba.__version__` is stored in the
   `.nbi` and a mismatch silently empties the index).
2. `NUMBA_CPU_NAME=x86-64-v3`, `NUMBA_CPU_FEATURES=""` at build *and* run time, plus a runtime
   guard (`llvmlite.binding.get_host_cpu_features()` contains `+avx2`) or the process SIGILLs.
3. The agent directory is read-only on the platform, so `InTreeCacheLocator.ensure_cache_path()`
   (which `makedirs` then writes a temp file) fails and numba silently falls through to the
   path-dependent user-wide locator. Work-around, all in plain python at the top of `agent.py`
   before numba is imported: compute `sub = basename(d) + '_' + sha1(abspath(d)).hexdigest()`,
   copy the shipped cache into `/tmp/nb/<sub>/`, set `os.environ["NUMBA_CACHE_DIR"] = "/tmp/nb"`.
4. `os.utime()` `agent.py`, `fastboard.py`, `fastsearch.py` to the fixed integer timestamp the
   cache was built with (zip mtimes have 2 s DOS granularity and no timezone, so the exact float
   stamp is lost).
5. Ship ~9 MB of `.nbc`.
6. **Get a ruling on "native binaries in the zip are rejected".** A `.nbc` is a pickle of compiled
   object code. It is not source a judge can read, and the "do not obfuscate" line points the same
   way. I am flagging this, not deciding it.

**Achievable init with option (a): ~15-18 s here (~27-32 s on the platform), and only if `search`
and `quiesce` stay uncached.** Given the segfault, the observed silent fallback, four separate
brittleness conditions and a probable rule violation, my recommendation is not to ship it, and to
treat the reduction levers below as the real plan.

## Options (b), (c), (d)

**(b) fewer specialisations -- measured, real, source-only.**
Every fastboard argument is annotated `Any`, so numba specialises per concrete integer type at each
call site. Signature counts read out of the cache index:

```
attackers_to 9   is_attacked 9   rook_attacks 9   bishop_attacks 9
feature 6        _add 6          bit 5            _acc_row_one 4
_acc_row 3       gen_legal 3     occupancy 3      score_moves 3
make_full 2      make_light 2    rebuild 2
```

That is **61 redundant compiles** in fastboard alone, and each redundant specialisation is also a
separate body that `search` inlines. Fix = eager explicit signatures on the leaf helpers (as
`agent.py` already does for its five kernels) plus consistent `np.int64`/`np.uint64` at call sites.
Direct saving is bounded by `fb.warm_up`'s 7.05 s; the indirect saving in `search`'s compile is the
interesting part and is untested.

**(b') constant-folding the settled switches -- measured.** Twenty of the `ctrl` slots are now
permanently on or off in v8.5 (`HYGIENE/FUTILITY/PVS/LMR/SEE/SAFE/SEE_MAIN/TT_BUCKETS/LMR_AGGR/
LAZY_ACC/PRUNE_V2/SINGULAR/HISTORY2` on; `LMP/NMP_GUARD/RFP_PHASE/IIR/TT_KEEP/QS_CACHE/TT_OFF` off)
but `search` still compiles both arms of all of them. Replacing 34 `ctrl[C_*]` reads with literals:

```
fs.warm_up  23.28 s  ->  18.99 s     (-4.30 s, -18%)
fb.warm_up   7.05 s  ->   7.05 s     (unchanged, as expected)
```

**(c) lazy compilation after move one -- not viable.** 73% of init is `search`, which is needed for
move one. `fb.warm_up`'s 7 s is also all needed by `search`. The only genuinely deferrable kernels
(ponder helpers, `rebuild` for non-initial refreshes) are worth well under a second, and deferring
anything moves it onto the game clock, which is the thing `warm_up` exists to prevent.

**(d) `numba.pycc` AOT.** `from numba.pycc import CC` imports fine in 0.67.0, so it is present.
It is also deprecated, requires a C toolchain at build time, does not support the whole `njit`
feature set, and produces a `.so`/`.pyd`. AGENTS.md: *"Native binaries in the zip are rejected.
Ship source; take compiled code from public packages."* This is a clearer rule violation than the
`.nbc` cache. Flagging, not deciding -- but I would not build it.

## Node-time profile at depth 8

Frozen snapshot, 40 positions, **1,491,095 nodes** in every row (node identity is the exactness
proof: a build that changes the tree is not measuring what it claims to).

| build | s | knps | us/node | delta |
|---|---:|---:|---:|---|
| baseline (repeated) | 4.6 / 4.9 / 4.4 | 326 / 306 / 337 | 3.09 - 3.29 | - |
| `evaluate` body run twice | 7.1 | 209 | 4.76 | **+1.68 us/node** |
| `POLL_MASK` 255 -> 4095 | 4.7 | 316 | 3.15 | +-0 (noise) |
| `NUMBA_CPU_NAME=x86-64-v3` | 5.0 | 297 | 3.35 | -8% |
| `QS_EVAL_CACHE = True` | 4.2 | 351 | 2.82 | **+4.2%** |

Component micro-benchmarks (real weights where relevant, best of 5):

| kernel | ns |
|---|---:|
| head 32x1024 float32 (`w2t` layout, current) | 866 - 886 |
| head 16x1024 float32 (half the output layer) | 414 |
| head 32x1024 int16 weights x int16 acc | 1675 |
| head 32x1024 int8 weights x int16 acc | 1662 |
| head 32x1024 float32, sparse input-major, skip zeros | 2534 |
| generic 4 x 1024 dot, float32 + fastmath | 66 |
| generic 4 x 1024 dot, int16 | 194 |
| `np.zeros(32, int64)` inside `njit` | 64 |

Clipped-ReLU sparsity over 6 representative positions: **3850 / 6144 = 62.7% exact zeros.**

### Reading of the profile

* **`evaluate` is the engine.** The double-eval build costs +1.68 us on a 3.1 us node, and the head
  alone measures 866 ns, so evaluation is roughly **35-50% of node time** -- consistent with the
  29.4% + 15.4% accumulator split in the source comment. Everything else fights for the rest.
* **Quantisation is dead.** int16 and int8 are ~1.9x *slower* than float32 at the real head shape,
  and 2.9x slower on a clean L1-resident dot. LLVM does not recognise the int16 MAC idiom, while
  float32 + `fastmath` vectorises to AVX2 FMA at ~62 GFLOP/s (4096 MACs in 66 ns). The 128 KB of
  weights read per evaluation is *not* the binding constraint either: halving the neuron count
  halves the time almost exactly (866 -> 414 ns), which is the signature of a compute bound, not a
  bandwidth bound. Do not spend a night on int8 NNUE here.
* **Sparsity is real but unexploitable as written.** 62.7% zeros, yet the input-major skip loop is
  2.9x slower: the 32-wide inner loop is too short to amortise an unpredictable branch and a
  128-byte row gather. It would need a two-pass form (collect non-zero indices, then process four
  at a time) to have any chance, and that is speculative.
* **The clock poll is free.** `POLL_MASK` 255 -> 4095 changed nothing measurable, so the `objmode`
  `time.monotonic()` costs under 2% and removing `objmode` buys node rate only in theory. (It also
  does *not* fix the cache segfault -- tested.)
* **SEE's allocation is small.** `np.zeros(32, int64)` in `njit` is 64 ns; `fastboard.see` does
  allocate one per call (line 1029, confirmed). My in-search A/B was invalid -- LLVM deleted the
  duplicate allocation as dead -- but at at most one `see` per node, 64 ns is <=2% of a 3.1 us node.
* **Move generation has little headroom.** `gen_legal` is already legal-by-construction (pins +
  check masks, only en passant verified by making it) and already has a `captures_only` stage, so
  the "pseudo-legal + king safety" and "staged captures-first" suggestions are already implemented,
  and better than the alternatives. `STAGED_MOVEGEN` is a separate, off, root-side switch.
* **`boundscheck` is already off** and `error_model` cannot help (all divisions are by literals).
* **`astack` traffic** is 2 x 512 float32 = 4 KB written per pending ply in `sync_acc` and 4 KB
  restored in `unmake_move`, plus ~16 KB of `W1` rows read per accumulator update. Under
  `LAZY_ACC` this is already paid only on lines that reach an evaluation. Worth an experiment, but
  it is second order behind the head.
* **Python-side root overhead** is ~35 moves x 8 iterations x one python-chess legality check per
  root move over a 4.6 s bench: sub-1%. Not worth attention.

## Ranked table

Elo estimates use the house calibration (a node-rate doubling = +65 Elo at 8 s, +32 at 120 s), so
+10% knps ~ +9 Elo at 8 s. "Init" is the saving on this box; multiply by ~1.8 for the platform.

| # | change | effort | init saved | knps | est. Elo | risk |
|---|---|---|---:|---:|---:|---|
| 1 | `QS_EVAL_CACHE = True` (already implemented, one flag) | 15 min | 0 | +4.2% | +4 | none -- node-identical |
| 2 | Constant-fold the 20 settled `ctrl` switches in `fastsearch` | 3 h | **-4.3 s** | +0-2% | init insurance | low -- must stay node-identical |
| 3 | Eager signatures on fastboard leaves (kill 61 specialisations) | 4 h | -2 to -4 s (est.) | +0-3% | init insurance | low |
| 4 | Halve the output layer 32 -> 16 neurons | 1 night (retrain) | -1 s | **+15%** | +13 at 8 s, minus eval quality | high -- needs a gauntlet |
| 5 | Split `search` into 2-3 njit functions to cut LLVM's superlinear cost | 4 h | -3 to -8 s (untested) | -0 to -5% | init insurance | medium -- may lose inlining |
| 6 | Shrink `ACC_SIZE` 512 -> 384 | 1 night (retrain) | -2 s | +12% (est.) | +11 minus eval quality | high |
| 7 | Prebuilt numba cache (all but `search`/`quiesce`) | 2 days | **-14 s** | -8% (portable codegen) | -7, or -inf if rejected | **very high** -- see above |
| 8 | Allocation-free `see` (threshold form, no `np.zeros(32)`) | 3 h | 0 | +0-2% | +1 | low |
| 9 | Two-pass sparse head (gather non-zeros, 4 at a time) | 1 day | 0 | unknown, naive form is 2.9x slower | speculative | medium |
| 10 | Remove `objmode` (watchdog thread sets `ctrl[C_STOP]`) | 4 h | 0 | +0-2% | +1 | medium -- clock safety is the #1 risk |
| 11 | int8 / int16 quantised NNUE | 1 week | 0 | **-48%** | negative | **do not do** |
| 12 | `numba.pycc` AOT `.so` | 2 days | -30 s | 0 | -inf if rejected | **rule violation** |
| 13 | `error_model='numpy'`, more `fastmath`, `POLL_MASK` | 1 h | 0 | ~0 | 0 | not worth it |

Ordered by Elo per hour: **1, 8, 2, 3, 10, 5, 4, 6**. Ordered by init saved: **7 (blocked), 5, 2,
3, 4**. Items 2, 3 and 5 buy no Elo directly; they buy the difference between an init that fits in
60 s on a contended 1.8x-slower box and one that does not, which is worth every game.

## Scoping the top four

### 1. `QS_EVAL_CACHE = True`

* **File / function.** `agent.py:956` `QS_EVAL_CACHE: Final = False` -> `True`. Nothing else moves:
  `fastsearch.quiesce` already reads `ctrl[C_QS_CACHE]` (line 430) and the `ec_key`/`ec_val` arrays
  (2^20 slots) are already allocated by `new_eval_cache()` on every path.
* **Switch.** The flag *is* the switch.
* **Exactness.** The memo is keyed on the full position key and the static evaluation is a pure
  function of the position, so the same position always yields the same score. `testing/bench.py
  --depth 8` must report **exactly 1,491,095 nodes** and `testing/check_fastsearch.py` must be
  clean; if either moves, the memo is colliding and the change is not exact.
* **Expected bench.** 4.2 s / 351 knps against 4.4 s / 337 knps. Confirm with three alternating
  runs before believing +4%.

### 2. Constant-fold the settled `ctrl` switches

* **Files / functions.** `fastsearch.py`, inside `search`, `quiesce`, `make_move`, `unmake_move`,
  `sync_acc`, `evaluate`. Replace the reads of the thirteen permanently-on and seven
  permanently-off slots with module-level `Final` booleans that mirror `agent.py`'s flags, e.g.
  `_LAZY_ACC: Final = True` and `if _LAZY_ACC:` instead of `if ctrl[C_LAZY_ACC] != 0:`. numba
  folds a module-level python bool at compile time, so the dead arm never reaches LLVM.
  `ctrl[C_QS_CAP]`, `ctrl[C_CHECK_CAP]` and the four `C_PH_*` percentages are values, not switches;
  leave them, or fold them too if the tuning is settled (`QS_CAP = 14`, `CHECK_EXT_CAP = 0`).
* **Data layout.** Unchanged. The `ctrl` slots stay allocated so `agent.py` needs no edit; the
  compiled side simply stops reading them.
* **Switch to add.** One module-level `_FOLD_SWITCHES: Final = True` in `fastsearch.py` that
  selects between the folded constants and the `ctrl` reads, so a bisect can turn it off. It costs
  compile time while it exists (both arms compile), so remove it once the gauntlet is green.
* **Exactness.** This must be bit-identical by construction: the folded values are exactly what
  `agent.py` writes into `ctrl` today. Verify with `testing/check_fastsearch.py` (which compares
  the compiled search against the reference search move for move) and with `bench --depth 8
  --json`, diffed against a baseline json: **every position's nodes, score and best move must
  match**. Any mismatch means a switch was folded to the wrong value.
* **Expected numbers.** `fs.warm_up` 23.3 s -> 19.0 s (measured); total init 31.7 -> 27.4 s here,
  ~49 s on the platform. Node rate unchanged to +2% (smaller I-cache footprint), bench 4.4 -> 4.3 s.

### 3. Eager signatures on the fastboard leaves

* **Functions, in order of waste.** `attackers_to` (9), `is_attacked` (9), `rook_attacks` (9),
  `bishop_attacks` (9), `feature` (6), `_add` (6), `bit` (5), `_acc_row_one` (4), `_acc_row` (3),
  `occupancy` (3), `score_moves` (3), `gen_legal` (3).
* **How.** Give each an explicit signature in the decorator, the way `agent.py:332` does, e.g.
  `@njit(uint64(uint64[:], uint64, int64, int64), cache=False)` for `attackers_to`. Where a call
  site passes the wrong width (a python int literal, an `int32` from `sq`, a `uint64` from a
  bitboard), fix the call site with an explicit `np.int64(...)`/`np.uint64(...)` rather than adding
  a second signature. Nine specialisations of `is_attacked` means nine different argument-type
  combinations are reaching it; the signature will make each mismatch a hard error at import,
  which is exactly the feedback wanted.
* **Data layout.** Unchanged.
* **Switch.** None needed -- a signature cannot change behaviour, only reject a call. But do it in
  one commit per function group so a bisect is cheap.
* **Exactness.** Types are the only thing that changes, and a wrong cast would change results, so
  run `testing/check_fastboard.py` **and** `check_fastsearch.py`, plus `bench --depth 8 --json`
  diffed for identical nodes/scores/moves on all 40 positions. Watch specifically for
  `uint64` -> `int64` narrowing on bitboards: `bit(63)` and any key arithmetic must stay unsigned.
* **Expected numbers.** `fb.warm_up` 7.05 s -> ~4 s, plus an unquantified reduction in `search`'s
  compile from inlining one body instead of several. Node rate flat to +3%.

### 4. Halve the output layer, 32 -> 16 neurons

* **Files.** `training/` (the net export), `agent.py` `_eval_kernel` / `_eval_bucket_kernel` head
  loops (`for j in range(0, 32, 4)` -> `range(0, 16, 4)`), `fastsearch.evaluate` (the same loop,
  verbatim), and the `W2`/`W2T`/`B2`/`W3` shapes.
* **Data layout.** `W2T` becomes `(8, 16, 1024)` and `W3` `(8, 16, 1)`; per-evaluation weight
  traffic drops from 128 KB to 64 KB and the MAC count from 32,768 to 16,384. The 4-at-a-time
  unrolling stays; `hidden` and the accumulator are untouched.
* **Switch.** This one is a weights change, so the switch is the weights file: keep the 32-neuron
  net and gate on `W2.shape[1]` so both nets load with the same code. That also lets the gauntlet
  run the two head sizes head to head.
* **Exactness.** Not exact and not meant to be -- node counts and scores will move. Verify only
  that the *plumbing* is right: export the 32-neuron net through the new shape-general kernel and
  confirm `bench --depth 8` reproduces the current 1,491,095 nodes bit for bit before touching the
  16-neuron net. Then judge the smaller net by gauntlet Elo, not by bench.
* **Expected numbers.** Head 866 -> 414 ns, so ~-0.45 us on a 3.1 us node: bench 4.4 -> ~3.8 s,
  **~+15% knps, worth ~+13 Elo at 8 s** -- against an unknown evaluation-quality loss, which is the
  whole question and only a gauntlet can answer it. Init also drops ~1 s.

## Reproduction

Scratch variants (all cut from the frozen `speed/cpuv3/` snapshot unless noted):

```
speed/timed/      instrumented import, per-stage prints
speed/cpuv3/      frozen baseline (== working tree agent.py at 17:35, md5 57b579e7...)
speed/eval2x/     evaluate body duplicated  (node-identical: measures one extra evaluate)
speed/poll4095/   POLL_MASK = 4095
speed/qsc/        QS_EVAL_CACHE = True
speed/fold/       34 ctrl switch reads constant-folded
speed/cached/     every njit cache=True                      -> segfaults on warm import
speed/nomode/     cache=True, objmode removed from timed_out -> still segfaults
speed/cp2/        cache=True except search/quiesce           -> warm import works, 17.8 s
speed/micro.py speed/head2.py speed/sparse.py                   kernel micro-benchmarks
```

Command used throughout: `.venv/Scripts/python.exe -m testing.bench --depth 8 --agent <dir>`
from the repo root. Repeat each timing and take the best; the node count is the thing to trust.
