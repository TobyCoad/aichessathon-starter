# Where init time actually goes, and the one lever left (measured 6 Sep 14:20-14:32, iter 33)

Instrument: `overnight/eval/initprof.py` (numba event listener attributing exclusive
compile seconds per dispatcher; then `cres.metadata['pipeline_times']` per compiler
pass and `cres.library.get_llvm_str()` line count as an IR-size proxy). All numbers
taken on this laptop WITH the `160-v95` gauntlet running, so absolutes are inflated
~20%; every comparison below is between two runs under the same load.

## 1. Init is one function, and it is type inference

Cold import of the champion tree with INIT_FOLD **True** (what ships):

    import agent            37.4 s
    numba compile total     37.3 s   (64 functions, 76 compiles)
    search                  24.5 s exclusive / 32.6 s inclusive   <-- 66% / 87%
    gen_legal 2.3   quiesce 2.1   make_full 1.3   make_light 0.7   make_move 0.6
    ... 59 more functions, none above 0.5 s

Breaking `search`'s own compile down by numba pass:

    nopython.17_nopython_type_inference   21.95 s   (71%)
    nopython.24_native_lowering            6.71 s   (22%)
    everything else                        2.17 s

This settles the open question in NOTES: LLVM is NOT the cost, which is exactly why
NUMBA_OPT 1 / 2 / default all measured 36-40 s. The cost is numba's own type-inference
fixpoint over a single 650-line function, and that fixpoint is superlinear in the size
of the function it runs on.

## 2. How superlinear -- measured, not assumed

INIT_FOLD prunes 27 settled switch branches before typing, which gives a clean
within-function pair (same source, same machine, same load):

                        LLVM lines   inference   lowering
    INIT_FOLD False        17,627      30.6 s      8.2 s
    INIT_FOLD True         15,953      23.5 s      7.0 s
    delta                   -9.5%      -23.3%     -14.6%

Elasticity of inference to IR size = ln(0.767)/ln(0.905) = **2.65**. Lowering's is 1.6.
A cross-function check at one point in time agrees on the shape (llvm lines ->
inference): make_move 4,879 -> 0.91 s, quiesce 10,578 -> 4.15 s, search 15,953 ->
23.5 s; the implied exponents are 2.0 (quiesce/make_move) and 4.2 (search/quiesce).
Call it "between quadratic and cubic". The direction is not in doubt: **the same code
costs dramatically less to compile when it is spread over several njit functions than
when it sits inside one.**

That also explains INIT_FOLD's whole gain, and it means lever (c) from NOTES ("delete
closed switches outright rather than folding them") is nearly exhausted -- folding
already removes them before typing, and there are only 27 of them.

## 3. The lever: split `search` into njit helpers (SEARCH_SPLIT)

`search` is `fastsearch.py:746-1394`. Four blocks in it are self-contained and,
crucially, **do not call `search`** -- so moving them out creates no mutual recursion,
which is the thing numba handles badly:

  A. TT probe            807-841   ~35 lines
  B. eval / improving / RFP / razor / futility flags   859-943   ~85 lines
  C. move generation + ordering    1059-1141  ~80 lines
     (gen_legal, score_moves, history2 base, conthist base, CAPTURE_ORDER rescore,
      the pvs/lmr/aggr/lmp/prune2 flag block)
  D. TT store            1356-1391 ~35 lines

That is ~235 of 650 lines, ~36% of the body. NMP (944-1020), the singular /
extend_hash block (1021-1058) and the move loop (1142-1347) all recurse into `search`
and must stay where they are.

Predicted saving at the measured exponent 2.65: 23.5 s * (0.64^2.65) = 7.5 s, plus
maybe 2-3 s for the four helpers' own (small, cheap) inference. **Net ~-13 s local**,
which at the platform's measured 2.1x is **~-27 s there**: typical platform init
~63 s -> ~36 s, worst observed 90+ s -> ~51 s. Even at a conservative exponent of 2.0
it is ~-8 s local / ~-17 s platform.

Nothing else on the board is worth this. It is bigger than every search bundle we have
shipped, because a game lost at init is a whole point and we have measured one
(4 platform init samples: 74.1 / >90 GAME LOST / 88.1 / 64.1 s against a 90 s budget).

### Why it is safe

It is pure code motion: the arithmetic is unchanged, the helper takes the same arrays
and returns the same locals. It is exact by construction and the existing gates prove
it -- `check_fastsearch --depth 4 --random 30` (70/70 + 40/40 table-on) and a depth-8
bench node count that must come back **bit-identical** to 1,110,289.

### The one risk, and how to read it

njit -> njit calls are real calls at the IR level; LLVM normally inlines small ones,
and LLVM opt is cheap here (section 1), so the runtime cost should be nil. But it must
be MEASURED, not assumed: bench depth 8 must show the same node count AND knps within
noise. If knps drops, add `inline='always'` to the offending helper -- but note that
IR-level inlining happens BEFORE type inference, so an `inline='always'` helper hands
the compile time straight back. That is the whole trade: keep helpers out-of-line for
the compile win, and only inline one back if it costs measurable knps.

### Order of work (biggest block first, one at a time, gate after each)

C (ordering, ~80 lines) -> B (eval/pruning flags, ~85) -> A (TT probe) -> D (TT store).
After each split: ruff, mypy, `check_fastsearch`, bench depth 8 node count identical,
and re-run `overnight/eval/initprof.py` to record the new `search` inference seconds.
Do it behind no switch at all -- there is nothing to switch, it either compiles to the
same nodes or it is wrong -- but commit each block separately so a knps regression can
be bisected.

## 4. Second-order notes

- `gen_legal` compiles 3 times (2.3 s total) and `make_full` / `make_light` /
  `score_moves` / `qs_tt_store` twice each. Those are separate signatures forced by
  differing argument types at the call sites. Unifying the types would save ~1.5 s --
  real but an order of magnitude below the split. Worth doing only if it falls out of
  the split work.
- `quiesce` (4.2 s inference on 10,578 lines) is the second-largest function and the
  same treatment would apply to it later; do `search` first.
