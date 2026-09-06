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

## 5. Block C, worked out exactly (do this one first)

Read at 14:31, iter 33, so the next iteration does not have to re-derive it. `search`
lines 1059-1141 split as follows.

STAYS in `search` (the early return has to happen there):

    mv = moves[ply]
    n = fb.gen_legal(bb, sq, meta, mv, False)
    if n == 0:
        return -MATE + ply if in_check else 0
    sc = scores[ply]

MOVES OUT, verbatim, as lines 1064-1117 (`history2 = ...` through the end of the
CAPTURE_ORDER rescore loop):

    @njit(cache=False, nogil=True)
    def order_node(bb, sq, meta, undo, mv, n, sc, hash_move, k0, k1,
                   butterfly, counter, conthist1, ctrl):
        ...                      # fills sc in place
        return base, ch_base     # int64 pair

Call site: `base, ch_base = order_node(bb, sq, meta, undo, mv, n, sc, hash_move,
killers[ply, 0], killers[ply, 1], butterfly, counter, conthist1, ctrl)`.

LIVE-OUT ANALYSIS (grepped over 1118-1394, the rest of `search`):
  - `base`     used at 1154, 1192, 1305, 1310  -> must be returned.
  - `ch_base`  used at 1155-1156, 1194-1195, 1317, 1324, 1331 -> must be returned.
  - `counter_move` is NOT used after 1117 -> stays local to the helper, do not return it.
  - `history2`, `conthist_on`, `capture_order` ARE used later (1193, 1217, 1219, 1299,
    1335) but are one-line fold ternaries over `ctrl`. RECOMPUTE them in `search` rather
    than returning them: it keeps the INIT_FOLD constant-folding working at both sites
    and costs a couple of IR nodes, against a tuple return that would cost more.
  - Everything else the block touches (`mv`, `sc`, `killers`, `butterfly`, `counter`,
    `conthist1`) is an array mutated in place, so it needs no return.
  - Module globals it reads (`MVV`, `CAPTURE_BONUS`, `_FOLD`, the `_F_*` constants) are
    visible from any njit function in the module -- no plumbing needed.

Give the helper `nogil=True` to match `search`. Name it `order_node`, NOT `order_moves`:
`fb.order_moves` already exists and is a different thing.

Gate this one commit on: ruff, mypy, `python -m testing.check_fastsearch --depth 4
--random 30` (70/70 + 40/40), `python -m testing.bench --agent <dir> --depth 8` returning
**1,110,289 nodes exactly** with knps within noise of the champion's, and `initprof.py`
re-run to record `search`'s new inference seconds against the 23.5 s baseline.
