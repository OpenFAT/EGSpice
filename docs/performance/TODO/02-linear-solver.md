# Linear solver at scale

What the sparse LU costs on large circuits, what already ships, and every
change that would make it cheaper. Anchors are `src/...` paths in this
tree. Numbers marked *measured* come from the pages named beside them;
anything else is an estimate.

Constraints every item below respects (owner decisions, `dev/solvers/gpu-lu.md`
§1 and `dev/research/plans/README.md`):

- The host path is the oracle. A device or threaded factor must be bitwise
  the host's `SparseLu.refactor` and `solve`, so a pivot failure peels to the
  host and the run continues as a host-only run would.
- No float atomics anywhere. Every sum has a fixed order.
- Every kernel goes through gompute, written once in Zig for CUDA and HIP.
  cuDSS, cusolverRf and rocSOLVER are performance references only.
- The published Newton iterate, the convergence test and the time grid are
  ngspice's in the default mode. An inexact solve is not a default-mode
  option (`dev/solvers/gpu-convergence.md` §10.1).

## What ships

| Piece | State | Anchor |
|---|---|---|
| Flat KLU-style left-looking LU, threshold pivoting τ = 1e-3, BTF + AMD order per block, block boundaries discarded | default | `src/solver/sparse_lu.zig:390-577`, `src/solver/order.zig:173-218` |
| Value bypass (memcmp), pivot-tape replay, growth monitor, probe solve, full re-pivot on failure, fill cap 3x | default | `src/solver/direct.zig:228-363` |
| Flat flop tape for factors ≤ 2,048 flops; supernode panels from a 32-row L column | default | `src/solver/sparse_lu.zig:1030-1134`, `:42` |
| Tridiagonal Thomas engine, BBD engine (per subcircuit instance) | by pattern | `src/solver/tridiag.zig`, `src/solver/bbd.zig:37-41` |
| Multicore refactor: the GPU kernel body on host threads, ticketed, sync-free | **off by default**: needs `ESPICE_SOLVER_THREADS` or `ESPICE_THREADS` > 1, both default 1 (`src/main.zig:171-172`) | `src/solver/kernels/lu.zig:688-761`, admission `src/solver/direct.zig:511-566` |
| Device LU, stage 1: host pivot tape replayed on the device, both solves on the device, bitwise | `auto` only under a non-cpu backend, n ≥ 10k, F/n ≥ 500 or F ≥ 50M; **the CLI default backend is cpu** (`src/main.zig:42`) | `src/analysis/gpu/lu.zig:337-362`, kernels `src/solver/kernels/lu.zig:155-582` |
| LaneLu(W = 4) frequency lanes, GPU frequency lanes on equal tapes | AC family | `src/solver/lane/lu.zig`, `src/analysis/gpu/freq.zig` |
| Masked per-point GMRES on the device for HB | HB lanes | `src/solver/kernels/gmres.zig`, `src/analysis/gpu/krylov.zig` |

## Where the LU is the bottleneck

Share of wall time in the LU (factor plus solve), *measured*
(`dev/devices/gpu-evaluation.md`, "Where the time goes"; RTX 4060 Laptop,
i9-14900HX, one host thread, device LU off):

| Deck | n | fill | F per refactor | LU, CPU eval | LU, GPU eval |
|---|---:|---:|---:|---:|---:|
| logic_bsim4_1k | 4.0k | | | 53% | 58% |
| chain_bsim4_10k | 45k | 2.6x | 2.9M | 32% | 50% |
| logic_bsim4_10k | 39k | 9.2x | 170M | 82% | 94% |
| sram_bsim4_10k | 17k | 8.2x | 40.7M | 70% | 83% |
| logic_psp103_10k | 124k | 3.8x | 201M | 87% | 93% |
| chain_bsim4_100k | 446k | 2.9x | 74M | 31% | 46% |
| c7552_sky130 (real) | 117k | 1.2x | 2.0M | 17% | 26% |
| asap7 24-stage chain (sweep workload) | 293 | | | 2.2% | |
| vacask_mul, n = 12 (tran) | 12 | | | 28% | |

Two regimes. **Thick factors** (F/n in the hundreds and up: random logic,
SRAM, meshed supplies) are flop-bound and already have two parallel engines
that win (below). **Thin factors** (F/n under 100: chains, rings, RC trees
with ideal supplies) are span-bound: the columns form a chain and every
handoff is a wait, so 8 host threads run them at 0.39-0.57x and the device
at 0.47-0.78x of one host thread (`dev/solvers/gpu-lu.md` E2). Nothing that
ships helps a thin factor; items 2.4 and 2.5 are the only levers for it.

What the two parallel engines buy today, *measured* (`gpu-lu.md` "E3, host";
wall seconds; `lu8` = multicore refactor on 8 threads, `cudalu` = device
eval plus device LU):

| Deck | F/n | serial | lu8 | cuda, host LU | cudalu |
|---|---:|---:|---:|---:|---:|
| sram_bsim4_10k | 2,400 | 38.6 | 27.3 | 32.9-52.2 | 19.8-26.1 |
| logic_bsim4_10k | 4,400 | 183.6 | 85.7 | 228-237 | 51.9-63.1 |
| logic_psp103_10k | 1,600 | | | 357.2 | 86.1 |
| chain_bsim4_10k | 63 | | 21.7 | 9.9 | 11.6 |
| ring_bsim4_10k | 87 | | 36.2 | 20.1 | 24.4 |

## Items

Tags: **P0** build next, **P1** after the P0s, **P2** research or opt-in.
**[big]** one large circuit, **[freq]** AC family, **[fp]** changes the
summation order of every full factor (within rounding) and needs the
one-time decision `dev/solvers/solver-perf-2026-09.md` "Not done" asks for.

### 2.1 Turn the multicore refactor on by default — P0 [big]

**Today.** `HostRefactor` runs only when a thread variable is set; the CLI
defaults both to 1 (`src/main.zig:171-172`, `src/analysis/executor.zig:235`).
The admission bar (F ≥ 500k, F/n ≥ 300, F ≤ 400M, `src/solver/direct.zig:511-566`)
already keeps thin factors serial, where threads lose.
**Cost.** *Measured* E3: 1.07x sram_1k, 1.26x logic_1k, 1.41x sram_10k,
2.14x logic_10k left on the table at default settings.
**Change.** Default `lu_threads` to min(8, P-cores) when the bar admits the
epoch; take the threads from the shared budget of
[05](05-scheduling-and-parallelism.md) so `--jobs=auto` times 8 does not
oversubscribe. Pin the refactor tasks to P-cores: E-core lanes ran PSP
evaluation 2x slower than P-core lanes (`gpu-evaluation.md`, "Eight host
threads"); the LU was not measured per core type.
**Gate.** Bitwise already. **Effort.** 0.5 d plus the budget.

### 2.2 Default `--backend auto` — P0 [big]

**Today.** The default backend is `cpu` (`src/main.zig:42`), so neither the
device LU nor device evaluation is ever admitted unless the user asks.
**Cost.** logic_bsim4_10k 183.6 s → 51.9-63.1 s, logic_psp103_10k 357 → 86 s
with both admitted (*measured*, E3).
**Change.** Default to `auto`. `auto` prices the query before the first
driver call and declines small decks (`gpu-evaluation.md`, "Choosing the
backend"), so the 2.6 ms small-deck floor must not move; `bench` on the
corpus is the check. The cold-JIT refusal (`ImageNotCompiled`) is the risk,
see [01](01-gpu-resident-newton.md) 1.7.
**Gate.** `zig build test` under the new default equals `cpu` within the
`test-gpu` tolerance; `zig build bench` medians on the small decks unchanged.

### 2.3 Resident A: assemble and factor on the device without the planes round trip (GPU LU stage 2) — P0 [big]

**Today.** Stage 1 only. With device evaluation, every Newton iteration
downloads the planes (16·nnz + 16·n bytes), the host combines
`G + ag0·C`, uploads A (8·nnz) for the device refactor, downloads dx; one
sync per call, no graph (`src/analysis/gpu/lu.zig:213-283`).
**Cost.** chain_bsim4_100k: plane download 3.2 ms of a 19.6 ms eval
(*measured*, `gpu-evaluation.md`), plus the A upload. At n = 10^6,
nnz = 5·10^6: about 96 MB down and 40 MB up per iteration, 12 ms at the
measured 11 GB/s, more than the eval kernels on an FP64-class card.
**Change.** `gpu-lu.md` §3(f) and §4.4 stage 2: a scatter kernel forms
`A = g + ag0·c` straight into the factor's value slots on the device;
host-evaluated batches form an **overlay** (a compact slot list: the device
gathers those slots, the host adds its stamps in today's order and uploads
values that replace, not add, so the sum order is the host's); the ground
pin joins the overlay. The transient companion still needs the q history
the host owns, so stage 2 keeps one extra sync for the companion rows;
stage 3 moves `companionAt` onto the device (cross-reference
[01](01-gpu-resident-newton.md) 1.2). Capture scatter, refactor and solves
in the iteration's graph.
**Expect.** Bytes per iteration from O(nnz) to O(n + overlay): 96 MB → ~32 MB
at the sizes above (estimate). Syncs unchanged until 01 lands.
**Gate.** `ESPICE_GPU_LU_CHECK` bitwise; raw output byte-identical with the
device LU off (bitwise by construction). **Effort.** 1-2 weeks.
**Depends.** Nothing; 01 for the graph.
**Status (2026-10-22): not started, measured.** logic_bsim4_10k under
`cuda` (ReleaseFast, `ESPICE_GPU_STATS`): per Newton iteration the planes
download is 3.5-7.0 MB and A up 1.5 MB, and the device LU call's wall is
0.4-0.5 ms over its kernels (17.9 vs 17.5 ms): resident A would save about
1-2 ms of a ~20 ms iteration (5-10%) on the thick decks. It touches
Circuit.zig, converger.zig, integrator.zig and context.zig (all hot or
owned elsewhere). Next: land it with 01 1.2 (companion on the device), where
the q history moves too.

**Status (2026-10-23): done, exact.** `Circuit.evalNewtonGC` (tran) asks
the context for A = G + ag0·C: the device gathers g and c only at the
diagonal (ground pin, host loads) and downloads rhs and q; with the
device LU resident-ready `GpuLu.solveResident` forms A on the device
(`arp_gc_combine` + the diagonal overlay), else the device forms A and A
comes down instead of g and c. `slv.vcopy` downloads only when a host
refactor needs it. ESPICE_GPU_LU_CHECK checks the device-formed A bitwise
(0 mismatches on the five 10k/30k decks); raws byte-identical base vs
new; CPU corpus A/B 836 SAME / 104 `exit 1/1`. KB down per Newton eval:
logic_bsim4_10k 4290 -> 1970, chain_bsim4_10k 5279 -> 3689,
chain_bsim4_100k 52,295 -> 26,116. Wall (ReleaseFast, cuda, median of 3,
load 15-30): chain_bsim4_10k 8.32 -> 8.07 s, chain_bsim4_100k 81.3 ->
76.6 s (2 runs), logic_bsim4_10k within noise (30.9 vs 31.9 s).

### 2.4 Order for span as well as fill — P1 [big] [fp]

**Status (2026-10-22): done without a new order, exact.**
`src/solver/subtree.zig`: `subtree.Plan` splits each pivot epoch at cut
steps into the connected components of the column DAG (dealt to lanes,
largest first) and a serial tail, priced by an O(nnz(L+U)) cost model;
every column runs `SparseLu.Replay.column` on final inputs, so the
factors are bitwise `refactor`'s. Policy: automatic whenever
`lu_threads` >= 2, the ticketed kernel declines the epoch and the model
prices >= 1.5x. AMD's forest is wide enough: model 2.1-2.6x at 8 lanes
on chain/ring_bsim4_10k (the supply-column tail holds 24-38% of F);
measured 1.7-2.6x per refactor in isolation where the ticketed kernel
ran 0.39-0.57x; factor time over the run 4.10 -> 2.66 s (chain) and
6.41 -> 4.16 s (ring), 5-run medians at load 12-40. A level-structure
nested dissection was built and measured: 116x fill on chain_bsim4_10k
against AMD's 2.6x (8.4x vs 8.0x on the 100x100 grid), so it was
removed; a multilevel partitioner is the remaining route (research).
Tail (round 2, exact): the tail runs pipelined when the model prices it
faster: the lanes take its steps by ticket in order, and a step waits
only for the tail steps in its U[:,k] before reading their L columns
(`Replay.columnAfter`, the ticketed kernel's schedule on the host column
replay, so no `kernels/lu.zig` entry point). Per refactor at 8 lanes
(ReleaseFast, medians of 15, load 12): chain_bsim4_10k 2.27 -> 1.17 ms
(lanes with serial tail -> pipelined; serial refactor 5.42 ms),
ring_bsim4_10k 2.80 -> 1.95 ms, chain_bsim4_100k 70.1 -> 41.4 ms
(serial 138 ms). Factor time over the run, default threads, 3 interleaved
runs at load 26-45: chain 4.92 -> 3.76 s, ring 6.32 -> 5.28 s, raw output
identical. Full corpus A/B (Debug, 8 threads): 836 SAME, 104 DIFF all
`exit 1/1`.

**Today.** BTF then AMD or natural per block, flat
(`src/solver/order.zig:173-218`). AMD's elimination trees are deep:
`inverter_chain_256` has 259 levels for 261 columns, and on the replicated
decks the supply and bias columns land last and depend on every instance
(4 columns held 49% of the opamp OP work). The 100k chain's longest column
chain carries 31% of its refactor, and host threads cannot split one
column's updates (`gpu-lu.md`, "Multicore host refactor").
**Cost.** Thin factors are the regime where neither parallel engine pays
(0.39-0.78x), and they are the common post-layout shape (RC trees per net,
F/n 17-87 on every chain, ring and real deck measured).
**Change.** A second ordering, nested dissection (own implementation; the
separators from a multilevel partition of the filled graph), chosen per
pattern by price: predict F and the refactor span S_r for AMD and for ND
with the O(nnz(U)) host passes the E1 census already has (`gpu-lu.md` §4.2),
and take the order with the lower max(F / threads, S_r · h). On ladder, tree
and mesh graphs ND gives O(log n) depth for a bounded fill increase; on a
dense-tail logic block AMD stays. With BTF blocks kept (2.5) each block is
ordered on its own.
**Expect.** Unmeasured. The parallel refactor and the device LU become
admissible on chains and rings, where today they lose. Build the span
predictor first and read S_r for ND against AMD on the post-layout set
before any kernel work.
**Gate.** Every full factor's summation order changes, so corpus raw files
move within rounding: the pass set and `VERDICTS.tsv` must not move, and
the per-deck F and S_r before and after are recorded. **Effort.** 2-3 weeks.

### 2.5 Keep the BTF blocks through factorization — P1 [big] [fp], census first

**Status (2026-10-22): census negative, dropped.** `ZP_LU_STATS` prints
`lu-btf`: chain, ring, logic and sram 10k, rc_ladder_100k, grid
100x100, divider chain and vacask_ring are each one SCC of n - 1
columns plus one singleton, off-block nnz 0.

**Today.** `order()` returns only the permutation; the block boundaries are
scratch and one flat `SparseLu` eliminates the off-block entries too
(`dev/solvers/btf-permutation.md`, "Our variant";
`docs/performance/bench/2026-10-20/xyce-research/solver-source-audit.md`).
**Change.** KLU's form: factor each diagonal block separately, keep the
off-block entries (`Offp/Offi/Offx`) for block substitution. Blocks are
independent, so they run on host threads or device blocks with no tickets,
each with its own replay tape; fill is confined to blocks.
**Census first.** Extend `ZP_LU_STATS` with the SCC count, the largest block
and the off-block nnz per deck. A flat extracted netlist with a meshed supply
is probably one giant SCC, in which case this buys nothing and 2.4 is the
lever. **Gate.** As 2.4. **Effort.** 1 week after a positive census.

### 2.6 BBD from a partitioner instead of subcircuit instances — P2 [big]

**Status (2026-10-22): host side subsumed by 2.4's lanes** (components
are the blocks, the tail the border, factored sparse and bitwise). The
dense-border device form stays research and needs the partitioner 2.4
did not find.

**Today.** One block per subcircuit instance (`src/frontend/builder.zig:126-217`);
`Bbd.init` needs ≥ 8 blocks of ≤ 64 rows and a border ≤ min(512, n/2)
(`src/solver/bbd.zig:37-41`). Every post-layout deck is declined and no
corpus deck takes the engine (`dev/solvers/overview.md`). Blocks are dense
full factors with no replay.
**Change.** Blocks from the separators of 2.4 (thousands of rows, sparse LU
per block, bitwise per block), Schur complement reduced in fixed block order
(a batched dense shape that fits the device), the border factored dense.
The border bounds it: a 4k border is 128 MB and ~43 GFLOP per factor, which
is 0.2 s on this card's FP64 and 2 ms on an H100. Datacenter-only, and only
if 2.4's census finds large borders. **Effort.** research.

### 2.7 Device LU kernel work — P1 [big]

Each is small and measured in `gpu-lu.md` or marked there:

- **Markstein division.** A reciprocal multiply (not bitwise) saved 12% on
  logic_bsim4_10k; a Markstein sequence with the exactly-rounded slow path
  keeps the bits. The dense tail spends about a thousand IEEE f64 divisions
  per column on the critical path at FP64 1/64.
- **Graph capture.** Stage 1 launches the scatter, refactor, status and
  solves directly with one sync; put them in the iteration graph with 2.3.
- **Peel hysteresis.** Designed (demote after more than one peel per 20
  device refactors), not built: the code only counts peels
  (`src/analysis/gpu/lu.zig:213-283`).
- **Refit the admission bars.** Nothing was measured at F/n 166-1,600 or
  F 5-69M (`src/analysis/gpu/lu.zig:350`, `:360`); one F/n bar serves every
  thread count (`src/solver/direct.zig:561`); the tail-cut and gather
  constants are guesses (`src/solver/kernels/lu.zig:766`, `:1009`).
  Run the post-layout generator at 3k and 30k to fill the gap.
- **`dmap` width.** Split columns into a u16 list and a u32 list when
  `dmap` traffic is above 30% of refactor bytes (`gpu-lu.md` §4.1 note).
- **f32 refactor on the device** (P2, opt-in). The host f32 + GMRES-IR mode
  lost 2-32% and was deleted (`gpu-lu.md`, "fast_mode"). On a consumer card
  the division chain is the critical path, so the device economics differ;
  it cannot be bitwise, so it lives only behind an opt-in solver mode with
  f64 fallback on a pivot or Newton failure.

**Status (2026-10-22, gpu-lu).** Measured on ReleaseFast GPU builds, RTX 4060
Laptop, `ESPICE_GPU_LU_BENCH` medians, interleaved runs:
- Markstein division: built, bitwise (4M-pair test), and 5-7% *slower* on the
  refactor (logic_bsim4_10k 11.04 -> 11.61 ms, sram_bsim4_10k 6.55 -> 7.05).
  Removed; ptxas's `div.rn.f64` stays.
- `dmap` width: u16 column-relative slots for every epoch whose widest
  column has under 65,535 slots (one width per epoch, no split). Tables
  halve (F=594M re-pivot epoch: 2.4 -> 1.2 GB); refactor 0 to -2%.
- Table build: the u16 dmap is filled directly and the gather-form analysis
  runs only for columns whose U entries average under half a block, which
  halved `build` (logic OP: 12.2 -> 6.3 s, 5 epochs; the host multicore
  refactor's tables use the same build).
- Peel hysteresis as designed (demote after >1 peel per 20 refactors): lost
  (logic OP 12.0 -> 15.1-16.9 s; the host refactors the F=594M re-pivot
  epochs instead). Not built.
- Graph capture: the call wall is 0.4-0.5 ms over the kernels, mostly the A
  upload, so capture saves under 1%; deferred to 2.3.
- Tail solves (2.9 device side): panels of 64 rows across the grid replace
  the one-block column sweep: solves -5 to -16%.
- A lone operating point now takes the device LU under explicit `cuda`/`hip`
  (it was excluded whenever the GPU lane policy was on).
- Admission bars at 3k and 30k: not measured yet.

**Graph capture (2026-10-23): measured, not worth it.** About 8 launches
per device LU call, enqueued in ~20-40 us that overlap the kernels,
against 6-17 ms per call (<0.5%); the per-call `stamp` scalar would also
have to move to device memory (or the done stamps be cleared per call).

### 2.8 Host micro-work — P1, small each

**Status (2026-10-22).** Probe scan: done, exact: `SparseLu.replay_mult`
carries max|L| out of the column replay (the rounded max|v| / |d|;
rounding is monotonic, so it equals the scan) and `maxMultiplier` skips
its O(|L|) pass, 290-560 us per replay on chain/ring_bsim4_10k, 7-10% of
a serial replay. Partial refactor: census (`lu-reach`) says no, 31.6% of
A entries change per factor but the redo reach holds 99.2-99.4% of F.
Thomas 2x2: does not pay, the LU is 9% of rc_ladder_100k and its branch
unknown sits last. Dense 2-D tiling: does not pay, every HB/PSS corpus
deck runs under 0.25 s. Full-factor scratch: not done (a few page
faults per full factor against 44n resident bytes). `factorEach`
allocations: `freq_solve.zig`, the freq owner's.
Round 2: permuted assembly dropped; supernodal DFS done. Permuted
assembly would only remove the `prow` load from the replay's scatter of
A, which is nnz(A) = 215k entries against F = 21M plus 0.9M L/U slots
on chain_bsim4_10k (about 1%). The full factor's DFS is about 40% of
every full factor (204 of 498 ms on a 300x300 grid OP; 231 of 588 ms
over chain_bsim4_10k's 14 full factors, 3% of its run). An
order-preserving symmetric pruning was built first and gained nothing
(the walk is bound by node visits, and `lend` already prunes supernode
chains). The supernodal walk (`SparseLu.enter`): a row whose step lies in
a panel marks the panel's later steps up to the first already reached,
scans the rows below the block once, and emits the run in step order.
It engages per factor once panels hold 4/5 of the L entries so far
(`sn_dfs`: 0.86-0.98 on the resistor grids, at most 0.68 on the
chain/ring re-pivots, which keep the plain walk, bitwise). Grid 300x300
OP factor 352 -> 255 ms (min of 5, ReleaseFast, load 25), DFS about 195
-> 55 ms; full corpus A/B byte-identical (836 SAME, 104 identical
`exit 1/1`), including the 100x100 grid.

- **Ordering once per pattern.** OP and transient executors each run
  BTF + AMD on the same frozen pattern: two `computeOrdering` calls, 95M Ir
  each on `scaling_rc_ladder_100k` (1.5% of the run, ~2% on the 100x100
  grid; *measured*, `solver-perf-2026-09.md`). Compute it with the prepared
  pattern; `direct.Solver.init` takes it (`src/solver/direct.zig:574-583`).
- **Full-factor scratch in the workspace.** Every full factor allocates and
  frees nine n-length u32 arrays and an n-length `rscale`
  (`src/solver/sparse_lu.zig:291-310`), 4.4 MB at a 100k ladder. Rare outside
  OP ladders (the 100k chain's OP peeled 23 times), so cheap but small.
- **Assembly in permuted coordinates.** Refactor replay makes the solver
  cheap enough that the scatter pass through `prow` is visible. Stamps
  written straight into permuted slots delete it (`dev/solvers/README.md`
  open question 1). Needs the tapes to carry permuted slots, which the
  frozen GPU boundary complicates: an extra host-side slot map, not a tape
  change.
- **Partial refactor.** Replay only the columns whose reach includes a
  changed A entry. Pairs with bypass ([04](04-newton-and-timestep.md) 4.1):
  on the 4k chain 0.5% of MOSFETs are active per evaluation, so most
  columns' values do not move. Census first: changed-entry fraction and the
  reach size per iterate on the chains and the adder.
- **Probe scan.** `replayHolds` scans all of L after every successful
  replay (`src/solver/direct.zig:318-363`). The refactor kernel already
  computes each column's max; carry the running max multiplier out of it
  and the scan becomes O(1). Measure its share first (the NaN scan it
  replaced cost 0.4-3.6% of a replay, `src/solver/direct.zig:323`).
- **Supernodal DFS, or the reach sorted by pivot step** [fp]. The DFS is the
  larger half of the 100x100 grid factor after the panels
  (`solver-perf-2026-09.md`, "Not done"). Meshed supplies are the
  post-layout case where it matters.
- **`FreqSolver.factorEach` allocations.** It frees and reallocates its
  LaneLu groups (five buffers each) on every call: once per HB Newton step
  and once per PAC input frequency (`src/solver/freq_solve.zig:841-851`).
  Keep capacity.
  **Status (2026-10-23): done** (with 09 9.4(b)): `replayHeldNumbers` and
  `factorEachComplex` reuse their stacked and complex groups while the
  tape's L/U lengths hold, and `held_id` keeps its capacity; a group is
  reallocated only when a new tape changes those lengths (a re-pivot).
- **Thomas with 2x2 block pivots** (`src/solver/direct.zig:272`). A ladder
  with a V source leaves the O(n) tridiagonal path today.
- **Dense LU 2-D register tiling** (`src/solver/dense_lu.zig:11`). Rank-8
  panels are the only blocking; the HB deck profiled spent 95% of its time
  here before them (36.7 → 12.8 ms at n = 640).

### 2.9 Solves

**Status (2026-10-22), host side: nothing pays.** Solves are 4-5% of the
chain/ring runs; the subtree lanes could split the U back-solve bitwise
(not the forward one), worth about 1%. No host caller batches
right-hand sides (sens, tf, xf solve one each).

A single right-hand side is span-bound: on circuit matrices a GPU solve lost
3.16x to one core (Chen, TPDS 2015), and the host solve is scalar and
single-threaded (`src/solver/sparse_lu.zig:1140-1224`). The device solve
stays on the device only because the factors live there, and its sync-free
heads landed (#141: 80 → 21 ms on the 100k chain). The remaining lever is
batching right-hand sides: sweep lanes ([06](06-sweeps-and-lanes.md)),
adjoint and sensitivity columns ([09](09-frequency-and-periodic.md)).
No single-circuit item.

### 2.10 A complex sparse LU for the AC family — P1 [freq]

**Today.** `FreqSolver` solves the real 2n system `[G, -ωC; ωC, G]`: a
stacked pattern with 4·nnz entries and its own BTF + AMD, dense below
n ≤ 16 (`src/solver/freq_solve.zig:173-223`). Every ω factors; the bypass
copy is dropped; LaneLu(4) replays W frequencies per pass.
**Cost.** On `sweep_opamp_wl_5000` the AC solve was about 59% of the run
(refactor 22.6%, lanes 20.3%, the pivot refresh 16%; *measured*,
`solver-perf-2026-09.md`, before the refresh fix).
**Change.** A complex-valued `SparseLu` on the pattern of G ∪ C: half the
index traffic (one index per complex entry instead of per real entry) and
a complex multiply-add is 4 real multiplies against the 2x2 block's 8.
Pivoting on |z|. The same kernel body serves LaneLu(W) over ω and the GPU
frequency lanes.
**Expect.** Up to 2x on the AC refactor and lane replay (estimate), so about
a quarter of that deck. **Gate.** AC corpus within tolerance (the bits
change; it is a new factorization, not a replay of the real one).
**Effort.** 2 weeks.
**Status (2026-10-22): done on the CPU**, see 09 9.1.

## Closed, with the reason

| Idea | Verdict | Where |
|---|---|---|
| GMRES on a stale LU inside Newton | more instructions than a refactor on 12 of 13 decks; conformance | `gpu-convergence.md` §10.4 |
| f32 host refactor with GMRES-IR (`--lu-fast`) | lost 2-32% on every deck; deleted | `gpu-lu.md` "fast_mode" |
| Level-scheduled host refactor | 1.3-2.1x upper bound at 8 threads; replaced by sync-free tickets | `solver-perf-2026-09.md` |
| LaneLu(8) | Ir -22%, wall 3-7%; ragged tails | `solver-perf-2026-09.md` |
| Right-looking GLU kernel | float atomics reorder the sum; Newton did not converge on a non-deterministic stamp | `gpu-lu.md` summary |
| cuDSS, cusolverRf, rocSOLVER | gompute-only policy; two sets of numerics would fork a peel | `gpu-lu.md` §3(c) |
| Dense LU for tiny systems (n ≤ 25) | 2-4x slower than sparse replay | `solver-perf-2026-09.md` |

## Docs that disagree with the code

Fix when the `dev/` pages move: `overview.md` and the solver README say
every solve runs on the host (the device LU, lane LUs and device GMRES
exist); `structured-preconditioners.md` says there is no spectral
preconditioner (HB, QPSS and PAC use the block-diagonal one,
`src/solver/freq_solve.zig:746-862`); `lptv-block-solves.md` says PAC is
dense (only below n·n_sb = 64); `amd-ordering.md` says there is no natural
order (`src/solver/order.zig:209-216` picks it per block); `gpu-lu.md` calls
the multicore refactor on by default and describes a peel hysteresis the
code does not have.
