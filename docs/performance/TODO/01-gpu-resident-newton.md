# One circuit on the GPU: toward a resident Newton step

The single-circuit GPU path (`--backend cuda`, `hip`, or `auto` when the
cost model admits it) evaluates devices on the device and does everything
else on the host, with a round trip per Newton iteration. This page lists
what would move the rest of the step onto the device and how much each
move buys. Anchors are `src/...` paths. The lane backend
([06](06-sweeps-and-lanes.md)) already runs a whole Newton tick on the
device for N lanes; several of its kernels are the bodies this page needs
at N = 1 with rows instead of lanes as the parallel axis.

## The step today

Per device evaluation (`src/analysis/gpu/context.zig:1291-1338`), *measured*
in µs (`dev/devices/gpu-evaluation.md`, `ESPICE_GPU_STATS=phases`):

| Stage | chain_bsim4_10k | c7552_sky130 | chain_bsim4_100k | E1 bsimcmg, N = 1024 |
|---|---:|---:|---:|---:|
| upload x (8n B) + clear staging | 241 | 391 | 2,892 | 1,032 |
| eval kernels, one launch per resident batch | 1,513 | 2,058 | 11,096 | 10,353 |
| reduce (two launches) | 213 | 419 | 2,416 | 1,646 |
| download planes (16·nnz + 16·n B) | 321 | 944 | 3,155 | 1,980 |

Then the host: ground pin, non-resident batches stamped *after* the download
(`context.zig:1232-1235`), `G + ag0·C`, companion rows, factor and solve
(host, or the device LU with A uploaded again), update, limit kernel,
gates. Per converged solve one more state-half launch and wait, the state
query wait, and one or two charge evaluations (there is no charge-only
kernel: `src/analysis/Circuit.zig:865-875`). Syncs per accepted step were
5.3-6.2 after the fused waits, before #15 added the state-half launch
(unmeasured since). The floor is 25-60 µs per iterate whatever the circuit
size: eight dependent stream operations at ~15 µs per wait, which is why a
4-diode, million-step deck runs 40-80x slower on the device
(`stress/vacask_graetz` 2.9 s CPU, 184 s cuda) and `auto` keeps it on the
CPU. No CUDA graph is captured on this path (only the lanes capture), four
copies still run synchronously on the default stream, no constant baseline
exists on the device (resident R and C restamp every eval,
`Circuit.zig:940-947`), and the eval kernel always uses the wide derivative
basis (`src/device/eval.zig:2694-2697`).

Everything below is **exact** (same arithmetic in the same order; raw files
bitwise equal to the unfused path) unless marked.

## Items

### 1.1 One sync per Newton iteration, and a sync census — P0

**Today.** The `evalFollows` prefetch already fuses the limit pass with the
next eval when the next eval is certain (`src/solver/converger.zig:481-483`,
`context.zig:1244-1263`). Still separate waits: the state-half launch per
converged solve (#15), the `stateCtl(.query)` before `evalQ`, the charge
eval itself, and a second charge eval under stateful charge
(`Circuit.zig:1169-1175`, `context.zig:1617-1672`).
**Change.** `dev/solvers/gpu-convergence.md` §2.1: queue the state half,
the query, the charge eval and the tape kernel as one chain with one wait;
they act at the same x. Add the census first: syncs and bytes per Newton
iteration and per accepted step in `ESPICE_GPU_STATS`, which no run has
recorded since #15. **Expect.** I + 1 syncs per step (from 2I + 4 before
the fusion, ~I + 3 now). Each sync is ~15 µs: 10-15% of a bsim4 iteration
at 2,000 instances, most of a mos1 iteration. **Effort.** 2-3 days.

**Status (2026-10-22): done.** `ESPICE_GPU_STATS` prints a `gpu-census`
line (Newton and charge evals, steps, syncs and KB each way per Newton
eval and per step) and the waits by cause. The state half no longer
launches for a batch with nothing to stage (limit-only models), and
`GpuContext.acceptStates` queues the accepted point's charges, the state
halves and the query on one wait, keeping the query's answer for the
`stateCtl(.query)` that follows. Syncs per step, ReleaseFast: chain_bsim4_10k
7.06 → 5.03, parallel_inverters_2000 7.31 → 5.29 (lim ≈ I + 1, two charge
passes: the accepted point and the stateful-charge rebase). Bitwise.

### 1.2 Resident planes, assembly and companion on the device (stages 2 and 3) — P0

[02](02-linear-solver.md) 2.3 covers the assembly into the device LU's
slots and the overlay for host batches. The companion rows
(`src/analysis/tran/integrator.zig:239-249`) need the charge history, which
the lane backend already keeps on the device (`d_ring`, `d_ip`,
`companionRow` in `src/solver/kernels/lane.zig`); at N = 1 that is one
thread per row. With both, a Newton iteration moves x up and dx, the
residual and diag(A) down (3n + n doubles) instead of 16·nnz + 16·n + 8·nnz.
**Expect.** At chain_bsim4_100k the download alone is 16% of an eval; at
n = 10^6 the bytes dominate the kernels on an FP64-class card
(`gpu-lu.md` §4.4). **Gate.** Bitwise against the host companion
(`ESPICE_GPU_LU_CHECK` extends to A). **Effort.** 1 week after 2.3.

**Status (2026-10-22): not started; waits on 02 2.3.** The census puts
the planes download at 5.3 MB per Newton eval on chain_bsim4_10k (x up
1.9 MB). The device side needs the factor's slot map from gpu/lu.zig;
`d_planes` and `Layout` in context.zig are what a scatter would read.

**Status (2026-10-23): measured, not worth it now.** With 02 2.3 in, the
only bytes 1.2 still saves are q (n doubles, ~360 KB / ~40 us per eval on
chain_bsim4_10k against a 15-20 ms iteration, <=0.3%); the residual gate
reads rhs every iteration, so rhs stays. On the chain decks the device LU
declines in transient and A comes down for the host factor (most of the
3.7 MB per eval); on logic_bsim4_10k the device LU is ~19 s of 24.7 s.

### 1.3 Newton gates on the device and k iterations per launch (stage 4) — P1

**Today.** `finalizeStep` runs on the host after every solve: the delta
norm, the plausible bound, the residual gate (off in transient), the
device-convergence flag (`converger.zig:794-819`). The lane backend has
these as kernels (`arp_lane_update`, `gates`, `finalize`,
`src/solver/lane_device.zig:164-225`), one thread per lane looping over n
rows, which is serial at N = 1.
**Change.** Row-parallel forms: the delta norm and the plausible bound are
maxima, so a block max tree is exact in any order; the device-convergence
flag is an OR. Then unroll k iterations into one graph with early-exit
nodes (every later node checks a `done` word), k from the deck's measured
iterations per solve (2.0-3.9). HIP has no conditional graph nodes, so the
unrolled form is the portable one (`FEATURE.md` item 5); CUDA 12.4's while
node is a later experiment. Preconditions: every batch resident (all 64
built-in models have images; only runtime `.hdl` lacks one, 1.8), the
device LU resident (2.3), the companion on the device (1.2), and a peel
inside the k iterations stopping the rest with the host resuming at that
iteration.
**Expect.** Syncs per step from I + 1 to about 1 per k iterations plus one
per accepted step; the 15 µs floor stops mattering for decks whose kernels
take 100 µs-2 ms (1k-10k instances), where waits are 15-40% of the
iteration today. A 4-diode deck still loses: its kernel is 19 µs against a
microsecond on the host. **Gate.** Bitwise against the host gates (max is
order-free). **Effort.** 2 weeks.

**Status (2026-10-22): not started.** Its preconditions (device LU
resident, 2.3; companion on the device, 1.2) are not in the tree. The
`../ESPice-gpu-newton-resident` experiment (CUDA conditional WHILE/IF
nodes around the lanes' A/B graphs, `cuda_newton_graph.zig`,
`newton_control*.zig`) was not ported: it targets the lane loop
(gpu/lanes.zig), its only logged build failed (a function named
`resume`, renamed since) and it was never run on a GPU;
its driver bindings (`CUgraphNodeParams` 256 B, conditional payload at 16)
are the piece worth taking when this item starts.

**Status (2026-10-23): measured, not worth it now.** Where the host
factors (chain decks, device LU declined) a device Newton cannot run;
where the device LU runs (logic_bsim4_10k) waits are ~5 per step at
~15 us against ~40 ms of LU per step (~0.2%). Revisit for decks whose
kernels take 100 us-2 ms with the device LU admitted.

### 1.4 Graph capture of the eval chain — P1

No graph is captured on the single-circuit path; the lanes capture two per
tick. The eval kernels take `SimState` and `limiting` by value, so their
node parameters change every iteration: re-capture per iteration on the
host and `cuGraphExecUpdate` in place, or move those scalars into a device
buffer (a VerA ABI change at the frozen boundary). Removes the ~20 µs of
driver enqueue per iterate measured on mosamp, not the wait itself; the
eval chain's 8 submissions become one. Price re-capture against plain
launches first (`gpu-lu.md` E2, launch strategy). **Effort.** 2-3 days.

**Status (2026-10-22): measured, not worth it now.** Host enqueue per
eval, ReleaseFast: 7.8 us on parallel_inverters_2000 (2,485 evals: 19 ms
of a 0.8 s run, the most a graph could remove), 50-100 us on
chain_bsim4_10k, most of it the 400 KB memcpy of x into pinned memory,
which a graph does not remove. Revisit with 1.3.

### 1.5 Pinned mirrors, asynchronous copies, no default-stream traffic — P0, small

Four copies run synchronously on the legacy default stream and so fence
the whole device: the host-stepped batch push and pull
(`context.zig:1771-1810`), the seeded lim upload (`:1496`), the parameter
re-upload (`:1887-1903`) and the tape permutation upload (`:1016-1018`).
The host-stepped push compares that batch's whole host Instance column
against its last-pushed copy twice per eval (`:1248`, `:1293`, `:1797`), and
a wline_4 Instance carries about 2 MB of inline history
(`src/device/eval.zig:1087-1088`). Pinned mirrors plus async copies on the
context's stream, and a dirty flag per stepped Instance in place of the
byte compare (the byte-granular merge is `context.zig:1917-1919`).
gompute has `allocPinned` and `copyFromAsync`. **Effort.** 1 day.

**Status (2026-10-22): done, bitwise.** Every copy after `init` goes on
the context's stream (`GpuContext.upload`/`download`), the tape
permutation upload included, and the stepped Instance scratch is pinned,
so a pull queues all batches and waits once. The byte compare stays
(ponytail: a dirty flag needs a host hook; no measured deck has a stepped
batch).

### 1.6 Constant batches: a device baseline and no staging for them — P1

**Today.** On the host, batches whose generated code declares a constant
Jacobian (resistor, capacitor, inductor, vcvs, vccs, cccs, bsource_q,
kinduc) put G and C into a baseline computed once and skip those stamps in
Newton (`eval.zig:2284-2293`, `Circuit.zig:958-989`); their residual and
charge are still evaluated. On the device there is no baseline: every
resident batch restamps everything, and every contribution takes a
staging cell (8 B × (pattern entries + n_u), doubled with charge). On a
post-layout deck the million R and C own most of the staging cells, and the
clear plus reduce is 27% of the eval at chain_bsim4_100k (2,892 + 2,416 of
19,559 µs).
**Change.** Stamp the constant batches' G and C once into a device
baseline, exclude them from the staging permutation, and start each eval's
reduce from the baseline instead of zero (the host already adds "the
constant baseline for `.newton`" as the initial value in ParEval's reduce,
`src/analysis/par_eval.zig`, so the order is defined). Their residual
`G_lin·x` and charge `C_lin·x` become one SpMV kernel in a fixed row order,
Xyce's `SEPARATELOAD` form (`xyce-research.md` candidate 1). Sources stay
per instance (no constant declaration: `vsource`, `isource`).
**Expect.** Clear and reduce shrink with the staging cells they lose, up to
the 27% above on RC-heavy decks (estimate). **Gate.** The residual sum
order changes for those rows: tolerance class, not bitwise. **Effort.** 1 week.

**Status (2026-10-22): done, automatic.** A resident batch with
`has_const_jacobian` whose image has `arp_evalc_<model>` (the eval kernel
without g/c stamps) leaves the slot staging; its g and c sit in a
baseline region behind the stage (`GpuContext.uploadBase`, host-stamped
once and after every parameter repack), reduced as one piece ahead of
each cell's run. chain_bsim4_10k: stage 3.62 M → 2.78 M cells. Off under
`ESPICE_GPU_NOFUSE` (A/B). Tolerance: rawdiff worst 6.1e-12 V on
chain_bsim4_10k (with 04 4.4); GPU corpus tran/ and stress/ pass.

### 1.7 Cold JIT: ship SASS, warm in the background — P0

**Today.** Images are PTX, JIT-compiled by the driver on first use and
cached in `~/.nv/ComputeCache` (cap raised to 4 GiB): bsim4va 15 s, psp103
35 s, hisimhv 437-1,001 s and 13 GB of RAM cold. Under `auto` an image over
1 MB that the driver has not compiled yet sends the whole query to the CPU
(`ImageNotCompiled`, `context.zig:789-796`), so the first run of each model
never uses the GPU. The driver cache also keys on the driver version
(`context.zig:255`).
**Change.** Build SASS for the release's `-Dcuda-arch` beside the PTX (a
fatbin; `ptxas` already runs at build time for the frame measurements and
takes 39-181 s and 2-11 GB per model, which is CI time, not run time), so a
matching card loads without a JIT. For other cards, when `auto` declines
with `ImageNotCompiled`, spawn a detached process that compiles the deck's
images into the cache while the deck runs on the CPU, so the second run
takes the GPU. **Effort.** 2-3 days.

### 1.8 Runtime `.hdl` models on the device — P2

A loaded `.hdl` model gets a GPU payload when eligible but no kernel image,
because images are embedded at build time (`context.zig:785-788`,
`gpu-evaluation.md`). The `.hdl` build already runs VerA to a host `.so`;
add the NVPTX kernel root to that build and `cuModuleLoadData` the PTX at
load time (gompute's `loadModuleFromMemory`), cached under the `.hdl` cache
key. Then a deck mixing built-in and user models no longer keeps one batch
on the host.

### 1.9 The `auto` cost model — P1

**Today** (`context.zig:897-964`, constants at `:139-166`): decline below 20
expected evals; host price = best of three serial evals at a spread x over
`1 + 0.1·(threads − 1)`; device price = 40 µs + bus bytes at 11 GB/s +
staging bytes at 100 GB/s + 0.12 × host time; the GPU must win 2x per eval
and 1.3x over the query including 350 ms of driver setup. Expected evals
are a guess: 30 per OP, 3 per DC point, 4 per transient step, 1 per
small-signal analysis (`src/analysis/executor.zig:933-947`). Known misses:
`scaling_rc_ladder_100k` ran slower on the device than its eval times
predicted (the reduce), bsimcmg's kernel ratio is 0.07 not 0.12.
**Change.** (a) A per-model kernel ratio measured once per card and cached
beside the JIT marker (`~/.cache/espice/gpu-cost/<card>`); (b) re-price
after the first 20 evals of a query from the measured eval and wait times
and demote to the host when the device is losing, using the existing
fault-fallback path ("after a fault the rest of the query runs on the
host"); (c) price the staging clear and reduce from the cell count, not a
flat bandwidth. **Gate.** No corpus deck where `auto` picks a slower path
(`zig build bench` with `--backend auto` against `cpu`). **Effort.** 3 days.

**Status (2026-10-22): (b) done, (c) in part, (a) not done.** (b) Under
`auto` (`GpuContext.Audit`), wall time inside the evaluating entry points
per device eval, evals 10-40, is held to the host price `Cost.admit`
measured; a device that costs more brings its state home
(`syncHostState`) and finishes the query on the host through the fault
fallback, quietly (chain_bsim4_10k: 1,532 us against 6,325, stays; a
forced demotion completes within 3.9e-12 V of the CPU run). (c) The staging
price leaves out constant batches' slots (1.6). A started driver
(05 5.12) prices the setup at zero. (a) needs a per-card cache file; the
audit covers the cases it would fix.

**Status (2026-10-23): (a) done (exact on the CPU path).** `CostCache`
in gpu/context.zig keeps each audited query's measured host and device
microseconds per eval in `~/.cache/espice/gpu-cost/<key>` (key: card,
circuit shape and models, kernel images, threads, expected evals). The
next run prices from it without the 2x safety ratio (the query margin and
setup stay) and declines a circuit the device lost without starting the
driver. Only audited (admitted) queries write records; a priced decline
still probes. Measured: chain_bsim4_10k auto 5.30 -> 4.99 s (probe
skipped; one run each). Not fixed: scaling_inverter_chain_4k, which `auto`
declines (priced 250 vs 173 us) though cuda runs 2.8 s against 3.9 s on
the CPU: no record exists for a declined circuit.

### 1.10 The staging clear — verify

With one staging cell per contribution and every resident instance writing
all of its cells every eval, the device memset before the eval
(`context.zig`, "clear staging") may be redundant: the reduce reads only
cells the ranges name. Check what the clear protects (the trash tail, cells
of a batch that fell back to the host, bypass later) and drop it where it is
not needed; it is 10-15% of an eval at 100k. If bypass (04 4.1) lands,
latent instances keep their cells, which also needs no clear.

**Status (2026-10-22): done, bitwise.** The device `Sink` stores each
staging cell (each is written once per pass, the property the host's
staged ParEval positions rely on) and the reduce starts from +0.0, so
`+0.0 + v` sums exactly as `+0.0 + (0 + v)` did; the stage is cleared at
`init` and after a parameter repack only. chain_bsim4_10k, phases: upload
+ clear 178 → 45 us per eval; byte-identical raw with
`ESPICE_GPU_NOFUSE` against the old binary.

### 1.11 Several GPU queries and CPU work beside the device — see [05](05-scheduling-and-parallelism.md)

5.3 (host batches stamped during the device eval; the sweep pool feeding
CPU and GPU together), 5.4 (in-process GPU concurrency), 5.12 (driver start
overlapping the parse).

## Device-side costs this page does not fix

The kernels themselves: 255 registers, a 20.8 KB frame per thread, 17%
occupancy, spills to DRAM (128 MB of stack at full residency against a
32 MB L2), FP64 at 1/64 on this card. Those are
[03](03-device-evaluation.md) 3.1-3.3. On an FP64-class card the
kernels shrink 4-69x (projected) and every item above grows in share.
