# Scheduling and parallelism

How independent work is spread over the CPU cores and the GPU today, and
what would let more of it run at once. Anchors are `src/...` paths.

## What ships

| Piece | State | Anchor |
|---|---|---|
| Query DAG: one row per query plus its implicit OP, keyed by (template, tolerances with variant and temperature, TRANOP, warm start); only OP edges | default | `src/analysis/session.zig:310-349, 1250-1304` |
| `run_all`: a pool of `max_parallel` workers, whole-query quanta, LIFO urgent order (a point's OP then its analysis), breadth-first under GPU lanes | `--jobs=auto` = logical CPUs | `src/espice.zig:942-952`, `src/analysis/session.zig:807-829, 1141-1177` |
| Refilling pool for sweeps: a point's OP and transient back to back, rows assembled in order | default (1.2.0) | `src/analysis/session.zig:281-298, 674-721` |
| ParEval: device stamp over T − 1 worker lanes with dynamic chunk claims and a deterministic staged reduce; limiting, staging, commit and convergence hooks stay serial | `ESPICE_THREADS`, default 1 | `src/analysis/par_eval.zig:348-385, 537-541, 622-665` |
| Multicore refactor | `ESPICE_SOLVER_THREADS`, default 1 | [02](02-linear-solver.md) 2.1 |
| One GpuContext and one stream per query; one primary context per process; no cross-query GPU lock | explicit `cuda`/`hip`, or `auto` | `src/analysis/executor.zig:871-889`, `src/analysis/gpu/context.zig:173-175, 549-552` |
| CPU lanes (W = 4) over sweep points | off by default (break-even, #211) | `src/analysis/executor.zig:99-107` |
| GPU lanes over sweep points | explicit `cuda`/`hip`; `auto` only when FP64 ≥ FP32/4 | `src/analysis/gpu/lanes.zig:63-80` |

Per query the executor builds its own `Circuit` copy, planes, solver
workspace with its own ordering and first full pivoting factor, constant
baseline, ParEval, GpuContext and GpuLu (`src/analysis/executor.zig:218-248, 724-744`;
`src/analysis/Circuit.zig:455-462`). Dependents never inherit the OP's
workspace (`Circuit.zig:418, 592-593`). Every query runs on its own OS
thread with a 512 MiB stack reservation, never reused
(`src/analysis/worker.zig:60, 132-188`). Every session and executor
allocation goes through one mutex over glibc malloc
(`src/espice.zig:134-135, 1529-1561`).

## Items

### 5.1 One thread budget — P0

**Today.** `--jobs=auto` is one query per logical CPU (32 here);
`ESPICE_THREADS = T` gives every running query T − 1 spinning ParEval lanes
plus min(T, 16) LU tasks, and nothing caps jobs × T
(`src/main.zig:144, 171-172`, `src/analysis/executor.zig:235`). So the
defaults leave a single big transient on one core, and a user who sets
`ESPICE_THREADS=8` with a sweep oversubscribes 8x.
**Change.** A process budget B = physical cores. The scheduler hands each
running query a share: with k queries ready, each gets max(1, B / k) lanes
for ParEval and the LU, re-evaluated when the ready count changes (a query
reads its allotment at its next step boundary; ParEval's dynamic chunk
claims and the LU's per-epoch admission already tolerate a changing lane
count). One big transient then gets the whole machine; 32 sweep points get
one core each. **Expect.** The measured per-query gains at default
settings: ParEval 1.5-3.3x on PSP and BSIM4 eval (`src/analysis/gpu/context.zig:152-157`),
the multicore refactor 1.07-2.14x on thick factors, with no oversubscription
on sweeps. **Effort.** 2-3 days.

### 5.2 P-cores for the numerics — P1

*Measured* (`dev/devices/gpu-evaluation.md`, "Eight host threads"): four
ParEval lanes on P-cores beat four with three on E-cores by 1.35-1.38x; a
worker lane on an E-core stamps PSP103 about 2x slower; unpinned, 4 and 8
threads tie because the scheduler fills E-cores. On this i9-14900HX the
P-cores are CPUs 0-15 (two per core), E-cores 16-31.
**Change.** Read the topology once (`/sys/devices/system/cpu/cpu*/topology`,
or `lscpu`'s cache sharing), pin ParEval and LU lanes to P-cores, and count
E-cores at 0.6 of a P-core in the budget of 5.1. Queries' coordinator,
output encoding and `.meas` streaming can take E-cores. The static weight
split (`count · n_u²`, 16x for nonlinear) was replaced by dynamic claims
(#138), so stragglers are already bounded. **Effort.** 1-2 days.
**Gate.** None needed (scheduling only); measure eval time per thread
placement as the page above did.

### 5.3 CPU and GPU at the same time — P1

**Today.** One backend policy per Problem. Under `cuda`, a query's resident
batches run on the device and its host batches stamp *after* the plane
download instead of beside it (`src/analysis/gpu/context.zig:1232-1234`),
and the CPU cores sit idle while the device evaluates. Under GPU lanes,
every eligible transient point joins the device batch and the CPU workers
only run OPs and ineligible points.
**Change.** Two forms. (a) Single circuit: launch the device eval, stamp the
host batches on the CPU meanwhile, then add on the downloaded planes (the
add order is fixed, so the sum stays the host's). Free for sources; it
hides a heavy ineligible batch's eval time. (b) Sweeps: the refilling pool
feeds both a GPU lane batch and the CPU workers from one queue, each taking
points as it frees slots. On this card the GPU adds +60-80% to the 24-core
CPU (`dev/research/lane-batched-simulation.md` §9, estimate); on an
FP64-class card the CPU share is small but still free. Points must stay
bitwise their scalar run on each backend, which they are (per-lane control
is exact), so the mix changes nothing but order.
**Effort.** (a) 1 day; (b) 1 week, after [06](06-sweeps-and-lanes.md) 6.1.

**Status (2026-10-22): (a) measured, not done.** The host batches' stamp
after the download costs 16-44 us per eval on chain_bsim4_10k (sources
only; ~30 ms of a 6-10 s run) and 0.6-0.9 us on
parallel_inverters_2000. Stamping into private planes beside the device
eval needs a clear of those planes per eval (or a touched-cell list) and
an add after the download, which is not bitwise where two host batches
hit one cell (two current sources on a node: d + (i1 + i2) against
(d + i1) + i2). It pays only beside a heavy ineligible batch (runtime
`.hdl` models before 01 1.8).

### 5.4 Several GPU queries in one process — P1, measure first

**Today.** Each query owns a GpuContext and a stream on the one primary
context; nothing serializes them. Separate *processes* time-slice the GPU
badly: a latency-bound deck took 3.4 s alone, 16.3 s each with two copies
and 32.5 s each with four (`gpu-evaluation.md`, "Checking the GPU path"),
which is why `test-gpu` runs `--jobs 1`. In-process concurrency has not
been measured ("No measurement of several in-process GpuContexts").
**Change.** Measure k = 1, 2, 4, 8 concurrent GPU queries of the
latency-bound and of the throughput-bound kind in one process. Streams on
one context should interleave without a context switch; if they do, the
only policy needed is `auto`'s existing refusal of latency-bound decks. If
the driver serializes them anyway, add a GPU admission semaphore (k
concurrent device queries) and send the rest to the CPU workers. Also:
AC-family queries never qualify for device evaluation under `auto`
(`expectedEvals` = 1 < 20, `src/analysis/executor.zig:935-947`), which is
correct for one linearization and irrelevant to the lane path.
**Effort.** 1 day to measure.

**Status (2026-10-22): measured; no limit needed.** One process,
`--backend cuda`, lanes off, k `.step` points as k queries (ReleaseFast,
RTX 4060 Laptop): parallel_inverters_2000 (latency-bound) k = 2, 4, 8 at
`--jobs 1` 1.77, 2.86, 3.74 s against `--jobs k` 0.96, 1.05, 1.15 s;
chain_bsim4_10k (throughput-bound) k = 4: 23.7 s against 8.5 s. Streams on
the one primary context interleave, so no admission semaphore.

### 5.5 Multiple GPUs — P2

Nothing selects a device today; gompute has `initDevice`. Executors could
round-robin devices (one primary context each), and a lane batch could
shard its lanes across devices, since lanes share nothing but the tape and
the model rows. No second GPU is on this machine, so this is design only.

### 5.6 Share per-template setup across queries — P1

**Today.** Every query, including each of 10,000 sweep points, computes its
own BTF + AMD ordering, runs its own first full pivoting factor, builds its
own baseline and uploads its own device state. The ordering alone is 95M Ir
per call on the 100k ladder (1.5-2% of that run, twice: OP and transient);
`set_params` costs 6.1-6.3 ms per point against 85.5-90 ms before the
identity fast path (`dev/analysis/variants.md`), and the serial lane
materialize is marked as taking seconds at thousands of lanes
(`src/analysis/executor.zig:502-503`).
**Change.** Hang the ordering, the symbolic structure and the last pivot
tape on the prepared pattern (template), not the query: a dependent clones
the OP's factor as the CPU lanes already share one tape; a GpuContext per
template uploads the model rows, tapes and pattern once and gives each
query its own instance rows and planes (the lane backend already
deduplicates model rows by bytes). Measure the setup share first with
`--timing-in-depth` per query on a 1,000-point sweep.
**Expect.** Per-point setup drops to the instance rows and a tape replay
(estimate); the GPU upload per point drops from everything to the rows that
differ. **Effort.** 1-2 weeks.

### 5.7 Worker threads and stacks — P1

One OS thread with a 512 MiB virtual stack per query, never reused
(`src/analysis/worker.zig:60, 76-77`). Thread creation plus the mmap and
the first page faults cost tens of microseconds per query; at 10k points
that is noise next to the solve, but it is the kind of cost a points/s
benchmark exposes. A pool of reusable workers with the same stack size is a
small change; the retained-stack design (paused queries keep their stack)
stays.

### 5.8 The allocator mutex — P1, measure first

All session and executor allocations take one mutex over glibc malloc
(`src/espice.zig:134-135, 1529-1561`). Each query has arenas, so steady
state should be quiet, but `perf lock` or `perf record -e lock:contention`
on a 32-job sweep has not been run. If it shows, give each worker its own
arena page source (`std.heap.smp_allocator` is already used by
`ProtoStore`).

### 5.9 Delivery blocks the pool — P0, small

Output delivery and `.meas` printing run between the coordinator's wait and
its next fill (`src/espice.zig:946-952, 1451-1507`), so slots freed by
finished queries idle while a plot is encoded and written, and in-order
delivery makes a fast later query wait behind an earlier one
(`:1463-1472`). Fill first, then deliver, or hand delivery to one output
thread. **Effort.** 0.5 day.

### 5.10 Checkpoint quanta — P1

A `.checkpoint` quantum parks the worker at every accepted step: two thread
handoffs per step, measured at 2-11x slower on small transients
(`dev/Problem/3)Parallel Query Execution.md`). `run_all` already uses
`.completion`; `--timing-in-depth` and C API callers of `advance_ready`
with the default quantum pay it. Make the checkpoint coarser (every k steps
or every few milliseconds of wall time) and say so in the API docs.

### 5.11 More edges in the query DAG — P1

Only OP prerequisites are graph nodes (`dev/Problem/migration-notes.md`).
`.four` and `.fft` re-run the deck's `.tran` (`src/frontend/analyses.zig:1504`,
`src/analysis/post/four.zig:118`, `post/fft.zig:75`): a `.tran` plus `.four`
deck integrates twice. Add a "reads the result of" edge for post analyses
whose transient card matches, and later for PAC/PXF/pnoise sharing one
PSS (that needs the state and trajectory contract the migration notes
ask for).

### 5.12 Overlap process startup — P1, small

Driver setup is about 350 ms once per process (cuInit, the context, module
loads), and `auto` prices the query before the first driver call
(`gpu-evaluation.md`, "Choosing the backend"). When the deck's size says the
GPU will be admitted, start the driver on a thread while the frontend
parses and binds (`gompute` OVERHEAD.md "Startup" suggests the same). On
c7552-class decks setup takes 0.7-10 s on the host, so the whole driver
start hides behind it. **Effort.** 0.5 day.

**Status (2026-10-22): done.** `Problem.init` starts the driver and the
primary context on a thread (`gpu.Driver.prestart`) before the parse when
the backend is `cuda`/`hip`, or `auto` with a deck file of at least
1 MiB; `auto` then prices the setup at zero. chain_bsim4_10k under
`auto`: 6.6 → 6.2 s (medians of 3). Below 1 MiB `auto` mostly keeps the
CPU, and a started context costs that run its teardown
(parallel_inverters_2000: 0.6 → 1.0 s at a 128 KiB bar), hence the bar.

### 5.13 Lane batches on one worker — note

A CPU lane batch of W points runs on one worker (`src/analysis/session.zig:517-552`),
which is acceptable only because CPU lanes are off by default
(`dev/research/plans/README.md`, risk 6). If they come back on (an AVX-512
build, or cheap-device decks), the budget of 5.1 must count a W-lane batch
as one worker's work, not W.
