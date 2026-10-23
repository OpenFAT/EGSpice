# Many points: sweeps and lanes

Ten thousand to a hundred thousand independent analysis points of one
circuit (sizing sweeps, Monte Carlo, corners, `.data`), as CPU workers and
as GPU lanes. `docs/performance/gpu-batching.md` is the plan in dependency
order and `FEATURE.md` (repo root, 2026-10-21) its verified review; this page
condenses both, adds what the code surveys found, and ranks. Anchors are
`src/...` paths.

## What ships

- Every sweep point is its own query with its own implicit OP; the refilling
  pool runs a point's OP and transient back to back; `--jobs=auto`;
  measures-only, early stop and warm start turn on when nothing reads the
  waveforms (1.2.0). Points API: `espice_append_points`,
  `espice_copy_point_measures`.
- CPU lanes (W = 4) over OP, `.dc`, `.tran`, `.pss` and the AC family,
  bitwise the unbatched run; off by default (break-even, #211).
- GPU transient lanes (`src/analysis/gpu/lanes.zig`): one thread per lane,
  lane-major (`i·N + l`), each thread walking every instance of a batch in
  tape order into its own planes (no staging, no reduce), model rows
  deduplicated by bytes, two captured graphs per Newton tick, host step
  control per lane. On for explicit `cuda`/`hip`; under `auto` only when
  FP64 ≥ FP32/4. GPU AC-family, shooting PSS and HB lanes beside it.

## Where a round goes at 10,000 lanes

Verified in `FEATURE.md` from `lanes.zig:767-1150` and `tran.zig:1714-1860`:

| Phase | Host, one thread | Device fences |
|---|---|---|
| begin | `Sim.begin` per lane | 0 |
| predict | | 4 synchronous uploads |
| newton | | 4 uploads, 2 download-and-sync per tick, 1 download at the end |
| probe | probe scatter per lane | 2 uploads, 3 async downloads, 1 sync |
| end | `Sim.end` per lane, waveform, measures | 0 |
| apply | | 6 uploads, 1 sync |

The uploads fence because they are synchronous copies from pageable memory
on the legacy stream (`lanes.zig:769-772, 980-985, 1124-1141`); two async
downloads land in stack variables (`:990-992, 1003-1005`), against
gompute's pinned-memory requirement. Estimates (E1's kernel rate, ~1 µs of
host control per lane per pass):

| Lane circuit | device per tick, RTX 4060 | host per round | host share |
|---|---|---|---|
| one transistor | 1-2 ms | 20-40 ms | over 90% |
| 48-transistor chain | ~90 ms | 20-40 ms | 20-30% |
| either, H100 class | 7-69x smaller | unchanged | dominant |

Host RAM caps N before device RAM: each lane materializes a whole circuit
(every batch copied), a run and step machine (~20 n-vectors), the matrix
copy, charge rings and a solver workspace with its own factors, roughly
0.5-1 MB per lane at 1,000 unknowns, materialized serially
(`src/analysis/executor.zig:502-503`). Device side a 1,000-unknown lane is
~300 KB. The probe download carries every deck probe for every lane each
round, not the measures-only subset (`lanes.zig:1091-1105`,
`src/analysis/tran/tran.zig:1761-1762`). Every round runs a separate charge
pass (+44% evaluations, `dev/research/plans/README.md` risk 3). Every
recorded run had zero peels.

Ceiling on this card: E1 puts the 4060 at the work of 12-15 host cores per
instance evaluation, so a perfect device-resident loop lands near one
16-core CPU here. The 10k-100k case pays on FP64-class parts (4-69x
projected), or on consumer cards only if the f32 Jacobian experiment
([03](03-device-evaluation.md) 3.3) works.

## Items, cheapest first

### 6.1 Pinned host mirrors and async copies on the lane stream — P0, exact

Removes about 16 fences per round with no numerical change; also fixes the
stack-variable async downloads. Smallest diff on this page. **Effort.** 1 day.

### 6.2 LTE tape ring lane-major — P0, exact

The ring is the one lane-minor array, so the LTE, charge, accept and rebase
kernels read addresses a whole history block apart per adjacent lane: 32
cache lines per warp load. Lane-major keeps each lane's operation order, so
the bodies stay bitwise their host oracle. LTE cost half of eval in the
recorded trace. **Effort.** 1-2 days.

### 6.3 Download only the probes the point needs — P0, exact

Pass the measures-only probe subset to `runGpuLanes` instead of the deck's
full probe list; under measures-only most sweeps need a handful of columns.
Bytes per round drop from 8 B × probes × N to 8 B × cards × N. **Effort.** 0.5 day.

### 6.4 The two host passes on the pool's threads — P0, exact

`Sim.begin` and `Sim.end` are per-lane independent and write disjoint
slots: run them on the worker threads for roughly a 10x cut of the host
share while 6.5 is built. Two half-batches on two streams would overlap one
half's host pass with the other's Newton, but only after 6.1 (synchronous
uploads fence both streams). **Effort.** 2 days.

### 6.5 Step control on the device — P1, exact

Move `Sim.begin` and `Sim.end` into one control kernel per round, one
thread per lane: LTE decision, step choice, breakpoint landing, accept or
reject, ring rotation, early stop. The scalar bodies exist std-only in
`src/solver/kernels/lane.zig` (`stepBound`, companion coefficients, gates),
which is what makes the CPU lane path the bitwise oracle. The one
non-scalar piece is the breakpoint search, a device hook per batch on the
host today: enumerate each lane's breakpoints once at init into a sorted
table with a cursor. Waveform rows go to a bounded device ring drained per
launch; a reject-at or bound-step request parks that lane with a yield
flag. The host then uploads nothing per round and downloads one counter.
**Effort.** 2-3 weeks (the long pole).

### 6.6 Multi-round graphs with masks — P1, exact

Capture R rounds with the iteration limit unrolled; every kernel already
exits on the active and round masks; the three unconditional memsets and
the counter zeroing need lane-masked clears. A lane that failed its
refactor keeps its matrix until the host peels at the launch boundary.
Waste is launch cost on idle ticks (~14 nodes per tick at a few µs). The
coordination probe measured unrolled chunks faster than a conditional
WHILE graph, and HIP has no conditional nodes; conditional nodes stay a
later experiment (about five optional driver entry points in gompute).

### 6.7 Peel without downloading the world — P1, exact

A gather kernel compacts the failed lanes' A and rhs so the download scales
with failures; for small n a dense partial-pivot LU from a device scratch
pool removes the host from the path. No recorded run peeled, so this is
about not stalling 9,999 peers.

### 6.8 Measures on the device, subset — P1, exact

Port the per-step accumulators (find-at, max, min, avg, rms, integ, when
and trig-targ with linear crossing) of the 2,247-line measure module; other
cards keep the probe ring. Early stop becomes a device flag. With 6.5 the
common sweep is fully resident.

**Status (2026-10-22): done, automatic, exact.** `src/output/scan.zig` holds
the one-row passes `measure.Stream` and the lane kernel `arp_lane_meas`
share; a lane feeds them its accepted solution each round, early stop reads
the round that finished them, and the lanes that leave hand them back
(`Stream.adoptScans`). Applies when every open `.tran` card is a one-row
pass and no measure bound is set. On the RTX 4060 it is neutral (drain -46 /
-27 ms, the extra kernel +57 / 0 ms on 578 / 1,184 lanes); it removes the
per-row host work that dominates once device rounds shrink.

### 6.9 OP and DC lanes on the device — P1

No GPU lane path exists for operating points or `.dc`; the structural sweep
solves points serially per circuit and the DC lanes are CPU-only. The
Newton kernels already serve this case; the missing piece is a DC driver
over them with a CPU fallback for lanes that need gmin or source stepping
(`FEATURE.md`, "DC and OP sweeps"). Until then every GPU transient lane
starts from a CPU operating point.

**Status (2026-10-22): not done, measured not to pay.** A 10k-point OP
sweep of a 98-device deck takes 0.28 ms per point on the CPU pool; the lane
setup alone (`Run.init` + `GpuLanes.init`) measured 0.23 ms per lane before
any tick. Inside a transient sweep the OP is under 2 % of a point.

### 6.10 Bound the host: waves, shared setup, no whole circuit per lane — P1

Materialize one resident wave at a time (N sized to the card), free a
lane's circuit when it finishes, keep only the instance rows the latch
check reads back, and refill freed slots from the pool as the CPU pool
already does. Share the pattern, ordering, pivot tape and model rows per
template ([05](05-scheduling-and-parallelism.md) 5.6) so a lane's
materialization is its instance rows and vectors. Per-field model
overrides ([07](07-frontend-and-preparation.md) 7.1) keep a sized sweep
from carrying one 10-22 KB Model row per (instance, lane): the parasitics
note's bound is 85 GB at 8,192 lanes and 100k elements
(`dev/research/spef-parasitics.md` §3.6). **Effort.** 1-2 weeks.

### 6.11 Occupancy on datacenter parts — P2

At 255 registers the eval kernel holds 8 warps per SM: 10k lanes are 1.6
waves on the 4060 but ~30% of an H100's resident threads, so thread-per-lane
underfills a datacenter part below 30k-70k lanes. The alternative is a
thread per (instance colour, lane) with hub staging (phase-4 G10): colour
instances so no two in a colour share a non-hub cell, with level order
keeping the host's per-cell sum order; hub cells (supply rows) take a small
staging run plus a reduce. The lane `Sink` was written so this mapping
reuses it.

**Status (2026-10-22): done, automatic.** `gpu/lane_color.zig`: level
colours in instance order plus per-touch staging cells for hubs, reduced in
the host's (instance, entry) order, so the planes are bitwise the
thread-per-lane walk (raw byte-identical on four decks). Measured on the RTX
4060: Newton -29 % at 512 lanes, +13 % at 1,335, +40 % at 8,665; the policy
colours below 1,024 lanes on a consumer part and 4,096 on an FP64-class
part. The charge kernel still walks thread-per-lane.

### 6.12 The charge re-read per round — P1

[04](04-newton-and-timestep.md) 4.4: fuse `acceptQ` into the state half so
`q_fresh` lanes skip the extra eval (+44% evaluations on the chain).

**Status (2026-10-22): nothing to do on the lane path.** The lanes already
re-read q with a values-only charge kernel (`LaneChargeKernel`), once per
round, exactly the host's `evalQ` when `!q_fresh`.

### 6.13 CPU and GPU from one pool — P1

[05](05-scheduling-and-parallelism.md) 5.3(b).

**Status (2026-10-22): done, automatic.** Whenever GPU lanes are on, a
batch keeps ceil(f * points) and the CPU workers run the rest beside it,
f = card / (card + host threads), card = 13 x 64 / FP64 ratio host cores
(E1). 10k-point mos6 sweep under `cuda` on this machine: 19.3 s GPU-only,
13.5 s CPU-only, 10.05 s shared.

### 6.14 AC family, PSS and HB lanes resident — P1

[09](09-frequency-and-periodic.md) 9.2-9.4: the frequency lanes return every
solve to the host, the HB sampled evaluation uploads x and downloads f, q,
g and c per call (`lanes.zig:1409-1443`).

**Status (2026-10-22):** 9.2's quantum spans landed in `gpu/freq.zig` (freq
agent); no lane plumbing was needed. Resident factors and HB Newton remain.

### 6.15 Points API residue — P1, small

Per point the C API keeps a formatted label and the variant row until
`deinit` (`src/espice.zig:783-785`); `espice_copy_point_measures`
re-evaluates every card on each call (`src/espice.zig:705-752`), formatting
every number through an exact big-integer conversion even into the discard
writer (`src/output/measure.zig:1710-1778`); `espice_lane_width` always
returns 1 (`src/c_api.zig:660-665`); the C API never sets `final_plan`, so
it gets no state moves, no early circuit release and no streaming
(`src/analysis/executor.zig:222-230`, `src/analysis/session.zig:734-739`).
Store a point's measures once when it finishes, release its label and row
with `release_points`, and expose the final-plan behaviours as
`espice_points_options` fields.

## What to expect

Rough budget for 10k single-transistor transients of 3,500 steps
(`FEATURE.md`, estimates): today ~80 s on the RTX 4060; after 6.1-6.6 ~12 s
on the 4060 and 1-2 s on an H100-class card; a 16-core CPU is comparable to
the 4060 result. Measure first with `ESPICE_GPU_STATS=1` under an explicit
`cuda` backend (the automatic policy keeps lanes on the CPU for this card):
it prints the host, predict, Newton, probe and apply split and the idle
fractions, which confirms the host-bound regime before anything is built.

**Status (2026-10-22): done except stored measures.** A value-only
`evaluate` no longer computes the exact decimal; `espice_lane_width` returns
the executor's batch width; `append_points` marks each point's queries and
OP final, so the transient moves its OP's state and circuits are freed as
nothing reads them (final-plan behaviour, no option). Labels and variant
rows still live until `deinit` (~100 B per point).
