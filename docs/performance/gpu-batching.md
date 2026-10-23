# Batching thousands of analysis points on one GPU

How ESPice runs many independent transient points of one circuit as GPU
lanes, with the host out of the inner loop. The design is in
`src/analysis/gpu/lanes.zig` (device state and rounds),
`src/solver/lane_device.zig` and `src/solver/kernels/lane.zig` (the
model-independent kernels and their std-only bodies), the per-model lane
kernels in `src/device/eval.zig`, and `tran.runGpuLanes` (the host driver).
Measured numbers are from an RTX 4060 Laptop (FP64 at 1/64 of FP32) on Debug
host builds; kernels always compile at `.fast`. Estimates are marked.

## Layout

N lanes stay resident, lane-major: element i of lane l at `i * N + l`, so a
warp holds 32 lanes of one instance. The charge rings and the LTE tape ring
are lane-major too (entry j of tape slot k at `(k * n_qt + j) * N + l`),
which keeps every kernel's loads coalesced. The host mirrors of anything
that moves per round are page-locked, and every per-round copy is an async
copy on the lane stream, so no copy fences the device.

## A round under device step control

Every stepping lane runs one `LK.LaneStep` machine on the device: the
scalars of `tran.Sim` (t, dt, the step history, order, breakpoint state,
statistics) and three bodies that restate `Sim.begin` and `Sim.end`
statement for statement for the shapes lanes admit (order <= 2, no VPI,
tangents, delay lines or uic).

| Phase | Kernels |
|---|---|
| begin | `step_begin`: idle lanes start an attempt (scalars, `SimState`, LTE inputs); a lane whose Newton is unfinished carries it over |
| predict | predictor, predictor limit pass, per-lane lim-active words |
| Newton | `newton_init` for the new lanes, then ticks: eval, assemble, refactor, solve, dense peel, update, limit pass, gates, state pass, finalize. Each lane counts its own ticks (`Gate.tick`) and leaves at `max_iter` |
| settle | lanes whose Newton ended this round become the decide mask |
| probe | flip query, charges at x into the tape, LTE pair, probe rows |
| decide | `step_end_a`: `Sim.end` to the commit (Newton/flip/LTE rejects, order promotion, ring rotation, accept) |
| apply | latches commit or revert, accepted histories, stateful-charge rebase |
| land | `arp_lbrk_<model>` per breakpoint batch, then `step_end_b`: landing, the next step's clamp |
| report | `ring_put`: each lane's report and probe row into this round's ring slot |

A launch runs up to `GpuLanes.max_rounds` (16) rounds. With every failed
replay solvable on the device (below) and graphs available, one round is
captured once, with 3 masked Newton ticks, and replayed R times; R doubles
per launch up to the ring. Otherwise a launch is one round whose ticks the
host drives with one sync each. After a launch the host downloads the ring
once and drains it per lane in round order on worker threads: waveform
rows, the `.meas` stream, the measure bound and early stop. That is the only
host work per round.

Breakpoints are exact, not tabulated: a PULSE or PWL corner rounds from
the accepted time it is requested at (`spicePulseBreak`), so the lane
breakpoint kernel keeps the host's per-instance `{req, at}` cache on the
device, seeded from `GpuPayload.brk`. A lane whose device asks for
`reject_at` or a finite `$bound_step` is parked and reruns on the CPU.

## Failed replays

All lanes replay one LU plan (lane 0's pivot order). A lane whose replay
fails its pivot guard used to come back to the host, which downloaded every
lane's matrix, and the plan was never refreshed, so a plan built at the
first step's tiny dt went stale as dt grew: on a 2,001-lane MOS sweep about
90 % of lanes fell back on every tick. Now:

- **Refresh.** When most lanes fail in one tick (or across a multi-round
  launch), the plan is rebuilt from a lane the host has just factored on
  its own matrix, as the CPU lane solver's lead follows its own solver.
- **Device dense peel.** For n <= 64 a failed lane factors its own matrix
  with partial pivoting in a device scratch slot (256 MB pool). Its
  elimination is `dense_lu`'s unblocked loops op for op, so pivots and
  singularity agree with it exactly. With one slot per lane no tick needs
  the host, which is what allows multi-round launches.
- **Compacted host peel.** Past n = 64 or past the pool, `arp_lane_pack`
  gathers only the failed lanes' A and rhs, and `arp_lane_unpack` scatters
  their dx back.

## Host memory

`Executor.executeGpuBatch` runs a batch in waves sized by
`gpuLaneBytes` against a 4 GiB budget, with at least 256 lanes per wave:
each wave's circuits are materialized, run and, when nothing reads them
later (`Executor.no_readers`, set under a final plan), released before the
next wave. A lane's run and step machine are freed as soon as it finishes.
What remains per lane is its result. Measured: about 174 KB of host memory
per lane on a 72-unknown deck; a 2,000-lane run with a forced small budget
went from 496 MB to 363 MB peak with identical results.

## Measured

2,001 lanes of a one-transistor inverter, measures-only with early stop:

| Configuration | Device Newton | Host | Host-side LU peels |
|---|---|---|---|
| before (stale plan, host peels) | 675 ms | 823 ms | 424,212 |
| plan refresh | 272 ms | 737 ms | 2,001 |
| plus device dense peel | 40 ms | 750 ms | 0 |
| plus device step control, multi-round | 49 ms (whole rounds) | 84 ms | 0 |

301 lanes, full waveforms (2,068 rounds): host step control 456 ms Newton
plus 1,228 ms host (1,980 ms total); device step control one round per
launch 1,124 ms; multi-round with per-lane Newton 585 ms. Unrolling every
round to the full `max_iter` instead cost 4,750 ms, because each masked
tick is about 25 graph nodes at launch latency.

## Exactness

The CPU lane path remains the oracle within tolerance. Within the GPU
path, each step is checked against the one before it bit for bit:
`ESPICE_GPU_HOST_STEP=1` keeps `tran.Sim` on the host over the same device
numerics, and device step control, multi-round launches and waves all
reproduce its output exactly (measures and full `-r` waveforms on the decks
above; they also match the CPU there). Two places can differ in the last
bits from the host-controlled path: a plan refresh that happens at a
different tick, and dense versus sparse LU for a failed lane.

Early stop is decided by the host's drain, so a multi-round launch can run
a lane past its resolving row. The host stops reading it there and finishes
it from that row's report; its x is scratch it frees. The extra rounds are
unseen unless a model can latch a status `refuseLatched` reads off the
device rows (a `status` hook), in which case such batches run one round per
launch.

## f32 Jacobian (removed)

An experiment carried the device kernels' derivative lanes in f32
(`-Dgpu-jac-f32`): device Newton -22.5 % on a 512-lane 10-stage BSIM4
inverter chain, about 1 % on small models, waveforms within 1.1e-8 of a
column's peak. It was removed rather than made automatic: the card class
it targets (FP64 slower than 1:4) is where `auto` keeps lanes on the CPU,
the f32 partials also reached the single-circuit device eval, whose
results are bitwise the host's (`ESPICE_GPU_EVAL_CHECK`), and a per-card
choice would build every device kernel twice. VerA's `--jac-f32` remains.

## What is left

- **Large circuits.** Past n = 64 a failed replay still needs the host, so
  those batches run one round per launch with a sync per tick. A device
  sparse LU fallback, or a larger dense limit with a bigger pool, would lift
  that.
- **Refill within a wave.** Waves run one after another; a wave's finished
  lanes leave their slots idle until the wave ends.
- **Device-side measures.** Card resolution is the host's: porting the
  measure scan machinery exactly would let early stop end lanes on the
  device and let latching models batch rounds.
- **Occupancy.** At 255 registers the eval kernel holds 8 warps per SM, and
  thread-per-lane underfills a datacenter part below roughly 30k to 70k
  lanes (estimate). Threads per instance and lane with colouring, and
  VerA's frame reduction, are the remaining levers.
- **Conditional graph nodes** were not used: setting the condition from
  device code needs the CUDA device runtime library at link time, which a
  driver-only machine lacks. Per-lane Newton carry-over gives the same
  effect portably.
- **DC and OP sweeps** still have no GPU lane path; the Newton kernels
  would serve them behind a thin driver with a CPU fallback for gmin or
  source stepping.

## Knobs

| Knob | Effect |
|---|---|
| `ESPICE_GPU_STATS=1` | Per-batch round, tick, peel, retape and dense-peel counts and phase times |
| `ESPICE_GPU_HOST_STEP=1` | Step control on the host (the oracle) |
| `ESPICE_GPU_NOGRAPHS=1` | Direct launches, so one round per launch |
| `ESPICE_LANES=0` | No lanes |
