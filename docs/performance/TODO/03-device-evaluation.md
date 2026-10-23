# Device evaluation

The model cores (VerA-compiled Verilog-A) and the stamp around them, on the
CPU and on the GPU. Evaluation is 45-64% of a nonlinear transient on the
CPU (`dev/analysis/evaluation-profile.md`, `dev/research/lane-batched-simulation.md` §1)
and the kernels are 78% of a GPU lane tick (`dev/research/gpu-lane-e1.md` §4).
Anchors are `src/...` paths.

## What ships

- One batch per device type, instances of all cards sharing it through
  `model_of` (`src/frontend/builder.zig:284-294`); columns of Model rows,
  Instances, States, lim plane, charge tape and the flat u32 tapes `gath`,
  `rhs_idx`, `slots` (`src/device/eval.zig:1575-1622`). Equal Model rows are
  deduplicated by bytes at finalize (`eval.zig:1230-1267`).
- `evalRange` per instance: gather n_u unknowns, limiting correction,
  one fused `evalQ` core call, scatter residual, G, q and C straight into
  the CSC value slots (`eval.zig:764-904`, `:2610-2613`). About 4
  instructions per stamp, ~45 stamps per mos1 instance; the tape layout is
  frozen and AVX2 has no scatter.
- Derivatives as `@Vector(width, f64)` dual lanes, one value's lanes being
  the unknowns it depends on, padded to a power of two on the host; **f64
  only** (`eval.zig:170`). The `jac_f32` / `gpuJacFloat` path exists only at
  HEAD and the build never passed VerA's `--jac-f32`, so the f32 Jacobian
  items in `docs/performance/gpu-batching.md` (item 5) and
  `dev/devices/gpu-evaluation.md` (`-Djac-f32-gpu`) describe code that is
  not in this tree.
- A narrow 4-lane instantiation for fully collapsed instances of ten
  models (mos1/2/3/6/9, bsim1, bsim3, hfet2, jfet, mes; `eval.zig:709-736`);
  `canNarrow` refuses n_u > 8, so BSIM4 and PSP always run wide, and the
  GPU always runs wide (`eval.zig:2694-2697`).
- Constant-Jacobian batches (resistor, capacitor, inductor, vcvs, vccs,
  cccs, bsource_q, kinduc) stamp G and C once into a baseline on the host;
  their residual and charge are still evaluated every iterate
  (`eval.zig:2284-2293`, `src/analysis/Circuit.zig:958-989`).
- ParEval: the stamp over T − 1 worker lanes with dynamic chunk claims and a
  deterministic staged reduce; limiting, state staging, `stateCtl` and
  `advanceIteration` stay serial (`src/analysis/par_eval.zig:392-514, 623-654`).
  Default `ESPICE_THREADS=1`.
- GPU: one thread per instance in deck order, 64-wide blocks, every
  contribution to its own staging cell, two-level ordered reduce;
  `gpu_host_only` is empty (`build.zig:741`), so every built-in model
  has an image (the changelog line that keeps psp103_nqs and psp104_nqs on
  the host is stale).
- No device bypass of any kind.

*Measured* kernel facts (E1, RTX 4060 Laptop): bsimcmg 0.19-0.25 µs per
instance-eval on the device against 2.89-3.13 µs on one host core (0.066-0.081);
255 registers, 20.8 KB stack, 17% occupancy; FP64 issue is 45-90% of the
kernel; a random instance order costs +18-34%, point order +8%, distinct
Model rows +10.5%. ParEval on 10k PSP103: 12.4 ms per Newton iteration on
one thread, 4.4 ms on 8 P-cores; a lane on an E-core stamps 2x slower.

| Model | n_u | pattern entries per instance | derivative lanes | Model row |
|---|---:|---:|---:|---:|
| mos1 | 8 | 32 | 6 | ~0.9 KB |
| bsim4va | 18 | 100 | 11 | ~10.5 KB |
| psp103 | 19 | 92 | 12 | ~11 KB |
| bsimcmg | 30 | 235 | 20 | ~22.6 KB |
| hisimhv_va | 32 | 194 | 26 | ~11.8 KB |

## Items

### 3.1 The narrow derivative basis for BSIM4 and on the GPU — P0

**Today.** `canNarrow` refuses n_u > 8 with a comment naming BSIM4 as
unpriced (`eval.zig:726-752`). The generated BSIM4 has 18 local unknowns, 11
derivative-read nodes, and four distinct lanes after maximal topological
collapse, which is what the post-layout cards select (RDSMOD, RGATEMOD,
RBODYMOD, TNOIMOD all zero); its four limiter-written nodes map to four
distinct roots, so the existing non-aliasing rule holds
(`xyce-research/device-source-audit.md`, experiment 1). The GPU kernel never
takes the narrow path (`eval.zig:2696`, the `narrow_count` note).
**Change.** Admit BSIM4 (and PSP, whose topology flags are similar) to the
narrow instantiation on the host, and give the lane and single-circuit
kernels the two-kernel form of the phase-4 plan (G2 b: a narrow kernel over
the collapsed prefix, the wide one over the rest, each with its own
register allocation). `ProtoStore.finalize` already sorts collapsed
instances first.
**Expect.** 11 → 4 derivative lanes: the dual arithmetic, which is most of
the FP64 work, shrinks ~2.5x; the frame shrinks with it, raising occupancy.
Estimate 1.3-1.8x on BSIM4 evaluation, host and device. Unmeasured.
**Gate.** Tolerance: tied ports, polarity, limiting, every gate/body/
resistance mode, DC, AC, noise and transient fixtures; the wide path stays
for the other topology settings. **Effort.** 1 week.

### 3.2 Frame and spills: the device ceiling — P1

bsimcmg's kernel holds a 20.8 KB frame per thread; at 6,144 resident threads
that is 128 MB of stack, four times the L2, so spills go to DRAM, and on an
FP64-class card the memory part is what remains (E1 §3). The phase-4 G2
matrix: `maxrregcount` 128/168 (occupancy 17% → 25-33%, more spills, a
gamble), launch bounds, the narrow basis (3.1), and the real fix, fewer
live derivative lanes per expression in VerA's codegen (projected 20.8 KB →
~5 KB for bsim4 after ABI 5's sparse lanes). Decision rule: keep a variant
when µs per instance-eval drops ≥ 5% at N = 4096 and the lane parity test
holds. A field-major (SoA) Model row would coalesce the distinct-row reads
(+10.5% today) but needs VerA to emit field accessors: P2.

### 3.3 f32 derivative lanes on consumer cards — P2, opt-in

Does not exist in this tree (above). The record: an f32 GPU Jacobian
measured 1.16x on `parallel_inverters_2000` with Newton 1356 → 1354, then
failed four decks (mos1/mos6, 0.8-10% off) once the MOS models were
resident; the host f32-planes census cost +0.22% iterations and lost two
decks for want of an f64 fallback (`gpu-convergence.md` §9.5). The fixed
point of Newton is set by the f64 residual, so this trades Newton ticks for
a 64x cheaper Jacobian on a card whose FP64 is 1/64. Re-add behind a build
flag for the GPU kernels only, with f64 fallback on a Newton failure,
measure tick inflation per model on the lane corpus, and keep it off where
the inflation exceeds the saving. An FP64-class card needs none of it.

### 3.4 Thread the serial halves of the step — P1

ParEval threads the stamp only. Limiting (7-9% of Ir), state staging
(4-7%), `stateCtl` and the accepted-point charge read run as serial walks
with ParEval on (`par_eval.zig:623-654`), and the charge read and staging
run as two passes instead of the host's fused one (`Circuit.zig:1167-1175`).
On 8 threads those serial 15% cap the Amdahl gain at ~4.5x where 3.3x is
measured for bsim4. Each is per instance with a flag OR, so they take the
same chunked lanes. **Exact.** **Effort.** 2-3 days.

### 3.5 Default device threads from the budget — P0

`ESPICE_THREADS` defaults to 1 and ParEval is built only above it
(`src/analysis/executor.zig:724-728`). [05](05-scheduling-and-parallelism.md)
5.1 gives a lone query the cores; 5.2 pins the lanes to P-cores (1.35-1.38x
over a mixed set).

### 3.6 Instance order for cache and warps — P1, tolerance

Instances stamp in deck order (`ProtoStore` keeps it). On the device a warp
mixing transistors costs +8% (point order) to +34% (random); on the host
the scatter hits the planes in netlist order, whose locality is whatever the
extraction tool wrote. Sort each batch's instances at finalize by collapse
class and Model row (warp coherence) and then by first slot (locality), and
make the staging permutation and the host stamp use the same order, so CPU
and GPU stay bitwise with each other while the corpus moves within
rounding. Measure LLC misses in the stamp with `perf stat` first on the
100k decks; the GPU number is E1's.

**Status (2026-10-22): dropped for now.** The BSIM4 eval kernel is 1.13 ms
of a full eval on chain_bsim4_10k, 0.6 s of a ~10 s GPU run; +8% from
warp mixing is under 0.5% of the run, while the reorder moves every CPU
deck's stamp order (tolerance across the corpus). Revisit once LU stops
dominating these decks.

### 3.7 Model rows unshared by sweeps — P0, see [07](07-frontend-and-preparation.md) 7.1

`collectParams` gives every instance its own Model row whenever a deck has
variants, `.mc`, `.sens`, dcmatch, acmatch or tangents (`eval.zig:1703-1713`,
`src/frontend/prepare.zig:414`): +10.5% kernel time from uncoalesced rows,
10.5 KB per BSIM4 instance in memory and on the bus, and on the lanes one
row per (instance, lane) for the fields that differ. Per-field overrides
over a shared row fix all three.

### 3.8 Charge tape written on every eval — P1, small, exact

The host writes each instance's LTE charge tape on every evaluation
(`eval.zig:895-901`); the transient reads it at acceptance only. Write it on
the charge pass and the accepted-point pass. Also [04](04-newton-and-timestep.md)
4.4, the charge-only kernel.

### 3.9 Host-stepped batches — P1

Switches, delay and transmission lines, Laplace and pole filters, S and W
elements, mosvar and r3_cmc evaluate on the device but step their state on
the host, with the whole Instance column compared twice per eval and
synchronous default-stream copies ([01](01-gpu-resident-newton.md) 1.5).
Device stepping for them is #204 and #206.

### 3.10 Linear devices on the CPU — note

The 1M R and C of a post-layout deck cost about 6 stamps × 4 instructions
each per iterate, 0.2% of what 100k BSIM4 cost; the CPU has nothing to gain
from a separate linear path. The GPU does, through the staging cells
([01](01-gpu-resident-newton.md) 1.6).

## Closed, with the reason

| Idea | Verdict | Where |
|---|---|---|
| CPU SIMD across instances for compact models (the VerA batch family) | 1.0-1.25x on bsimcmg evalQ, ~1.05x end to end, three VerA contract changes; cheap devices only, and only when a profile is dominated by them | `lane-batched-simulation.md` §4 |
| Hand-written Zig devices, the R/C host cache | removed in ABI 34; every model is Verilog-A | `CHANGELOGS.md` |
| Larger GPU blocks | 256-wide ran bsim4va 866 vs 679 µs at 64 | `gpu-evaluation.md` |
| 8-wide host dual padding | spills on AVX2 | VerA measurement |
