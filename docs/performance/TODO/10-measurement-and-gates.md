# Measurement and gates

How every item in this directory is to be measured, on what, and what must
not move. Nothing here is a change to the simulator; it is the protocol
the other pages assume.

## Machine and what it cannot measure

The development machine is an i9-14900HX (8 P-cores at CPUs 0-15, 16
E-cores at 16-31, 36 MB L3, one NUMA node), 31 GB, and an RTX 4060 Laptop
(sm_89, 24 SMs, 8 GB, FP64 at 1/64 of FP32, driver 595.71). What it cannot
tell:

- **FP64-class GPUs.** Every datacenter number in `dev/research/gpu-lane-e1.md`
  §3 is a projection (A100 4-26x, H100 7-69x this card on device
  evaluation). The lane work ([06](06-sweeps-and-lanes.md)) and the
  per-instance kernel costs ([03](03-device-evaluation.md)) change their
  ranking on such a card. Run E1 (`gpu-lane-e1/run.sh … cuda`) on one before
  sizing anything around those numbers.
- **HIP.** gompute's AMDGCN path compiles and has never run on AMD hardware.
- **CUDA profilers.** `ncu` and `nsys` are blocked here
  (`RmProfilingAdminOnly=1`); set the kernel module parameter
  `NVreg_RestrictProfilingToAdminUsers=0` or use CUDA events
  (`ESPICE_GPU_STATS=phases` already syncs after each stage).

## Workloads per regime

| Regime | Decks | Generator |
|---|---|---|
| One large circuit, post-layout | chain, ring, logic, sram × bsim4, psp103 × 1k, 10k, 100k (300k chain); real c7552_sky130, c7552_ihp, tdc_sky130 | `tests/conformance/external/spice-bench/postlayout/gen.py`, `fetch.sh` |
| Large linear | `stress/scaling_rc_ladder_100k`, `scaling_resistor_grid_100x100`, the power-grid suite | fixtures, `tests/conformance/external/spice-bench/powergrid` |
| Small nonlinear, many steps | `stress/vacask_{rc,mul,graetz,ring}`, `tran/bench_ngspice_mosamp`, `device_mesa_oscillator` | fixtures |
| Many points of a small circuit | asap7 24-stage bsimcmg chain at N = 1k-32k points; `stress/sweep_opamp_wl_5000` | `gen_lanes.py`, `gen_chain.py`, `lockstep.py` (HEAD `dev/research/gpu-lane-e1/`, `dev/research/lane-batched-simulation/`: restore them under `tests/benchmark/` when the `dev/` move lands) |
| AC family | `ac/bench_medium_ladder_filter`, `stress/sweep_opamp_wl_5000`, the 222 AC-family corpus decks | fixtures |
| Periodic | the 66 periodic corpus decks; no swept `.pss` deck exists | fixtures |
| Whole corpus | 940 decks in `tests/conformance/tests/netlist/` (the VAMS Conformance Suite submodule), `cmcqa` (5,645 single-device decks) | `zig build test`, `bench-suites` |

## Counters and tools that exist

| Tool | Gives | Build |
|---|---|---|
| `--timing-in-depth` | per query: Newton split into device eval (GPU wait included), matrix load, factor, solve, step update; n and LU fill; preparation phases | any |
| `ZP_TRAN_STATS=1` | accepted, attempted, Newton iterations, LTE and Newton rejections | any |
| `ZP_LU_STATS=1` | per full factor: n, nnz, fill, F, S_r, level count, widest U column, BbdInfo verdict | `-Dlu-instrument` |
| `ESPICE_LU_PAR_STATS=1` | the multicore refactor's admission verdict per epoch | any |
| `ESPICE_GPU_STATS=1` / `=phases` | admission, rounds, ticks, graph replays, peels, idle fractions; per-stage µs with a sync after each stage | GPU build |
| `ESPICE_GPU_LU_BENCH=k` | device refactor and solve ms, host comparison | `-Dlu-instrument` |
| `ESPICE_GPU_EVAL_CHECK`, `ESPICE_GPU_LU_CHECK` | bitwise replay of every device eval or refactor against the host | GPU build |
| `ESPICE_GPU_NOGRAPHS`, `ESPICE_GPU_NOFUSE`, `ESPICE_LANES=0`, `ESPICE_GPU_LU=0/1`, `ESPICE_LU_PAR=0/1`, `ESPICE_NO_BBD`, `ESPICE_SOLVER=…` | A/B switches | |
| `ZP_OPDBG`, `ZP_NEWTON_DEBUG`, `ZP_TRAN_LTE_TRACE` | per-iterate traces | any |
| callgrind | instruction counts (Ir), the stable metric on a loaded machine | `-Doptimize=ReleaseFast -Ddebug-info -Dcpu=x86_64_v3 -Dcuda-arch=none -Dhip-arch=none` (native emits GFNI, which valgrind cannot decode) |
| `perf stat` | FP64 operation counts (`fp_arith_inst_retired.*_double`), cache and TLB misses | any |
| `ptxas -v`, `cuobjdump -sass`, `tools/asm_probe.py`, `llvm-readobj --notes` | kernel registers, stack frame, spills, instruction mix; host codegen probes | GPU build |
| `zig build test-conformance` (oracles, then espice CPU/GPU against every simulator on PATH), `bench`, `bench-postlayout`, `bench-suites` | medians against ngspice 45, VACASK, Xyce 7.10 (`nix develop .#benchmarking`) | ReleaseFast |

## Counters to add

- **A one-line split at the end of every run** under `--timing`: parse and
  bind, OP, device eval, assembly, factor, solve, step control, output, and
  bytes moved over the bus. Today the split needs `--timing-in-depth` and
  the output share is reported nowhere.
- **Sync and byte census per Newton iteration** in `ESPICE_GPU_STATS`
  (uploads, downloads, syncs, bytes), the metric of
  [01](01-gpu-resident-newton.md).
- **Activity census** (`ESPICE_CENSUS`, `ESPICE_CENSUS_ACT`) and the
  `lu_trig` counter stayed off `main`; bring the activity counter in as a
  `ZP_TRAN_STATS` field for [04](04-newton-and-timestep.md) 4.1.
- **Iterations spent on rejected attempts** (4.3).
- **Per-Problem memory accounting.** `ZP_MEM_STATS` was removed with the
  global allocator; per-executor peak bytes are needed for
  [08](08-output-and-memory.md) and the lane host cap.
- **Changed-entry fraction and reach per iterate** for the partial refactor
  ([02](02-linear-solver.md) 2.8).
- **SCC count, largest block, off-block nnz** in `ZP_LU_STATS` (2.5).
- **Resolved step ceiling and LTE policy** in the bench reports (4.2).
- **A KLU oracle harness**: dump the Newton matrices and right-hand sides of a
  deck, factor and refactor them with KLU out of tree, and compare per-call
  times and fill with `SparseLu` (`docs/performance/bench/2026-10-20/xyce-research/solver-source-audit.md`,
  experiment 3). The only matrix-level comparison that separates the kernel
  from the step policy.

## Protocol

- **Load.** Record the 1-minute load average with every number; the repo's
  measurements were taken at loads from 2 to 215 and say so. Below 3 for
  wall time; any load for Ir and for device event times (the 4060 held
  2640 MHz at a CPU load of 150).
- **Pinning.** `taskset -c 2` for single-thread CPU runs; `taskset -c 0-15`
  for P-core-only threaded runs.
- **Repetitions.** One warm-up, then medians of at least 3 (7 for the
  published comparisons); keep minima and maxima. Read Ir for deltas under
  10% and wall for anything else.
- **Builds.** ReleaseFast for timing, Debug for correctness; one global
  cache per Zig store path (AGENTS.md).
- **Small decks.** Process startup is 2.6 ms and most of the corpus; a
  change that moves small-deck medians is a startup regression, not a
  solver one.

## Gates

| Change class | Gate |
|---|---|
| Exact (bitwise): kernel hoists, threading with fixed order, device replay, graph capture, scheduling | all corpus decks byte-identical in raw file, stdout and exit code against the parent build (`tools/lane_diff.py`'s form), plus the bitwise checkers above |
| Tolerance (same grid and test): new factorization (complex LU), new ordering, bypass | `zig build test` pass set unchanged, `VERDICTS.tsv` unchanged; `test-gpu` tolerance (6e-9 of rtol 1e-3 is the recorded worst) |
| Opt-in mode (grid, test or published iterate changes) | pass set a superset of the conformant one at each deck's oracle tolerance, with a recorded exemption list for the nine decks whose oracles encode ngspice's Newton truncation (`gpu-convergence.md` §10.1), or a tighter reference for them; `--backend cpu` and the GPU must pass the same gate |
| Any | `zig build bench` medians before and after on the regime's decks, load noted; no deck where `auto` picks a slower path |

Every item in this directory names which class it falls in.
