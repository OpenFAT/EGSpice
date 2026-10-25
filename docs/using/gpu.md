# GPU backend

ESPice can evaluate device models and run batches of independent analyses
on an NVIDIA (CUDA) or AMD (HIP) GPU. Device kernels use the same generated
model code as the CPU. GPU and CPU results are compared within numerical
tolerance; different transient step decisions can produce different grids.

```sh
espice big.sp --backend=auto     # GPU if present and worth it, else CPU
espice big.sp --backend=cuda     # CUDA or fail
espice big.sp --backend=hip      # HIP or fail
espice big.sp --gpu              # like auto, but fail when there is no GPU
```

The default is `--backend=cpu`.

## What runs on the GPU

For a single circuit, the GPU evaluates the devices per Newton iteration: every instance's
currents, charges and their derivatives, summed into the circuit matrix
and right-hand side in a fixed order. The sum is deterministic, so a
Newton iteration sees the same matrix every time. The matrix solve, the
Newton update, convergence checks and transient step control run on the
CPU. Batches of independent points have additional device paths:

| Analysis | GPU work | Host work |
|---|---|---|
| Transient lanes | Device evaluation, assembly, sparse LU replay, Newton updates and gates, charge history, LTE and probes | Each point's step control, breakpoints, measurements and early stop; exceptional LU re-pivoting |
| AC, noise, SP, AC transfer functions and loop stability | Flattened `(point, frequency)` matrix assembly and forward/adjoint solves, including live margin refinement requests | Operating points, output construction, stability brackets and refinement control |
| Shooting PSS | Fixed-step trapezoidal period integrations, including device commit/revert | Shooting iteration, finite-difference columns, per-lane period and Krylov directional integrations |
| Driven HB/MHB lanes | Independent `(point, sample)` device evaluation, sampled Jacobian products and spectral transforms, harmonic factor replay, masked GMRES | Each point's nonlinear iteration and backtracking; sparse pivot candidates and output construction |
| Periodic small signal | Retained harmonic preconditioner solves | Frequency sweep, outer Krylov control and Jacobian products |

Transient Newton uses two captured graphs around the exceptional host LU
re-pivoting step. Capture failure reports a warning and uses direct launches.
Both paths keep the same arithmetic and convergence controls. Shooting PSS
retains its finite residual gate and gmin. It preserves the final assembled
charge of converged lanes while peers continue, and stages all device state
downloads before publishing an accepted period to the host.

For large circuits (10,000 unknowns or more) ESPice may also factor the
matrix on the GPU: when the factorization is dense enough per column, or
large overall (about 50 million multiply-adds, which post-layout netlists
reach near 100,000 transistors). `ESPICE_GPU_LU=1` forces that on under a GPU backend,
`ESPICE_GPU_LU=0` keeps the factorization on the CPU.

The build no longer excludes MVSG, ASM-HEMT or BSIM-SOI solely for kernel
frame size. Single-circuit evaluation can mix supported GPU batches with
host batches. Devices with state across time points can update their state
on the CPU. `auto` also estimates the
cost: launching kernels and copying the solution vector each iteration has
a fixed price, and a small circuit is faster on the CPU.

GPU transient lanes require device payloads and supported device state
kernels. UIC, snapshots, OPtran, tangents, gear orders above two, delay
lines, coupled inductors and host-only Newton callbacks fall back to the
scalar path. A lane requesting an unsupported step rejection reruns on the
CPU. Shooting GPU lanes additionally refuse stateful charge and device
state without GPU control kernels; a state rejection reruns the period on
the CPU from its unchanged start. Frequency lanes require a compatible sparse
pattern; small dense frequency solves remain on the CPU. HB/MHB sampled
evaluation rejects history-dependent device state, host callbacks and device
status hooks until their per-sample state and status can be combined safely.
Those points retain CPU sampling. Finished or failed points are masked out
of device evaluation and Krylov work. A forced GPU backend therefore does
not imply every analysis used GPU lanes.

Explicit `cuda`/`hip` enables eligible lanes unless `ESPICE_LANES=0`.
`auto` admits lanes only when the reported FP64 rate is at least one quarter
of FP32. It keeps CPU lanes on the tested RTX 4060 Laptop; this device's
measurements do not predict datacenter GPU performance. A binary carrying
both image families currently selects CUDA first and rejects an explicit
HIP request when HIP is not its selected lane runtime. Use a HIP-only build
for HIP lanes.

## Lane controls and verification

| Setting | Effect |
|---|---|
| `ESPICE_LANES=0` | Disable lane batching |
| `ESPICE_GPU_STATS=1` | Admission, rounds, Newton ticks, graph replay, peels and idle fractions |
| `ESPICE_GPU_STATS=phases` | Direct launches with waits between stages; useful for diagnosis, changes timing |
| `ESPICE_GPU_NOGRAPHS=1` | Direct launches without graph capture, for comparison |

The execution configuration's `gpu_lane_cap` defaults to 16,384 device
lanes. It has no environment-variable override.

`zig build test -- --backend cuda --jobs 1` runs the fixture corpus under `--backend cuda`, one
deck at a time; GPU results are compared within each fixture's tolerance.

## What fails, and when

| Request | No GPU, or the build has no kernels | Nothing in the circuit runs on the GPU | GPU slower than CPU (estimate) |
|---|---|---|---|
| `--backend=cpu` | CPU | CPU | CPU |
| `--backend=auto` | CPU, with a warning | CPU, with a warning | CPU, silently |
| `--gpu` | error | CPU, with a warning | CPU, silently |
| `--backend=cuda` or `hip` | error | error | GPU anyway |

An explicit `cuda` or `hip` that cannot run fails the deck, with an error
naming the hardware ESPice found. Here a CPU-only binary (no kernel images)
runs on a machine with an NVIDIA card:

```text
$ espice --backend=cuda divider.sp
note: GPU batch 'resistor' (2 instances) stays on the CPU: no kernel image in this build
warning: --backend cuda: GPU unavailable (CircuitNotEligible); detected GPU hardware: cuda
Error: divider.sp: CircuitNotEligible
```

The `note:` lines name each device type that stays on the CPU and why.
The reason in parentheses is one of:

| Reason | Meaning |
|---|---|
| `CircuitNotEligible` | no device in the circuit has a GPU kernel in this binary |
| `NoGpuArtifacts` | the binary was built with no device kernels at all |
| `NotEnoughGpuWork` | the cost estimate says the CPU is faster (`auto` only) |
| `ImageNotCompiled` | a device type's kernel (over 1 MB of PTX) has not been compiled by the driver yet, so the whole run stays on the CPU (`auto` only; one `--backend cuda` run compiles it) |
| a driver error | the CUDA or HIP runtime refused |

With `--gpu` the same deck falls back, because the machine has a GPU and
only the circuit was ineligible:

```text
$ espice --gpu divider.sp
note: GPU batch 'resistor' (2 instances) stays on the CPU: no kernel image in this build
warning: GPU unavailable (CircuitNotEligible); running on the CPU
Voltage divider: 3 devices
  Operating Point: 1 points, 3 variables
```

The Nix package and any `-Dgpu=false` build carry no kernels, so they
always run on the CPU.

## Getting a GPU build

Linux release artifacts select one GPU target each: a `cuda-sm_*` build
contains only that CUDA target, and a `hip-gfx*` build contains only that
HIP target. CPU-only variants carry neither. See the
[release catalogue and portable bundles](install.md#release-tarballs) for
the available CPU instruction sets and GPU targets. To build for your own card:

```sh
nix develop                      # CUDA and ROCm toolchains
zig build -Doptimize=Debug -Dgpu=true -Dcuda-arch=sm_89 -Dhip-arch=none
zig build -Doptimize=Debug -Dgpu=true -Dcuda-arch=none -Dhip-arch=gfx90a
```

`-Dcuda-arch=none` or `-Dhip-arch=none` leaves a target out. The machine
running ESPice needs the CUDA driver or the ROCm runtime; the toolkit is
only needed to build.

## Threads instead

On the CPU, `ESPICE_THREADS=N` evaluates devices on N threads and
`ESPICE_SOLVER_THREADS=N` (up to 16) factors the matrix in parallel blocks.
For many independent analyses in one deck, `--jobs=N` runs them at the
same time. These combine with each other and with a GPU backend.
