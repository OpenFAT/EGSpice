# Performance TODO

Every change that would make ESPice faster, with what it costs today, what
it would buy, and how it is gated. Written 2026-10-21 against the working
tree at 1.2.0 (`ec53a724` plus the uncommitted `src/analysis/gpu/`,
`src/solver/kernels/` and `src/solver/lane/` moves). Assumptions: a CUDA
GPU is present at all times, and independent work is free to run in
parallel. The development card is an RTX 4060 Laptop (FP64 at 1/64 of
FP32); items whose payoff depends on FP64-class hardware say so.

Numbers marked *measured* come from the page cited beside them; the rest
are estimates. Anchors are `src/...` paths in this tree.

## Status (2026-10-22)

Implemented in the working tree, each checked byte-identical against the
1.2.0 baseline binary on the decks it touches (Debug builds, CPU):

| Item | What changed | Measured (Debug build) |
|---|---|---|
| 05 5.9 | the pool refills freed slots before delivering output | |
| 05 5.1, 02 2.1 | a per-Problem thread pool: a query whose device weight or matrix is large borrows up to 7 extra lanes for ParEval and the multicore refactor when no thread variable is set | `parallel_inverters_2000` 43.4 → 11.0 s |
| 03 3.4 | limiting and state staging run on ParEval's lanes (range hooks, ABI 36) | `parallel_inverters_2000` → 7.6 s; `chain_bsim4_1k` 60.4 → 13.1 s |
| 05 5.2 | ParEval workers pinned to the P-cores of a hybrid CPU | |
| 07 7.2 | subcircuit-local `.model` cards memoized per parameter environment | 3,000-instance PDK-style deck 1.05 → 0.69 s |
| 07 7.1 (part) | the sweep planner addresses parameters through `Hooks.param_layout` instead of a 64-byte `ParamRef` each | `chain_bsim4_10k` + `.step`: 1.81 → 0.96 GB, 6.1 → 4.9 s |
| 07 7.4 | Model rows deduplicated at append | `chain_bsim4_100k` preparation 1.93 → 0.90 GB, 26.2 → 20.4 s |
| 02 2.8 (ordering) | BTF + AMD computed once per template and shared by every query | |
| 08 8.3 | `.meas` cards evaluated once per printed block | |
| 02 2.2 | the CLI defaults to `auto` in builds with GPU kernels, quietly; the lane FP64 probe runs only for decks that can batch | |
| 01 1.7 | Gompute 1.2.0 embeds SASS for `-Dcuda-arch` automatically whenever `ptxas` and `fatbinary` are on PATH (the default shell; release GPU variants now build there), and `auto` trusts a matching SASS image | |
| 06 6.1 (part), 6.3 | pinned per-tick counters; lanes download only the probes a point reads | |
| 04 4.1 (automatic) | latency bypass on every host-evaluated query: an instance whose limited unknowns sit within reltol/10 of its last evaluation replays the recorded stamps, residuals moved by J·Δx (SPICE's predicted-current test guards it), charges re-read exactly; each batch drops it during the transient when under 1 in 5 evaluations replay; an allowlist of models that read no time (VerA metadata to replace it) | `scaling_inverter_chain_256` 19.6 → 13.1 s (edges within 5e-18 s); BSIM4 and the adder gain nothing |
| 08 8.2 | under `final_plan` (the CLI) `run_all` frees each result once its plot and `.meas` block are out; `Problem.summary` keeps name and shape. Not with `.save`/`.stim`; Monte Carlo trials and tangent runs stay | 200-point stepped RC ladder with `-r`: 3.36 → 1.01 GB peak, files identical |
| 03 3.1 | the narrow derivative basis admits any model whose narrow dual needs fewer AVX2 registers; BSIM4 (11 → 4 lanes) narrows in its linked device object, the test-built vtable stays the forced-wide reference | `chain_bsim4_1k`/`10k` (ReleaseFast) about 3-4% faster end to end, output bit-identical |
| 03 3.8 | Newton passes no longer write the LTE charge tape; the full, charge-only and accepted-point passes still do | exact (full corpus A/B) |
| 05 5.7 | queries run on a per-Problem pool of reusable 512 MiB-stack threads (`worker.Threads`) | 200-point sweep (Debug): 401 → 1 thread creations, 2.16 → 1.58 s, output identical |
| 05 5.10 | a new `.interval` quantum parks at most once per millisecond of work (two thread handoffs per park); `--timing-in-depth` uses it; `.checkpoint` keeps its per-step contract for API stepping | |
| 06 6.13 | a lane batch keeps ceil(f × points) rows on the GPU and hands the rest to CPU workers at once, f = card / (card + host threads) from the FP64 ratio; lanes gather in row order (reproducible) | nd10000 `cuda`: GPU only 19.3 s, CPU only 13.5 s, shared 10.05 s |
| 06 6.11, 6.8, 6.15; 01 1.8 | thread per (instance colour, lane) below the measured crossover (bitwise); `.meas` one-sample passes on the device (`output/scan.zig`, bitwise); points API final-plan lifetimes; runtime `.hdl` compiled for the device | 512 lanes: Newton −29%; inv_2000 host per-round row work 66.6 → 21.0 ms |
| 02 2.7, 2.9 (device) | u16 `dmap` per pivot epoch (two refactor kernels), epoch tables built in half the time, 64-row panel tail solves; a lone OP takes the device LU under explicit `cuda` | `--backend cuda`: logic_bsim4_10k 28.4 → 26.6 s, sram_bsim4_10k 16.3 → 15.8 s, logic_psp103_10k 44.0 → 35.2 s, bitwise |
| 01 1.1, 1.5, 1.6, 1.9 (b, c), 1.10, 4.4, 5.12 | sync census, one wait per accepted point, no default-stream traffic, device baseline for constant batches, write-not-add staging (no per-eval clear), charge-only kernel, `auto` demotes a losing device query to the CPU, driver started before the parse | `cuda`: chain_bsim4_10k 7.29 → 5.22 s, logic_bsim4_10k 29.3 → 24.6 s, syncs per step 7.06 → 5.03; 1.6 and 4.4 within rounding (full GPU corpus owed) |
| 05 5.4 | measured: GPU queries in one process run concurrently, no admission limit needed | 8 × parallel_inverters_2000: 3.74 s sequential → 1.15 s concurrent |
| 07 7.1, 7.5, 8.1, 8.8 | copy-on-write Model rows (`Hooks.write_params`), builder and collapse memos, every transient/AC plot streams (side files), per-Problem query memory | chain_bsim4_1k 16-point `.step` 20.4 s / 335 MB → 7.4 s / 96 MB; chain_bsim4_100k creation 1.9-2.6 → 0.76-0.92 s; 200-point stepped `-r` 3.37 GB → 49 MB |
| 05 5.6 (part) | solver workspaces pooled per Problem: a query takes one built for its pattern and knobs and `reset`s it (bitwise a fresh one) instead of building and freeing its own | full corpus byte-identical; with 7.1 and the warm start a 1000-point `.op` sweep runs in 0.04-0.05 s (was 0.77 s) |
| 04 4.5 (part) | `minDel`'s maxes and min without NaN handling (NaN already returns through `poison`) | transient-family decks byte-identical |
| 02 2.4, 2.6 (host) | `subtree.Plan`: per pivot epoch, independent column groups run as lanes with their own accumulators, the supply tail serially; bitwise `SparseLu.refactor`; automatic at lu_threads >= 2 when the ticketed kernel declines (F/n < 300) and the model predicts >= 1.5x | refactor at 8 lanes: chain_bsim4_10k 1.7-2.6x, ring 1.5-2.4x, chain_bsim4_100k 2.06x; factor time chain 4.10 → 2.66 s |
| 02 2.8 (part) | the replay carries its largest L multiplier, so `maxMultiplier` skips its pass over L | 6.5-10% of a serial replay |
| 04 4.6 (automatic) | every sweep point (`.step`, `.data`, `.mc`, `.alter`) starts its OP from the nominal one and converges 1000x tighter than the deck's tolerances; points that change temperature stay cold (path-dependent oracles) | 100-point `.op` sweep of a 20-stage chain: 145 → 2 Newton iterations per point, worst difference 4.2e-7 V; default output unchanged |
| 02 2.10, 09 9.1 | complex lane LU (`ComplexLaneLu`) for the AC family, PAC and single-host HB preconditioners; per-frequency fallback to the stacked real path; GPU keeps the stacked tape (tolerance) | AC query `sweep_opamp_wl_5000` 1570 → 1039 ms, `chain_bsim4_10k` AC 8067 → 5539 ms (ReleaseFast) |
| 09 9.2 (part) | GPU frequency lanes take up to 64 frequencies per round (byte-identical to the CPU stacked path) | `chain_bsim4_1k` 8-row GPU sweep 22.4 → 3.4 s (CPU still faster on this card) |
| 09 9.3 | shooting PSS Jacobian from the trapezoid's sensitivity recurrence over one recorded period (tolerance) | 400-stage driven ladder 0.90 → 0.26 s |
| 09 9.4 (b, d) | HB preconditioner lane groups reused; single-tone transforms use real-signal symmetry | `.hb 1k 32` at n = 403: 3.27 → 1.48 s |
| 09 9.6 (trannoise) | METHOD=SDE transient noise on the sparse LU instead of a dense one (exact) | 200-node ladder (Debug) 33.9 → 2.15 s |

VerA is at v0.12.1 and Gompute at 1.1.0. The other session in this
checkout owns the GPU lane items of [06](06-sweeps-and-lanes.md) (pinned
mirrors, lane-major LTE ring, host passes on threads, device step control,
multi-round graphs, device peel and retape).

Dropped with a measurement: 05 5.6 rest (per-point setup is 15-17 µs of a 1.5 ms 1000-point `.step` transient point, ReleaseFast: about 1%; a recycled Circuit would add a reset hook to every device batch for that), 05 5.8 (a 1000-point sweep's user time stays 1.60-1.63 s from 1 to 32 jobs: the allocator mutex costs nothing measurable), 04 4.3 and 4.8 (115 transient runs, 3.3M attempts: 2 Newton rejections, LTE rejections 1% of attempts, so neither an early divergence cut nor speculative retry lanes has work to save), 04 4.7 (the built modified-Newton mode failed 9 oracles and lost 1.06-1.54x on the chains; under the no-switch rule it cannot ship as an opt-in), 04 4.9 (a higher-order or TR-BDF2 default moves every grid; no automatic policy passes the oracle gate), 02 2.5 (every census deck is one SCC plus a singleton), 02 2.8 partial refactor (the changed columns hold 99% of the work), 04 4.6 rung racing (plain Newton fails at once on a singular Jacobian, gmin stepping is sequential), 04 4.5 history swaps (one n-copy per step, under 1%).

Deferred with a reason: 05 5.11's `.four`/`.fft` reuse (with 8.1 the CLI streams a `.tran` to the file instead of holding its rows, so a `.four` has nothing to read without keeping the waveform; the corpus `.four` decks run in milliseconds), 04 4.4 (under `cuda` eval is 3% of the run and LU
93%, so the charge-only kernel saves about 1%; the host re-read already runs
values only), and 05 5.11 / 04 4.5's `.four`/`.fft` reuse of the `.tran`
result (every corpus deck with `.four` also has `.tran`, but reuse needs a
second dependency kind in the session DAG, which today knows only OP
prerequisites).

Not done yet: the remaining P0/P1/P2 items below. `flake.nix`'s `zigDeps`
hash still pins the old dependency set: `bsim.berkeley.edu` refused unattended
downloads (401) when it was recomputed.

## Pages

| Page | Covers |
|---|---|
| [01 GPU-resident Newton](01-gpu-resident-newton.md) | one circuit on the device: syncs, bytes, graphs, device gates, JIT, the `auto` cost model |
| [02 Linear solver](02-linear-solver.md) | host and device LU, ordering, BTF blocks, complex LU |
| [03 Device evaluation](03-device-evaluation.md) | model kernels, derivative lanes, ParEval, instance order |
| [04 Newton and timestep](04-newton-and-timestep.md) | bypass, step ceiling, charge re-read, OP ladder, opt-in solver modes |
| [05 Scheduling and parallelism](05-scheduling-and-parallelism.md) | thread budget, P-cores, CPU beside GPU, GPU concurrency, per-template setup, the DAG |
| [06 Sweeps and lanes](06-sweeps-and-lanes.md) | 10k-100k points as device-resident lanes |
| [07 Frontend and preparation](07-frontend-and-preparation.md) | parse, flatten, bind, the variant path, SPEF |
| [08 Output and memory](08-output-and-memory.md) | streaming, retention, encoding, growth |
| [09 Frequency and periodic](09-frequency-and-periodic.md) | AC family, HB, shooting, PAC, pz |
| [10 Measurement and gates](10-measurement-and-gates.md) | workloads, counters, protocol, what must not move |

Tags on every item: **P0** build next, **P1** after the P0s, **P2** research
or opt-in; **exact** (bitwise), **tolerance** (same grid and test, values
within the oracle tolerance), **opt-in** (a named mode; grid, test or
published iterate changes). [10](10-measurement-and-gates.md) defines the
gates.

## Where the time goes

*Measured*, from `dev/devices/gpu-evaluation.md`, `dev/analysis/evaluation-profile.md`,
`dev/research/lane-batched-simulation.md`, `dev/solvers/solver-perf-2026-09.md`
and `docs/performance/gpu-batching.md`:

| Regime | Example | Split | Floor today |
|---|---|---|---|
| One large post-layout circuit, CPU | chain_bsim4_100k, 1.14M devices, 446k unknowns | eval 38%, LU 31%, setup 10%, rest 21% | one thread for eval and LU unless env vars are set |
| The same under `cuda` | logic_psp103_10k | eval 3%, **LU 93%** | the device LU is admitted only under a non-cpu backend; the CLI default is cpu |
| Real PDK deck under `cuda` | c7552_ihp | eval 23%, LU 30%, setup 28% | per-instance `.model` re-reads |
| Small nonlinear, many steps | mos6_inverter, vacask_mul | eval 55%, limiting + staging 14%, LU 9-28%, charge re-read 5% | 25-60 µs per iterate on the GPU, so these stay on the CPU |
| Many points of a small circuit | asap7 bsimcmg chain, 10k points | per point: eval 64%, limiting + gates 21%, LU 2%; GPU lanes: host control over 90% of a round for one-transistor lanes | 16 fences and two scalar host passes per round |
| Sweep with one swept parameter | chain_bsim4_10k with `.step` | Problem creation 171 ms → 1.54 s, 191 MB → 2.02 GB | every Model row unshared |
| AC family | sweep_opamp_wl_5000 | AC refactor 23%, lane replay 20%, pivot refresh 16% | stacked-real 2n system |
| Harmonic balance | profiled HB deck | dense LU 95% before the panels | dense Jacobian below n·nf = 512 |

## Build first (P0)

Cheap, exact, and each removes a floor the measurements point at:

| Item | Page | Buys (measured or estimate) | Effort |
|---|---|---|---|
| Default `--backend auto` and the multicore refactor on by default, under one thread budget | 02 2.1-2.2, 05 5.1 | logic_bsim4_10k 183.6 s → 52-63 s with device eval + device LU; 1.07-2.14x on thick factors from the refactor alone (*measured*) | 3 d |
| Copy-on-write Model rows; collect only the swept fields | 07 7.1 | sweep setup 9x faster and 10x smaller; GPU rows coalesce again (+10.5% kernel) | 1 wk |
| Local `.model` cards folded once per parameter environment | 07 7.2 | c7552_ihp setup from ~8 s toward 0.7 s | 2 d |
| Lanes: pinned mirrors, lane-major LTE ring, probe subset, host passes on the pool | 06 6.1-6.4 | removes ~16 fences per round and most of the host share at 10k lanes | 1 wk |
| One sync per Newton iteration plus a sync and byte census | 01 1.1 | ~I + 3 → I + 1 syncs per step at 15 µs each | 3 d |
| Resident assembly and factor on the device (GPU LU stage 2) | 02 2.3, 01 1.2 | bytes per iteration O(nnz) → O(n): 96 MB → ~32 MB at n = 10^6 | 2 wk |
| Latency bypass, host first | 04 4.1 | the 4k chain evaluates 1 MOSFET in 200; ~3x on chains (estimate); tolerance class | 2 wk |
| Narrow derivative basis for BSIM4, host and device | 03 3.1 | 11 → 4 derivative lanes: 1.3-1.8x on BSIM4 eval (estimate) | 1 wk |
| SASS in the release, background JIT warm | 01 1.7 | the first run of each model takes the GPU; no 15-1,000 s cold compile | 3 d |
| Stream every plot; release delivered results | 08 8.1-8.2 | steps × probes × points → one window | 1-2 wk |
| Delivery after fill, not between wait and fill | 05 5.9 | freed pool slots stop idling behind encoding | 0.5 d |
| Default device threads from the budget, pinned to P-cores | 05 5.1-5.2, 03 3.5 | ParEval 1.5-3.3x on PSP/BSIM4 eval for a lone query (*measured*) | 3 d |

## Then (P1), by regime

- **One large circuit.** Newton gates on the device and k iterations per
  launch (01 1.3); graph capture (01 1.4); constant batches out of the
  staging (01 1.6); ordering for span (02 2.4, **fp**) and BTF blocks kept
  (02 2.5); device LU kernel work (02 2.7); ordering once per pattern and
  the other host micro-items (02 2.8); the serial halves of the step
  threaded (03 3.4); instance order (03 3.6); charge-only kernel (04 4.4);
  host batches beside the device eval (05 5.3a); in-process GPU concurrency
  measured (05 5.4); per-template setup (05 5.6); parallel preparation
  (07 7.5); staging row dedup at append (07 7.4).
- **Many points.** Device step control (06 6.5), multi-round graphs (6.6),
  peel compaction (6.7), device measures (6.8), OP/DC lanes (6.9), host
  memory waves (6.10), CPU and GPU from one pool (05 5.3b), points API
  residue (6.15).
- **Small decks.** Bypass (04 4.1) is the only lever that changes their
  work; the GPU does not help them.
- **AC and periodic.** Complex sparse LU (02 2.10, 09 9.1), resident
  frequency lanes (9.2), shooting without finite-difference integrations
  (9.3), resident HB iteration and the averaged preconditioner (9.4).
- **Memory.** Encoding once and off the coordinator (08 8.3), waveform
  growth (8.4), the batch-after-finish copy (8.6).

## Research and opt-in (P2)

Modified Newton and iterate-0 acceptance (04 4.7, blocked by the oracles
that encode ngspice's Newton error), speculative retry lanes (4.8),
timestep-ceiling presets (4.2, accuracy decision pending), f32 derivative
lanes on consumer cards (03 3.3, the code is not in the tree), f32 device
refactor (02 2.7), BBD from a partitioner (2.6), SoA model rows (3.2),
colored thread mapping for datacenter occupancy (06 6.11), multi-GPU
(05 5.5), runtime `.hdl` kernels (01 1.8), PAC/PXF matrix-free and sparse
pz (09 9.5-9.6), huge pages (08 8.10).

## Dependency order

```
05 5.1 thread budget ──> 02 2.1 refactor default ──> 05 5.2 P-core pinning
02 2.2 backend auto ───> 01 1.7 SASS/JIT warm ───> 01 1.9 cost model refit
02 2.3 resident A ─────> 01 1.2 companion on device ──> 01 1.3 gates + k iterations
                                                     ├─> 01 1.4 graph capture
                                                     └─> 01 1.6 constant batches off the staging
06 6.1 pinned mirrors ─> 06 6.4 host passes threaded ─> 06 6.5 device step control ─> 6.6 ─> 6.8
07 7.1 COW rows ───────> 06 6.10 host memory waves; 03 3.7
04 4.1 bypass (host) ──> 02 2.8 partial refactor ──> bypass on the GPU (compaction)
02 2.4 ordering for span (fp decision) ─> 02 2.5 BTF blocks ─> 2.6 BBD from a partitioner
```

## Constraints every item respects

- The host path is the oracle: a device or threaded kernel is bitwise the
  host's, so a peel continues as a host-only run would.
- No float atomics; every sum has a fixed order.
- Every kernel goes through gompute, in Zig, for CUDA and HIP; no vendor
  libraries.
- The default mode publishes ngspice's iterate x_k, applies its convergence
  test and takes its LTE grid; anything else is a named opt-in mode gated
  on the corpus pass set (`dev/solvers/gpu-convergence.md` §10.1).
- CPU SIMD across instances stays off for compact models; the lane axis is
  the CPU oracle and the GPU the throughput lever
  (`dev/research/plans/README.md`).

## Already shipped, do not re-propose

Pivot-tape replay with a growth monitor and probe; value bypass; the flat
flop tape and supernode panels; BTF + AMD; the Thomas and BBD engines; the
multicore refactor and the device LU (stage 1, with sync-free solve heads
and no size cap); constant-Jacobian baseline on the host; shared Model
rows; fused waits and the `evalFollows` prefetch on the GPU path; the
deterministic staged reduce; ParEval with dynamic claims and a futex park;
CPU lanes (W = 4) and GPU lanes for tran, AC, PSS and HB with graph capture
per tick; the refilling pool; measures-only, early stop, warm start, the
points API; streaming of the first CLI binary-raw plot; the OP ladder's
gmin and source-step rules; the binder's model-row and field fixes
(c7552_sky130 setup 30.5G → 2.9G instructions); the radix pattern sort.

## Docs found stale by the surveys

`dev/solvers/overview.md` and `README.md` ("every solve runs on the host"),
`structured-preconditioners.md`, `lptv-block-solves.md`, `amd-ordering.md`,
`gpu-lu.md` (refactor default, peel hysteresis), `transient-integration.md`
(ITL4 as the attempt cap), `operating-point-homotopy.md` (a damping formula
not implemented), `migration-notes.md` and `4)Module APIs and main.md`
(streaming, `publish`/`finish`), `frontend.md` (sorted-slice source index),
`gpu-evaluation.md` and `docs/performance/gpu-batching.md` (f32 Jacobian
options, `-Djac-f32-gpu`, `gpu_host_only` entries), `CHANGELOGS.md` 1.2.0
(psp103_nqs/psp104_nqs on the host), `dev/research/spef-parasitics.md`
(the O(E²) name scan is fixed). Each page's last section lists its share.
