# Remaining performance work

What is still open after the 2026-10-22 round, with what was learned and the
next concrete step. Done items are in [README](README.md)'s status table;
the agent reports behind each line are in the session scratchpad
(`agents/<name>/REPORT.md`).

Rules that apply to everything below: no new user-facing flags or environment
variables (each change picks itself by whether it applies); exact changes keep
the corpus byte-identical; tolerance-level changes must pass the full oracle
corpus (the user accepted tolerance-level differences between a run alone and
the same circuit as a sweep point or lane).

## Gates still owed before integration

- **Full GPU corpus** (`zig build test-gpu`, hours) for 01 1.6 and 04 4.4,
  which change results within rounding and so far passed only `tran/` and
  `stress/` on the GPU.
- **Full CPU oracle corpus and A/B on the final tree** once every agent has
  stopped: the last full runs were on intermediate trees.
- **`flake.nix` `zigDeps` hash**: still the old dependency set;
  `bsim.berkeley.edu` refused unattended downloads (401).
- **Release builds**: decide whether `-Dgpu-jac-f32` (the earlier lane
  session's experiment, off) becomes automatic or is removed. SASS is done:
  Gompute 1.2.0 embeds it automatically and the release GPU variants build in
  the CUDA shell (one architecture per variant; other cards JIT the PTX).
- **VerA**: the VerA agent's commits sit on a local `perf/espice` branch;
  tagging, pushing and the pin bump are the user's call.

## Wanted features (planned later)

- **One huge circuit split across GPUs** (multi-GPU stage 4): device evaluation
  partitioned by instances with piece-ordered boundary sums (exact), the LU on
  one device. Estimated 1.1-1.3x on 2-8 GPUs over PCIe, up to ~2x on 8 NVLink
  GPUs (logic_bsim4_10k); multi-GPU LU loses on thin circuit factors (14,106
  dependent steps on chain_bsim4_100k). Needs its own design pass; see
  `$S/agents/multigpu-research/REPORT.txt` section 3. Preferred first: make the
  circuit smaller on the device (compaction: fewer bytes per instance, row and
  index widths, field-major Model rows, shared rows), which helps every card.

## 01 GPU-resident Newton

- **1.2 resident planes, assembly and companion on the device** and **1.3
  Newton gates on the device, k iterations per launch**: not started; both
  wait on 02 2.3 (resident A). The CUDA conditional-graph bindings in
  `../ESPice-gpu-newton-resident` are the part worth taking for 1.3 (that
  worktree is an unfinished, never-run experiment; do not merge it).
- **1.9 (a) per-card cost cache** for `auto`: not done; (b) the mid-run
  demotion to the CPU and (c) are in.
- **5.3 (a) host batches stamped beside the device eval**: measured 16-44 µs
  per eval on chain_bsim4_10k, and not bitwise where two host batches share a
  cell; only worth it with an ordered add.

## 02 Linear solver

- **2.3, 2.7, 2.9 (device)**: the GPU LU agent's items; see its section below.
- **2.4 tail**: the serial supply-column tail holds 24-38% of the work; run
  the ticketed kernel on those columns (needs an entry point in
  `src/solver/kernels/lu.zig`).
- **Supernodal DFS ([fp]) and permuted assembly**: not started.
- **`FreqSolver.factorEach` per-call allocations**: small; not done.

## 03 Device evaluation

- **3.2 the device frame** (20.8 KB per thread for bsimcmg): the real fix is
  in VerA's codegen; see the VerA section below. The ESPice-side
  `maxrregcount`/launch-bounds experiment was not run.
- **Bypass allowlist**: `bypass_models` in `src/device/eval.zig` is a
  hand-kept list of models that read no time; VerA metadata should replace it.

## 04 Newton and timestep

- **4.6 source stepping rebuilds the constant baseline per λ** and **JFNK's
  full assemble per finite-difference matvec**: not done (the ladder now
  rarely runs on sweeps, which warm-start).

## 05 Scheduling

- **5.6 rest**: dropped (round 2): setup is 15-17 µs of a 1.5 ms stepped
  transient point (about 1%); device evaluation is 70% of the point.
- **5.11 `.four`/`.fft` reading the `.tran` result**: superseded by 8.1
  streaming for now; would need the waveform kept when a `.four` reads it.

## 06 Sweeps and lanes

- The GPU lanes agent's items: see its section below.

## 07 Frontend and preparation

- **7.3 flattening without re-reading the text**: about a week; the frontend
  is now 0.6 s of 1.4 s on 1.14M devices.
- **7.5 threads**: memoization took creation to about the parse time; what
  remains is text scanning and the pattern sort.
- **7.7 SPEF/DSPF as indexed parasitics**: needs the eight owner decisions in
  `dev/research/spef-parasitics.md`; the per-point write path it needs
  (`Hooks.write_params`) exists.
- **7.8 `buildTapes` merge** (about 2% of Debug preparation) and the rest of
  the small list.
- **`.sens`, dcmatch/acmatch, CPU Monte Carlo lanes, tangents** still unshare
  their own query's model rows (per-instance writes).

## 08 Output and memory

- **8.1 rest**: plots still held in memory: those `.meas`/`.save`/`.stim`/the
  optimizer read, lane batches, the C API, and non-raw formats.
- **8.2 C API**: the final-plan behaviours (state moves, early release,
  streaming) as automatic C API behaviour.
- **8.7 per-point residue**: about 5.9 KB per point, mostly the 424-byte
  `Query` held at least twice; needs a query-options storage refactor.

## 09 Frequency and periodic

- **9.2 rest**: OPs on the device, factors resident between rounds, batched
  adjoint and port right-hand sides on the device; the GPU frequency kernel
  is still on the stacked-real tape (complex on the CPU only).
- **9.4 (a)**: HB's nonlinear iteration resident on the GPU.
- **9.5 PAC/PXF/pnoise**: not pursued (already matrix-free; a sideband FFT
  pays only above ~10 sidebands, default 7).
- **9.6 `.pz` at large n**: a sparse Arnoldi method returns only the roots
  nearest a shift, which changes what `.pz` reports: **user decision** (how
  many roots, which shift).
- **9.3**: Krylov recycling (GCRO-DR) not done; oscillators still take one
  finite-difference period for the period column.

## 10 Measurement and gates

- The counters list (run split under `--timing`, per-iterate sync census is
  in, rejected-attempt iterations, changed-entry fraction, SCC census is in
  as `lu-btf`, KLU oracle harness) is not done: no runtime gain, and most
  need debug switches.

## Agent results (all agents have stopped)

### GPU LU (02 2.3, 2.7, 2.9 device)

Done and exact (in README's table): u16 `dmap` per epoch, faster epoch table
build, 64-row panel tail solves, a lone OP on the device LU under explicit
`cuda`/`hip`. Dropped: Markstein division (bitwise but 5-7% slower), peel
hysteresis as designed (the host then refactors the 594M-flop re-pivot
epochs). Remaining:

- **2.3 resident A**: not started. The device LU call spends 0.4-0.5 ms beyond
  its kernels, mostly uploading A; each Newton eval moves 3.5-7 MB of planes
  down and 1.5 MB of A up (5-10% of a ~20 ms iteration on the 10k thick decks,
  more at 100k+). Next: with 01 1.2, a device kernel forming `G + ag0·C` from
  the eval's planes plus a small host overlay (ground pin, host batches);
  touches Circuit, converger, integrator, context.
- **2.7 graph capture**: launch overhead is under 1% of a thick-deck call;
  capture with 2.3 once A no longer uploads.
- **2.7 admission bars at 3k/30k**: decks generated, `~/.cache/espice-gpu-lu/bars.sh`
  compares host vs forced device LU; run on a fresh GPU build and refit
  `min_per_col`/`min_flops`; epoch load costs 2-4 ns per flop (`load_ms` in
  `ESPICE_GPU_STATS`), usable for a cost-based demotion.
- **2.7 tail cut**: `tail_step_cost` still prices the old one-block sweep;
  refit for panels (2048 rows is best on logic_bsim4_10k).
- **2.7 f32 device refactor**: dropped under the no-switch rule unless an
  automatic policy passes the oracle gate.
- **Check at integration**: the agent saw logic_bsim4_10k's OP fail
  (`OpDidNotConverge`, `lu_nnz=n`) on an intermediate shared tree with the
  device LU off; verify it is gone. `~/.cache/espice-gpu-lu/` holds 2 GB of its
  binaries and logs (delete when done).

### GPU lanes (06 6.8, 6.9, 6.11-6.15; 01 1.8; 03 3.9)

Done (README's table): 6.13 CPU and GPU from one pool, 6.11 colour-per-lane
eval threads, 6.8 measures on the device, 6.15 points-API residue, 1.8 runtime
`.hdl` on the device. Remaining:

- **Gate owed**: GPU Debug test-espice after 6.8 and the row-order lane
  gather (not rerun: the Gompute 1.2.1 pin makes the next GPU build cold).
- **6.9 OP/DC lanes**: measured not to pay here (a 10k-point OP sweep costs
  0.28 ms per point on the CPU pool, lane setup alone 0.23 ms per lane). Next:
  `GpuLanes.initStatic` without the step machine, CPU ladder from rung 2;
  measure first on an FP64-class card with a large `.dc` sweep.
- **6.12**: nothing on the lane path; the remaining lever is colouring the
  charge kernel as 6.11 colours eval.
- **6.11 follow-ups**: thresholds come from one card and one model; Gompute
  needs a `Context.multiprocessorCount` (FP64 class stands in for SM count).
- **6.8 follow-ups**: host measuring remains when `abort_after` is set or a
  card has stages or reads a `par()` waveform.
- **6.15 residue**: point labels and parameter rows until `deinit` (~100 B
  per point).
- **3.9 host-stepped batches on the device**: not started; needs a
  per-instance history store on the device and VerA's history-state code for
  the device (#204/#206). Start from the delay line.
- **Note**: an install's `share/espice` can lag its binary (a stale `abi.zig`
  makes every runtime `.hdl` load fail with LayoutMismatch); runtime `.hdl`
  needs `ZIG=` pointing at the nix Zig 0.18.

### VerA (model metadata, 03 3.2 codegen, field-major rows)

- **Model metadata: done in VerA, waiting on the pin.** VerA commit
  `b05b9b9d` on the local branch `perf/espice` (parent `322cee37`, not
  pushed) adds `contract.SimReads` and `pub const sim_reads` on every
  generated device. `$S/agents/vera/espice-sim-reads.patch` replaces the
  `bypass_models` allowlist with `!readsTime(D)`; apply it with the pin bump
  (url + hash to `b05b9b9d`). Gated: VerA test-device and test-all pass; the
  ESPice corpus is byte-identical (836/104). The old allowlist was keyed by
  module name while `baseName` returns the file name, so bsimcmg, bsimsoi_va,
  hicumL0, hicumL2_va, hisimhv_va, hisimsoi, hisimsotb, l_utsoi, mextram,
  mextram_t and mvsg were never bypass-eligible; the declaration admits them
  (and ccvs).
- **3.2 device frame: cause found, fix saved, not landed.** The frame is not
  live derivative lanes: 11 KB of bsimcmg's 15.2 KB local memory is three
  copies of the 3.7 KB `evalQ` result, from the generated functions' two early
  zero returns (status latch) plus the real one. A single result variable and
  one return cut the frame 20,848 → 9,840 B with identical arithmetic
  (`$S/agents/vera/vera-status-single-return.patch`). Next: apply in VerA,
  run its tests, build ESPice for the GPU with `--fork`, measure bsimcmg,
  bsimbulk and hisimhv at N = 4096 under the GPU lock (keep at ≥ 5% per
  instance-eval), CPU corpus A/B. Frames today (sm_89): bsim4va 4.6 KB,
  psp103 6.3 KB, bsimbulk 17.2 KB, bsimcmg 20.8 KB, hisimhv 24.9 KB.
- **Field-major Model rows (P2): not started.** One rewrite in `render.zig`
  where it spells `model.`; measure the distinct-row cost on bsimcmg after
  the frame fix first.
- **Note:** the user's uncommitted `flake.nix` change in VerA adds an unlocked
  input, so `nix develop` fails inside the VerA tree until it is locked.
