# Frequency-domain and periodic analyses

The AC family (ac, noise, sp, tf, xf, stb, lstb, acmatch, dcmatch,
sens), harmonic balance, shooting PSS, PAC, PXF, pnoise, pz and
transient noise: where their time goes and what would cut it. Anchors are
`src/...` paths.

## What ships

| Piece | State | Anchor |
|---|---|---|
| `FreqSolver`: (G + jωC)x = b as the real 2n system `[G, -ωC; ωC, G]`, dense below n ≤ 16, else a 4·nnz stacked pattern with its own BTF + AMD; every ω factors; no value bypass | default | `src/solver/freq_solve.zig:1-10, 173-223, 204-207` |
| `solveBatch`: W = 4 frequencies per LaneLu replay of one pivot tape, failed lanes peel to the scalar per-ω path; the per-chunk scalar pivot refresh was removed (Ir 1,654M → 772M on the lifted fourbitadder; `sweep_opamp_wl_5000` 3,584 → 2,739 ms) | default | `src/solver/freq_solve.zig:366-435`; `dev/solvers/solver-perf-2026-09.md` |
| GPU frequency lanes over (point, ω) with equal tapes: fill, refactor and solve on the device, one thread per lane, every solve returned to the host | explicit `cuda`/`hip`, `auto` on FP64-class | `src/analysis/gpu/freq.zig`, `src/solver/freq_solve.zig:480-493` |
| Block-diagonal G0 + jω_j·C0 preconditioner, factored as frequency lanes over up to 3 tapes, unreplayable lines held as identity | HB, multi-tone HB, QPSS, PAC | `src/solver/freq_solve.zig:746-862` |
| Masked per-point GMRES on the device; resident HB Jacobian products and preconditioner solves for lanes | HB lanes | `src/solver/kernels/gmres.zig`, `src/analysis/gpu/krylov.zig`, `src/analysis/gpu/periodic.zig` |
| Dense LU with rank-8 panels from n = 40 | HB dense path, PAC dense path, pz, tf, disto, MATEX projections | `src/solver/dense_lu.zig:215-218` |
| Radix-2 complex FFT | `.four`, PAC, HB transforms | `src/solver/fft.zig` |

Linear-solve choices per analysis (`survey`, from the code):

| Analysis | Solve | Anchor |
|---|---|---|
| PSS shooting, n < 50 | dense LU of a finite-difference Jacobian: one period integration per column | `src/analysis/pss/pss.zig:32-35` |
| PSS shooting, n ≥ 50 | unpreconditioned GMRES(30), 10 restarts, tol 1e-3, one period integration per matvec | `src/analysis/pss/pss.zig:456`, `src/core/query.zig:406-415` |
| HB, one tone, n < 10 and n·nf < 512, or an oscillator | dense Jacobian LU | `src/analysis/pss/hb.zig:70-71` |
| Other HB, MHB, QPSS | preconditioned GMRES(40), 20 restarts, tol 1e-7 | `src/analysis/pss/mhb.zig:449-452` |
| PAC, n·n_sb < 64 | dense conversion-matrix LU | `src/analysis/pss/pac.zig:151` |
| Other PAC | preconditioned GMRES(60), 20 restarts, tol 1e-11 | `src/analysis/pss/pac.zig:237-239` |
| pz | dense QZ, O(n³) | `src/analysis/eigen/pz.zig:133-165`, `qz.zig` |
| transient noise | dense n × n K and a dense LU per step, O(n³) a step | `src/analysis/tran/tran_noise.zig:203-204` |

Where the time went on the one AC stress deck profiled
(`sweep_opamp_wl_5000`, 5,000 operating points plus AC, 2n = 30k;
*measured* Ir shares before the refresh fix, `solver-perf-2026-09.md`):
refactor 22.6%, lane replay 20.3%, pivot refresh 16%, full factor 2.4%,
solve 2.8%. On the HB deck profiled, the dense LU was 95% before the panels
(36.7 → 12.8 ms per factor at n = 640).

## Items

### 9.1 A complex sparse LU — P1

[02](02-linear-solver.md) 2.10: the stacked-real form costs 4·nnz indices
and 8 real multiplies per complex multiply-add where a complex LU on the
pattern of G ∪ C costs one index and 4. Up to 2x on the AC refactor and the
lane replay (estimate). The same kernel body serves LaneLu(W) and the GPU
frequency lanes.

**Status (2026-10-22): done on the CPU.** `src/solver/lane/clu.zig` `ComplexLaneLu(W)` replays a real
SparseLu tape laid from the moduli |G + jωC| on the n x n source pattern; `FreqSolver.solveBatch` runs it
first (failed lanes peel to the stacked scalar path) and `factorEach` holds complex factors for PAC/HB on the
host (`complex_held`). AC query time 1.46-1.5x faster on sweep_opamp_wl_5000 and chain_bsim4_10k+AC
(ReleaseFast), roundoff-level differences. The device frequency lanes keep the stacked tape.

### 9.2 Frequency lanes that stay resident — P1

**Today.** `gpu/freq.zig` groups only lanes with equal pivot tapes and
returns every solve to the host; the operating points that feed them run
on the CPU (no OP lane path on the device, [06](06-sweeps-and-lanes.md)
6.8). The frequency quantum is a fixed 64 (`src/analysis/ac/freq.zig:62-63`).
**Change.** With device step control in place for transients (06 6.1), the
same resident-lane shape serves the AC family: keep the point's G, C, the
factors and the spectra on the device, read back a point's spectrum once
when it finishes, and size the chunk by the lanes resident rather than by a
constant. Noise's adjoint solves (`solveRhsT`) and sp's port right-hand
sides are batched right-hand sides on factors that already live there,
which is where device solves win outright (`gpu-lu.md` §3(b)).
**Expect.** Per-point host work falls to the OP and output construction
(estimate). **Gate.** Corpus AC decks within tolerance against the CPU
lanes (bitwise on the CPU today).

**Status (2026-10-22): partial.** The batch precompute (`gpu/freq.zig` `run`) launches up to a quantum
of frequencies per point per round, speculatively on the current tape, and accepts chunks up to the first
failed lane: bitwise the CPU stacked lanes, 22.4 -> 3.4 s on an 8-row chain_bsim4_1k AC sweep (RTX 4060).
Still open: the operating points on the device, factors resident across rounds, batched adjoint/port
right-hand sides.

**Status (2026-10-23): complex device lanes done.** The sweep precompute (`gpu/freq.zig` `run`) replays
the host's complex tapes (`FreqSolver.beginComplexChunk`/`finishComplexChunk`, the same tape, centre
and peel decisions as `complexBatch`) with `kernels/frequency.zig`'s complex bodies (`complexFill`,
`complexRefactor`, `complexSolve`: `ComplexLaneLu`'s operation order, re/im planes), so `--backend cuda`
is bitwise `--backend cpu` on the complex path; a point whose complex lanes turn off continues on the
stacked kernels. All of a group's right-hand sides (noise's adjoints, sp's ports) solve in one launch,
with one wait per group (the solve skips failed lanes on the device). Gates: CPU byte-identical on the
full corpus (836 SAME + 104 `exit 1/1`; `complexBatch` only split into shared helpers); `cuda` vs `cpu`
byte-identical on `.ac`/`.noise`/`.sp` `.data` sweeps of a 150-stage diode ladder, chain_bsim4_1k, and an
LC ladder with 291 peeled lanes; host oracle test `kernels/frequency.zig` bitwise `ComplexLaneLu(4)`.
Not done: OPs on the device (each point's OP stays a host solve), factors kept across rounds (each
round's chunks replay a fresh tape per point group, so there is nothing to keep; the held preconditioner
factors of HB/PAC stay stacked), the live round driver (`solveRound`) stays stacked.

### 9.3 Shooting PSS without finite-difference period integrations — P1

**Today.** Below 50 unknowns the shooting Jacobian is n period
integrations per Newton step; above, each GMRES matvec is one period
integration and the solve is unpreconditioned (table above). GPU shooting
lanes integrate fixed-step trapezoidal periods on the device, but the
finite-difference columns and the Krylov directional integrations stay on
the host per lane (`docs/using/gpu.md`, "Shooting PSS").
**Change.** `dev/solvers/monodromy-krylov.md`: the monodromy product Φ·v by
the sensitivity recurrence over the saved step factors (one pass over the
period with the stored LUs, no re-integration), GMRES on (Φ − I) with
GCRO-DR recycling across shooting iterations, and the adjoint replay for
pnoise and PXF. On the device, the recurrence is a chain of the lane solves
that already exist.
**Expect.** One period integration per shooting iteration plus m cheap
solves instead of m + 1 integrations (estimate: several-fold on
`pss`/`pnoise` decks with n ≥ 50). **Gate.** Periodic corpus within tolerance.

**Status (2026-10-22): done.** `pss.zig` `Mono`: the base period records its trajectory, G_k and C_k are
evaluated once per shooting iteration, and Φ·v is the trapezoid sensitivity recurrence (dense columns in one
pass, or GMRES matvecs). 3.5x on a 400-stage driven ladder (ReleaseFast), shooting-tolerance differences.
The pnoise/PXF adjoint replay does not apply (they solve the frequency-domain conversion matrix).

**Status (2026-10-23): rest done.** An oscillator's period column d x_N / d(ln T) is one more recurrence
column (`Mono.apply`'s `period_col`, driven by α's dependence on T and the sources' t·∂f/∂t at the
recorded x_k) instead of one finite-difference period. GCRO-DR (`src/solver/gcrodr.zig`) recycles 10
harmonic Ritz vectors in the host HB solve (`mhb.zig`), refreshed only after a cycle that restarts:
`.hb 1k 32` at n = 403 GMRES applications 783 -> 621, ReleaseFast 1.75 -> 1.46 s (median of 5, load
14-16), `.hb 1k 8` unchanged. Not used by PAC (its solves converge in one cycle, ~18 iterations; the 10
recycled directions cost 10 operator applications per frequency, measured slower) nor the shooting
Krylov solve (tolerance 1e-3, few iterations). Gate: tolerance; full-corpus A/B 822 SAME + 104
`exit 1/1` + 14 DIFF (oscillators and multi-tone HB, all within 2.5e-10 section-relative).

### 9.4 HB: resident nonlinear iteration and the averaged preconditioner — P1

**Today.** The HB sampled evaluation (`Samples.evaluate`) uploads x and
downloads f, q, g and c on every call (`docs/performance/gpu-batching.md`
item 7); the nonlinear iteration and its backtracking run on the host per
point while the masked GMRES and the Jacobian products are resident
(`src/analysis/gpu/periodic.zig`). `hb.zig:179-181` notes the single-tone
path is all on the CPU and that the GPU would batch the nt independent
samples. The block-diagonal preconditioner reallocates its LaneLu groups
on every HB Newton step and every PAC frequency
(`src/solver/freq_solve.zig:841-851`). The averaged block-circulant
preconditioner (FFT-diagonalized, `dev/solvers/structured-preconditioners.md`)
and Krylov recycling do not exist.
**Change.** (a) Keep the sampled evaluation's planes on the device and run
the per-point Newton (update, line search, convergence) there next to the
GMRES that already is, as the transient lanes do for their step control.
(b) Keep the preconditioner's buffers across calls. (c) Add the
block-circulant averaged-Jacobian preconditioner behind a pin and compare
GMRES iteration counts against the block-diagonal one on the HB corpus.
(d) A mixed-radix or Bluestein FFT (`src/solver/fft.zig` is radix-2 only), so
nf need not be a power of two, and a device FFT for the lane transforms.
**Gate.** HB corpus within tolerance; `ESPICE_HB_TRACE` iteration counts.

**Status (2026-10-22):** (b) done (held LaneLu groups reused; byte-exact). (c) already exists: G0 and C0 are
the sampled planes' time averages, so the block-diagonal preconditioner is the FFT-diagonalized averaged one.
(d) one-tone transforms folded by symmetry (`mhb.zig` `Fold`, a quarter of the multiply-adds; 2.2x on
`.hb 1k 32` at n = 403); a radix-2/mixed-radix FFT does not fit nt = 2(2K+1). (a) not done (GPU).

### 9.5 PAC, PXF and pnoise matrix-free — P2

Below n·n_sb = 64 PAC factors a dense conversion matrix; above, GMRES with
the block preconditioner. `dev/solvers/lptv-block-solves.md` gives the
matrix-free apply (block-Toeplitz structure through the FFT) and the
transpose for PXF and pnoise, and `monodromy-krylov.md` the time-domain
adjoint. PAC, PXF and pnoise also re-prepare their periodic state inside
each query (`dev/Problem/migration-notes.md`): a shared PSS node in the
query DAG ([05](05-scheduling-and-parallelism.md) 5.11) removes the repeat.

**Status (2026-10-22): not pursued.** The Krylov path is already matrix-free over the pattern (n_sb²·nnz per
product); an FFT apply only wins past roughly M = 10 sidebands, and the default is M = 3. The repeated PSS per
query is 05 5.11.

### 9.6 Pole-zero and transient noise at large n — P2

pz runs dense QZ, O(n³) on n of at most a few hundred, and is not laned by
design (`dev/research/lane-batched-simulation.md` §6). For larger n a
sparse shift-invert Arnoldi on the existing LU (the "future: sparse/Arnoldi"
of `dev/solvers/README.md`) replaces the dense pencil. Transient noise forms
a dense n × n K and a dense LU per step (`src/analysis/tran/tran_noise.zig:203-204`);
a sparse or low-rank noise injection makes `.trannoise` usable past a few
hundred unknowns.

**Status (2026-10-22):** transient noise done: the METHOD=SDE covariance uses the circuit's sparse LU
(15.8x on a 200-node ladder in Debug, byte-identical on the corpus).

**Status (2026-10-23):** pz done (user decision: partial results above a size threshold). Up to 1000
unknowns the dense QR/QZ runs unchanged (corpus byte-identical); above it `.pz` returns the 20 poles and
20 zeros nearest s = 0 by shift-invert Krylov-Schur Arnoldi on the sparse LU (`src/analysis/eigen/arnoldi.zig`,
`pz.zig` `sparseRoots`), and the plot name records the count wanted, the shift and the counts found.
ReleaseFast: 100k-node RC ladder pz query 0.88 s (dense: 196 s at 2.5k nodes, O(n³)); 5000-OTA deck
(n = 15k) 0.50 s. Matches the dense nearest-20 to <= 1e-8 relative on RC ladders (0.5k-2.5k stages) and a
200-OTA deck.

### 9.7 LaneLu width on AVX-512 — P2

W = 4 is the native f64 width on AVX2; W = 8 cut Ir 22% but wall only 3-7%
there and was retired (`solver-perf-2026-09.md`). `dev/decisions.md` S6:
revisit when an `x86_64_v4` build ships separately.

**Status (2026-10-22):** the i9-14900HX here has no AVX-512, so nothing was measured; LaneLu(8) and
ComplexLaneLu(8) are now covered by unit tests on every host.

### 9.8 Small items

- The per-ω full factor has no value bypass by design (`src/solver/freq_solve.zig:1037`);
  a sweep over points at one ω list could share tapes across points, which
  the GPU frequency lanes already do by `sameTape`.
- `sp` runs a dense O(n³) on at most 16 legs (`src/analysis/ac/sp.zig:302`):
  fine.
- `.four` and `.fft` re-integrate the deck's `.tran`
  ([04](04-newton-and-timestep.md) 4.5).
