# Xyce 7.10 linear-solver research

This is a read-only source investigation, not a causal performance profile. The measured program is Xyce 7.10.0 at `/nix/store/wssn1r1k10dlnkra48kc5nil5lqrv0m3-xyce-7.10.0/bin/Xyce`, SHA256 `313a99f6b912ede19ee142f094617d238368e1f420c007f30dc995c769bccd46`. Its installed `Xyce_config.h` leaves `Xyce_PARALLEL_MPI` undefined, and `ldd` resolves Amesos, Epetra and TrilinosSS to Trilinos 16.1.0. Xyce release source is `/tmp/xyce-7.10-research`, tag `Release-7.10.0`, commit `a592a42ae7472151d1c8e3bca0ca62d27476f2f3`.

The ESPice comparison binary is the parent’s baseline ReleaseFast build at commit `ec53a724d1e67adcb54927e46fbd13af40aed1eb`. `git diff` confirms that `direct.zig`, `order.zig`, `sparse_lu.zig`, `lane_solver.zig` and `analysis/Circuit.zig` are unchanged from that baseline. Lane implementation changes elsewhere must not be credited with the baseline’s measured performance.

## What was measured, and what is not yet established

The supplemental run in `zig-out/comparison-scaling-2026-10-20` measured the following medians, with three repetitions, one worker per simulator and matching output selections:

| Post-layout deck | ESPice seconds | Xyce seconds | ESPice / Xyce time |
|---|---:|---:|---:|
| chain_bsim4_10k | 15.227 | 12.821 | 1.188 |
| chain_bsim4_1k | 2.667 | 2.197 | 1.214 |
| ring_bsim4_1k | 4.920 | 2.018 | 2.438 |
| logic_bsim4_1k | 8.803 | 22.572 | 0.390 |

Thus Xyce was faster in three of four post-layout decks under those shared-machine conditions. These times alone do not establish a linear-solver cause. All four Xyce post-layout comparisons missed the deliberately strict small-signal waveform screen; that is relevant to equal-accuracy work counts, even though absolute voltage differences were small. The parent is separately collecting timestep/Newton counts and profiles.

## Source-proven mechanisms

**Serial solver selection.** Xyce’s ordinary serial, matrix-based transient path defaults to KLU. The factory can instead select KSparse, AztecOO, Belos, Amesos2 or other compiled solvers by option, but the comparison adapter does not set a different linear solver. A 1-by-1 problem has a special simple solver. [Xyce 7.10 transient solver factory](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/LinearAlgebraServicesPKG/N_LAS_TranSolverFactory.C#L90-L153).

**Symbolic work is retained; numerical pivots are not reused by default.** The Xyce Amesos wrapper constructs its solver and calls `SymbolicFactorization()` once. Ordinary solves call `NumericFactorization()` unless a caller explicitly supplies `reuse_factors`. The constructor sets `repivot_ = true`; the serial setup sets `TrustMe = true` and `Refactorize = !repivot_`. Consequently, the default is a new numerical pivot choice on each factorization, not KLU’s faster fixed-pivot refactor path. `.OPTIONS LINSOL KLU_REPIVOT=0` is a valid ablation, not a description of the measured default. [Xyce Amesos wrapper](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/LinearAlgebraServicesPKG/N_LAS_AmesosSolver.C#L80-L98), [factorization setup and calls](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/LinearAlgebraServicesPKG/N_LAS_AmesosSolver.C#L309-L396).

The ordinary Newton caller explicitly uses `solve(false)`, so the caller-side `reuse_factors` exception does not bypass numeric factorization there. The NOX interface delegates its Newton step to that same method. [Xyce Newton linear solve](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/NonlinearSolverPKG/N_NLS_NonLinearSolver.C#L948-L969), [NOX delegation](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/NonlinearSolverPKG/N_NLS_NOX_Interface.C#L2290-L2305).

**KLU keeps the BTF partition through factorization.** The embedded KLU defaults enable BTF and AMD, with diagonal pivot tolerance 0.001. Analysis computes row/column permutations and retains block boundaries, maximum block size and off-block entry counts. AMD acts inside each diagonal block; very small blocks use natural ordering. [KLU defaults](https://github.com/trilinos/Trilinos/blob/trilinos-release-16-1-0/packages/common/auxiliarySoftware/SuiteSparse/src/KLU/Source/trilinos_klu_defaults.c), [KLU analysis](https://github.com/trilinos/Trilinos/blob/trilinos-release-16-1-0/packages/common/auxiliarySoftware/SuiteSparse/src/KLU/Source/trilinos_klu_analyze.c).

The numeric routine factors each diagonal block separately and stores off-block entries in `Offp/Offi/Offx`. Those entries participate in block substitution rather than diagonal-block elimination. That structurally avoids elimination fill in the off-block region and makes the largest block, rather than total circuit size alone, relevant to factorization cost. [KLU factorization](https://github.com/trilinos/Trilinos/blob/trilinos-release-16-1-0/packages/common/auxiliarySoftware/SuiteSparse/src/KLU/Source/trilinos_klu_factor.c#L126-L246), [KLU block solve](https://github.com/trilinos/Trilinos/blob/trilinos-release-16-1-0/packages/common/auxiliarySoftware/SuiteSparse/src/KLU/Source/trilinos_klu_solve.c#L207-L277).

**The Amesos path avoids repeated sparse-format conversion.** For storage-optimized serial Epetra matrices, it binds directly to existing CSR indices/values during symbolic setup. Under `TrustMe`, later numeric calls skip export and reconversion. CSR(A) is passed as CSC(A-transpose), so the normal solve calls `trilinos_klu_tsolve`; transpose solves call `trilinos_klu_solve`. This can change numerical pivot/fill behavior relative to factoring CSC(A), beyond the cost of copying itself. When fixed-pivot refactor is explicitly enabled, failed refactor or a poor diagonal condition estimate causes a fresh pivoting factorization. [Trilinos 16.1 Amesos_Klu](https://github.com/trilinos/Trilinos/blob/trilinos-release-16-1-0/packages/amesos/src/Amesos_Klu.cpp).

**Do not credit default row scaling to this Xyce build.** Although KLU’s standalone default is max-row scaling, Amesos replaces `common_->scale` with `ScaleMethod_`; the Amesos control constructor initializes that to zero. The Xyce wrapper does not set another value. The effective default here is no scaling. [Amesos control defaults](https://github.com/trilinos/Trilinos/blob/trilinos-release-16-1-0/packages/amesos/src/Amesos_Control.h#L24-L33).

**Device matrix addresses are bound ahead of the load loop.** Xyce’s resistor model stores Jacobian pointers during setup and directly adds conductances through them in a model-wide instance loop, with an indexed compile-time alternative. This eliminates sparse matrix searching during loads, but is not a distinction from prebound scatter slots at the algorithm level. [Xyce resistor pointer setup](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/DeviceModelPKG/OpenModels/N_DEV_Resistor.C#L689-L735), [resistor master matrix load](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/DeviceModelPKG/OpenModels/N_DEV_Resistor.C#L1306-L1355).

## Already present in ESPice

| Feature | Existing implementation | Consequence for the explanation |
|---|---|---|
| BTF plus fill-reducing order | `src/solver/order.zig:77`: Tarjan SCC, then AMD or natural order according to symbolic fill estimate | “Add BTF/AMD” is not a new improvement. |
| Frozen sparse structure | `direct.zig:127`, `sparse_lu.zig:44` | No per-Newton matrix graph rebuild. |
| Fixed-pivot numeric replay | `direct.zig:265`, `sparse_lu.zig:890` | Already more aggressive reuse than Xyce’s default; includes failure fallback and stability checks. |
| Repivot fill cap | `sparse_lu.zig:241`, `direct.zig:289` | Historical bad-pivot fill explosions on chain_bsim4 are already mitigated; their old timings are not evidence of today's cause. |
| Unchanged-matrix factor bypass | `direct.zig:228` | Linear circuits can skip numeric factorization while values remain identical. |
| Tridiagonal/BBD engines | `direct.zig:127`, `bbd.zig` | Some layouts already have stronger structural specialization than general LU. |
| Small-matrix operation tape and large-column replay | `sparse_lu.zig:92`, `sparse_lu.zig:1087` | Avoids rediscovering elimination operations on subsequent factors. |
| Supernode handling in full factors | `sparse_lu.zig:30` | “Add panels” alone is not a new idea here. |
| Prebound matrix slots | `device/eval.zig:2643` | Direct scatter is indexed, but it does not search a sparse structure. |
| Constant G/C baselines | `analysis/Circuit.zig:597`, `analysis/Circuit.zig:727` | Constant-Jacobian caching already exists; linear residual/state-update bypass is a separate question. |
| CPU lane factor/solve | `lane_solver.zig:1`, `lane_solver.zig:79` | Exploits repeated independent points with matching tapes; it does not intrinsically accelerate an isolated unswept benchmark. |

The meaningful BTF gap is narrower: ESPice’s `order()` returns only `q`. Its block boundaries are temporary scratch, and `direct.computeOrdering()` passes that order into one flat `SparseLu`. Off-block entries remain in that flat elimination. The project already documents this exact distinction in `dev/solvers/btf-permutation.md`, under “Our variant.”

## Hypotheses and useful next experiments

1. **Prioritize by observed profile.** Compare numeric-factor and triangular-solve time per Newton iteration, not just total runtime. Record matrix dimension, original nonzeros, factor nonzeros, full-factor/replay/bypass counts, marginal-replay probes, SCC count, largest block and off-block nonzeros. If device evaluation or timestep count dominates, a solver rewrite is the wrong first intervention.
2. **Prototype retained BTF blocks only if the census supports it.** Extend ordering to return block boundaries and off-block maps, retain independent block factors, and perform block substitution. The likely benefit is reduced fill/work and smaller working sets on reducible patterns. A giant single SCC limits the opportunity. This changes pivot admissibility and arithmetic order; require residual/accuracy and difficult-pivot regressions, not just faster elapsed time.
3. **Use KLU as a controlled matrix-level oracle.** Export the same ESPice Newton matrices and RHS values, then compare ESPice’s full factor and replay with KLU’s factor/refactor, factoring both A and A-transpose where useful. This separates circuit formulation and timestep choices from the linear kernel. Keep this out of production until accuracy and dependency costs are understood.
4. **Ablate Xyce’s `KLU_REPIVOT` without mislabeling it.** A quiet paired run with 0 versus the default 1 can show how much fixed-pivot reuse matters for Xyce on these decks. It does not establish why baseline Xyce beats ESPice, which already replays pivots.
5. **Pointer stamping is a lower-priority microbenchmark.** Precompute host matrix pointers for selected batches and compare with existing u32 slots only if load/scatter time is material. Pointers enlarge metadata and complicate rebinding/circuit replicas; fixed slots already eliminate lookup, and changing representation may hurt cache footprint.

No measured causal percentage is claimed by this note. Retrieved official Trilinos source files and their URLs are preserved in `/tmp/espice-xyce-linear-sources/`; no solver source or benchmark settings were changed.
