# Lane fixes and validation — 20 October 2026

These checks concern the modified workspace. The 49-circuit CPU comparison
uses the separate, frozen baseline binary identified in the main report.
The measurements must not be attributed to the lane changes.

## Issue #215: the apparent nondeterminism was in the assertion

[Issue #215 is closed](https://github.com/OmarSiwy/ESPice/issues/215).
The [resolution comment](https://github.com/OmarSiwy/ESPice/issues/215#issuecomment-6297341314)
records the evidence and explicitly notes that the workspace changes have
not yet been committed or pushed.

The deliberately failing point consistently returns `TimestepTooSmall`.
Zig's self-hosted Debug backend miscompiles the runtime optional-error
comparison used by `std.mem.indexOfNone(?anyerror, ..., &.{null})`.
Disassembly shows a comparison reading a nonexistent optional tag beyond
the two-byte stored value.

The corrected test records `QueryInfo.failure`, checks null explicitly,
unwraps the error, and requires `TimestepTooSmall` on the intended row.
It retains the bitwise waveform and failure checks for all 19 points in
three packing orders. The old assertion failed in 12/12 fresh processes;
the corrected test passed in 12/12 fresh processes. A small standalone
reproducer distinguishes the self-hosted backend from LLVM.

Evidence: [old runs](validation/espice-215-alone.log),
[fixed runs](validation/espice-215-fixed-repeat.log),
[self-hosted reproducer](validation/espice-215-optional-error-debug.log),
and [LLVM reproducer](validation/espice-215-optional-error-llvm.log).
The [standalone source](validation/issue215-compiler-reproducer.zig)
is preserved with these results.
The assembly explanation and reproducer source are also recorded in
the repository's `.debug-knowledge-base.md`.

## Changes exercised by the checks

- Transient GPU Newton iterations use captured graphs around exceptional
  host pivot recovery. Each point retains its own timestep, rejection,
  convergence, measurement, and failure decisions.
- GPU shooting-period integration preserves the final assembled charge
  of converged lanes while other lanes continue. The earlier implementation
  cleared that charge and produced large nonlinear PSS errors. State
  downloads are now staged before an accepted period is published, so a
  failed download leaves the host state available for fallback.
- AC frequency batches and live LSTB margin refinements retain each
  point's original solver and pivot history. Recovery does not restart
  that history or silently discard an allocation failure.
- Driven HB/MHB batches have point-local nonlinear steps, device sample
  evaluation, sampled Jacobian products, harmonic factors, and masked
  Krylov work. Finished or failed points leave the active work.
- The comparison tools require actual GPU admission and graph replay
  where requested. The CPU corpus tool explicitly selects `--backend=cpu`,
  rejects nonfinite output, and detects sweep statements without mistaking
  comments or single-temperature cards for sweeps.

## Completed focused checks

Hardware: NVIDIA GeForce RTX 4060 Laptop, 8,188 MiB, driver 595.71.05,
CUDA architecture `sm_89`. These are Debug correctness checks.

| Check | Result |
|---|---|
| CPU solver suite | 158 passed |
| CPU analysis suite | 264 passed, 5 GPU-only tests skipped |
| CPU facade/analysis suite | 112 passed, 1 GPU-only test skipped |
| CUDA analysis oracles | 14 passed, none skipped |
| CPU genuinely swept corpus | 6/6 decks bitwise equal to scalar execution |
| CPU replicated corpus | 452/468 bitwise passes; 5 setup/unsupported cases and 11 Debug timeouts; zero observed numerical differences |
| Mixed GPU transient facade | 20 points in each of 3 packing orders; expected failure isolated; repeated points bitwise equal |
| Focused GPU transient graph fixtures | 7 decks × 64 replicas; actual graph replay, zero measured waveform difference |
| GPU frequency fixtures | 3 cases × 17 points × 19 frequencies; includes live LSTB refinement |
| GPU shooting PSS | Linear and nonlinear cases, each 9 points × 64 steps |
| GPU HB/MHB and periodic small signal | 7 cases × 3 mixed/repeated points |

The mixed transient test recorded 2,302 graph ticks per packing order and
zero exceptional LU peels. The nonlinear PSS case's worst error is 0.0247
times its declared tolerance
`1e-6*abs(reference) + 1e-7*reference_peak + 1e-12`.
The linear PSS, frequency, and periodic fixture comparisons measured zero
waveform difference. These different gates have explicit tolerances in
their respective runners; they are not all byte-equality claims.

The tested Debug CLI has SHA256
`0958895c98b7e50557fbe4aaca2d4af94f26486c77806d519bb2c136b81ef9df`.
The [focused validation manifest](validation/gpu-focused-validation.json)
links the hardware logs. The CPU build summaries are preserved in
[analysis](validation/espice-integrated-analysis-recovery-final.log) and
[facade](validation/espice-integrated-espice-final.log) logs. Negative
fixtures intentionally emit errors on stderr; the final build summaries
report successful completion.

The complete bounded CPU replication queue is recorded in the
[corpus report](validation/espice-lane-corpus-validation.md),
[JSON](validation/espice-cpu-replicate-summary.json), and
[per-fixture CSV](validation/espice-cpu-replicate-summary.csv).
It covers 221/222 AC-family, 66/66 periodic, and 167/182 transient decks;
the family selections overlap by two decks. Timeouts remain unverified.

## Scope still requiring work

These results do **not** close all four lane issues.

| Issue | Remaining implementation or gate |
|---|---|
| #207, transient lanes | Resolve the bounded corpus's unsupported/setup cases and 11 timeouts; perform a controlled ReleaseFast compile-time comparison against the +20% limit |
| #208, GPU backend | Full admitted-corpus coverage and completed end-to-end throughput measurements; HIP and FP64 datacenter hardware are untested |
| #209, AC family | GPU `.lin` phases, real DC/TF/transfer and finite-difference sensitivity/matching solves, frequency-dependent dynamic stamps, and subsequent noise resweeps |
| #210, periodic analyses | GPU PAC/PXF/periodic-noise outer operators and Krylov work, shooting directional Krylov integrations, and dense/autonomous HB paths |

Unsupported GPU work continues on its existing CPU path. Some such paths
are silent, so a forced backend alone cannot establish device execution;
only runs with explicit device-admission evidence count as GPU coverage.
Even admission does not universally exclude a later per-point CPU rerun.
PZ remains scalar by design. The automatic lane policy retains CPU execution on this
consumer GPU; forced CUDA runs exercise the device implementation.

The full CPU corpus selection contains 468 unique replicable decks
(182 transient, 222 AC-family, 66 periodic, with two overlaps), in addition
to the six genuinely swept decks. The exact selection is preserved in
the [scope manifest](validation/espice-lane-corpus-scope.json).
The GPU corpus has 68/182 assessed cases before being parked for the
controlled timing window: 18 actual GPU parity passes, 27 `NoLaneTape`
outcomes, 17 refusals, three MOSRA restrictions, one VPI dependency, and two
timeouts. These categories are not all GPU coverage. The remaining 114
cases and ReleaseFast throughput remain outstanding.

## Source provenance

The frozen ReleaseFast lane build starts at
`ec53a724d1e67adcb54927e46fbd13af40aed1eb` plus the
[source overrides](validation/lane-source-overrides.tar.gz).
The [manifest](validation/lane-source-manifest.json) identifies every
source file in that snapshot; its tree digest is
`cf8510c5118e928023763e4586d461da3cf1e2a4d1091f4712ba11a9ffa3edd0`.
The snapshot includes device fixes made by another concurrent workspace
task. The archive preserves the executable's inputs without claiming that
all those changes belong to this task.

No oracle verdicts were changed to make these GPU gates pass. Other
workspace work has changed corpus fixtures and `VERDICTS.tsv`, so the
working tree as a whole is not identical to HEAD's verdict file.
