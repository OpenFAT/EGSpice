Xyce 7.10 transient control versus ESPice — source research, 2026-10-20

The strongest established explanation to test first is unequal maximum-step semantics, not a demonstrated faster per-step algorithm. For `.tran 2p 2000p`, ESPice resolves a 2 ps maximum step; Xyce resolves a 200 ps default ceiling before breakpoint, device and LTE limits. The first number has different meanings. Adding the fourth argument explicitly to both decks removes that confound without changing either simulator.

This report uses official Xyce release source at commit `a592a42ae7472151d1c8e3bca0ca62d27476f2f3` (`Release-7.10.0`), read from `/tmp/xyce-7.10-research`, and the current ESPice checkout. Xyce's official GitHub documentation/source was also checked through the web. No simulator source or build was changed for this research.

**Maximum-step semantics are confirmed in documentation and executable code.**

Xyce documents the first `.TRAN` number as an initial-step input. Its omitted ceiling defaults to `(final-start)/10`; breakpoints impose another limit. Its guide explicitly distinguishes this from SPICE's printing interval. [Xyce `.TRAN`, lines 38–53 and 95–113](https://github.com/Xyce/Xyce/blob/Release-7.10.0/doc/Reference_Guide/TRAN_Command.tex#L38-L113)

For these zero-start decks, the implementation sets that default when no maximum is supplied, then takes minima with schedule, breakpoint and device limits. `TSTEP` is stored as `initialTimeStep`; `DTMAX` is stored separately and marks the maximum as explicit. [StepErrorControl, lines 899–927](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/TimeIntegrationPKG/N_TIA_StepErrorControl.C#L899-L927), [TIAParams, lines 352–357](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/TimeIntegrationPKG/N_TIA_TIAParams.C#L352-L357)

A nonzero-output-start caveat: the code subtracts `initialTime`, while `.TRAN TSTART` is parsed into `initialOutputTime`. Thus the documentation's displayed start-time formula should not be generalized to a nonzero third argument without checking the effective code path. This does not affect the zero-start postlayout comparison.

Another documentation caveat: the guide describes ten steps between breakpoints as automatic, but this release's `updateStopTime` only computes `maxTimeStepBP` when `minTimeStepsBPGiven` is true; that flag defaults false. Ordinary breakpoint landing still constrains a step. For strict control, set `MINTIMESTEPSBP` explicitly if its between-breakpoint subdivision is intended. [Breakpoint ceiling guard](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/TimeIntegrationPKG/N_TIA_StepErrorControl.C#L426-L432), [defaults](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/TimeIntegrationPKG/N_TIA_TIAParams.C#L75-L90)

ESPice's deck frontend explicitly resolves the omitted limit to `min(dt_init, (stop-start)/50)`, unless a fourth `.TRAN` argument or `DELMAX` supplies it: `src/frontend/analyses.zig:515–534`. Its transient driver then also respects device delay constraints at `src/analysis/tran/tran.zig:621–624`. This follows the ordinary ngspice initialization rule. [ngspice `TRANinit`](https://github.com/ngspice/ngspice/blob/master/src/spicelib/analysis/traninit.c#L29-L34)

The parent experiment reports raw point counts:

| Case | ESPice | ngspice | Xyce |
|---|---:|---:|---:|
| chain10k | 270 | 270 | 69 |
| chain1k | 532 | 532 | 139 |
| logic | 775 | 775 | 708 |
| ring | 1014 | 1014 | 129 |

These counts are evidence of different saved time grids, not by themselves an exact tally of solver attempts. Successful/failed step counters and Newton counts should establish the actual work. The direction and size of the difference make the maximum-step control a strong hypothesis, especially for the ring, but timing after an explicit common ceiling is still needed to quantify how much of the win it explains.

**The integration method is similar; the error policy is materially different.**

Both use variable-order trapezoidal integration by default and begin with backward Euler. Xyce's default is `ONESTEP`, which the option parser maps from `TRAP`; the official options table confirms order 1–2 and time-integration tolerances `RELTOL=1e-3`, `ABSTOL=1e-6`. [Transient constructor/parser](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/AnalysisPKG/N_ANP_Transient.C#L127-L147), [time integration options](https://github.com/Xyce/Xyce/blob/Release-7.10.0/doc/Reference_Guide/timetbl.tex#L15-L81)

Xyce estimates error from the weighted RMS norm of the difference between corrected and predicted solution, multiplied by the integration coefficient `ck`. Default `NEWLTE=1` uses the largest absolute signal at the current time as the relative reference for every unmasked unknown. The weight is `RELTOL * reference + ABSTOL`. `MASKIVARS` defaults false; device-provided masks can independently exclude entries. The returned LTE norm is the solution norm; the separately computed charge norm is not the ordinary return value. [OneStep error estimate](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/TimeIntegrationPKG/N_TIA_OneStep.h#L93), [DataStore weights, norm and correction, lines 1310–1472, 1495–1553 and 1739–1742](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/TimeIntegrationPKG/N_TIA_DataStore.C#L1310-L1553)

ESPice instead implements ngspice's CKTterr-style bound over individual device charge states, taking the tightest state bound. It combines current-scale and charge-scale tolerances with `TRTOL`, then compares the proposed bound with `0.9*dt`. Relevant code is `src/analysis/tran/integrator.zig:282–312,389–419` and `src/analysis/tran/tran.zig:859–903`; per-device charge selection is described at `tran.zig:461–479`. Its defaults include `RELTOL=1e-3`, current `ABSTOL=1e-12`, voltage `VNTOL=1e-6`, charge `CHGTOL=1e-14`, `TRTOL=7` (`src/core/numerics.zig:377–398`).

Consequently, matching a few numerical tolerance values does not make the criteria equivalent. A weighted RMS solution norm with a common global amplitude can allow a different distribution of local error than the strictest device charge-state bound. Whether that provides useful speed at acceptable waveform/measurement error is an experiment, not a conclusion that Xyce is inaccurate or merely "loose."

Source-reading caveat: `TIAParams::populateMetadata` contains a `RELTOL=1e-2` metadata value, while the actual constructor and release documentation give the time-integration default as `1e-3`. The options parser copies metadata only for input parameters that the deck supplied, so it does not install that entire metadata map as runtime defaults. [Options parser, lines 468–484](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/IOInterfacePKG/N_IO_PkgOptionsMgr.C#L468-L484)

**Newton convergence differs, but default Jacobian reuse does not explain the result.**

Xyce's transient nonlinear defaults are `ABSTOL=1e-6`, `RELTOL=1e-2`, weighted update limit `0.33`, RHS limit `1e-2`, maximum 20 iterations and full Newton steps. These are separately configured from time-integration tolerances. The native transient loop calculates update weights once per solve, tests a weighted maximum update, checks the maximum residual, and checks device convergence. [NLParams defaults](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/NonlinearSolverPKG/N_NLS_NLParams.C#L98-L111), [DampedNewton loop and convergence](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/NonlinearSolverPKG/N_NLS_DampedNewton.C#L429-L485), [residual/update checks](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/NonlinearSolverPKG/N_NLS_DampedNewton.C#L1230-L1333)

ESPice uses type-specific current/voltage update tolerances on each row and device/load convergence checks (`src/solver/converger.zig:451–490,787–810`). Its transient path deliberately disables the additional residual gate to follow ngspice acceptance (`src/analysis/tran/tran.zig:783–788`), and defaults to 10 Newton iterations per attempted step. Therefore a numerical comparison of `ABSTOL` alone is particularly misleading: it does not describe the same complete convergence test.

Xyce's native Newton loop invokes `jacobian_()` every iteration; that calls the loader, and the linear solve uses `solve(false)`. The Amesos backend retains the symbolic solver object but requests numerical factorization when reuse is false. This supports symbolic reuse, not a claim that default transient Newton skips the Jacobian or reuses unchanged numeric factors across iterations. [Jacobian load and solve call](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/NonlinearSolverPKG/N_NLS_NonLinearSolver.C#L902-L969), [Amesos symbolic and numeric factorization](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/LinearAlgebraServicesPKG/N_LAS_AmesosSolver.C#L309-L363)

ESPice already reuses its symbolic structure and pivot sequence, with monitored fallback to a fresh factorization, and bypasses numeric factorization when matrix values are exactly unchanged (`src/solver/direct.zig:215–241,289–297`). "Add factor reuse like Xyce" would not identify a missing capability here. Any remaining per-step gap should be assigned using model-evaluation, assembly and factor/solve timings.

**Accepted/rejected steps and possible next experiments.**

Xyce normally requires a non-failed Newton solve plus LTE estimate no greater than one; special first-step/restart and explicit options affect which test runs. Newton failure shrinks the step by eight and returns to the minimum order. LTE failure also lowers order and adjusts the step from the estimated ratio. Accepted-step growth uses a bounded ratio and promotes order when profitable. [Acceptance logic](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/TimeIntegrationPKG/N_TIA_StepErrorControl.C#L460-L569), [rejection](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/TimeIntegrationPKG/N_TIA_OneStep.C#L1965-L2014), [accepted-step control](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/TimeIntegrationPKG/N_TIA_OneStep.C#L2191-L2235)

ESPice already has error-based adaptive stepping, rejection rollback, predictor/history state, a factor-eight Newton retry, order promotion, device bounds and breakpoint landing. Its LTE rejection normally preserves the order and uses the calculated bound, which differs from Xyce's order reduction (`src/analysis/tran/tran.zig:811–818,881–903,934–948,997–1036`). Both have conservative breakpoint restart logic; no source basis was established for claiming Xyce uniquely avoids all small restart steps.

1. First run every postlayout case with an explicit common fourth-argument maximum step, retain both original and controlled results, and compare delay, edge time, oscillator period/phase, extrema and both native sample grids against a finer-step reference. Record accepted steps, failed attempts, Newton iterations, Jacobian loads, factorizations and wall time. This isolates the demonstrated input-semantics difference.
2. Then hold that ceiling fixed while varying only Xyce `NEWLTE=0` versus `1`, with explicit time-integration tolerances. This tests whether global amplitude weighting, rather than the ceiling, still explains a substantial step-count difference. A numerical value shared with ESPice remains an imperfect cross-policy match.
3. If the remaining gap is mostly Newton work, vary transient nonlinear tolerances independently from LTE and compare iteration distributions and output error. Do not simply transplant Xyce's numbers into ESPice's differently scaled tests.
4. If fewer accepted steps survive accuracy checks, an optional alternative error-control policy is a plausible research direction. Keep existing SPICE-compatible semantics as the default, calibrate the new policy against independent fine-step truth, and include small-amplitude critical nodes and rare switching spikes so an RMS/global scale cannot hide local failures.
5. Improve benchmark and CLI diagnostics immediately: display the resolved initial/max step and error policy in evidence. ESPice already permits an explicit larger `dt_max`, so testing that policy does not require a simulator implementation change.

Existing counters are sufficient for the first attribution pass: ESPice `ZP_TRAN_STATS` prints accepted/attempted/Newton/LTE-rejected counts at `src/analysis/tran/tran.zig:1097–1101`; Xyce's analysis summary prints successful/failed steps and solver-work counters. [Xyce summary source](https://github.com/Xyce/Xyce/blob/Release-7.10.0/src/AnalysisPKG/N_ANP_AnalysisBase.C#L338-L356)

No speedup or accuracy claim is made here for an unmeasured algorithm change. The paired explicit-ceiling and work-count experiments are being performed by the parent agent.
