# Newton and timestep policy

Levers that change how much work a transient does per step and how many
steps it takes, as opposed to how fast each piece runs. Most of them touch
the conformance contract: ESPice publishes ngspice's iterate x_k, applies
ngspice's convergence test and takes ngspice's LTE-controlled grid by
default, and nine corpus oracles encode ngspice's per-step Newton error
(`dev/solvers/gpu-convergence.md` §10.1: a solver that is *more* accurate
per step fails them at rtol 3e-3). So every item below is marked **exact**
(bitwise or within rounding, default-mode), **tolerance** (same grid and
test, values move within the oracle tolerance) or **opt-in** (a named
`--solver`/`.options` mode, gated on the pass set staying a superset plus a
recorded exemption list).

## What a step costs today

*Measured* (`dev/analysis/evaluation-profile.md`, `dev/research/lane-batched-simulation.md` §1,
`docs/performance/bench/2026-10-20/xyce-research.md`):

| Deck | eval | limiting + gates + staging | charge re-read | LTE | factor + solve | rest |
|---|---:|---:|---:|---:|---:|---:|
| mos6_inverter (callgrind Ir) | 54.5% | 13.9% | 5.3% | 2.7% | 11.4% | |
| parallel_inverters_100 (Ir) | 55.8% | 13.1% | 5.8% | 4.7% | 9.2% | |
| asap7 24-stage bsimcmg chain (wall) | 64% | 21% | | | 2.2% | 13% |
| ring_bsim4_1k (wall) | 52% | | | | 21% | |

Per accepted step: about 4 device evaluations (4.3 on mos1_2000, 5.2 on
vacask_ring, `src/analysis/executor.zig:928-931`), 2.0-3.9 Newton
iterations (2.0 on quiescent decks, 3.9 on the inverter chains, 3.05 on the
ring; `gpu-convergence.md` §9.1), and one charge re-read when the device
state moved the charges (2.4-8.0%, `dev/analysis/transient-integration.md`).
Every iterate stamps every batch and runs every limiter
(`src/analysis/Circuit.zig:852-863, 940-952`); there is no bypass. The
per-attempt Newton cap is 100, ngspice's own floor (`src/solver/converger.zig:81-96`);
the residual gate is off in transient (`tran.zig:798-803`).

## Items

### 4.1 Latency bypass: evaluate only the instances whose voltages moved — P0, tolerance

**Today.** None. The activity census (`gpu-convergence.md` §9.2, ngspice's
bypass reference: an instance is active when a gathered unknown moved more
than 1e-3·|v| + 1e-6 since its own last active evaluation):

| Deck | model instances | median active | evals under 20% active |
|---|---:|---:|---:|
| scaling_inverter_chain_4k | 8,000 | 0.5% | 100% |
| scaling_inverter_chain_256 | 512 | 6.6% | 100% |
| scaling_parallel_inverters_2000 | 4,000 | 0% (mean 27%) | 73% |
| bench_tran_fourbitadder | 288 | 30.6% | 44% |
| vacask_ring | 18 | 100% | 0.2% |

**Change.** Host first, behind a pin, then default when the gate holds:
per instance keep `v_ref` (the gathered unknowns at its last active
evaluation) and its last stamp; skip eval, limit and `updateState` when
every unknown is within the tolerance. ParEval's staging already holds every
instance's contributions at its own positions and reduces them in tape order
(`src/analysis/par_eval.zig`), so a skipped instance's stamp is simply not
rewritten; the serial path needs the same per-instance staging (one plane
copy's worth) or an undo-and-restamp. The stamp error of a skipped instance
is bounded by its local Jacobian times its voltage change, which is below the
Newton tolerance by construction of the threshold; record the bound per
step in `ZP_TRAN_STATS`. On the GPU the eval kernel reads an `active` byte
per instance (the lane kernels already do, `LaneEvalKernel`'s `active[l]`)
and the staging keeps latent contributions; compaction (a scan to an index
list) pays only once the active set is under one wave (6,144 threads on this
card at 255 registers).
**Expect.** The 4k chain evaluates about one MOSFET in 200: eval, limiting
and staging (70-85% of its run) shrink toward zero and the LU and step
control remain. Estimate 3x on the chains, nothing on the ring or the
parallel inverters at an edge. Then [02](02-linear-solver.md) 2.8's partial
refactor is the next lever on the same decks.
**Gate.** Full corpus at oracle tolerance, pass set a superset; `.meas`
values on the chain decks against the exact run. ngspice ships bypass on by
default, so the conformance direction is right. **Effort.** 1-2 weeks.

### 4.2 The default step ceiling — P1, accuracy decision

**Today.** `.tran tstep tstop` resolves the maximum step to
min(tstep, span/50) (`src/frontend/analyses.zig:515-534`), ngspice's rule.
Xyce resolves span/10. On the BSIM4 ring that is 1,013 accepted steps
against 128, and the LTE never binds on the asap7 chain at any tstep from
0.25 to 10 ps: the ceiling alone sets both cost and accuracy there
(`lane-batched-simulation.md` §1, cost linear in the step count). Raising
the ring's ceiling to 200 ps moved one output by 79 mV (`xyce-research.md`).
**Change.** Three steps. (a) Print the resolved ceiling and the LTE policy in
`--timing-in-depth` and the bench reports, so a comparison names it. (b)
Finish the ceiling experiment: 2, 5, 10, 50, 200 ps against 0.1-0.2 ps
references at tight tolerances on the post-layout set, judged on delay,
edge time, period and small-signal nodes, not on default-to-default
agreement (the 256-stage chain carries a 295 ps default-setting error that
default ngspice shares, `docs/performance/comparison.md`). (c) If a wider
ceiling plus a tighter `trtol` or a weighted-RMS LTE (Xyce's `NEWLTE`) holds
accuracy on that set, ship it as an opt-in `.options` preset, not the
default. Dense output (the integration polynomial at the reference times)
is needed for the corpus harness to score a coarser grid fairly
(`gpu-convergence.md` §6).
**Expect.** On step-ceiling-bound decks the cost is linear in the ceiling;
the ring would be 3-8x cheaper at 10-50 ps if accuracy held. Unmeasured.

### 4.3 Up to 100 Newton iterations on a failing attempt — P2, opt-in

**Today.** `optionsFromTolerances` never lets the cap fall below 100
(`src/solver/converger.zig:85-96`), because ngspice's NIiter raises any
smaller `maxIter` to 100 itself (niiter.c:38); ITL4 = 10 only steers the
callers' heuristics. So this is conformant, and `transient-integration.md`
is wrong where it says ITL4 bounds the attempt. The cost is real all the
same: a non-converging attempt burns up to 100 iterations, two syncs each
on the GPU path, before the dt/8 retry.
**Change.** Count the iterations spent on rejected attempts
(`ZP_TRAN_STATS` has `rej_newton`; add the iteration sum) on the
rejection-heavy decks (device_tline 664 rejected attempts,
bench_tline_ideal_tline 758, device_switch_hysteresis 464, device_urc 320:
`gpu-evaluation.md`, "Host-stepped"). If the sum is material, an early
divergence cut (give up when the scaled step grows for k iterates in a
row) is an opt-in mode: it changes which attempts are rejected, so the grid
moves on those decks.

### 4.4 Charge re-read without a second full evaluation — P1, exact

**Today.** When the accepted state moved the charges (`!q_fresh`, Meyer
models and every stateful charge), `end` re-evaluates the devices at the
accepted x for q alone (`tran.zig:885, 1223-1230`). On the CPU that is
`evalQOnly`, 2.4-8.0% of the run; on the GPU there is no charge-only
kernel, so it is one more full device evaluation per step, about +44% of
the evaluations on the chain (`dev/research/plans/README.md`, risk 3).
**Change.** A charge-only kernel `arp_q_<model>` (the value lanes only, no
derivatives: a fraction of the eval frame) or fuse `acceptQ` into the state
half of the state kernel, as the host's fused `acceptQ` already does
(`src/device/eval.zig`, `acceptQ`). **Expect.** Evaluations per step on the
GPU from ~2.25 + 1 to ~2.25 on the chain. **Gate.** Bitwise against the
unfused path (same arithmetic, same order). **Effort.** 2-3 days.

**Status (2026-10-22): done.** `arp_qonly_<model>` (eval.zig
`ChargeKernel`) writes the q rows and the LTE tape at x, values only, and
`GpuContext.evalQ` reduces and downloads the q plane alone; g, c and rhs
keep the last eval, as on the host. Taken whenever every resident batch
with charge has the kernel; `ESPICE_GPU_NOFUSE` keeps the full pass.
Not bitwise against the unfused GPU path where limiting is active at the
accepted point: the full pass read the lim plane (x_k+1's limited
voltages), the kernel reads x unlimited, as the host's charge pass does.
parallel_inverters_2000 stays byte-identical; chain_bsim4_10k within
6.1e-12 V (with 01 1.6). Device time per charge pass on chain_bsim4_10k
1,328 → ~330 us.

### 4.5 Per-step O(n) host passes — P1, exact, small each

- `integrator.stepBound` is 2.7-4.7% of the small decks; 2 of its ~8
  instructions per `@max`/`@min` are NaN handling. The constant operands
  (`chgtol`, `abstol`) can use a plain select; the data-data maxes must
  keep NaN propagation (`evaluation-profile.md`, "Still open").
- History rebase and rotation loops are 1.5-2.4% (`simulateInto` self);
  the x ring already rotates by pointer. The remaining copies are
  `q_vec` (n), the tape (n_qt) and one more n copy for stateful charge per
  accepted step (`tran.zig:1227-1228, 1262`).
- `.tran` plus `.four` or `.fft` integrates the same transient twice:
  `.four` takes the deck's own `.tran` options and calls `tran.simulate`
  again (`src/frontend/analyses.zig:1504`, `src/analysis/post/four.zig:118`,
  `post/fft.zig:75`). Make the post analyses read the `.tran` result when
  the cards match ([05](05-scheduling-and-parallelism.md) 5.11).
- The results arena doubles the waveform buffer and leaves the old rows
  resident (`src/analysis/tran/types.zig:136-153`): see
  [08](08-output-and-memory.md).

### 4.6 Operating-point ladder — P1

The 4k chain's OP went from 308 s to 15 s with the gmin and source-step
rules (`dev/analysis/operating-point-homotopy.md`). What is left:

- Source stepping rebuilds the constant baseline per λ
  (`src/analysis/dc/op.zig:230-275`); JFNK (rung 4) runs a full assemble per
  finite-difference matvec; OPtran is a 1 µs `uic` transient at 10 ns steps.
- **Exact rung racing** (`gpu-convergence.md` §2.3): rungs 2-5 start cold,
  so they can run on worker threads from cloned circuits with the first in
  ladder order taken. Serial result by construction; needs the latches not
  to leak between rungs (`converger.run` clears limits). Worth it only on
  the `convergence/` decks and DC sweeps that fall to the ladder per point.
- **Consensus multi-start** (§7.6) is speculative and OP-only; latch decks
  make it unsafe as a default. Opt-in, last rung before OPtran.
- Dependents never inherit the OP's workspace, ordering or first pivoting
  factor (`src/analysis/Circuit.zig:418, 592-593`): every transient after an
  OP repeats them. [05](05-scheduling-and-parallelism.md) 5.6.

### 4.7 Modified Newton and acceptance at iterate 0 — P2, opt-in

The census (`gpu-convergence.md` §9.1): iterate 0 already passes every gate
except the first-iterate floor in 65% of `tran/` solves and 95% of
`stress/` solves (42% of all evaluations), and a kept LU would fire
CVODE's refresh trigger on 5-20% of attempts against 1.1-3.9 factors per
attempt today. The built mode (`--solver=native`, retired, branch
`retired/native-modified-newton`) cut factors 2-40x but raised iterations
22-80% on the always-active decks, won 0.55-0.97x on quiescent and
factor-heavy decks, lost 1.06-1.54x on the chains, the ring and the adder,
and failed 9 oracles because it published a more accurate iterate (§10).
Re-open only as an opt-in mode with (a) the exemption list or a tighter
reference for those 9 decks and (b) the lagged-LU half alone, which is the
part that pays on LU-bound post-layout decks. Chord iterations need only
the residual, which on the GPU is a residual-only kernel (n + 1 doubles
down): pair with [01](01-gpu-resident-newton.md).

### 4.8 Speculative lanes on the retry chains — P2, exact

Known in advance: dt/8 after a Newton failure, the gmin retreat factors,
source-step halving (`gpu-convergence.md` §2.3). Evaluate the chain as lanes
from the same accepted state and commit the first candidate in serial order.
Pays only where rejections are frequent (`ZP_TRAN_STATS` `rej_newton`,
`rej_lte`) and the lanes are free (the E2 headroom knee). The asap7 chain
had zero rejections at every tstep; the switch and line decks above have
hundreds. GPU dt-candidate lanes (dt, 2dt, 4dt, 8dt, accept the largest that
passes LTE) remove ngspice's 2x growth cap after a breakpoint, which costs
about 4 serial steps per edge; that is a grid change, so opt-in.

### 4.9 Integration methods — P2

Trap with xmu is the default, Gear 2-6 exist, TR-BDF2 does not
(`src/analysis/tran/integrator.zig:40-136`). Higher order cuts steps on
smooth decks but changes the grid: opt-in. MATEX (`.matex`) has no
nonlinear path and costs one full `eval` at x = 0 plus three G solves, an
Arnoldi basis of up to 80 and a dense LU per step
(`src/analysis/tran/matex.zig:312-319, 443-563`); routing linear decks to it
was dropped because it fails pulse edges by 50-89,000x the tolerance and
ignores `uic` (`gpu-convergence.md` §9.4). Revisit only after it handles a
fast source edge and `uic`.

### 4.10 DC sweep and lane rows — P2

**Status (round 2, CPU half): done.** A `.dc` block whose first point
sets every swept device to its netlist value takes the OP prerequisite's
answer (`Sweep.isOpPoint`): that cold ladder is the prerequisite's own
solve (same circuit, the `.dc` tolerances, the same nodesets). Measured
on a 4,000-stage chain `.dc Vdd 1.8 1.5 -0.1`: 2.2-3.4 to 1.2-1.3 s in
ReleaseFast (at load 26-36), 11.0 to 5.3 s in Debug, raw output
identical. An MC, `.dc DATA=` or temperature lane whose plain Newton
fails now takes the OP ladder from its own start (`lanes.rescue`)
instead of being dropped. Gate: full corpus A/B, 835 SAME and 105 DIFF,
104 of them identical `exit 1/1` failures and one
(dc/device_mesa_inverter) within 1.5e-16; the correctness runner fails
only the 34 known decks. The GPU `.dc` lane driver is 06 6.9's.

The first point of every `.dc` block runs the cold full ladder; later
points use a 2x − x_prev predictor and warm Newton
(`src/analysis/dc/dc.zig:292-347`). MC and temperature lane rows run a full
`recompute` and plain Newton only (`src/analysis/sweep/lanes.zig:55-67`):
a lane that fails plain Newton is dropped as non-converged, where the
ladder would recover it (yield, not speed). `.dc` has no GPU lane path
(`FEATURE.md`, "DC and OP sweeps"): the Newton kernels already serve it,
missing is a DC driver over them with a CPU fallback for lanes that need
gmin or source stepping.

## Closed, with the reason

| Idea | Verdict | Where |
|---|---|---|
| Finite-difference JFNK as the GPU solver | the Jacobian is free from AD; m + 1 syncs per step; 2-18x slower measured | `gpu-convergence.md` §1.3, §2.7 |
| Parareal, MGRIT, waveform relaxation | different grid; K large on oscillators and switching decks | `gpu-convergence.md` §2.8, §7.9 |
| Nonlinear elimination of internal nodes | internal unknowns lead 8.6% of hard-OP iterates; keep only as a 3-deck prototype after 4.7 | `gpu-convergence.md` §9.3 |
| MATEX routing for linear decks | 2-3x more work and wrong on pulse edges today | `gpu-convergence.md` §9.4 |
| Higher-order (Halley) Newton | saves at most one iteration where Newton is already quadratic; globalization is the limit | `gpu-convergence.md` §2.9 |
| Rosenbrock/W-methods as the default integrator | nowhere to apply `$limit`; tiny-circuit ensembles only | `gpu-convergence-fields.md` §4 |
