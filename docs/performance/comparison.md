# Why simulator results can differ

A benchmark `DIFFER` means the saved outputs exceeded the comparison
tolerances. Fixture correctness is checked separately against its
`.expected.json` oracle. The published performance reports were measured on
older ESPice binaries; they are not a verdict on the current source.

## Model differences in the references

These three labels explain some differences without changing the circuit or
relaxing its oracle. VACASK observations below apply to revision
`bcd48e2dd25182f5aaa3392c4e27b4e198372744`, pinned in
[`nix/vacask.nix`](https://github.com/OmarSiwy/ESPice/blob/main/nix/vacask.nix).

| Label | Cases | Evidence |
|---|---|---|
| **VACASK diode high-injection omission** | `reference/diode_high_injection` | Its `devices/spice/diode.va` initializes `DIOforwardKneeCurrentGiven` to zero and never enables it. Changing `IKF` leaves VACASK's output at 0.752086 V. With `IKF=.01`, ESPice and ngspice give 0.798301 V. |
| **VACASK MOS1 overlap-capacitance omission** | `tran/device_mos1_large_signal` | Without `TOX`, `devices/spice/mos1.va` sets `meyer_scale=0`, also suppressing the supplied overlap capacitances. VACASK's gate current is zero; ESPice/ngspice peak near 0.568 mA. An artificial large `TOX` isolated the cause; it is not a replacement model card. |
| **Different model versions or variants** | HICUM, SOI, VBIC and PSP cases | A matching model name or `LEVEL` does not guarantee matching equations. HICUM/L2 is 3.2.0 in ESPice versus 2.4.0 in ngspice 45; VBIC is 1.3 versus 1.2. ESPice's SOI level mappings select a different implementation. VACASK's PSP103.4 and ESPice's PSP103.8.2 also differ. |

The PSP version mismatch is established, but does not establish the whole
cause of each historical post-layout failure. Full-size BSIM4 chain and
SRAM reproductions instead identify numerical startup and initial-state
differences; see the controlled experiments below.

### Remaining ESPice model gap: BSIM4 channel noise

The optional BSIM4 `NOISE_DERIVATIVES` block stays disabled because its
hand-written derivatives previously differed from ngspice by up to 6× in
noise power. [#119](https://github.com/OmarSiwy/ESPice/issues/119) is closed:
`noisePsd` now receives seeded derivatives. However, BSIM4 still has no
replacement wiring for `Gm/Gds/Gmb`, so the `tnoiMod=1` channel thermal
contribution remains zero. A fresh fixture run gives 0.3332 nV/√Hz versus
ngspice's 1.7523 nV/√Hz at 10 Hz. The remaining work is in the model, and
simply enabling the inaccurate hand-written block is not a verified fix.

## Implementation and benchmark corrections

| Area | Verified error and correction |
|---|---|
| **ESPice LTRA AC** | The recursive-convolution line supplied its DC resistance during AC. The device frequency hook now supplies the exact distributed RLC/RC two-port, including phase and attenuation. DC and transient convolution retain their existing equations. The two existing analytic AC oracles and an additional distributed RC case cover the response. |
| **SFFM translation** | ngspice 45 orders its arguments as `VO VA FM MDI FC`. The VACASK adapter previously treated the third argument as the carrier, inherited different defaults, and omitted the modulation-index clamp. It now maps the frequencies, supplies ngspice defaults from `TSTOP`, and limits the index to `[0, FC/FM]`. Unsupported delay/phase arguments are refused. |
| **HSPICE dialect** | The runner told ESPice to use HSPICE syntax but gave the original deck to references using ngspice semantics. This changed temperature defaults, `GSHUNT`, transient segments and SFFM. The ngspice and VACASK adapters now explicitly refuse HSPICE input until they can translate it faithfully. Such a row is unavailable with a reason, rather than a comparison of different circuits. |
| **Behavioral power** | In ngspice 45's default mode, both `pow(x,y)` and `x^y` use `abs(x)` as the base. ESPice's constant-integer optimization wrongly preserved its sign. The optimization now agrees with the general power and constant evaluator. The general power also uses `pow` directly, avoiding `log(0)*0` at Newton's zero start. Fixtures intended to test a signed cubic use explicit multiplication and retain their numeric oracles. |
| **Pole/zero names** | ngspice wraps roots as `v(pole(k))` / `v(zero(k))`. Normalizing those names lets the existing unordered-root comparison work. Wrong or missing roots still fail. |

The SFFM convention is specified in §4.1.5 of the
[ngspice 45 manual](https://ngspice.sourceforge.io/docs/ngspice-45-manual.pdf).
Default power behavior is explicit in that release's
[`PTpower` and `PTpowerH`](https://github.com/imr/ngspice/blob/ngspice-45/src/spicelib/parser/ptfuncs.c).
The transmission-line equations also appear in
[`LTRAacLoad`](https://github.com/imr/ngspice/blob/ngspice-45/src/spicelib/devices/ltra/ltraacld.c);
the fixture oracles use the independent closed-form telegrapher solution.

A targeted rerun of 13 fixtures with the corrected adapters found agreement
in all nine available ESPice/ngspice comparisons, including both LTRA AC
regressions, the RC line, powers and pole/zero ordering. SFFM also agreed
with VACASK. The four HSPICE references were explicitly unavailable, and
the VACASK diode and MOS1 model differences remained reproducible.

## Numerical differences

Equal tolerances and `.tran` text do not impose equal accepted timesteps,
startup history or integration errors. Refine both simulations and compare
against an analytic solution where possible before assigning blame.

Research on the stored failures found:

- **Stiff RC and diode reverse recovery:** VACASK's initial step after a
  breakpoint was too coarse for the waveform comparison. On the analytic
  1 ns RC ramp, changing only `tran_fbr` from 0.2501 to 0.01 reduced VACASK's
  maximum voltage error from 21.2 mV to 88.2 µV.
- **256-stage inverter chain:** confirmed accumulated numerical timing
  error under the default integration controls. Three fresh ReleaseFast
  runs give the same first stage-256 rising crossing, 24.934747756 ns.
  An independent MOS1/RK4 calculation converges to 24.639601135 ns:
  a 295.146621 ps default error. With a 0.125 ps maximum step and tighter
  tolerances, ESPice gives 24.639602779 ns, within 1.65 fs of that limit.
  The earlier Debug timeouts are superseded by completed refinement runs.
  Default ngspice agreement therefore does not establish timing accuracy.
- **BSIM4 chain, 100k transistors:** confirmed UIC startup-resolution
  difference. ESPice/ngspice start at 20 fs and agree within 4.25 pV;
  VACASK starts at 500 fs, producing a 130.75 mV difference. Starting
  both engines at 0.2 fs reduces the full 200 ps comparison to 1.77 mV.
  That remaining difference is within the 2–3 mV changes seen during each
  engine's own refinement. Strict accuracy remains uncertified: tighter
  startup probes abort in both engines.
- **BSIM4 SRAM, 10k transistors:** confirmed ambiguity in the initial
  memory state. The original roughly 0.7 V difference is already present
  at time zero. Different initial guesses converge to different operating
  points even with tight tolerances. Selecting the same branch with
  nodesets reduces the ESPice/ngspice difference to 0.724 µV over the full
  1 ns run. ESPice/VACASK differ by 6.14 mV with those nodesets and by
  2.05 mV after step/tolerance refinement, concentrated immediately after
  a source breakpoint. This does not establish a BSIM4 model error.
- **Parallel inverters, RC ladders, clamp and rectifier:** representative
  cases passed the existing waveform thresholds after refinement. In the
  rectifier, a small voltage error was amplified by the exponential diode
  current. This does not establish a cause for every larger circuit in
  the same family.
- **PVT inverter chain:** its first two stages have no explicit capacitance
  and very sharp algebraic transitions. An independent MOS1 DC-law check
  of their saved points found maximum voltage errors of about 0.33 mV
  for ESPice/ngspice and 2.55 mV for VACASK. Stricter VACASK runs aborted
  with `Timestep too small`, including a shortened first-edge probe with
  trap and Gear integration. Full waveform convergence remains unproven.

These observations do not justify changing ESPice's solver defaults,
loosening the comparator, or adding capacitances to the original circuits.

## Sensitivity and historical comparison artifacts

`sens/diode_ngspice` intentionally omits `IKF` and `ISR` from its selected
oracle. ngspice reports a NaN for the disabled `IKF` parameter; the broad
benchmark rejects nonfinite output. Direct ngspice operating-point
perturbations at `ISR=1e-18`, `1e-17` and `1e-16` converge toward
−9.7393×10⁷ V/A, whereas its `.sens` column is about a million times
larger. There is no reason to reproduce that reference value in ESPice.
Instance/model parameter naming differences also remain.

The five published CPL/TXL transient differences came from repeated current
column names. The current comparator already selects the last occurrence
on both sides; fresh reproductions agree with that pairing. No further
change to duplicate-column handling is needed.

The comparator's thresholds and nonfinite checks remain unchanged. Its
transient interpolation and local time-shift allowance make agreement a
waveform comparison, not a general certificate of numerical accuracy.
