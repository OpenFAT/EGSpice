# CPU lane corpus validation

The harness explicitly selects `--backend=cpu`; output parsing rejects nonfinite samples, missing/truncated data and process failures. Sweep selection now ignores comment text and keeps a single-temperature card from consuming the next line. Focused Python regressions: **14/14 passed** (`/tmp/espice-lane-diff-unit-final.log`).

| Family | Existing swept decks | Unswept decks eligible for four-way replication |
|---|---:|---:|
| tran | 4 | 182 |
| ac_family | 1 | 222 |
| periodic | 1 | 66 |
| Combined unique | 6 | 468 |

The combined genuine swept gate passed **6/6**, with zero differences, failures or timeouts. It used two concurrent decks, one CPU worker per simulator, and the current Debug CLI `/tmp/espice-208-gpu/bin/espice`.

Log: `/tmp/espice-lane-corpus-swept.log`.
Binary SHA256: `0958895c98b7e50557fbe4aaca2d4af94f26486c77806d519bb2c136b81ef9df`.

Validated swept decks:
- `hb/sweep_two_tone_cubic.sp`
- `optimize/bisection_rc_max.sp`
- `optimize/rc_delay_fit.sp`
- `variants/data_lane_stimulus.sp`
- `variants/data_sweep.sp`
- `variants/monte_tran_meas.sp`

The complete bounded replication queue is finished: **452 bitwise passes from 468 unique selected decks, zero numerical lane differences**. The remaining 16 outcomes are five unsupported/setup cases and eleven Debug timeouts; they are not passes and are not classified as numerical mismatches. The driver exited nonzero because the complete selected corpus did not pass.

Each successful case compared the finite binary waveform payload, variables, dimensions and flags of a scalar run with all four copies from a generated unused-parameter `.step`, with CPU lanes enabled for that replicated deck. Both processes used `--backend=cpu --jobs=1`; two cases ran concurrently, with a 120-second cap per process. This was a correctness gate under concurrent development load, not a performance measurement. Successful waveform comparisons do not prove physical accuracy or GPU admission, and do not replace the focused streamed-measure/checkpoint tests.

| Family | Selected replication decks | Bitwise passes | Scalar setup/expected refusal | Unsupported replication | Timeouts |
|---|---:|---:|---:|---:|---:|
| AC family | 222 | 221 | 1 | 0 | 0 |
| Periodic | 66 | 66 | 0 | 0 | 0 |
| Transient | 182 | 167 | 1 | 3 | 11 |
| Unique combined | 468 | 452 | 2 | 3 | 11 |

Two transient decks also belong to the AC family, so family rows are not additive. The unique combined row deduplicates them.

The non-timeout unavailable cases are:

- `hspice/unknown_card.sp`: the fixture intentionally expects `UnsupportedCard` for `.stateye`; its scalar run is not a waveform oracle.
- `hdl/veriloga_vpi_callbacks.sp`: the generic runner does not load the required `vpi_gain` fixture application; scalar execution reports `UnregisteredSystf`. The plugin has process-global state, so a generic `.step` replication is not automatically an equivalent test.
- `mosra/nmos_aged_vth.sp`, `mosra/nmos_dc_stress.sp`, `mosra/nmos_dtemp_arrhenius.sp`: each scalar run succeeded, but the frontend intentionally rejects combining `.mosra` with `.step`, `.alter` or an analysis sweep.

The 120-second timeouts remain unverified:

| Deck | Timed-out stage |
|---|---|
| `stress/scaling_inverter_chain_4k.sp` | Scalar |
| `stress/scaling_parallel_inverters_2000.sp` | Four-way replica |
| `stress/scaling_rc_ladder_100k.sp` | Four-way replica |
| `stress/vacask_graetz.sp` | Four-way replica |
| `stress/vacask_mul.sp` | Four-way replica |
| `stress/vacask_ring.sp` | Scalar |
| `tran/bench_ngspice_mosamp.sp` | Scalar |
| `tran/cpl_tran_history_past_2048.sp` | Four-way replica |
| `tran/device_mesa_oscillator.sp` | Four-way replica |
| `tran/ltra_tran_long_run_past_8192.sp` | Scalar |
| `tran/txl_tran_long_run_past_2048.sp` | Four-way replica |

Initial `.hdl` attempts inherited ambient Zig 0.16 and failed compilation. The queue was parked and resumed under `nix develop .#ci -c`, using pinned Zig 0.18.0-dev.349. Those compiler failures were retried successfully and are retained only as historical attempts. The consolidated report counts the latest completed result per fixture; it does not double-count retries or parked runs.

Evidence:

- `/tmp/espice-cpu-replicate-summary.json`: consolidated outcomes and full metadata.
- `/tmp/espice-cpu-replicate-summary.csv`: one row per selected fixture.
- `/tmp/espice-cpu-replicate-results.jsonl`: all 478 recorded attempts, including retries.
- `/tmp/espice-cpu-replicate-evidence/`: per-attempt source decks, stdout/stderr, raw hashes and compressed raw outputs for unavailable/differing cases.
- `/tmp/espice-cpu-replicate-progress.json` and `/tmp/espice-cpu-replicate.log`: final progress and execution log.

The corpus processes exited after the bounded queue finished. No additional simulator runs were started; consult the experiment timestamps for overlap with other measurements.

Exact selection and per-kind counts: `/tmp/espice-lane-corpus-scope.json`.
