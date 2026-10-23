# Measured simulator comparison

All ratios are reference wall time / ESPice wall time; above 1 means ESPice is faster.
The strict screen uses interpolation without time shifts, voltage/current absolute floors, RMS ≤ 0.1% of reference peak and maximum ≤ 1% of peak. It screens disagreement; it does not establish ground truth.

ESPice completed 49/49 selected circuits.

| Reference | Both completed | Historical agreement | Strict screen agreement | Speed ratio, all completed | Speed ratio, strict agreement | Scaling ratio (passing cases) |
|---|---:|---:|---:|---:|---:|---:|
| ngspice (Sparse) | 48/49 | 48 | 47 | 2.81× | 2.88× | 3.08× (12) |
| VACASK | 40/49 | 31 | 27 | 2.80× | 2.66× | 2.83× (4) |
| Xyce | 39/49 | 28 | 26 | 12.88× | 15.59× | 14.15× (4) |

Ratios above are geometric means. Failed or refused runs are excluded from speed ratios and retained in coverage counts.
Scaling is the predetermined `stress/` and `postlayout/` subset; the other cases are mostly small, where process startup dominates.
Every measured repetition, minimum, maximum and percentile is in `timings.csv`.

| Circuit | ESPice ms | ngspice (Sparse) ms | VACASK ms | Xyce ms |
|---|---:|---:|---:|---:|
| ac/bench_bjt_common_emitter.sp | 2.898 | 7.153 | 7.137 | 42.854 |
| ac/bench_medium_ladder_filter.sp | 2.934 | 7.191 | 7.353 | 43.609 |
| ac/device_bsim4_capmod2.sp | 2.772 | 6.963 | 7.053 | 43.375 |
| ac/device_diode_capacitance.sp | 2.747 | 7.113 | 6.967 | 42.588 |
| ac/rc_highpass.sp | 2.705 | 7.157 | 7.098 | 42.578 |
| ac/rlc_lowpass.sp | 2.727 | 7.054 | 7.050 | 42.260 |
| ac/rlc_near_resonance_0p1.sp | 2.692 | 7.117 | 7.115 | 42.317 |
| ac/unbuffered_rc_ladder.sp | 2.684 | 7.146 | 7.076 | 42.978 |
| dc/device_bjt_npn_gummel.sp | 2.926 | 7.481 | 7.504 | 46.346 |
| dc/device_bjt_npn_output.sp | 5.016 | 10.021 | 11.202 | 77.512 |
| dc/device_bsim3_output.sp | 3.835 | 55.021 | — | 66.914 |
| dc/device_bsim4_output.sp | 4.658 | 33.008 | — | 57.729 |
| dc/device_diode_high_injection.sp | 2.976 | 7.723 | 7.644 | 50.370 |
| dc/device_jfet_transfer.sp | 3.000 | 7.657 | 7.685 | 49.695 |
| dc/device_mos1_transfer.sp | 3.104 | 7.852 | 8.066 | 51.663 |
| dc/device_mos3_transfer.sp | 3.106 | 7.853 | — | 52.282 |
| dc/diode_forward.sp | 2.659 | 7.118 | 6.890 | — |
| dc/resistor_sweep.sp | 2.671 | 7.121 | — | 42.538 |
| noise/bench_noise_bjt_flicker.sp | 2.815 | 7.135 | 7.377 | — |
| noise/rc_1e-06.sp | 2.648 | 7.063 | 7.055 | — |
| op/balanced_bridge.sp | 2.629 | 7.153 | 6.802 | 42.420 |
| op/bench_bjt_cascode.sp | 2.753 | 7.167 | 7.149 | 42.878 |
| op/bench_medium_resistor_mesh.sp | 2.950 | 7.893 | 7.629 | 45.453 |
| op/bench_mosfet_nand2.sp | 2.730 | 7.235 | 6.971 | 42.868 |
| op/divider_default.sp | 2.622 | 7.158 | 6.910 | 42.385 |
| op/parallel_resistors.sp | 2.579 | 7.215 | 6.894 | 42.558 |
| postlayout/chain_bsim4_10k.sp | 10454.756 | — | — | — |
| postlayout/chain_bsim4_1k.sp | 1610.150 | 3850.183 | 3967.485 | — |
| postlayout/logic_bsim4_1k.sp | 4660.725 | 33595.355 | 14465.075 | — |
| postlayout/ring_bsim4_1k.sp | 4112.597 | 10185.420 | 8900.825 | — |
| stress/scaling_divider_chain.sp | 5.731 | 14.404 | 15.699 | 90.100 |
| stress/scaling_inverter_chain_256.sp | 410.789 | 532.078 | 4415.993 | 1632.144 |
| stress/scaling_inverter_chain_4k.sp | 6529.457 | 9201.919 | — | 34500.479 |
| stress/scaling_parallel_inverters_100.sp | 48.222 | 71.726 | 188.959 | 145.114 |
| stress/scaling_parallel_inverters_2000.sp | 935.903 | 1327.778 | 4144.710 | 2459.683 |
| stress/scaling_rc_ladder_1k.sp | 32.638 | 59.058 | 104.704 | 129.899 |
| stress/scaling_resistor_grid.sp | 5.555 | 19.537 | 15.595 | 79.657 |
| stress/scaling_resistor_grid_100x100.sp | 38.269 | 2125.931 | 114.017 | 492.023 |
| stress/scaling_resistor_grid_32x32.sp | 5.508 | 27.946 | 15.492 | 76.257 |
| tran/bench_digital_rc_filter_chain.sp | 2.955 | 7.990 | 7.862 | 47.472 |
| tran/bench_medium_rc_ladder_50.sp | 4.092 | 10.633 | 12.861 | 49.121 |
| tran/bench_mosfet_nmos_cs.sp | 3.029 | 7.677 | 7.733 | 44.302 |
| tran/bench_ngspice_mosamp.sp | 1975.216 | 1755.233 | — | — |
| tran/bench_ngspice_mosmem.sp | 3.990 | 8.354 | — | — |
| tran/bench_power_rectifier.sp | 7.819 | 20.221 | 19.776 | 147.283 |
| tran/bench_tran_rc_pulse.sp | 3.445 | 9.393 | 9.228 | 56.604 |
| tran/device_mos1_large_signal.sp | 4.608 | 10.474 | 11.937 | 54.642 |
| tran/pwl_triangle.sp | 4.772 | 15.519 | — | 96.889 |
| tran/rc_sinusoidal_startup.sp | 15.045 | 51.366 | 46.243 | — |

## Strict waveform disagreements

| Circuit | Reference | Signal | Maximum absolute difference | RMS / peak |
|---|---|---|---:|---:|
| dc/device_bjt_npn_gummel.sp | Xyce | incomplete coverage: missing: v(q1_baseprime), v(q1_collectorprime), v(q1_emitterprime) | — | — |
| dc/device_bjt_npn_output.sp | VACASK | incomplete coverage: missing sweep axis: outersweep | — | — |
| dc/device_bjt_npn_output.sp | Xyce | incomplete coverage: missing: v(q1_baseprime), v(q1_collectorprime), v(q1_emitterprime) | — | — |
| dc/device_diode_high_injection.sp | VACASK | i(v1) | 0.00072874 A | 1.66% |
| dc/device_diode_high_injection.sp | Xyce | i(v1) | 0.000232855 A | 0.69% |
| dc/device_jfet_transfer.sp | Xyce | incomplete coverage: missing: v(j1_drainprime), v(j1_sourceprime) | — | — |
| op/bench_mosfet_nand2.sp | Xyce | i(vdd) | 3.84635e-12 A | 27.7% |
| postlayout/chain_bsim4_1k.sp | VACASK | v(c9s49_0) | 0.000169046 V | 1.79% |
| postlayout/logic_bsim4_1k.sp | VACASK | v(g11_5_0) | 0.000166852 V | 1.42% |
| postlayout/ring_bsim4_1k.sp | VACASK | v(r0s15_0) | 6.34397e-05 V | 0.431% |
| stress/scaling_inverter_chain_256.sp | VACASK | i(vdd) | 0.00038081 A | 20.3% |
| stress/scaling_inverter_chain_256.sp | Xyce | i(vdd) | 0.00038023 A | 21.5% |
| stress/scaling_inverter_chain_4k.sp | Xyce | v(s310) | 1.79127 V | 17.9% |
| stress/scaling_parallel_inverters_100.sp | VACASK | i(vdd) | 0.00175848 A | 1.1% |
| stress/scaling_parallel_inverters_100.sp | Xyce | i(vdd) | 0.00188405 A | 1.98% |
| stress/scaling_parallel_inverters_2000.sp | VACASK | i(vdd) | 0.0351695 A | 1.09% |
| stress/scaling_parallel_inverters_2000.sp | Xyce | v(out1) | 0.124919 V | 1.93% |
| stress/scaling_rc_ladder_1k.sp | VACASK | i(vin) | 2.23236e-05 A | 0.542% |
| stress/scaling_rc_ladder_1k.sp | Xyce | i(vin) | 7.9755e-06 A | 0.236% |
| tran/bench_digital_rc_filter_chain.sp | VACASK | v(n1) | 0.026892 V | 0.135% |
| tran/bench_medium_rc_ladder_50.sp | Xyce | v(n1) | 0.0096464 V | 0.105% |
| tran/bench_ngspice_mosamp.sp | ngspice | i(vb) | 0.00349601 A | 0.133% |
| tran/bench_power_rectifier.sp | VACASK | i(vac) | 0.146708 A | 0.122% |
| tran/bench_power_rectifier.sp | Xyce | i(vac) | 0.14654 A | 0.232% |
| tran/bench_tran_rc_pulse.sp | VACASK | v(out) | 0.0147457 V | 0.106% |
| tran/device_mos1_large_signal.sp | VACASK | i(vin) | 0.000568408 A | 1.24e+10% |
| tran/device_mos1_large_signal.sp | Xyce | i(vin) | 0.000245285 A | 3.28% |
