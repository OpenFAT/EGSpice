# Simulator benchmark

- espice: `/tmp/espice-baseline-release/bin/espice` — espice 1.2.0
- ngspice: `ngspice` — ngspice-45 : Circuit level simulation program
- vacask: `vacask` — This is vacask unknown.
- xyce: `Xyce` — Xyce Release 7.10.0-opensource
- column espice: `ESPICE_THREADS=1 ESPICE_LU_PAR=0 espice --jobs=1 --backend=cpu --full-waveforms`

Median of 7 measured runs after one warm-up; milliseconds include process startup and output. Runs are serial, with engine order rotated each repetition. VACASK conversion is outside timing.

| Fixture | espice ms | ngspice ms | VACASK ms | Xyce ms | espice/ngspice | espice/VACASK | espice/Xyce | ngspice/VACASK | ngspice/Xyce | VACASK/Xyce |
|---|---:|---:|---:|---:|---|---|---|---|---|---|
| ac/bench_bjt_common_emitter.sp | 2.898 | 7.153 | 7.137 | 42.854 | agree | agree | agree | agree | agree | agree |
| ac/bench_medium_ladder_filter.sp | 2.934 | 7.191 | 7.353 | 43.609 | agree | agree | agree | agree | agree | agree |
| ac/device_bsim4_capmod2.sp | 2.772 | 6.963 | 7.053 | 43.375 | agree | agree | agree | agree | agree | agree |
| ac/device_diode_capacitance.sp | 2.747 | 7.113 | 6.967 | 42.588 | agree | agree | agree | agree | agree | agree |
| ac/rc_highpass.sp | 2.705 | 7.157 | 7.098 | 42.578 | agree | agree | agree | agree | agree | agree |
| ac/rlc_lowpass.sp | 2.727 | 7.054 | 7.050 | 42.260 | agree | agree | agree | agree | agree | agree |
| ac/rlc_near_resonance_0p1.sp | 2.692 | 7.117 | 7.115 | 42.317 | agree | agree | agree | agree | agree | agree |
| ac/unbuffered_rc_ladder.sp | 2.684 | 7.146 | 7.076 | 42.978 | agree | agree | agree | agree | agree | agree |
| dc/device_bjt_npn_gummel.sp | 2.926 | 7.481 | 7.504 | 46.346 | agree | agree | incomplete | agree | incomplete | incomplete |
| dc/device_bjt_npn_output.sp | 5.016 | 10.021 | 11.202 | 77.512 | agree | agree | incomplete | agree | incomplete | incomplete |
| dc/device_bsim3_output.sp | 3.835 | 55.021 | — | 66.914 | agree | unavailable | agree | unavailable | agree | unavailable |
| dc/device_bsim4_output.sp | 4.658 | 33.008 | — | 57.729 | agree | unavailable | agree | unavailable | agree | unavailable |
| dc/device_diode_high_injection.sp | 2.976 | 7.723 | 7.644 | 50.370 | agree | agree | incomplete | agree | incomplete | incomplete |
| dc/device_jfet_transfer.sp | 3.000 | 7.657 | 7.685 | 49.695 | agree | agree | incomplete | agree | incomplete | incomplete |
| dc/device_mos1_transfer.sp | 3.104 | 7.852 | 8.066 | 51.663 | agree | agree | agree | agree | agree | agree |
| dc/device_mos3_transfer.sp | 3.106 | 7.853 | — | 52.282 | agree | unavailable | agree | unavailable | agree | unavailable |
| dc/diode_forward.sp | 2.659 | 7.118 | 6.890 | — | agree | agree | unavailable | agree | unavailable | unavailable |
| dc/resistor_sweep.sp | 2.671 | 7.121 | — | 42.538 | agree | unavailable | agree | unavailable | agree | unavailable |
| noise/bench_noise_bjt_flicker.sp | 2.815 | 7.135 | 7.377 | — | agree | unavailable | unavailable | unavailable | unavailable | unavailable |
| noise/rc_1e-06.sp | 2.648 | 7.063 | 7.055 | — | agree | unavailable | unavailable | unavailable | unavailable | unavailable |
| op/balanced_bridge.sp | 2.629 | 7.153 | 6.802 | 42.420 | agree | agree | agree | agree | agree | agree |
| op/bench_bjt_cascode.sp | 2.753 | 7.167 | 7.149 | 42.878 | agree | agree | agree | agree | agree | agree |
| op/bench_medium_resistor_mesh.sp | 2.950 | 7.893 | 7.629 | 45.453 | agree | agree | agree | agree | agree | agree |
| op/bench_mosfet_nand2.sp | 2.730 | 7.235 | 6.971 | 42.868 | agree | agree | agree | agree | agree | agree |
| op/divider_default.sp | 2.622 | 7.158 | 6.910 | 42.385 | agree | agree | agree | agree | agree | agree |
| op/parallel_resistors.sp | 2.579 | 7.215 | 6.894 | 42.558 | agree | agree | agree | agree | agree | agree |
| postlayout/chain_bsim4_10k.sp | 10454.756 | — | — | — | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| postlayout/chain_bsim4_1k.sp | 1610.150 | 3850.183 | 3967.485 | — | agree | agree | unavailable | agree | unavailable | unavailable |
| postlayout/logic_bsim4_1k.sp | 4660.725 | 33595.355 | 14465.075 | — | agree | agree | unavailable | agree | unavailable | unavailable |
| postlayout/ring_bsim4_1k.sp | 4112.597 | 10185.420 | 8900.825 | — | agree | agree | unavailable | agree | unavailable | unavailable |
| stress/scaling_divider_chain.sp | 5.731 | 14.404 | 15.699 | 90.100 | agree | agree | agree | agree | agree | agree |
| stress/scaling_inverter_chain_256.sp | 410.789 | 532.078 | 4415.993 | 1632.144 | agree | DIFFER | unavailable | DIFFER | unavailable | unavailable |
| stress/scaling_inverter_chain_4k.sp | 6529.457 | 9201.919 | — | 34500.479 | agree | unavailable | unavailable | unavailable | unavailable | unavailable |
| stress/scaling_parallel_inverters_100.sp | 48.222 | 71.726 | 188.959 | 145.114 | agree | DIFFER | unavailable | DIFFER | unavailable | unavailable |
| stress/scaling_parallel_inverters_2000.sp | 935.903 | 1327.778 | 4144.710 | 2459.683 | agree | DIFFER | unavailable | DIFFER | unavailable | unavailable |
| stress/scaling_rc_ladder_1k.sp | 32.638 | 59.058 | 104.704 | 129.899 | agree | DIFFER | DIFFER | DIFFER | DIFFER | DIFFER |
| stress/scaling_resistor_grid.sp | 5.555 | 19.537 | 15.595 | 79.657 | agree | agree | agree | agree | agree | agree |
| stress/scaling_resistor_grid_100x100.sp | 38.269 | 2125.931 | 114.017 | 492.023 | agree | agree | agree | agree | agree | agree |
| stress/scaling_resistor_grid_32x32.sp | 5.508 | 27.946 | 15.492 | 76.257 | agree | agree | agree | agree | agree | agree |
| tran/bench_digital_rc_filter_chain.sp | 2.955 | 7.990 | 7.862 | 47.472 | agree | DIFFER | agree | DIFFER | agree | DIFFER |
| tran/bench_medium_rc_ladder_50.sp | 4.092 | 10.633 | 12.861 | 49.121 | agree | agree | DIFFER | agree | DIFFER | agree |
| tran/bench_mosfet_nmos_cs.sp | 3.029 | 7.677 | 7.733 | 44.302 | agree | agree | agree | agree | agree | agree |
| tran/bench_ngspice_mosamp.sp | 1975.216 | 1755.233 | — | — | agree | unavailable | unavailable | unavailable | unavailable | unavailable |
| tran/bench_ngspice_mosmem.sp | 3.990 | 8.354 | — | — | agree | unavailable | unavailable | unavailable | unavailable | unavailable |
| tran/bench_power_rectifier.sp | 7.819 | 20.221 | 19.776 | 147.283 | agree | DIFFER | DIFFER | DIFFER | DIFFER | agree |
| tran/bench_tran_rc_pulse.sp | 3.445 | 9.393 | 9.228 | 56.604 | agree | agree | agree | agree | agree | DIFFER |
| tran/device_mos1_large_signal.sp | 4.608 | 10.474 | 11.937 | 54.642 | agree | DIFFER | agree | DIFFER | agree | DIFFER |
| tran/pwl_triangle.sp | 4.772 | 15.519 | — | 96.889 | agree | unavailable | agree | unavailable | agree | unavailable |
| tran/rc_sinusoidal_startup.sp | 15.045 | 51.366 | 46.243 | — | agree | agree | unavailable | agree | unavailable | unavailable |

- dc/device_bjt_npn_gummel.sp / espice/Xyce: complete=false, normalized max=5.854511183331357e-6, RMS=1.5717595895416608e-6
- dc/device_bjt_npn_gummel.sp / ngspice/Xyce: complete=false, normalized max=5.854511182387667e-6, RMS=1.5717595894758118e-6
- dc/device_bjt_npn_gummel.sp / VACASK/Xyce: complete=false, normalized max=1.9775586389554622e-5, RMS=2.231800239339952e-6
- dc/device_bjt_npn_output.sp / espice/Xyce: complete=false, normalized max=1.7536603736762694e-6, RMS=4.79324420912698e-7
- dc/device_bjt_npn_output.sp / ngspice/Xyce: complete=false, normalized max=1.7536603771899518e-6, RMS=4.7932214187905e-7
- dc/device_bjt_npn_output.sp / VACASK/Xyce: complete=false, normalized max=1.3080577003351732e-6, RMS=4.780604500113931e-7
- dc/device_bsim3_output.sp / VACASK: SimulatorFailed
- dc/device_bsim4_output.sp / VACASK: SimulatorFailed
- dc/device_diode_high_injection.sp / espice/Xyce: complete=false, normalized max=2.3285515536853804e-4, RMS=1.7052435081994847e-4
- dc/device_diode_high_injection.sp / ngspice/Xyce: complete=false, normalized max=2.3285515536853457e-4, RMS=1.7052435081994798e-4
- dc/device_diode_high_injection.sp / VACASK/Xyce: complete=false, normalized max=5.257560855882817e-4, RMS=2.575004417408995e-4
- dc/device_jfet_transfer.sp / espice/Xyce: complete=false, normalized max=1.1688606526813405e-10, RMS=8.471411240896187e-12
- dc/device_jfet_transfer.sp / ngspice/Xyce: complete=false, normalized max=1.1688606526675762e-10, RMS=8.471411240630602e-12
- dc/device_jfet_transfer.sp / VACASK/Xyce: complete=false, normalized max=9.910981033152302e-8, RMS=6.951878341490856e-8
- dc/device_mos3_transfer.sp / VACASK: MissingRaw
- dc/diode_forward.sp / Xyce: unsupported Xyce directive (only standalone OP/DC/AC/transient decks)
- dc/resistor_sweep.sp / VACASK: .dc sweeps 'r2', which is not an independent source
- noise/bench_noise_bjt_flicker.sp / Xyce: unsupported Xyce directive (only standalone OP/DC/AC/transient decks)
- noise/rc_1e-06.sp / Xyce: unsupported Xyce directive (only standalone OP/DC/AC/transient decks)
- postlayout/chain_bsim4_10k.sp / ngspice: Timeout
- postlayout/chain_bsim4_10k.sp / VACASK: Timeout
- postlayout/chain_bsim4_10k.sp / Xyce: SimulatorFailed
- postlayout/chain_bsim4_1k.sp / Xyce: SimulatorFailed
- postlayout/logic_bsim4_1k.sp / Xyce: SimulatorFailed
- postlayout/ring_bsim4_1k.sp / Xyce: SimulatorFailed
- stress/scaling_inverter_chain_256.sp / espice/VACASK: complete=true, normalized max=9.730668796729188e-1, RMS=1.1320145921229656e-1
- stress/scaling_inverter_chain_256.sp / ngspice/VACASK: complete=true, normalized max=9.730668800404206e-1, RMS=1.1320145922707703e-1
- stress/scaling_inverter_chain_4k.sp / VACASK: Timeout
- stress/scaling_parallel_inverters_100.sp / espice/VACASK: complete=true, normalized max=3.31033732927892e-2, RMS=5.6319672917669265e-3
- stress/scaling_parallel_inverters_100.sp / ngspice/VACASK: complete=true, normalized max=3.3103373292789506e-2, RMS=5.631967291767309e-3
- stress/scaling_parallel_inverters_2000.sp / espice/VACASK: complete=true, normalized max=3.391022423388254e-2, RMS=5.55710138966616e-3
- stress/scaling_parallel_inverters_2000.sp / ngspice/VACASK: complete=true, normalized max=3.391022423380413e-2, RMS=5.557101389666539e-3
- stress/scaling_rc_ladder_1k.sp / espice/VACASK: complete=true, normalized max=1.766857007264211e-2, RMS=3.0111324651428454e-3
- stress/scaling_rc_ladder_1k.sp / espice/Xyce: complete=true, normalized max=7.975500098309274e-3, RMS=2.2970679952459038e-3
- stress/scaling_rc_ladder_1k.sp / ngspice/VACASK: complete=true, normalized max=1.766857007264272e-2, RMS=3.0111324651426893e-3
- stress/scaling_rc_ladder_1k.sp / ngspice/Xyce: complete=true, normalized max=7.975500098309274e-3, RMS=2.2970679952459072e-3
- stress/scaling_rc_ladder_1k.sp / VACASK/Xyce: complete=true, normalized max=1.2402382139163781e-2, RMS=3.3246283844940404e-3
- tran/bench_digital_rc_filter_chain.sp / espice/VACASK: complete=true, normalized max=5.455896071409966e-3, RMS=1.0527611568830037e-3
- tran/bench_digital_rc_filter_chain.sp / ngspice/VACASK: complete=true, normalized max=5.455896071409239e-3, RMS=1.0527611568828784e-3
- tran/bench_digital_rc_filter_chain.sp / VACASK/Xyce: complete=true, normalized max=5.531839191614333e-3, RMS=1.4812514022594758e-3
- tran/bench_medium_rc_ladder_50.sp / espice/Xyce: complete=true, normalized max=9.646396324536621e-3, RMS=1.4307380257783408e-3
- tran/bench_medium_rc_ladder_50.sp / ngspice/Xyce: complete=true, normalized max=9.646396324535955e-3, RMS=1.430738025778273e-3
- tran/bench_ngspice_mosamp.sp / VACASK: .options trtol= changes the answer and has no known VACASK spelling
- tran/bench_ngspice_mosamp.sp / Xyce: unsupported Xyce directive (only standalone OP/DC/AC/transient decks)
- tran/bench_ngspice_mosmem.sp / VACASK: SimulatorFailed
- tran/bench_ngspice_mosmem.sp / Xyce: unsupported Xyce directive (only standalone OP/DC/AC/transient decks)
- tran/bench_power_rectifier.sp / espice/VACASK: complete=true, normalized max=4.989866757642113e-2, RMS=1.431409102646946e-3
- tran/bench_power_rectifier.sp / espice/Xyce: complete=true, normalized max=4.984218450208274e-2, RMS=2.6621764816177978e-3
- tran/bench_power_rectifier.sp / ngspice/VACASK: complete=true, normalized max=4.989866757641977e-2, RMS=1.4314091026383995e-3
- tran/bench_power_rectifier.sp / ngspice/Xyce: complete=true, normalized max=4.984218450208191e-2, RMS=2.6621764816168827e-3
- tran/bench_tran_rc_pulse.sp / VACASK/Xyce: complete=true, normalized max=2.9773333249014694e-3, RMS=1.0457379809441464e-3
- tran/device_mos1_large_signal.sp / espice/VACASK: complete=true, normalized max=1.2057072942051748e-2, RMS=1.6607437449402238e-3
- tran/device_mos1_large_signal.sp / ngspice/VACASK: complete=true, normalized max=1.2057072942052903e-2, RMS=1.6607437449403283e-3
- tran/device_mos1_large_signal.sp / VACASK/Xyce: complete=true, normalized max=1.2378972571920964e-2, RMS=3.776413017230618e-3
- tran/pwl_triangle.sp / VACASK: vin: PWL is not translated (VACASK's `type="pwl"` aborts on this build)
- tran/rc_sinusoidal_startup.sp / Xyce: unsupported Xyce directive (only standalone OP/DC/AC/transient decks)
