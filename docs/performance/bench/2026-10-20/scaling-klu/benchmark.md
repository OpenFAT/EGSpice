# Simulator benchmark

- espice: `/tmp/espice-baseline-release/bin/espice` — espice 1.2.0
- ngspice: `ngspice` — ngspice-45 : Circuit level simulation program
- xyce: `Xyce` — Xyce Release 7.10.0-opensource
- column espice: `ESPICE_THREADS=1 ESPICE_LU_PAR=0 espice --jobs=1 --backend=cpu --full-waveforms`

Median of 3 measured runs after one warm-up; milliseconds include process startup and output. Runs are serial, with engine order rotated each repetition. VACASK conversion is outside timing.

| Fixture | espice ms | ngspice ms | Xyce ms | espice/ngspice | espice/Xyce | ngspice/Xyce |
|---|---:|---:|---:|---|---|---|
| postlayout/chain_bsim4_10k.sp | 15226.548 | 24148.416 | 12820.940 | agree | agree | agree |
| postlayout/chain_bsim4_1k.sp | 2667.393 | 3067.342 | 2197.065 | agree | agree | agree |
| postlayout/logic_bsim4_1k.sp | 8803.179 | 21443.322 | 22571.984 | agree | agree | agree |
| postlayout/ring_bsim4_1k.sp | 4919.877 | 6003.973 | 2017.871 | agree | agree | agree |
| stress/scaling_divider_chain.sp | 7.412 | 21.588 | 113.902 | agree | agree | agree |
| stress/scaling_inverter_chain_256.sp | 675.426 | 818.164 | 2606.896 | agree | unavailable | unavailable |
| stress/scaling_inverter_chain_4k.sp | 5814.574 | 8863.392 | 38260.952 | agree | unavailable | unavailable |
| stress/scaling_parallel_inverters_100.sp | 42.251 | 63.960 | 128.245 | agree | unavailable | unavailable |
| stress/scaling_parallel_inverters_2000.sp | 834.760 | 1208.531 | 2371.549 | agree | unavailable | unavailable |
| stress/scaling_rc_ladder_1k.sp | 30.542 | 61.551 | 118.299 | agree | DIFFER | DIFFER |
| stress/scaling_resistor_grid.sp | 4.796 | 13.441 | 69.962 | agree | agree | agree |
| stress/scaling_resistor_grid_100x100.sp | 56.214 | 205.752 | 729.662 | agree | agree | agree |
| stress/scaling_resistor_grid_32x32.sp | 8.242 | 24.560 | 117.815 | agree | agree | agree |

- stress/scaling_rc_ladder_1k.sp / espice/Xyce: complete=true, normalized max=7.975500098309274e-3, RMS=2.2970679952459038e-3
- stress/scaling_rc_ladder_1k.sp / ngspice/Xyce: complete=true, normalized max=7.975500098309496e-3, RMS=2.2970679952459164e-3
