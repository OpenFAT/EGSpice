# Measured simulator comparison

All ratios are reference wall time / ESPice wall time; above 1 means ESPice is faster.
The strict screen uses interpolation without time shifts, voltage/current absolute floors, RMS ≤ 0.1% of reference peak and maximum ≤ 1% of peak. It screens disagreement; it does not establish ground truth.

ESPice completed 13/13 selected circuits.

| Reference | Both completed | Historical agreement | Strict screen agreement | Speed ratio, all completed | Speed ratio, strict agreement | Scaling ratio (passing cases) |
|---|---:|---:|---:|---:|---:|---:|
| ngspice (KLU) | 13/13 | 13 | 13 | 1.89× | 1.89× | 1.89× (13) |
| Xyce | 13/13 | 8 | 4 | 3.72× | 14.28× | 14.28× (4) |

Ratios above are geometric means. Failed or refused runs are excluded from speed ratios and retained in coverage counts.
Scaling is the predetermined `stress/` and `postlayout/` subset; the other cases are mostly small, where process startup dominates.
Every measured repetition, minimum, maximum and percentile is in `timings.csv`.

| Circuit | ESPice ms | ngspice (KLU) ms | Xyce ms |
|---|---:|---:|---:|
| postlayout/chain_bsim4_10k.sp | 15226.548 | 24148.416 | 12820.940 |
| postlayout/chain_bsim4_1k.sp | 2667.393 | 3067.342 | 2197.065 |
| postlayout/logic_bsim4_1k.sp | 8803.179 | 21443.322 | 22571.984 |
| postlayout/ring_bsim4_1k.sp | 4919.877 | 6003.973 | 2017.871 |
| stress/scaling_divider_chain.sp | 7.412 | 21.588 | 113.902 |
| stress/scaling_inverter_chain_256.sp | 675.426 | 818.164 | 2606.896 |
| stress/scaling_inverter_chain_4k.sp | 5814.574 | 8863.392 | 38260.952 |
| stress/scaling_parallel_inverters_100.sp | 42.251 | 63.960 | 128.245 |
| stress/scaling_parallel_inverters_2000.sp | 834.760 | 1208.531 | 2371.549 |
| stress/scaling_rc_ladder_1k.sp | 30.542 | 61.551 | 118.299 |
| stress/scaling_resistor_grid.sp | 4.796 | 13.441 | 69.962 |
| stress/scaling_resistor_grid_100x100.sp | 56.214 | 205.752 | 729.662 |
| stress/scaling_resistor_grid_32x32.sp | 8.242 | 24.560 | 117.815 |

## Strict waveform disagreements

| Circuit | Reference | Signal | Maximum absolute difference | RMS / peak |
|---|---|---|---:|---:|
| postlayout/chain_bsim4_10k.sp | Xyce | v(c0s49_0) | 0.000139817 V | 0.513% |
| postlayout/chain_bsim4_1k.sp | Xyce | v(c9s49_0) | 9.32812e-05 V | 1.07% |
| postlayout/logic_bsim4_1k.sp | Xyce | v(g11_5_0) | 4.35703e-05 V | 0.533% |
| postlayout/ring_bsim4_1k.sp | Xyce | v(r14s15_0) | 3.48711e-05 V | 0.226% |
| stress/scaling_inverter_chain_256.sp | Xyce | i(vdd) | 0.00038023 A | 21.5% |
| stress/scaling_inverter_chain_4k.sp | Xyce | v(s310) | 1.79127 V | 17.9% |
| stress/scaling_parallel_inverters_100.sp | Xyce | i(vdd) | 0.00188405 A | 1.98% |
| stress/scaling_parallel_inverters_2000.sp | Xyce | v(out1) | 0.124919 V | 1.93% |
| stress/scaling_rc_ladder_1k.sp | Xyce | i(vin) | 7.9755e-06 A | 0.236% |
