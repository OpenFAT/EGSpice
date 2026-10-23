# Independent review of the finalized comparison

Reviewed `tools/comparison_accuracy.py` and `tools/comparison_report.py` read-only,
plus the finalized timing and diagnostic JSONL in
`zig-out/comparison-2026-10-20`.

No denominator or aggregation error was found in the current data. Independent
standard-library calculation reproduces:

| Reference | Completed pairs | Strict pass | Complete numeric coverage | Geometric speed ratio, strict pass |
|---|---:|---:|---:|---:|
| ngspice | 48 | 47 | 48 | 2.8782937407512006 |
| VACASK | 40 | 27 | 39 | 2.6575357514685334 |
| Xyce | 39 | 26 | 35 | 15.586497514884295 |

There are 196 unique timing rows, 49 for each engine. There are no duplicate
fixture/reference diagnostics, no missing diagnostics for a completed pair,
no failed timing row with a median, and no pass flag inconsistent with plot
completeness or signal flags. Speed ratios exclude failed/refused runs;
coverage retains those runs. Strict agreement ratios only include the stated
passing subset and the graph marks nonpassing completed pairs.

The strict scorer rejects nonfinite/truncated data, missing reference plots
or public signals, mismatched DC coordinates, and incomplete transient spans.
Transient comparisons inspect both simulators' sample times without shifting;
matched discontinuities preserve each ordered limit. Complex differences use
the magnitude of the complex subtraction. Small currents use a 1 pA absolute
floor rather than a one-ampere scale. Noise PSD-to-amplitude conversion is
explicitly recorded.

Scope limit: matching is reference-to-candidate. Extra ESPice plots are not
required from the reference. In particular, VACASK noise agreements validate
the converted spectral densities, not ESPice-only integrated-noise totals.
This does not change the reported numerical agreement of the compared data,
but the report should not describe it as validation of every possible output.

General input assumptions are satisfied here: the first timing engine is
ESPice, every fixture has one row for every engine, and all completed medians
are positive. The report renderer would benefit from explicit schema validation
if reused with arbitrary files, but that is not a flaw in these results.
