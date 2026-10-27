# Output

A run produces three kinds of output:

| Where | What |
|---|---|
| stderr | one summary line per deck and per result (`AC Analysis: 41 points, 4 variables`), warnings and errors |
| stdout | `.meas` results, Monte Carlo statistics, and `--plan` / `--print-dag` graphs |
| `--rawfile FILE` | the results themselves, in the `--format` you choose |
| `--measures FILE` | the `.meas` results as JSON, labelled with analysis and variant |
| `--emir FILE` | each device terminal's average, RMS and peak current as JSON, per variant |

Without `--rawfile`, results stay in memory and nothing is written.

## Results, plots and columns

Each analysis produces one result, called a plot as in ngspice. A plot has
a name (`Transient Analysis`, `AC Analysis (rval=2000)`), a list of
variables (columns) and points (rows). The first column is the
independent variable (`time`, `frequency`, `v(v-sweep)`, `run`) except in
single-point plots such as `Operating Point`. AC-type plots are complex.

Column names follow ngspice: `v(node)` for node voltages, `i(vname)` for
the current through a voltage source (and inductors, B and E sources),
`v(x1.node)` inside subcircuit instance `x1`, and `v(n1#node)` for an
internal node of a Verilog-A instance `n1`.

## Formats

| `--format` | Aliases | File | Several plots |
|---|---|---|---|
| `binary` (default) | `raw` | ngspice binary raw | appended to one file |
| `ascii` | | ngspice ASCII raw | appended to one file |
| `print` | `text` | tab-separated text table per plot | appended to one file |
| `csv` | | comma-separated, one header row | `FILE`, `FILE.2`, `FILE.3`, ... |
| `touchstone` | `snp`, `s2p` | Touchstone `.sNp` | numbered files |
| `citi` | `citifile` | CITIfile | numbered files |
| `psf` | | Cadence PSF ASCII | numbered files |
| `fsdb` | | a minimal FSDB-style binary container | numbered files |
| `sst2` | `hspice` | HSPICE SST2-style binary records | numbered files |

Every file is replaced atomically: it holds either its old contents or the
complete new result, never half of one. In the formats that append, the
file is replaced once, when the run ends: until then it keeps its old
contents, and a run that fails or is interrupted leaves it so. ngspice
also writes its raw file at the end. A path that is a device or a FIFO
(`-r /dev/null`) is written in place, plot by plot.

### Raw files

`binary` and `ascii` are ngspice raw files: a text header (title, date,
plot name, flags, variable list) followed by the data. Every plot of the
deck goes into the one file, back to back, the way ngspice writes them, so
any reader of ngspice raw files can load it. The date field is fixed, so
the same deck gives a byte-identical file.

```text
$ espice divider.sp -r divider.raw --format=ascii
$ cat divider.raw
Title: Voltage divider
Date: Thu Jan  1 00:00:00 1970
Plotname: Operating Point
Flags: real
No. Variables: 3
No. Points: 1
Variables:
	0	i(v1)	current
	1	v(in)	voltage
	2	v(out)	voltage
Values:
0	-2.5e-3
	1e1
	7.5e0
```

### Text and CSV

`print` writes each plot as a table, with a header row, a rule and one row
per point; complex columns get an `(imag)` column after them. It is the
format the [Learn SPICE](../learn/index.md) chapters show. `csv` writes
one plot per file with a header row; a complex column `v(out)` becomes
`v(out)_re` and `v(out)_im`. A name with a comma or a quote in it, such as
`.lin`'s `S(1,1)`, is quoted as RFC 4180 asks (`"S(1,1)_re"`):

```text
frequency_re,frequency_im,i(v1)_re,i(v1)_im,v(in)_re,v(in)_im,v(out)_re,v(out)_im
1e1,0e0,-9.979542742416352e-8,-9.989267655685869e-6,1e0,0e0,9.999002045725759e-1,-9.98926765568587e-3
```

### Touchstone and CITIfile

These hold S-parameters only, so they accept only an `.sp` (or `.lin`,
`.net`) result; anything else fails with `NotSParameterData`, and a deck
with no ports with `NoPorts`. Touchstone needs the full S matrix, written
in real-imaginary form at the ports' reference impedance. Ports with
different reference impedances write Touchstone 2.0, with each port's z0
on the `[Reference]` line:

```text
$ espice two_port.sp -r lc.s2p --format=touchstone
$ head -3 lc.s2p
! LC low-pass as a two-port
# Hz S RI R 50
1e6 1.9921236660280783e-8 9.96883205564205e-7 9.99800389223915e-1 -1.9979532194577303e-2 9.99800389223915e-1 -1.9979532194577303e-2 1.9921236660280783e-8 9.96883205563849e-7
```

CITIfile declares `VAR FREQ` and one real-imaginary `DATA` line per
S-parameter in its header, lists the frequencies, then holds one
`BEGIN`...`END` block per `DATA`, in header order. scikit-rf reads it.

### PSF, FSDB and SST2

`psf` is Cadence's PSF ASCII: a multi-point plot writes its first variable
as the SWEEP and the rest as TRACEs. Quotes and backslashes in the title
and names are escaped with a backslash. `fsdb` and `sst2` follow the record
structure of those formats only. FSDB proper is a closed Synopsys format,
so vendor tools that need `libfsdb` will not read ESPice's files; SST2
holds at most 64 variables and is untested against HSPICE's readers.
Prefer `binary` or `psf` for interchange.

## Choosing what is written

ngspice dialect: `.save` lists the vectors to keep; everything else is
dropped from the results.

```text
.save v(out) i(vdd)
```

Without `.save`, every node voltage and branch current is written.

A device's operating-point variables are vectors too, at the operating
point, along a `.dc` sweep and in a `.tran`. Save or measure them by name:

```text
.save @m1[gm] @m.x1.mn1[vdsat] all
.meas op gm1 find @m1[gm]
```

HSPICE names them `.probe op` (every variable of every device, written as
`@m1[gm]`) and `.probe lx7(m1)`, Spectre `save m1:gm x1.mn1:vdsat`, and
each dialect's `.meas` takes the same names (`.meas dc gm1 find lx7(m1)
at=0.9`). A vector is written under the name it was saved as. HSPICE's
element template numbers, after its MOSFET output templates: `lx1` vbs,
`lx2` vgs, `lx3` vds, `lx4` id, `lx7` gm, `lx8` gds, `lx9` gmbs, `lx18`
cgg, `lv9` vth, `lv10` vdsat; another number refuses the deck. A Spectre
`inst:t` with a one-character or numeric `t` is a terminal current and is
not written.

A device has its model's own variables: each module-scope variable with a
`desc` or `units` attribute (the LRM's output variables, §3.2.1), such as `vth`,
`vdsat` or `region`. They are the instance's own, per unit of `m=`: VerA
leaves `$mfactor` to the host, and does not say which variables scale with
it (the CMC models mark that with `multiplicity`, which it does not read).
BSIM4 (`bsim4va`, sky130's model) has `vth` and `vdsat`, ngspice's `von`
and `vdsat` from b4ld.c, in the n-type frame, and `cgg`, ngspice's
intrinsic `cggb`: ∂Q_g/∂V_g of the channel's gate charge, without the
overlap. A MOSFET also has `id` (`ids`), `ig`,
`is`, `ib`, `gm`, `gds`, `gmbs`, `cgg`, `cgd`, `cgs`, `cgb`, `ft`, `vgs`,
`vds` and `vbs`, where the model has none of that name, with `m=` applied.
These come from
its terminals (d, g, s, b) with its internal nodes eliminated. Currents
and voltages are in ngspice's frame: type × the current into the
terminal and type × the voltage, type +1 for `nmos` and -1 for `pmos`,
so a saturated pfet's `id` and `vgs` are positive, as ngspice reports
them (`--emir` keeps the plain current into each terminal). `gm`, `gds`
and `gmbs` are positive at either polarity. `cgg` is ∂Q_g/∂V_g, `cgd`
is -∂Q_g/∂V_d (and so on), with the overlap charges. So `cgg` follows the
model: intrinsic for BSIM4, as ngspice's, and the total gate capacitance
for a model without its own. `ft` is gm / (2π·Cgg) of the total, overlap
included, either way. An unknown name refuses the deck and lists the
device's variables. A `.dc` that sweeps the temperature or a device of the
same type is not supported yet (#236).

`.print`, `.plot`, `.probe` and `.graph` are otherwise accepted and ignored for the
same reason. In the HSPICE dialect `.save` has HSPICE's meaning: it writes
the operating point as a loadable `.nodeset` or `.ic` file, beside the
output file.

## Measurements

`.meas` results print on stdout after the run, in ngspice's format:

```text
  Measurements for Transient Analysis

tphl                =  2.818778e-11 targ=  1.078188e-09 trig=  1.050000e-09
```

They are printed whether or not `--rawfile` is given. Over Monte Carlo
trials ESPice also prints the mean, sigma, min and max of each
measurement. [Chapter 7](../learn/measure.md) and the
[reference](reference.md#meas) cover the syntax.

### Measurements as JSON

`espice deck.sp --measures=meas.json` writes every `.meas` result of the
run to `meas.json` as well as to stdout, which is unchanged. The file is
written atomically when the run ends, as `-r` is. `--measures=-` writes the
JSON to stdout in place of the text blocks. The shape (values abridged):

```json
{
  "format": "espice-measures/1",
  "deck": "ota.sp",
  "results": [
    {"name": "gain", "analysis": "ac", "variant": {"kind": "nominal"}, "value": 144.242},
    {"name": "ugf", "analysis": "ac", "variant": {"kind": "step", "params": {"ib": 9e-6}}, "value": 30304900},
    {"name": "irms", "analysis": "tran", "variant": {"kind": "alter", "index": 2}, "value": 6.27185e-4, "from": 0, "to": 3e-6},
    {"name": "ph", "analysis": "ac", "variant": {"kind": "monte", "trial": 17}, "value": -1.667081},
    {"name": "off", "analysis": "dc", "variant": {"kind": "nominal"}, "value": null, "error": "out of interval"}
  ],
  "statistics": [
    {"name": "ph", "analysis": "ac", "kind": "monte", "n": 30, "failed": 0, "mean": -1.66, "sigma": 0.02, "min": -1.71, "max": -1.61}
  ],
  "soa": [
    {"instance": "x1.mn3", "quantity": "vds", "key": "vds_max", "limit": 1.95, "peak": 2.03, "from": 1.2e-6, "to": 1.31e-6, "analysis": "tran", "variant": {"kind": "nominal"}}
  ]
}
```

- `variant` names what produced the point: `nominal`, `step` (a `.step`
  or card `SWEEP` point, its parameter values in `params`), `data` (a
  `SWEEP DATA` row, its columns in `params`), `alter` (the run's `index`),
  `monte` (the `trial`), `temp` (an HSPICE `.temp` list entry, in `temp`).
  A point inside an `.alter` run also carries `alter`, and any point run at
  a `.temp` entry carries `temp`.
- `from`, `to` and `at` carry the window and position the stdout line
  prints.
- Numbers are the shortest text that reads back as the same f64, parameter
  values included.
- A failed measure has `"value": null` and an `error` string (`out of
  interval`, `no such vector`, or `analysis failed` for a sweep point whose
  analysis failed), never a sentinel number.
- Results keep the stdout order, so the JSON and the text agree line for
  line; a `_CONT` card's events are `name[1]`, `name[2]`, ... as printed.
- `statistics` holds, per card and Monte Carlo analysis, the trial count
  `n`, the trials the card `failed` in, and its mean, sigma (Bessel's
  correction), min and max. An analysis run with `SIGMA_SCALE=` has one
  object instead, `"kind": "sss"`: its `fail` text, the `trials` per
  scale, the `scales` and `failures` lists, then `p`, `lo`, `hi` (the 95 %
  interval) and `sigma`; these four are null with an `error` when the fit
  is refused ([analyses](analyses.md#scaled-sigma)). A `.wcd` card adds
  one `"kind": "wcd"` object per spec and `.step` point: `spec`, `bound`,
  `corner` (the `.step` label, `""` without), `beta` and `yield` (null
  when no parameter moves the measure), `converged`, `status`,
  `iterations`, `sims`, and `point`, a list of `{"name", "z"}` per
  statistical parameter ([analyses](analyses.md#wcd)).
- `soa` holds, under `.option soa`, every excursion past a model-card
  limit ([analyses](analyses.md#soa)), `maxwarns` aside: the instance, the
  quantity and the card key of its limit, the limit, the peak, and the
  window `from` and `to` (both null at an operating point; `to` null when
  the excursion lasts to the last point).

### Terminal currents as JSON

`espice hot.sp --emir=hot.emir.json` writes, for the deck's first `.tran`
(or, without one, its first `.op` or `.dc`), the current statistics of every device terminal and every independent
source, for electromigration and IR-drop sign-off (values abridged):

```json
{
  "format": "espice-emir/1",
  "deck": "hot.sp",
  "analysis": "tran",
  "variant": {"kind": "nominal"},
  "temp": 125,
  "window": [1e-06, 3e-06],
  "em_recovery": 1,
  "terminals": [
    {"instance": "x1.mn1", "model": "sky130_fd_pr__nfet_01v8", "terminal": "d", "net": "x1.out",
     "avg": 1.21e-04, "em_avg": 1.21e-04, "rms": 1.53e-04, "peak": 4.10e-04},
    {"instance": "x1.mn1", "model": "sky130_fd_pr__nfet_01v8", "terminal": "s", "net": "x1.tail",
     "avg": -1.21e-04, "em_avg": 1.21e-04, "rms": 1.53e-04, "peak": 4.10e-04}
  ],
  "sources": [
    {"instance": "vdd", "terminal": "p", "net": "vdd", "avg": -5.49e-04, "em_avg": 5.49e-04, "rms": 6.27e-04, "peak": 1.14e-03},
    {"instance": "vdd", "terminal": "n", "net": "0", "avg": 5.49e-04, "em_avg": 5.49e-04, "rms": 6.27e-04, "peak": 1.14e-03}
  ]
}
```

- Current is positive into the terminal. It is the current the solver's
  own Kirchhoff rows see, charge currents included, so at every internal
  net the terminals' `avg` sum to zero.
- `avg` is the signed mean over the window and `em_avg` the
  electromigration average with `.option em_recovery` applied, both as
  `.meas AVG` and `.meas EM_AVG` compute them over the same samples; `rms`
  is over the window, and `peak` the largest |i| at an accepted point.
- `window` holds the first and last accepted points read. `--emir-window=T1:T2`
  reads `[T1, T2]` as `.meas ... FROM=T1 TO=T2` does; the default is the
  whole run.
- At an operating point or a `.dc` sweep's first point, `analysis` is `op`
  or `dc`, `window` is null and `avg = em_avg = rms = peak = |i|` with the
  sign kept in `avg`.
- `instance` is the card's flattened name, `model` the `.model` card it
  names (a binned card's bin; null for a card without one), `terminal` the
  device's port and `net` the node it connects to. Internal nodes of a
  device are not terminals. A source lists both of its terminals, so every
  net's entries sum to zero.
- In the Spectre dialect an analysis's `emirformat` or `emirfile`
  parameter sets the flag: the file is `emirfile`, else
  `<deck stem>.emir.json`, and JSON whatever the format names. In any
  dialect `.options emirfile="FILE"` does the same; `--emir=FILE` names
  the file over both.
- Under `.step`, `.alter` or a sweep, each variant writes its own file
  (`FILE`, `FILE.2`, ...), the variant's label (the `--measures` object)
  under `variant`.
- The file is the same at any `--jobs`. The transient runs on the CPU, to
  its end (no early stop).

HSPICE's `.stim` writes transient signals as PWL sources or a `.data`
table, and `.save` (HSPICE dialect) writes the operating point; both go
beside the output file, or the working directory without one. HSPICE's own
listing files (`.mt0`, `.lis`, `.err`) are not written.
