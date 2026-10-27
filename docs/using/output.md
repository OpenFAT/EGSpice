# Output

A run produces three kinds of output:

| Where | What |
|---|---|
| stderr | one summary line per deck and per result (`AC Analysis: 41 points, 4 variables`), warnings and errors |
| stdout | `.meas` results, Monte Carlo statistics, and `--plan` / `--print-dag` graphs |
| `--rawfile FILE` | the results themselves, in the `--format` you choose |
| `--measures FILE` | the `.meas` results as JSON, labelled with analysis and variant |

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

A MOSFET's operating-point variables are vectors too, at the operating
point, along a `.dc` sweep and in a `.tran`. Save or measure them by name:

```text
.save @m1[gm] @m.x1.mn1[vds] all
.meas op gm1 find @m1[gm]
```

The variables are `id` (`ids`), `ig`, `is`, `ib`, `gm`, `gds`, `gmbs`,
`cgg`, `cgd`, `cgs`, `cgb`, `ft`, `vgs`, `vds` and `vbs`. They come from
the device's terminals (d, g, s, b) with its internal nodes eliminated:
currents flow into the device, `cgg` is ∂Q_g/∂V_g, `cgd` is -∂Q_g/∂V_d
(and so on), and `ft` is gm / (2π·cgg). An unknown name refuses the deck
and lists them. A model's own variables (`vth`, `vdsat`, `region`), other
device types, a `.dc` that sweeps the temperature or a device of the same
type, HSPICE's `.probe op` and Spectre's `save m1:gm` are not supported
yet (#236).

`.print`, `.plot`, `.probe` and `.graph` are accepted and ignored for the
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
  correction), min and max.

HSPICE's `.stim` writes transient signals as PWL sources or a `.data`
table, and `.save` (HSPICE dialect) writes the operating point; both go
beside the output file, or the working directory without one. HSPICE's own
listing files (`.mt0`, `.lis`, `.err`) are not written.
