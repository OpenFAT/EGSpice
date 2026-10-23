# Output and memory

Where results live during and after a run, how they reach the file, and
what scales with steps, probes, points or nodes. Anchors are `src/...` paths.

## What ships

- A transient records point-major f64 rows of time plus every probe
  (`src/analysis/tran/types.zig:64-160`); the result borrows the recording
  without a copy. Probes default to every branch current plus every named
  node (`src/frontend/builder.zig:708-729`); `.save` narrows them;
  measures-only keeps time plus the columns of cards that need a recording
  (`src/analysis/tran/tran.zig:1394-1411`).
- Rows are preallocated at twice the output window over TSTEP, clamped to
  64..2^22, untouched capacity virtual; past it the buffer doubles with a
  realloc in the results arena, which keeps the old block
  (`types.zig:18-21, 146-148`).
- `.meas` streams per step with no allocation (`src/output/measure.zig:287-308`);
  early stop ends the run when every card has resolved.
- **Streaming exists for exactly one plot**: the first requested output, a
  transient or AC, binary raw to a regular file, CLI only (`final_plan`), no
  `.meas` reading it, no `.save`/`.stim`/optimization, no variant or
  temperature (`src/espice.zig:1037-1070`; `dev/Problem/migration-notes.md`
  says there is none and is stale). Everything else is encoded whole after
  the query ends, on the coordinator thread, between `pool.wait()` and the
  next `pool.fill()` (`src/espice.zig:946-952, 1480-1507`).
- The CLI keeps every result until `deinit` (`src/main.zig:207-215`); the C
  API keeps them until `espice_destroy` or `espice_release_points` and never
  sets `final_plan`, so it gets no streaming, no state moves and no early
  circuit release.

*Measured* peak RSS (survey; `CHANGELOGS.md`, commits named there):

| Change | Deck | before | after |
|---|---|---:|---:|
| measures-only | 1000-row RC-ladder sweep | 6.5 GB | 168 MB |
| refilling pool, 16 jobs | 5000 points | 2.96 GB | 134 MB |
| transient streaming | scaling_rc_ladder_100k | 335 MB, 1.87 s | 149 MB, 1.53 s |
| transient streaming | vacask_graetz with `-r` | 60 MB | 14 MB |
| final-plan state moves | sweep_opamp_wl_5000 | 198 MB | 160 MB |
| C API malloc instead of the SMP allocator | per released point | 48 KB | 1.6 KB |

What still scales: steps × probes per transient whenever measures-only is
off (`-r`, `.save`, `.stim`, optimization, tangents, display, every API
point without `measures_only`), and in a `.step`/`.data` sweep with `-r`
every point's full waveform stays resident after it is written; the time
column per step even under measures-only; a 16-byte name header per probe
per point; the OP vector and the transient's copy per point in work arenas
until the executor dies; ~2.5 KB per released point (label and variant
row until `deinit`, `src/espice.zig:783-785`); the AC lane precompute
(frequencies × rhs per lane) in the work arena (`src/analysis/executor.zig:651-684`);
without `final_plan` a completed query's whole circuit (`src/analysis/session.zig:734-739`).

## Items

### 8.1 Stream every plot — P0

**Change.** Partial committed ranges: an appendable format (binary and
ascii raw, csv, print) takes a query's rows in chunks as it runs, from any
worker, into its own file region, and the recording is released behind the
write; a sweep writes a point's plot when the point finishes (in request
order for the single-file formats, so a fast later point buffers to a
temporary file, which numbered formats already do per plot). `-r` on a long
transient and `--full-waveforms` on a sweep then hold one window of rows,
not the run. Lift the four conditions on `openStream` that are not about
the format. **Expect.** steps × probes × points → one window per running
query. **Gate.** Byte-identical files. **Effort.** 1-2 weeks.
**Status (2026-10-22): done for transient/AC in binary and ASCII raw (CLI).** Each such plot that nothing reads after the run writes its rows to a side file from its worker (`Session.beginSide`/`endSide`, `Session.stream_opener`), published in request order with its real header. 200-point `.step` RC ladder with `-r`: 3.37 GB → 49 MB, file identical. Still kept in memory: plots a `.meas`/`.save`/`.stim`/optimizer reads, lane batches, the C API.
**Status (2026-10-23):** CSV and `.print` stream too (`Session.copySide`; CSV into each plot's own numbered file): `rc_step200.sp` `--format csv` 973 MB → 52 MB, `print` 1008 MB → 51 MB, files identical. Still in memory: psf/fsdb/sst2/Touchstone/CITI plots, plots a `.meas`/`.save`/`.stim`/optimizer reads, lane batches.

### 8.2 Release what was delivered — P0, small

The CLI reads every result after `run_all` and frees nothing until exit;
with 8.1 unnecessary, until then release a query's result and circuit as
soon as its plot is written. Give the C API the final-plan behaviours
(state moves, early circuit release, streaming) as options
([06](06-sweeps-and-lanes.md) 6.15).
**Status (2026-10-23): C API part closed by its contract.** Results stay valid until destroy and a later append may read any OP, so nothing is moved, released or streamed automatically; `append_points` marks its points final and `espice_release_points` frees them.

### 8.3 Encoding off the critical path, computed once — P1

Encoding and `.meas` printing run on the coordinator between wait and fill
([05](05-scheduling-and-parallelism.md) 5.9). `.meas` printing evaluates
every non-PARAM card twice per result (`src/output/measure.zig:31-53`) and
formats every number through an exact big-integer decimal conversion even
into the discard writer used to collect values (`measure.zig:1710-1778`);
`espice_copy_point_measures` re-evaluates the cards on every call
(`src/espice.zig:705-752`). Evaluate once when the point finishes, keep the
values, format once. Text formats call `print("{e}")` per value
(Ryu); fine per value, but a plot's encoding can run on a worker, and a
sweep's plots in parallel.

### 8.4 Waveform growth without the doubled arena — P1

Past the preallocation the buffer doubles in the results arena and the old
rows stay resident (`types.zig:146-148`): up to 2x the waveform. Chunked row
blocks (no realloc; the result iterates chunks) or an `mremap`-backed
growable mapping. With 8.1 the window is bounded anyway.

### 8.5 Probe download on the lanes — P0

[06](06-sweeps-and-lanes.md) 6.3.

### 8.6 Appending after a `finish` copies the whole file — P1

A batch published after `finish` first copies the existing file
(`src/output/session.zig:121-130`): an API caller running batch after batch
into one raw file rewrites everything earlier each time, the O(P²) #179
removed for a single run. Keep the replacement open across batches.

### 8.7 Per-point residue — P2

The 16-byte name headers per probe per point, the kept label and variant
row (2.5 KB per released point: 2.5 GB at a million points), the AC lane
precompute in the work arena. Intern and borrow; release with the point.
**Status (2026-10-23): session rows done.** A row holds a `Session.jobs` index (interned, deep-compared) and its variant instead of a 424-byte `Query`; OP keys are 12 bytes. CLI `.step` 6.9 → 5.0 KB per point; C API released point 2.5 KB → 0.84 KB.

### 8.8 Memory accounting — see [10](10-measurement-and-gates.md)

`ZP_MEM_STATS` went with the global allocator; per-Problem and per-executor
peak bytes are what the lane host cap and this page need.

### 8.9 Large-deck memory — pointers

Preparation peak 1.8 GB at 1.14M devices ([07](07-frontend-and-preparation.md) 7.4);
the 100k chain runs at 3.4-11.8 GB with the device LU's tables (`dmap` is
4 B per flop; the 300k chain's tables are 2.9 GB, [02](02-linear-solver.md) 2.7);
512 MiB of virtual stack per query ([05](05-scheduling-and-parallelism.md) 5.7);
0.5-1 MB of host state per GPU lane ([06](06-sweeps-and-lanes.md) 6.10).

### 8.10 Huge pages for the planes — P2, measure first

At 1M+ nonzeros the value planes, factors and `dmap` are hundreds of MB
walked by scatter and gather; `perf stat -e dTLB-load-misses` on the 100k
decks says whether `madvise(MADV_HUGEPAGE)` on those allocations is worth a
line. The CLI already raises glibc's mmap threshold to 128 KiB
(`src/main.zig:27-33`: 285 → 277 MB on rc_ladder_100k).
