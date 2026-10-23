# Frontend and preparation

Netlist bytes to a frozen `Problem`: parse, flatten, bind, pattern,
tapes, and the variant path a sweep point takes. Everything here is
single-threaded. Anchors are `src/...` paths.

## What it costs today

*Measured* by the frontend survey (ReleaseFast 1.2.0, `--plan --timing-in-depth`,
faster of two runs, load 7-71; the parse lap includes flattening, the
"expansion" lap is binding and freeze):

| Deck | devices | parse | Problem creation | peak RSS |
|---|---:|---:|---:|---:|
| grid 100x100 | 19.8k | 6 ms | 6 ms | 19 MB |
| rc_ladder_100k | 200k | 50 ms | 50 ms | 95 MB |
| chain_bsim4_10k | 115k | 41 ms | 166 ms | 191 MB |
| logic_psp103_10k | 90.5k | 38 ms | 183 ms | 193 MB |
| chain_bsim4_100k | 1.14M | 0.52 s | 1.82 s | 1.83 GB |
| ring_bsim4_100k | 1.13M | 0.56 s | 2.26 s | 1.67 GB |
| chain_bsim4_1k with one `{rv}` resistor and a 2-point `.step` | 11.5k | | 17 → **164 ms** | 28 → **248 MB** |
| chain_bsim4_10k, the same | 115k | | 171 ms → **1.54 s** | 191 MB → **2.02 GB** |

So on the synthetic post-layout decks preparation is 1-5% of a run and
about 0.45 KB per R or C and 13-14 KB per BSIM4 instance at peak. Two
things break that: a PDK deck, where c7552_ihp still spends 10.6G
instructions in `Problem.init`, 78% of it re-reading each PSP instance's
~390-pair `.model` card under its subcircuit parameters
(`dev/devices/gpu-evaluation.md`, "Setup on PDK decks"; c7552_sky130 went
30.5G → 2.9G with the binder fixes); and any deck with a sweep, where one
swept resistor makes creation 9x slower and 10x larger. A linear
extrapolation of the last row is about 18 GB at 100k transistors with one
swept parameter, unmeasured.

## Items

### 7.1 The variant fixed cost: stop unsharing every Model row — P0

**Today.** Any variant (`.step`, `.data`, live params, `.mc`, `.alter`,
dcmatch, acmatch, dcsens) runs `Planner.init`, which collects one `ParamRef`
(~56-64 B) for every sweepable field of every device, about 867 fields for a
BSIM4 or PSP103 instance, and unshares every batch's Model rows so each
instance owns a 10.5-11 KB copy (`src/frontend/variants.zig:219-230, 1289-1297`,
`src/device/eval.zig:1703-1713, 2148-2194`; `countParams` collects again,
`src/frontend/prepare.zig:414`). `duplicate` then copies the per-instance
rows into every query (`eval.zig:2363`). This is also the +10.5% GPU kernel
time of distinct rows and the per-(instance, lane) rows that bound the GPU
lanes ([03](03-device-evaluation.md) 3.7, [06](06-sweeps-and-lanes.md) 6.10).
**Change.** Copy-on-write rows: keep rows shared and unshare only a row a
variant actually writes, at write time. Collect `ParamRef`s only for the
names the sweep can reach (the `.step` targets, `.data` columns and live
params are known at parse; `.mc` touches primaries; dcmatch and dcsens need
the full set and can collect lazily per batch). An index → field table built
once per template makes the per-point write O(written fields)
(`dev/research/spef-parasitics.md` §3.3).
**Expect.** The `.step` rows above return to within a few percent of the
plain deck's creation time and memory; the GPU rows coalesce again.
**Gate.** Bitwise (the same values land in the same fields). **Effort.** 1 week.
**Status (2026-10-22): done.** `param_layout` is a read-only view over shared rows (`model_of`); `Hooks.write_params` writes a shared row in place when all its readers get the same writes, else appends a private row per written instance. chain_bsim4_10k + `.step`: creation 1.1-2.7 s → 0.45 s, 1.36 → 0.48 GB; chain_bsim4_1k 16-point `.step`: 20.4 s / 335 MB → 7.4 s / 96 MB (ReleaseFast). `.sens`, dcmatch/acmatch, CPU MC lanes and tangents still unshare their own query copy.

### 7.2 Local model cards read once per parameter environment — P0

**Today.** A subcircuit-local `.model` that reads instance parameters is
re-read from its line for every instance, then compared linearly against
every earlier variant of that model (`localRow`,
`src/frontend/netlist.zig:1809-1834`); `modelRow` scans the local models
newest to oldest and recomputes each candidate's bounds with four linear key
scans of the bin card (`netlist.zig:1738-1763, 4108-4113`). On c7552_ihp that
is 78% of a multi-second `Problem.init`.
**Change.** Key the folded card by (model line, hash of the instance
parameter values it reads) in a hash map; fold the bin bounds once per bin.
**Expect.** c7552_ihp setup near c7552_sky130's 0.7 s (estimate).
**Effort.** 2 days.

### 7.3 Flattening without re-reading the text — P1

`expand` runs once per X instance, builds a fresh scope map of every default
and instance pair, concatenates the path, and re-reads every body line
through `readDevice`, re-splitting fields and re-folding numbers per
instance (`netlist.zig:3861-3939`); subcircuit defaults stay as text and
each reference recompiles them (`netlist.zig:3109-3131`); `.func` calls are
inlined per site; `netOf` scans the port list with `mem.eql` per pin
(`netlist.zig:3268-3273`, pins × ports, which bites when one wrapper
subcircuit holds a whole extracted network); each `.lib` reference re-reads
the library file (`src/frontend/source.zig:631-655`); each `.alter` run
re-expands and re-parses (`prepare.zig:199-201`). Tokenize each subcircuit
body once into field slices and folded numbers, memoize a default's folded
value per (expression, scope signature), hash the port list, cache `.lib`
sections by (file, section). **Effort.** 1 week.

### 7.4 Stage a row once, not a Model copy per instance — P1

`ProtoStore.append` stages a full Model and Instance copy for every instance
(`eval.zig:1089-1097`) and finalize deduplicates by hash afterwards
(`:1230-1267`): 13-14 KB per BSIM4 instance at peak, most of the 1.8 GB at
1.14M devices. Hash the bound Model at append and stage a row index, copying
only the first instance of each distinct row. **Expect.** Preparation peak
toward ~0.5 KB per instance (estimate). **Effort.** 2 days.

### 7.5 Parallel preparation — P1

Every stage is single-threaded (`dev/frontend.md`). Independent work:
subcircuit instances after the declaration walk, per-batch binding
(`addSingleDevice` per instance, `src/frontend/builder.zig:2131-2181`),
pattern keys per batch, the LSD radix sort (`src/device/abi.zig:466-546`),
labels. At 1.14M devices creation is 3-4x the parse (1.8-2.3 s); with 8
P-cores and the items above, under 0.5 s is plausible (estimate). Take the
threads from the budget of [05](05-scheduling-and-parallelism.md) 5.1.
**Status (2026-10-22): partial, memoized instead of threaded.** Equal device cards bind once (`NetBuilder.memo`, `proto_add_row`) and `partitionCollapsed` reuses a collapse answer per (row, Instance): chain_bsim4_100k creation 1.9-2.6 s → 0.76-0.92 s (ReleaseFast). Threads not done.

### 7.6 Setup per query — P1

Each query copies the whole circuit (`src/analysis/Circuit.zig:548-552`),
installs its variant, and builds a fresh workspace with its own ordering
and first pivoting factor. [05](05-scheduling-and-parallelism.md) 5.6 and
[02](02-linear-solver.md) 2.8: ordering and symbolic structure on the
template, copy-on-write batches for the per-query circuit.

### 7.7 SPEF and DSPF as indexed parasitics — P1

GPurify's DSPF is already valid ESPice SPICE and 1M-element R + C decks run;
the cost is the per-point value path (7.1) and the lack of a scope and an
index contract. `dev/research/spef-parasitics.md` §3: a `.parasitics 'role.dspf' scope=x1`
directive, elements addressed by index `P + i` in the same `values` row as
the live params, written through the index → field table, R clamped at a
floor, C = 0 kept as a zero stamp, no variant row per point; a dedicated
linear-passive lane device (value-only SoA, 8 B per element per lane) when
lanes need it. ~1.5-2 weeks for v1; eight owner questions in that note.

### 7.8 Small items

- `buildTapes` binary-searches the CSC slot of every Jacobian entry
  (`eval.zig:501-522`); the radix output is already sorted, so a merge gives
  the slots in one pass.
- Every node and branch label is an `allocPrint` (`builder.zig:712-729`),
  every card name is copied into the session arena (`prepare.zig:349-352`),
  and every point keeps a 16-byte name header per probe
  (`src/analysis/tran/tran.zig:1386`): intern and borrow.
- Two Wyhash site keys per device even without Monte Carlo (`netlist.zig:3357`).
- The intern pool rehashes every name from its bytes when it grows; the
  reservation is twice the line count (`netlist.zig:1205-1206`), which a
  flattened deck exceeds.
- Includes are copied twice (`source.zig:44`); the lowercased copy and the
  joined continuation lines are copies too. Small next to the above.
- The timing laps: "frontend" includes flattening and "expansion" is
  binding; label them, and record `bench-frontend` results, which no doc
  holds for the current parser.

## Closed, with the reason

| Idea | Verdict | Where |
|---|---|---|
| AST parser | retired: 42.8-45.4 ms and a 143.6 MB arena on rc_ladder_100k against the three-walk parser | `dev/frontend.md` |
| Token SIMD | retired: 86.4M against 81.4M instructions; only `normalize` keeps a SIMD pass (0.05 vs 0.07 ns/byte) | `dev/frontend.md`, `src/frontend/lines.zig:275` |
| O(E²) live-name scan | fixed: names resolve through a hash map (`variants.zig:1013-1018`); the SPEF note describes the older code | survey |
| mmap the deck | whole-file reads are 0.5 s at 1.14M devices against 1.8 s of creation; not the bottleneck | survey |
