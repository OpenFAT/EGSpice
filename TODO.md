# TODO

Open issues and decisions from the 2026-10-22/23 performance round and the VerA
integration. Performance items in detail: [docs/performance/TODO/REMAINING.md](docs/performance/TODO/REMAINING.md).

## Decisions waiting on the owner

- **Conformance suite: merged (2026-10-23).** `tests/fixtures/` is now the submodule
  `tests/conformance` (VAMS Conformance Suite), decks under `tests/conformance/tests/netlist/`.
  dev/performance's 68 fixture changes are suite commit a46e5ca (local branch
  `espice-dev-performance`, not on GitHub). Before the push: push that suite commit, then
  `nix flake update conformance` (flake.lock still pins 166282e), and run `zig build test`
  for what the path move breaks. New or fixed decks go into the suite, then a submodule bump.
- **`.pz` at large n** (09 9.6): done. Above 1000 unknowns `.pz` uses shift-invert Arnoldi on the
  sparse LU and returns the 20 poles and 20 zeros nearest s = 0 (plot name says "partial"); the
  dense path and its output are unchanged below. Not done: complex shifts, a `.pz` parameter for
  k or the shift.

## Release and pins

Nothing is pushed or tagged until ESPice, VerA and Gompute sync perfectly (user decision); until
then pins are local: ESPice `dev/performance` (local), VerA via a local tarball of the integration
worktree, then one coordinated push of all three.

- **VerA v0.13.0**: session vera-26 merges the full updated VerA to main and pushes/tags it;
  the `vera` agent hands it its gated commits beyond ed42ca34. Then bump ESPice's pin and apply
  the ESPice patches that need it (`sim_reads` replacing the `bypass_models` allowlist, which also
  makes 11 models bypass-eligible that a name mismatch had excluded).
- **Gompute 1.4.0** (in progress): `multiprocessorCount` and multi-GPU stage 0 (device table,
  handles carrying their ordinal, fake-device test contexts); testing with Gompute's own Zig 0.17,
  then main, push, tag, ESPice pin, and the 6.11 colour-threshold patch (40 × SMs).
- **VerA `flake.nix`**: the user's earlier uncommitted change is no longer in the VerA working
  tree (flake.nix == HEAD); the `vera` agent is checking where it went.

- **VerA side, ready for the local pin**: the frame fix ed42ca34 (bsimcmg GPU frame 20.8 → 9.8 KB,
  bsimbulk 17.2 → 7.1 KB, eval −7.6% / −6.1% at N = 4096, bitwise) and `sim_reads`; the ESPice side
  is `$S/agents/vera/espice-sim-reads.patch`, applied with the VerA pin bump. The `vera` agent's own
  v0.13.0 candidate (4d392efa) is superseded by the VerA session's integration branch.
- **Gompute experiments for a later release** (from the `vera` agent): `-maxrregcount=128` on heavy
  roots (`gompute-maxrreg128.patch`: occupancy up to 2×, static frames +3–12%) and a bit-identical
  log without `?f64` (`gompute-log-optional.patch`: bsimcmg frame 9,840 → 9,072 B); measure with
  `$S/agents/vera/gpurun.sh` before adopting.

## Final gates still owed

- Full CPU oracle corpus and full-corpus A/B on the final tree.
- Full GPU corpus (`zig build test-gpu`) for the tolerance-class GPU changes (01 1.6, 04 4.4) and
  the round-2 GPU work.
- GPU Debug test-espice after 06 6.8 and the row-order lane gather.

## VerA ↔ ESPice integration (agent `bridge`, with session vera-26)

Done (bridge's patch, applied 2026-10-23, final gate running): ESPice builds as a lazy dependency
(`artifact("espice")`), runs as whatever `espice` the user has (VerA's `-Despice=PATH`, GPU options
forwarded), `-I/--include=DIR` for `.hdl` includes (in the cache key), §4.6.4.3 card-knot noise
tables, net initializers (`u_nodeset`) seeded as nodesets, `--version` with the VerA version and
commit, CSV output confirmed; VerA's 13 decks: 9 pass, the rest need VerA-side fixture fixes.
Remaining:

- **`$param_given` after the VerA `given__` bitset change** (with the VerA v0.13.0 pin):
  replace `markGiven`'s by-name `<field>__given` write with `target.setGiven(field)` guarded by
  `@hasDecl`; `castField` already handles the narrow integer fields.
- **Call `setup` before `say`** in every host path that can reach a display task (with the pin).
- **Build-cache gotcha**: after a VerA codegen change build once with `-Dva-cache=false`.
- **Loader Zig check**: check the `zig` version before compiling a runtime `.hdl` (today the
  system Zig 0.16 on PATH fails with "unrecognized optimization mode: 'debug'").
- **Relay to the VerA session** (vera-26's session ended before it arrived): noiseport's answer
  in `$S/agents/noiseport/to_vera26.txt` (resistor-noise constants, printed values, binary path),
  and the caveat that VerA should emit potential-contribution noise on the flow unknown only for
  pure potential branches, keeping the node form on switch branches (ESPice decides per instance
  whether a switch branch's unknown collapses). ESPice handles both forms without double-applying.
- **New fixture decks to move with the conformance-suite merge**: `noise/noise_va_vbranch.sp`,
  `noise/noise_table_card_knots.sp` (+ their `.expected.json`, `vseries_res.va`, `ntab_card.va`,
  `noiseless_res.va`); the compiled-in correctness runner sees them only after `zig build test` regenerates it.

## Performance round 2 (agents still running)

| Agent | Remaining |
|---|---|
| gpu-resident | 02 2.7 graph capture → multi-GPU stages 0–3 (device table and fake-device test knob in Gompute, sweeps across GPUs, per-device placement and constants, measured per-device rates) → 01 1.3 device Newton → 01 1.2 device history mirror (~0.15%) |
| gpu-tune | 02 2.7 admission bars, tail-cut refit, f32 refactor (likely dropped); 05 5.3(a) host stamping beside the device eval; `-Dgpu-jac-f32` removal gate |
| gpu-lanes2 | 06 6.9 OP/DC lanes gate, 6.12 charge-kernel colouring, 6.8 staged cards and `par()`, 6.15 residue, 03 3.9 host-stepped batches on the device (needs VerA's half) |
| host-solver | done this round: 02 2.4 pipelined tail, 04 4.10 CPU half (`.dc` first point reuses the OP: 4,000-stage chain sweep 2.2-3.4 → 1.2-1.3 s; failing MC/temperature/DATA lanes get the full ladder), supernodal DFS (300×300 grid factor 352 → 255 ms, exact); now: the chain_bsim4_100k cold OP (hours: ordering fill count, first-factor fill) |
| frontend2 | copy-on-write rows for dcmatch/acmatch/MC lanes/tangents (gating), 08 8.1/8.2 C API streaming, 05 5.11 `.four`/`.fft` reuse, 07 7.5/7.8, 07 7.3 flattening |
| freq2 | 09 9.3 GCRO-DR and oscillator period column, 9.2 GPU complex frequency kernel and resident factors, 9.4(a) HB on the GPU, `factorEach` allocations |
| vera | 03 3.2 frame fix measurement and spill reduction, field-major Model rows (P2), VerA half of 03 3.9, v0.13.0 release commit |

## Bugs found

- **chain_bsim4_100k with `.tran 2p 10p uic` stalls** (`--backend cuda`, ReleaseFast): Problem
  creation finishes in 0.6 s, then no progress for 10–50 min at ~124 MiB of GPU memory, on the
  binaries both before and after 02 2.3. The original deck (`.tran 2p 200p uic`) finishes in
  ~77 s. Not yet diagnosed (CPU path untested); deck at `$S/me/gmem/c100k_short.sp`.

## Wanted later

- **One huge circuit across GPUs** (multi-GPU stage 4): device evaluation partitioned by instances
  with ordered boundary sums, LU on one device; estimated 1.1–1.3× on 2–8 GPUs over PCIe, ~2× on
  8 NVLink GPUs; needs its own design pass. Compaction first (fewer bytes per instance, narrower
  indices, field-major and eval-only Model rows) helps every card.
- **Device-memory cap for lane waves**: waves cap host memory only (4 GiB); nothing caps device
  memory besides the 256 MiB dense-peel pool.

## Machine and tooling

- Build queue `$S/zb` (3 slots, -j6; GPU-shell builds one at a time at -j2, taking the GPU lock
  before a slot). ptxas needs up to 10 GB per large model now that SASS is embedded by default.
- Outside processes share the machine's 31 GB: the VAMS conformance harness (`-j 8`, iverilog,
  openvaf, `vera --run`) filled swap twice; one `ivlpp` reached 17 GB on
  `tests/verilog/19_compiler_directives`.
- Subagents cannot write `.md` files (a hook); their reports are `REPORT.txt`.
