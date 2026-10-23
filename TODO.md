# TODO

Open issues and decisions from the 2026-10-22/23 performance round and the VerA
integration. Performance items in detail: [docs/performance/TODO/REMAINING.md](docs/performance/TODO/REMAINING.md).

## Decisions waiting on the owner

- **Gate `main` by machine** (TESTING.md, AGENTS.md "Branches and releases"): a ruleset on `main`
  that requires `test.yml` (`release-safe`) and `nix.yml` (`nix`) and blocks direct pushes;
  the merge queue (`merge_group` is already wired) once a second contributor appears.
  `gh api -X POST repos/OmarSiwy/ESPice/rulesets` after the coordinated push.

## Release and pins

Nothing is pushed or tagged until ESPice, VerA and Gompute sync perfectly (user decision); until
then pins are local: ESPice `dev/performance` (local), VerA via a local tarball of the integration
worktree, then one coordinated push of all three.

- **VerA v0.13.0**: the candidate is VerA branch `dev/performance` (release commit f9ff7a2e,
  2759/2759 fixtures; includes sim_reads, the status-device fix and ab74c6ae's GPU-safe history;
  the suite at 3dad22b). The user refines it and merges to VerA main; pin ESPice to that merged
  main tip (not a branch commit), then apply `$S/agents/vera/espice-sim-reads.patch`
  (`sim_reads` replacing the `bypass_models` allowlist, 11 more models bypass-eligible).
- **Conformance suite pins**: ESPice pins suite `main` 4f4b219 (pushed); VerA pins 3dad22b. Bump
  VerA's after v0.13.0. New or fixed decks go into the suite, then a submodule bump and
  `nix flake update conformance`.
- **Gompute 1.4.0**: done and pushed (tag 0.17.0-v1.4.0), pinned in build.zig.zon. Still owed:
  the flake's zigDeps `outputHash` for that pin, and the 6.11 colour-threshold patch (40 × SMs).
- **Gompute experiments for a later release** (from the `vera` agent): `-maxrregcount=128` on heavy
  roots (`gompute-maxrreg128.patch`: occupancy up to 2×, static frames +3–12%) and a bit-identical
  log without `?f64` (`gompute-log-optional.patch`: bsimcmg frame 9,840 → 9,072 B); measure with
  `$S/agents/vera/gpurun.sh` before adopting.

## Final gates still owed

- Full-corpus A/B on the final tree (the CPU oracles run in `zig build test-conformance`).
- Full GPU corpus (`zig build test -- --backend cuda --jobs 1`) for the tolerance-class GPU changes (01 1.6, 04 4.4) and
  the round-2 GPU work.
- GPU Debug `zig build test` after 06 6.8 and the row-order lane gather.

## VerA ↔ ESPice integration (agent `bridge`, with session vera-26)

Done (bridge's patches v1 and v2, 2026-10-23): ESPice builds as a lazy dependency
(`artifact("espice")`), runs as whatever `espice` the user has (VerA's `-Despice=PATH`, GPU options
forwarded), `-I/--include=DIR` for `.hdl` includes (in the cache key), §4.6.4.3 card-knot noise
tables, net initializers (`u_nodeset`) seeded as nodesets, `--version` with the VerA version and
commit, CSV output, `setGiven` for `$param_given`, the loader's `zig version` check; VerA's 13 decks: 9 pass, the rest need VerA-side fixture fixes.
Remaining:

- **Call `setup` before `say`** in every host path that can reach a display task (with the pin).

## Performance round 2 (agents still running)

| Agent | Remaining |
|---|---|
| gpu-resident | 02 2.7 graph capture → multi-GPU stages 0–3 (device table and fake-device test knob in Gompute, sweeps across GPUs, per-device placement and constants, measured per-device rates) (01 1.2 and 1.3 measured and dropped) |
| gpu-tune | 02 2.7 admission bars, tail-cut refit, f32 refactor (likely dropped); 05 5.3(a) host stamping beside the device eval; `-Dgpu-jac-f32` removal gate |
| gpu-lanes2 | 06 6.9 OP/DC lanes gate, 6.12 charge-kernel colouring, 6.8 staged cards and `par()`, 6.15 residue, 03 3.9 host-stepped batches on the device (needs VerA's half) |
| host-solver | chain_bsim4_100k cold OP: gmin-stepping fixes (budget-relative thresholds, gentler back-off on LU refusals, ngspice's 1.00005 stop) and the gmin-assisted source start, unmeasured |
| frontend2 | 07 7.3 body-token memo and 7.5 parser threads (parked: need builds) |
| freq2 | 09 9.3 GCRO-DR and oscillator period column, 9.2 GPU complex frequency kernel and resident factors, 9.4(a) HB on the GPU, `factorEach` allocations |
| vera | field-major Model rows (P2) and spill reduction (frame fix, sim_reads, 3.9's VerA half and the release commit are in the v0.13.0 candidate) |

## Wanted later

- **One huge circuit across GPUs** (multi-GPU stage 4): device evaluation partitioned by instances
  with ordered boundary sums, LU on one device; estimated 1.1–1.3× on 2–8 GPUs over PCIe, ~2× on
  8 NVLink GPUs; needs its own design pass. Compaction first (fewer bytes per instance, narrower
  indices, field-major and eval-only Model rows) helps every card.
- **Device-memory cap for lane waves**: waves cap host memory only (4 GiB); nothing caps device
  memory besides the 256 MiB dense-peel pool.

## Machine and tooling

- Agents do not build; the coordinator runs the gates (`zig build test-conformance`, GPU builds at
  -j10 under `systemd-run -p MemoryMax=26G`). ptxas needs up to 10 GB per large model.
- After a VerA codegen change, build once with `-Dva-cache=false`.
- Outside processes share the machine's 31 GB: the VAMS conformance harness (`-j 8`, iverilog,
  openvaf, `vera --run`) filled swap twice; one `ivlpp` reached 17 GB on
  `tests/verilog/19_compiler_directives`.
- Subagents cannot write `.md` files (a hook); their reports are `REPORT.txt`.
