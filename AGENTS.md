# Agents

ESPice is a simulator whose consumers pin it by commit and hash, or follow `main` (`nix run github:OpenFAT/EGSpice`). Two rules decide everything below: **every test names its oracle**, and **main is green by machine**. Read [TESTING.md](TESTING.md) before adding a test layer, touching a gate, or claiming something is tested; it maps every bug class to its oracle, layer, file and tier. [CONTRIBUTING.md](CONTRIBUTING.md) holds the branch, pin and release policy for people, and [SECURITY.md](SECURITY.md) the reporting route and trust boundary.

## Skills to load first

Load each with the Skill tool (other agents: read
`~/.claude/skills/<name>/SKILL.md` and the companion files it names). Source:
https://github.com/OpenFAT/NixOSConfig/tree/main/skills

| Skill | Use it for here |
|---|---|
| `data-oriented-design` | Node, device-instance and matrix tables: the six questions, SoA, narrowest widths, handles |
| `codebase-clean-code` | Owners (`src/frontend`, `core`, `device`, `solver`, `analysis`, `output`), one spine per analysis, seams on the cheapest rung |
| `api-design` | Every `pub` function, `src/c_api.zig` and `include/espice.h`: usage code first, caller owns memory and flow |
| `code-style` | File order, ≤500 code lines per file, `check_style.py` |
| `simd-loops` | The `W`-lane device kernels: plain-loop forms or `@Vector`, with the `W == 1` instantiation as the scalar oracle |
| `doc-comments` | Every `pub` item |
| `ponytail` (ultra) | How little code; never what shape the data takes |
| `assertions` | Contracts, capacities, bounded Newton and timestep loops |
| `test-plan`, `tdd` | [TESTING.md](TESTING.md): the oracle map, the tiers, a feature's definition of done |

Language:

| Skill | Use it for here |
|---|---|
| `zig` | Zig 0.18 idioms and the zero-cost probe |

Project-specific:

| Skill | Use it for here |
|---|---|
| `spice-engine` | MNA stamps, sparse LU replay, Newton with limiting, timestep control, device derivatives, ngspice/Xyce oracles |
| `compiler-construction` | `src/frontend`: the SPICE and Spectre netlist readers, expressions, coded diagnostics |
| `gpu-kernel-optimization` | `--backend cuda` and HIP: device evaluation and solves on the GPU, where the data lives |
| `branchless` | Data-dependent branches in device evaluation (region selects, limiting), before `simd-loops` |

## The method

Binding on every change, in every project.

### Code

- Five commitments: lean storage, no runtime residue, cheap change,
  contained failure, proof by test.
- Answer the six data questions before writing code. Then: one array per
  field, the narrowest width the stated range allows, typed `u32` handles
  with a sentinel, existence tables instead of flags, hot/cold split, tables
  sized once from a named capacity, a compile-time size assertion on every
  hot struct.
- An abstraction on a hot path carries an assembly probe: FOLDED, IDENTICAL,
  EQUIVALENT or NEGATIVE pass; RESIDUE fails the build. A "no residue" claim
  cites the probe, a speed claim cites a benchmark. Dispatch per batch, never
  per element.
- A seam names what varies and needs a second case today; it sits on the
  cheapest rung (concrete, free function, table, closed enum with exhaustive
  switch, comptime, request table, function-pointer table, `dyn`, plugin).
- Tables, transforms, one spine per pipeline (ifs up, fors down), one owner
  writes each table and everyone else appends requests. Three likely changes
  each have a change radius of zero; the compiler lists every other site.
- APIs start from the caller's code. The caller owns memory and flow:
  requests instead of callbacks, an immediate twin for anything retained.
  Factor at the second real instance.
- Branches: delete, then branchless, then vectorize along a lane axis. The
  `W == 1` instantiation is the scalar oracle; vectorization is verified in
  the assembly.
- GPU: SoA always, the roofline ceiling named with its arithmetic before
  tuning, data kept on the device, fixed-order reductions, the dependency DAG
  recovered before porting. "This belongs on the CPU" is a valid verdict.
- An operating error returns to the caller that can decide; a bug hits an
  assertion. Two or more split, paired assertions per function, a bound on
  every loop, input validated once at the edge, fallible work before the
  commit.
- Pass order: boundary, placement, data, zero-cost proof, contracts, tests,
  branches, lanes. Data layout beats brevity. Every pass keeps NaN gates,
  allocation-failure paths, `errdefer`s, trust-boundary validation and
  tolerances.

### Testing

- The oracle is named before the code and never recomputes the answer the
  way the code does. Cheapest first: assertions, a reference
  implementation, a round trip, a certificate, a metamorphic relation, a
  model, a golden with recorded provenance, a spec example.
- The main bug class is a wrong number with no crash, which coverage and
  fuzzing cannot see. A bug found by a person is a hole in the oracle set:
  it lands with its regression test and the oracle or generator that would
  have caught its class.
- A new test goes red on the bug before it is trusted green.
- Unsupported fails and never skips. A known failure is a named marker in
  the tree; an unexpected pass fails until the marker goes. Name lists are
  compared, never counts.
- A tolerance follows the oracle's own accuracy. Determinism is a test:
  1 job and N jobs give the same bytes, and the order is fixed, not the test.
- A flaky test is fixed or quarantined the same day, never retried
  silently.
- Tiers: pre-commit under 10 s, merge under 15 min, nightly, release. A
  release ships on numbers a script checks. The map lives in `TESTING.md`.

### Builds

- A build is a cache keyed on its declared inputs. Every producing step has
  an output argument, every checking step an exit-code check, every file
  read at run time is declared (`addFileInput`) or the step says
  `has_side_effects`. Keys hash contents, never absolute paths.
- Cut the artifact graph where edits land: one object per generated unit,
  narrow imports for shared ones, kernels out of host graphs.
- Zig edit loop: `zig build check --watch -fincremental`, one watch per
  artifact across all worktrees; the result that counts comes from a plain
  build. One global cache per toolchain store path; a worktree symlinks its
  `.zig-cache` to the main checkout's.
- Rust: flags live in `.cargo/config.toml`, never a one-off `RUSTFLAGS`;
  `check`, `build`, `test` and rust-analyzer agree on features, profile and
  target directory (rust-analyzer gets its own).
- Nothing collects a cache. Keep a byte budget per cache and delete old
  entries only while no build runs (the `long-term-garbage-collection`
  skill); a warm cache is kept, never wiped by routine.
- Gate on the exit code: `cmd | tail` swallows it, so use
  `exit "${PIPESTATUS[0]}"`.

### Maintaining

- `main` is green by machine: work lands through a pull request whose merge
  result passed the required checks. No `dev` branch; topic branches are
  deleted after the merge.
- Bots never push to `main`. Generated output ships as a release asset or
  through a gated pull request.
- Every dependency is pinned by commit and content hash, the toolchain by
  `flake.lock`, actions by SHA. A third-party input you hash is mirrored
  somewhere you control.
- The changelog is written by hand, one line per change citing its issue,
  every number in it measured. A release refuses a version mismatch, builds
  every target, checks every artifact and smokes the bundle.
- CI holds: the merge gate, every target compiled, docs that execute,
  reference suites gated by name, a nightly, the downstream contracts, the
  release job.

## Testing

An **oracle** is the independent thing that says an output is wrong. The main bug class is a wrong number with no crash, which coverage and fuzzing cannot see.

- Name the oracle before writing the code. An expected value computed the way the code computes it is no oracle. Use a reference simulator, an analytic derivation, a certificate (KCL residual, scaled LU residual, finite-difference Jacobian), a metamorphic relation, or the `W == 1` instantiation of a SIMD kernel.
- Every bug fix lands with its regression test **and** the oracle or generator that would have caught it. A bug found by hand, by a user or by review means the oracle set has a hole.
- Check that a new test goes **red** on the bug before trusting it green.
- The corpus is measured, never typed. `tests/VERDICTS.tsv` holds each deck's verdict by name. Rewrite it only from a measured run (`zig build test -- --update-verdicts`), and only to move decks XFAIL → PASS. Compare name lists, never counts. A capture with zero names means the capture broke.
- An unsupported feature fails. It is never a skip. The deck carries `* KNOWN GAP: <cause> (#issue)`: XFAIL keeps the run green, and XPASS fails it until the line is deleted.
- Before changing the engine to match an `.expected.json`, confirm the expectation is right. Its provenance (tool, version, command, deck hash, or derivation) lives in the file.
- Gate on the exit code. `cmd | tail` swallows it; use `exit "${PIPESTATUS[0]}"`.
- A tolerance follows the oracle's own accuracy: ngspice converges at `reltol = 1e-3`, so it is no `1e-6` oracle. A deck the nightly reports LOOSE cannot see a 0.1% error; tighten it or derive an analytic oracle.
- Determinism is a test: when 1 thread and N differ, fix the reduction order, not the test. Fix a flaky test or quarantine it the same day. Never retry it silently.

## Maintaining

- `main` is the only long-lived branch. Work happens on topic branches (`dev/<topic>`) that merge through a PR whose required checks (`test`, `cross`, `nix`) test the merged result. No `dev` integration branch. Delete the branch after the merge.
- Bots never push to `main`. Generated output, such as bench charts, ships as a release asset and enters `docs/` through a gated PR.
- Pin every dependency by commit and content hash in `build.zig.zon`, and the toolchain in `flake.lock`. A third-party input lives in a place we control: vendored, as `models/` is, or mirrored as one of our own release assets.
- Pin actions by commit SHA with the tag in a comment. Dependabot bumps them in one grouped PR.
- `CHANGELOGS.md` is written by hand: one line per change, citing its issue. Every number in it is measured.
- A release is a `v*` tag whose version equals `build.zig.zon`. `release.yml` reruns the merge gate, builds every target, smokes the bundle and refuses an incomplete set. Releases meet the exit criteria in TESTING.md.

## Building

- Run every `zig build` inside `nix develop .#ci -c …` (or `nix develop` for the GPU toolchain). The shell gives each Zig store path its own global cache (`ZIG_GLOBAL_CACHE_DIR=~/.cache/zig-<store path>`). VerA pins a different build of the same Zig version, so with a shared `~/.cache/zig` the two evict each other's build runner and every `zig build`, even `--help`, takes about 2 minutes. The first build in a fresh cache pays a one-time compile of about 70 s.
- Edit loop: keep `zig build check --watch -fincremental` running. `check` type-checks every test root without emitting or running anything, so an edit is re-checked in under a second. On 0.18.0-dev.349 the compiler segfaults with `-fincremental` on every root that imports the device module; only the core, solver and output roots survive it. A plain `--watch` works everywhere.
- `zig build test` is the only test step. Narrow it with `-Dtest-filter=<unit test name>` and `-- --filter <deck path prefix>`. Run the corpus on a GPU with `-- --backend cuda --jobs 1`. Run it without `-fincremental`.
- Day-to-day: Debug, passed explicitly (build.zig defaults to ReleaseFast): `-Doptimize=Debug -Dgpu=false -Dcuda-arch=none -Dhip-arch=none`. The gate and the nightly run ReleaseSafe: every assertion on, LLVM codegen. `--fuzz=<N>` needs LLVM, so fuzz in ReleaseSafe. ReleaseFast is for benchmarking only.

Rebuild times per test root (measured 2026-10-19, when each root had its own step; 32 cores under load avg 11–17, `-Dgpu=false`; expect faster on a quiet machine):

| Change | Root | Time |
|---|---|---|
| nothing | core / device | 0.6 s / 2.4 s |
| `-D` values or build.zig edited (configure only) | any | 0.6–1.4 s |
| core file other than `core/ids.zig` | device | 6.5 s |
| `core/ids.zig`, `device/{eval,abi,bind}.zig` (all 87 device objects) | device | 27 s |
| core file | analysis | 21 s |
| analysis file | analysis | 14 s |
| one `models/*.va` | device | 6.5 s (that model only: ~0.1 s regeneration + its object) |
| edit under `check --watch -fincremental` (first pass 228 s) | all 115 roots | 0.2–8 s |
| first build in a new worktree / new global cache | device | ~350 s / +70 s |
| full `zig build test` (921-deck corpus) | test | ~26 min |
| ReleaseFast install, cold | — | ~16 min |
| ReleaseFast after a core edit | — | ~12 min: the `espice` exe and `libespice.a` are each one single-threaded LLVM module |

- Cold rebuild of everything: a different Zig binary, the first build of a new `-Doptimize` mode, or a new global cache (build runner only, ~70 s).
- Verilog-A regeneration (`vera --emit-zig`, cached per model on its args and inputs): one model when its `.va` or an `include` changes. Every model when the `vera` binary changes (a VerA pin bump or a Zig change), when VerA's `tools/contract.zig` changes, or when the `--zig` path changes. `-Doptimize` never triggers it.

## Worktrees

- Symlink a new worktree's `.zig-cache` to the main checkout's (`ln -s <main>/.zig-cache <worktree>/.zig-cache`) so they share build results. That needs VerA ≥ v0.20.0. Older VerA hashed absolute paths in its `simSources` WriteFiles, so each worktree regenerated every model and recompiled all device objects (230 s).
- Shared caches share incremental state: run one `--watch -fincremental` per artifact at a time across all worktrees. Take the lock: `flock -n ~/.cache/espice-incremental.lock nix develop <main>#ci -c zig build check --watch -fincremental -Doptimize=Debug -Dgpu=false -Dcuda-arch=none -Dhip-arch=none`. If it's held, use plain `zig build check` (still cached per root).
- Agents: one zig process each, and no build while `free -g` shows under 8 GB available.
