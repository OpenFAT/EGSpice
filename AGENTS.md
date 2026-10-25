# Agents

ESPice is a simulator whose consumers pin it by commit and hash, or follow `main` (`nix run github:OmarSiwy/ESPice`). Two rules decide everything below: **every test names its oracle**, and **main is green by machine**. Read [TESTING.md](TESTING.md) before adding a test layer, touching a gate, or claiming something is tested; it maps every bug class to its oracle, layer, file and tier. [CONTRIBUTING.md](CONTRIBUTING.md) holds the branch, pin and release policy for people, and [SECURITY.md](SECURITY.md) the reporting route and trust boundary.

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
