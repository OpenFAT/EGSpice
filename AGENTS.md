# Agents

## Building

- Run every `zig build` inside `nix develop .#ci -c …` (or `nix develop` for the GPU toolchain). The shell gives each Zig store path its own global cache (`ZIG_GLOBAL_CACHE_DIR=~/.cache/zig-<store path>`). VerA pins a different build of the same Zig version, so with a shared `~/.cache/zig` the two evict each other's build runner and every `zig build`, even `--help`, takes about 2 minutes. The first build in a fresh cache pays a one-time compile of about 70 s.
- Edit loop: keep `zig build check --watch -fincremental` running. `check` type-checks every test root without emitting or running anything, so an edit is re-checked in under a second. To run tests, use `zig build test-<x>` without `-fincremental`. On 0.18.0-dev.349 the compiler segfaults with `-fincremental` on every root that imports the device module (test-device, test-frontend, test-analysis, test-espice); only test-core, test-solver and test-output survive it. A plain `--watch` works everywhere.
- Day-to-day: Debug, passed explicitly (build.zig defaults to ReleaseFast): `-Doptimize=Debug -Dgpu=false -Dcuda-arch=none -Dhip-arch=none`, one test step at a time. ReleaseFast is for benchmarking only.

Rebuild times (measured 2026-10-19, 32 cores under load avg 11–17, `-Dgpu=false`; expect faster on a quiet machine):

| Change | Step | Time |
|---|---|---|
| nothing | test-core / test-device | 0.6 s / 2.4 s |
| `-D` values or build.zig edited (configure only) | any | 0.6–1.4 s |
| core file other than `core/ids.zig` | test-device | 6.5 s |
| `core/ids.zig`, `device/{eval,abi,bind}.zig` (all 87 device objects) | test-device | 27 s |
| core file | test-analysis | 21 s |
| analysis file | test-analysis | 14 s |
| one `models/*.va` | test-device | 6.5 s (that model only: ~0.1 s regeneration + its object) |
| edit under `check --watch -fincremental` (first pass 228 s) | all 115 roots | 0.2–8 s |
| first build in a new worktree / new global cache | test-device | ~350 s / +70 s |
| full `zig build test` (921-deck corpus) | test | ~26 min |
| ReleaseFast install, cold | — | ~16 min |
| ReleaseFast after a core edit | — | ~12 min: the `espice` exe and `libespice.a` are each one single-threaded LLVM module |

- Cold rebuild of everything: a different Zig binary, the first build of a new `-Doptimize` mode, or a new global cache (build runner only, ~70 s).
- Verilog-A regeneration (`vera --emit-zig`, cached per model on its args and inputs): one model when its `.va` or an `include` changes. Every model when the `vera` binary changes (a VerA pin bump or a Zig change), when VerA's `tools/contract.zig` changes, or when the `--zig` path changes. `-Doptimize` never triggers it.

## Worktrees

- Symlink a new worktree's `.zig-cache` to the main checkout's (`ln -s <main>/.zig-cache <worktree>/.zig-cache`) so they share build results. That needs VerA ≥ a9ca4351. Older VerA hashed absolute paths in its `simSources` WriteFiles, so each worktree regenerated every model and recompiled all device objects (230 s).
- Shared caches share incremental state: run one `--watch -fincremental` per artifact at a time across all worktrees. Take the lock: `flock -n ~/.cache/espice-incremental.lock nix develop <main>#ci -c zig build check --watch -fincremental -Doptimize=Debug -Dgpu=false -Dcuda-arch=none -Dhip-arch=none`. If it's held, use plain `zig build check` (still cached per root), and run tests with `zig build test-<x>` one step at a time.
- Main pins VerA a9ca4351 from the local tarball `~/.cache/espice-pins/vera-a9ca4351.tar.gz` (owner decision, f032f6cc). Branch new worktrees from main ≥ f032f6cc. That commit is not on GitHub, so CI can't fetch it.
- Agents: at most 8 at once, one zig process each, and no build while `free -g` shows under 8 GB available.
