# Testing

How ESPice proves a result is right, and which gate runs each proof. Rules every change obeys
are in [AGENTS.md](AGENTS.md#testing); this file is the map behind them, after the posts
"Exhaustive Testing" and "Large-Project Maintainer" (2026-10-21).

An **oracle** is the independent thing that says an output is wrong. Every layer below names one.
The main bug class in a simulator is a wrong number inside the plausible range, so certificates
and differential oracles come first; coverage and fuzzing alone cannot see it.

## Domains

| Domain | Components | What "correct" means |
|---|---|---|
| Compiler front end | `src/frontend/` (ngspice, HSPICE, Spectre dialects, `.meas`, expressions, `.vec`, Touchstone) | every deck parses to the netlist the dialect's reference defines, or is refused with a declared error |
| Device models | `models/*.va` through VerA, `src/device/` | currents, charges and their Jacobians match the model's equations |
| Numerical solver | `src/solver/` (sparse LU, lanes, GMRES), `src/analysis/` (Newton, integrators, AC, noise, PSS, HB, PZ) | each solve satisfies its equations to the stated tolerance |
| Stateful and parallel | lane batches, `--jobs` scheduler, `Problem` sessions, the C ABI | results equal the serial, one-at-a-time run |
| GPU kernels | CUDA and HIP device eval, LU, lane kernels (Gompute) | equal to the CPU path within each deck's tolerance |
| Output | raw (ASCII, binary), CSV, FSDB, SST2 writers | readers get back every bit written |

## Bug class → oracle → layer → where

| Bug class | Oracle | Layer | Where |
|---|---|---|---|
| Wrong number on a real deck | reference simulator or analytic derivation, recorded in each `.expected.json` with its provenance and the deck's hash | differential corpus | `tests/test_correctness.zig` over `tests/conformance/tests/netlist/` (`zig build test`) |
| A deck silently moving between PASS and XFAIL | the name list `tests/VERDICTS.tsv` | ratchet | full CPU run of `zig build test` |
| A tolerance too loose to see an error | every expected value moved 0.1%: a deck that still passes is LOOSE | perturbation (`--perturb`, reports) | `tests/test_correctness.zig`; nightly |
| A missing feature hidden as a skip | `* KNOWN GAP:` line in the deck: XFAIL keeps the run green, XPASS fails it | ratchet | same |
| Wrong Newton point, no crash | independent KCL residual under `abstol + reltol·max\|I\|` | certificate (`-Dcertify=true`) | `src/analysis/Circuit.zig` `kcl_cert`; nightly over the corpus |
| Wrong LU solve | LAPACK scaled residual ≤ 30 | certificate | `src/solver/tests.zig`, `src/solver/direct.zig` probes |
| Wrong device derivative | central finite difference at random voltages | property | `src/device/tests/jacobian_fd.zig` |
| Wrong integrator order | observed order `log2(e_h / e_{h/2})` on RC and RLC | convergence order | `src/analysis/tests/transient.zig` |
| Engine depends on node names, element order, impedance scale | rename, reorder, split, scale by k, 0 A source | metamorphic | `tests/metamorphic.zig` (in `test`) |
| Parser panic, leak, undeclared error | no panic, no leak, error from the declared set; same digest on reparse and after no-op rewrites | fuzz (`std.testing.fuzz`, structured and raw) | `src/frontend/tests/{fuzz,expr,measure,source,spectre,sparam,vec}.zig`; corpus replay in `test`, `--fuzz` nightly |
| Expression folding wrong | recursive reference evaluator | differential fuzz | `src/frontend/tests/expr.zig` |
| Swallowed out-of-memory | `checkAllAllocationFailures`: each allocation fails once | fault injection | 38 files under `src/` |
| SIMD lane differs from scalar | the `W == 1` instantiation of the same kernel, bitwise | differential | `src/solver/lane/`, `src/analysis/tests/` lane tests |
| Batch or lane leaks into a neighbour | each lane bitwise equal to its run alone, NaN and failing lanes included | differential | `src/analysis/tests/` "lane batches" tests |
| A `.four`/`.fft` reading a `.tran`'s result (#231) differs from its own run | the reader appended alone, bitwise; a `.tran` with other options, OP, columns or measures is not read | differential | `src/analysis/tests/session.zig` "read a matching .tran" |
| Parallel nondeterminism | `--jobs=1` against N and permuted ready sets: byte-identical | determinism | `src/analysis/tests/session.zig`; whole corpus nightly: `tools/determinism.sh` |
| Crash, hang or blow-up on huge or deep input | exit 0 or a diagnostic, never a signal or a timeout, under an 8 GB cap | adversarial sizes | `tools/adversarial.py`; nightly |
| C ABI misuse | model of valid states under random call sequences, forged handles | stateful fuzz | `src/c_api.zig` |
| Output corruption | round trip of every f64 bit pattern | round trip | `src/output/` tests |
| `--measures` JSON disagrees with the printed `.meas` or mislabels a variant | the printed text line for line; the deck's own `.step`/`.data` values bit for bit; RC closed forms | differential | `src/output/measure.zig`, `src/tests/espice.zig` "--measures" tests |
| `--emir` terminal current wrong, mis-signed or mislabelled | Ohm's law on a divider; Kirchhoff at every internal net; a sine's RMS and half-wave mean; `.meas EM_AVG` of the same branch and window; the `.step` values; 1 job against 4, byte for byte | analytic, certificate, differential | `src/tests/espice.zig` "--emir" tests |
| Browser build differs from native | the native espice at the same lane width, within 1e-9 | differential | `tools/playground/compare.js` (`pages.yml`) |
| A target or OS stops building | the compiler for every release target | cross compile | `cross.yml`; the gate on Linux and macOS |
| GPU differs from CPU | the CPU result of the same deck | differential | `zig build test -- --backend cuda --jobs 1` (local; no GPU runner) |
| HIP backend miscompiles | kernel symbols present after an LLVM AMDGPU compile | compile probe | `build.zig`, in `test` |
| Abstraction costs run time | hand-written twin's assembly | codegen probe | `tests/codegen.zig` + `tools/asm_probe.py` (in `test`) |
| Release ships a broken artifact | smoke decks from a clean directory, a container with no `/nix` | smoke | `tools/release_smoke.sh`, `release.yml` |
| Docs example rots | the example compiled and run; every docs deck runs to exit 0 | executed docs | `docs/embed/host.c` in `test`; `tools/doc_decks.sh` in the gate |

## Gates

| Tier | Where | Contents |
|---|---|---|
| Edit loop | local, seconds | `zig build check --watch -fincremental`; `zig build test -Dtest-filter=… -- --filter …` |
| Merge gate | `test.yml`, `cross.yml`, `nix.yml` on every PR, push to `main` and merge queue | on Linux and macOS: `zig build test` in ReleaseSafe (every suite, fuzz-corpus replay, the corpus with its name list, metamorphic rewrites, codegen and HIP compile probes, the C API example, bench-adapter tests), every docs deck, release-matrix unit tests; every release target compiled; `nix flake check` and smoke |
| Nightly | `nightly.yml`, 06:17 UTC | corpus with KCL certificates, the perturbed corpus (LOOSE list in the summary), `zig build test --fuzz=20M` (every fuzz target), `tools/determinism.sh` (1 thread against 8), `tools/adversarial.py`, `zig build bench` against ngspice, VACASK and Xyce |
| Docs deploy | `pages.yml` on `main` | the playground's espice.wasm against native espice over the corpus |
| Release | `release.yml` on `v*` | the merge gate via `workflow_call`, every target at every ISA level with `zig build test` run where the runner can, bundle smoke outside Nix, checksums; `bench.yml` attaches charts |
| Local only | a machine with a card | `zig build test -- --backend cuda --jobs 1` before any change to a GPU path lands |

A flaky test is fixed or quarantined the same day, never retried silently. A fuzz or determinism
failure becomes a checked-in input (`src/frontend/tests/fuzz_corpus/`, or a deck in the conformance suite).

## Exit criteria for a release

Each line is a yes or a number a script reads; none is typed from memory.

- `zig build test` passes in ReleaseSafe: the corpus at 100 percent with only `KNOWN GAP` XFAILs, `VERDICTS.tsv` unchanged or moved XFAIL → PASS only.
- The last nightly on the release commit is green: zero certificate misses, zero decks differ between 1 and 8 threads, no fuzz crash in 20M iterations per run.
- `zig build test -- --backend cuda --jobs 1` passes on CUDA hardware for the release commit. A backend that has never passed the corpus on a card is listed below as untested, and the README says so.
- Zero open crash bugs.

## Not yet tested

- **Debug builds**: the self-hosted backend diverges from LLVM (#25); the gate runs ReleaseSafe only.
- **GPU in CI**: no GPU runner; the GPU corpus runs locally, and HIP is checked by compilation only.
- **Merge gate budget**: a cold 4-core runner takes over an hour, past the 15-minute target.
- **The browser build**: wasm32 does not compile on this branch (the performance round's `u64`/`usize` sites in `src/solver/subtree.zig`, `src/analysis/executor.zig`, `src/analysis/gpu/devices.zig`), so the playground and its differential do not run; `pages.yml` deploys without them.
- **Tolerance ratchet**: the LOOSE list is reported, not gated; it becomes a name list like `VERDICTS.tsv` once triaged.
- **Coverage**: no line or branch tool for Zig; kcov over the test binary is the cheap first step.
- **Mutation testing**: no tool; the substitute is a script flipping one comparison at a time.
- **Fuzz bookkeeping**: CPU-hours and corpus growth per target are not recorded, and nightly finds are not saved automatically.
