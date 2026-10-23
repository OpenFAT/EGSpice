# Install

ESPice is one binary, `espice`, plus a static library and C header for
embedding. There are three ways to get it.

## Nix

The repository is a flake. Run it without installing:

```sh
nix run github:OmarSiwy/ESPice -- my_circuit.sp
```

Install it into your profile:

```sh
nix profile install github:OmarSiwy/ESPice
espice --version
```

Or use it as an input of your own flake:

```nix
{
  inputs.espice.url = "github:OmarSiwy/ESPice";

  outputs = { self, nixpkgs, espice, ... }: {
    # espice.packages.<system>.default is the espice package
    devShells.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.mkShell {
      packages = [ espice.packages.x86_64-linux.default ];
    };
  };
}
```

The flake package is a CPU-only build (`-Dgpu=false`). Its `espice` wrapper
sets `ZIG` to the Zig it was built with, so decks that load Verilog-A
through `.hdl` work out of the box. Named GPU packages use the release
catalogue, for example `nix run .#espice-linux-x86_64-v3-cuda-sm_89` from
a checkout. Build from source inside `nix develop` for other targets.

## Release tarballs

The release workflow builds a tarball per platform, plus a `SHA256SUMS`
file, for each tagged release on the
[GitHub releases page](https://github.com/OmarSiwy/ESPice/releases).

!!! note
    The 1.0.0 release has no tarballs: its release build failed. Until a
    release carries them, install with Nix or build from source.

The release catalogue is [nix/release-targets.json](https://github.com/OmarSiwy/ESPice/blob/main/nix/release-targets.json).
It currently configures **128 variants**, with separate CPU and GPU choices:

| Host | CPU targets | Backends |
|---|---|---|
| Linux x86-64 | baseline (v1), SSE4.2 (v2), AVX2/FMA (v3), AVX-512 (v4) | CPU, one CUDA target, or one HIP target |
| Linux ARM64 | ARMv8.0 baseline, Neoverse N1 | CPU or one CUDA target |
| macOS Intel | x86-64 v2 | CPU |
| macOS Apple silicon | Apple M1 baseline | CPU |

CUDA targets: `sm_75`, `sm_80`, `sm_86`, `sm_87`, `sm_89`, `sm_90`,
`sm_100`, `sm_120`. Each CUDA artifact embeds PTX for just that target;
the NVIDIA driver compiles it for the installed GPU. A lower target can
also run on later GPUs with a compatible driver; see
[NVIDIA's CUDA compatibility guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html).

HIP targets: `gfx900`, `gfx906`, `gfx908`, `gfx90a`, `gfx942`, `gfx950`,
`gfx1030`, `gfx1031`, `gfx1032`, `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`,
`gfx1150`, `gfx1151`, `gfx1152`, `gfx1200`, `gfx1201`.
Choose the exact GPU target reported by `rocminfo`. These are build targets;
GPU runtime compatibility also depends on the installed driver and AMD's
[ROCm support matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

A CPU artifact has no CUDA or HIP kernels. A CUDA artifact has no HIP
kernels; a HIP artifact has no CUDA kernels or other HIP targets. There
are no Windows builds or Metal/Intel GPU backends. An unlisted GPU can
still use the CPU build on a supported host.

The native Linux archives target glibc 2.28 or newer. Pick a CPU level your
machine supports: `grep -o 'avx512f\|avx2' /proc/cpuinfo | sort -u` helps
identify v3 and v4. A higher unsupported level can terminate with `SIGILL`.

```sh
tar xzf espice-1.2.0-linux-x86_64-v3-cuda-sm_89.tar.gz
./espice-1.2.0-linux-x86_64-v3-cuda-sm_89/bin/espice --version
```

Archives contain `bin/espice`, `lib/libespice.a`, `include/espice.h`, and
`share/espice/`, the evaluator sources that `.hdl` needs at run time. Keep
the directory layout intact. Runtime `.hdl` in an archive also requires
the pinned Zig compiler on `PATH` or in `ZIG`.

### Portable Linux executable

Every Linux variant also produces a self-extracting `.run` file using
[`nix bundle`](https://nix.dev/manual/nix/latest/command-ref/new-cli/nix3-bundle).
It includes the application's Nix runtime closure and the pinned Zig
compiler for `.hdl`; Nix does not need to be installed on the destination.
HIP bundles include the ROCm runtime. GPU drivers must be installed on the
host. Linux user namespaces must be permitted for the bundle launcher.
The launcher's host prerequisites are a shell, `tar` and `bzip2`; they are
needed before the bundled runtime can be extracted.

```sh
chmod +x espice-1.2.0-linux-x86_64-v1-cpu.run
./espice-1.2.0-linux-x86_64-v1-cpu.run circuit.sp
```

To build the same bundle locally, use the repository's locked bundler:

```sh
nix bundle --bundler . .#espice-linux-x86_64-v1-cpu
```

### Release validation and automation

Pushing a `v*` tag or publishing a GitHub release runs the release workflow.
The tag must equal the version in `build.zig.zon`. Shipped builds use
ReleaseSafe, matching the merge gate. All variants must build
and the ReleaseSafe test gate must pass before assets are uploaded to the
release. Each runnable host variant checks a built-in circuit and runtime
Verilog-A. Runnable CPU-only variants also run the core and solver unit
suites at their selected instruction-set level. Linux bundles repeat the
circuit checks in a container without Nix or
network access. Runners without the requested CPU features record that
execution was unavailable; hosted runners do not test GPU execution.

`BUILD.json` inside each archive and the release's `BUILD-MATRIX.json`
record the target, revision, locked dependencies and validation coverage.
`SHA256SUMS` covers all archives, bundles and the matrix. Configuring a
target is not evidence that its build or GPU execution has passed.

Use **Actions → release → Run workflow** to build all targets or one exact
variant without publishing a release. List the available names with
`python3 tools/release_matrix.py --list`. The matrix runs up to six variants
at once, with two compile jobs per variant; the full matrix is a large
build. Add hosts or GPU targets to the catalogue to expand it, subject to
GitHub's 256-job matrix limit.

## From source

You need Zig 0.18.0-dev.349 (the flake's `nix develop` provides it) and nothing
else; the Zig dependencies (VerA, Gompute) are pinned in `build.zig.zon` and
fetched by the build.

```sh
git clone https://github.com/OmarSiwy/ESPice
cd ESPice
nix develop     # pinned Zig plus the CUDA and ROCm toolchains
zig build       # zig-out/bin/espice
```

`zig build` compiles every Verilog-A model in `models/` and, by default,
device kernels for the GPU it detects. A CPU-only build compiles much
faster:

```sh
zig build -Dgpu=false
```

`zig build run -- FILE.sp` builds and runs in one step.

Build options (`zig build --help` lists them all):

| Option | Default | Effect |
|---|---|---|
| `-Doptimize=` | `ReleaseFast` | Zig optimize mode; use `Debug` for development |
| `-Dgpu=false` | `true` | omit all model and solver GPU kernels, overriding both GPU architecture settings |
| `-Dcuda-arch=` | `auto` | CUDA arch (`sm_75`, ...); `auto` probes the build machine, `none` omits CUDA |
| `-Dhip-arch=` | `gfx1100` | HIP arch; `none` omits HIP |
| `-Ddebug-info` | `false` | DWARF in release builds, for profiling (about 3x the LLVM time) |

The install prefix (`zig-out/` unless you pass `--prefix`) gets the same
`bin/`, `lib/`, `include/` and `share/espice/` layout as the release
tarballs.

## Check the install

```sh
espice --version
```

prints the ESPice version. Then try the [quick start](../index.md#quick-start).
