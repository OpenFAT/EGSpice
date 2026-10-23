# ESPice

A SPICE circuit simulator written in Zig. Every device model is Verilog-A, compiled at build time by
[VerA](https://github.com/OmarSiwy/VerA), and one device kernel builds for the CPU, CUDA and HIP
through [Gompute](https://github.com/OmarSiwy/Gompute).

```sh
espice deck.sp                                   # ngspice, HSPICE or Spectre deck, detected
espice -r out.raw deck.sp                        # also write waveforms
espice --format=touchstone -r out.s2p deck.sp
espice --backend cuda sweep.sp                   # sweep points as GPU lanes
```

Docs: <https://omarsiwy.github.io/ESPice/> · Changes: [CHANGELOGS.md](CHANGELOGS.md)

## Install

```sh
nix run github:OmarSiwy/ESPice -- deck.sp
nix profile install github:OmarSiwy/ESPice      # .#espice-cuda adds CUDA
```

From source (Zig 0.18.0-dev.349, provided by the devshell; VerA and Gompute are pinned in
`build.zig.zon`):

```sh
nix develop            # .#ci for CPU only
zig build              # zig-out/bin/espice; -Dgpu=false skips the GPU kernels
```

`espice --version` prints the installed version.

## Dependencies

For Ubuntu 20.04 or newer and Linux Mint 20 or newer. The release binaries link only glibc 2.28+,
so running ESPice on the CPU needs nothing beyond the base system.

| To | Install |
|---|---|
| Run on the CPU | nothing |
| Run on an NVIDIA GPU | the proprietary driver, which provides `libcuda.so.1`: `sudo ubuntu-drivers autoinstall` (or Mint's Driver Manager). The CUDA toolkit is not needed at run time. |
| Run on an AMD GPU | the ROCm HIP runtime from AMD's repository (`amdgpu-install --usecase=hip`), which provides `libamdhip64.so`. |
| Load `.hdl` models at run time | Zig `0.18.0-dev.349+d924f83bf` on `PATH`, or its path in `$ZIG` |
| Build from source with Nix (recommended) | `sudo apt install curl git xz-utils`, then Nix: `sh <(curl -L https://nixos.org/nix/install) --daemon`; `nix develop` then provides Zig, Python and the GPU toolkits |
| Build from source without Nix | `sudo apt install git curl xz-utils python3`, and Zig `0.18.0-dev.349+d924f83bf` from <https://ziglang.org/download/> |
| Embed precompiled CUDA kernels (optional) | `ptxas` and `fatbinary` on `PATH`, from NVIDIA's CUDA 12 toolkit repository (`cuda-toolkit-12-x`; Ubuntu's own `nvidia-cuda-toolkit` is too old for recent GPUs). Without them the driver compiles the kernels on first use. |
| Run the test corpus | `python3`; `nodejs` for the WebAssembly playground comparison |

## What it does

- **Analyses:** `op dc dcmatch sens tf temp`, `ac noise sp stb pz disto`,
  `tran tran_noise four envelope matex`, `pss pac pnoise pxf qpss hb`, `mc`.
- **Devices:** BSIM1–4, BSIM-SOI, BSIM-CMG, PSP, HiSIM2/HV, MOS 1/2/3/6/9, VDMOS, Gummel-Poon,
  VBIC, HICUM, JFET/MESFET/HFET, passives, controlled and behavioural sources, switches,
  transmission lines.
- **Your own models:** `.hdl "model.va"` plus an `N` card compiles a Verilog-A module at run time
  (needs that Zig on `PATH` or in `$ZIG`; cached under `~/.cache/espice/hdl`). OSDI is not loaded.
- **Output:** binary/ASCII raw, CSV, Touchstone, PSF, FSDB, SST2, CITIfile, print.
- **Parallel by default:** analyses form a DAG and run on every core (`--jobs`). `--plan` prints the
  DAG without solving.
- **Sweep throughput:** when nothing reads the waveforms, `.meas` cards stream, each transient stops
  once its measures resolve, and OPs warm-start from the nominal one. The printed values match a
  full run. `--full-waveforms` turns this off.
- **Lanes:** `--backend cuda` (or `hip`) runs sweep points as GPU lanes; `auto` picks the GPU only on
  FP64-class cards. CPU lanes (`--lanes`) are off by default.
- **Library:** `include/espice.h` is the C API, including the points API for optimizer loops
  (append points, read measures and tangents).

`espice --help` lists every flag. Exit code is 1 if any deck fails, 2 on a usage error.

## Performance

<!-- bench:start -->
![ESPice speedup by circuit size](docs/performance/bench/corpus/headline.svg)

| Compared with | Decks both ran | ESPice faster | Median speedup | Agree / differ |
|---|---:|---:|---:|---:|
| ngspice | 473 | 466 | 3.3x | 424 / 13 |
| VACASK | 270 | 270 | 3.5x | 234 / 17 |

Median of 3 runs after a warm-up, process startup included, CPU backend, 32 x Intel(R) Core(TM) i9-14900HX, commit `3a57edf3`, 2026-07-29 (local run; `.github/workflows/bench.yml` regenerates this on each release).
<!-- bench:end -->

Full report and per-analysis charts: [docs/performance](docs/performance/index.md).

## Tests

`zig build test` runs the unit suites and then the fixture corpus: about 920 decks in
`tests/fixtures/`, each scored against its checked-in `.expected.json` (oracles from ngspice and
VACASK). Per-area steps: `test-core`, `test-solver`, `test-device`, `test-frontend`,
`test-analysis`, `test-output`, `test-espice`, `test-c-api`. Benchmarks:
`nix develop .#benchmarking -c zig build bench`.

## License

Apache-2.0 ([LICENSE](LICENSE)). Models in `models/` keep their own licences
([models/LICENSES.md](models/LICENSES.md)); `models/bsim4va.va` (BSIM4, `LEVEL=14`/`54`) is
CC-BY-NC 4.0, so commercial users must replace it.
