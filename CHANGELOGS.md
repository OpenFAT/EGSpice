# Changelog

Newest first.

## Unreleased

- `.four` and `.fft` read the deck's `.tran` result instead of integrating the same transient again, where that run keeps its waveform with the same options and operating point (CPU backend); the values are bit for bit the same (#231).
- `--emir=FILE` writes each device terminal's and independent source's average, EM_AVG, RMS and peak current over the deck's first `.tran` (or `--emir-window=T1:T2`) as JSON, one file per `.step`/`.alter`/sweep variant (#237).
- `SWEEP MONTE=n SIGMA_SCALE=s1,... FAIL='expr'` (scaled-sigma sampling) runs n trials per sigma scale on `.tran`, `.ac` or a swept `.dc`, counts the trials whose `.meas` results fail, and extrapolates the failure probability to the real sigma with its 95 % interval and equivalent sigma; `--measures` writes it as `"kind": "sss"` (#239).
- `--measures=FILE` also writes every `.meas` result as JSON, labelled with its analysis and variant (`.step`, `SWEEP DATA`, `.alter`, Monte Carlo trial, `.temp` entry), a failed one as `null` with its error; `--measures=-` puts it on stdout (#235).
- `.option soa=1` (ngspice `warn=1`) checks every accepted point of an operating point, DC sweep and transient against the model-card SOA limits (`vds_max`, `vbe_max`, `bv_max`, ...), one line per excursion, `maxwarns` of them, then the count (#238).
- `.save @m1[gm]` and `.meas ... find @m1[id]`: MOSFET operating-point variables (terminal currents, gm, gds, gmbs, capacitances, ft, terminal voltages) at the operating point, along `.dc` and in `.tran`; an unknown name refuses the deck instead of being dropped (#236).
- `.save @m1[vth]` and `.meas ... find @m1[gm]`: device operating-point variables at the operating point, along `.dc` and in `.tran`: a model's own `desc` variables (`vth`, `vdsat`, `region`, BSIM-CMG's `ids`) and a MOSFET's terminal quantities (currents, gm, gds, gmbs, capacitances, ft, terminal voltages); an unknown name refuses the deck instead of being dropped (#236).
- VerA pinned at 1d599a0f (operating-point variables, `op_vars`/`opVars`).
- `espice --version` prints the ESPice and VerA versions and the GPU backends built in.
- `-I DIR` / `--include=DIR` adds a search directory for `.hdl` includes.
- `espice_release_points` also frees each point's parameter row and label (#225).
- `.pz` on circuits above 1000 unknowns returns the 20 poles and 20 zeros nearest s = 0 (the plot is named "partial").
- A `.tran ... uic` run no longer solves an operating point it does not use: a 100k-transistor chain started in seconds instead of hours.
- Operating points: gmin stepping and source stepping follow ngspice more closely; a few decks that needed them converge with slightly different iteration counts.
- Faster and leaner on large decks: quicker sparse LU, faster parsing of large hierarchies (a 100k-transistor chain parses in half the memory), CSV and `.print` sweeps stream to disk (a 200-point sweep: ~1 GB → ~50 MB), and Monte Carlo shares unchanged model rows (1.04 → 0.19 GB on a 1k chain).
- The compact models ship in the source tree; building no longer downloads them.
- Cached `.hdl` builds rebuild once (new device ABI).
- VerA v0.20.0 and Gompute 1.5.0.
- GPU lane batches also size their waves to each device's free memory, so a batch too large for the card runs in smaller waves instead of on the CPU (#217).
- Releases attach `.deb`, `.rpm` and Arch packages (`espice`; `espice-cuda` for NVIDIA GPUs; `espice-hip` with a build per AMD GPU target, picked at start), an AUR `PKGBUILD` (`espice-bin`) and a Homebrew formula; `tools/install.sh` installs a release from `sources.json` with a checked, cached download; `nix run .#latest` and `.#"<version>"` install release binaries.
- Removed: the `ESPICE_LINEAR_CACHE` variable, the `-Dbsim4-narrow` build option and the `test-gpu-*` build steps (`test-gpu` runs the corpus on CUDA).

## 0.3.0 (2026-10-20)

- Sweeps (`.step`, `.mc`, `.dc DATA=`/`MONTE=`, `.temp` ranges) run their points in parallel across `--jobs`, about 2x faster with flat memory; output is the same at every `--jobs`. Monte Carlo draws differ from 1.1 (#213).
- New CLI defaults: `--jobs` is one per CPU, and `--measures-only`, `--early-stop` and `--warm-start` turn on by themselves when nothing reads the transient waveforms; `--full-waveforms` turns them off. `--tokenizer` detects the dialect.
- `--measures-only`: `.meas` without storing waveforms (a 1000-row sweep: 6.5 GB → 168 MB). `--early-stop` ends a `.tran` once every `.meas` resolves; `--abort-after`/`--abort-measure` fail a point that has not.
- `--tangents=P,...` prints d(.meas)/d(P); `OPTIMIZE=` on `.tran` uses them for a cleaner Jacobian (#97).
- C API points: run many parameter points on one template and read each point's measures, tangents and status (`espice_append_points` and friends).
- `--lanes` batches sweep points (bitwise equal on the CPU); on the GPU each sweep point of a transient runs as one thread (#207–#211).
- `.meas op ... find i(vdd)` reads from the operating point.
- VerA v0.11.4: a user `.hdl` model writing `- -x` needs `-(-x)`. `.hdl` reads Verilog-AMS 2023, and 2.3-era models compile inside `begin_keywords "VAMS-2.3"`.
- A title-only deck succeeds, as in ngspice; espice.wasm works again (#8).

## 0.2.0 (2026-10-19)

- **Dialects:** Spectre netlists (`--tokenizer=spectre`); HSPICE quoting, braces, `$` comments and operators follow the HSPICE manual (#203); many ngspice parsing details match ngspice (#121).
- **New:** VPI applications (`--vpi app.so`) for Verilog-A `$systf` calls and analog callbacks (#197, #198); `.dc` over a `.param`; `.vec` files; `.qpss` on multi-tone HB; `OPTIMIZE=` grid steps; `live_params` / `set_params` to change `.param` values without re-parsing (#101).
- **Output:** `.hdl` display tasks print once per accepted point; Touchstone 2.0 with per-port z0; CSV, PSF and CITIfile fixes for downstream tools.
- **GPU:** BSIM4, VBIC, MOS 1–9, sources, lines and filters evaluate on the GPU; the device LU takes large post-layout circuits and is 4x faster there (#139–#141).
- **Accuracy:** dozens of fixes to match ngspice 45 on devices (MOS, BSIM2, BSIM4, JFET, diode, resistor TC), noise, `.tf`, `.sens`, `.four`, `.ac` sweep points and sources (#44–#168).
- **Robustness:** bad decks fail with an error naming the card instead of running wrong; out-of-memory and Newton runaway are reported instead of publishing zeros.
- **Models:** third-party compact models (PSP, BSIM-CMG/BULK/IMG/SOI, HiSIM, HICUM, Mextram, ASM-HEMT, MVSG and more) from their upstream releases, unmodified.
- `ESPICE_THREADS` gives the same output at any thread count (#137).
- Nix flake (`nix run github:OmarSiwy/ESPice`, CUDA package), the playground (`zig build wasm`) and the user documentation site.

## 0.1.0 (2026-10-05)

- First release: Zig 0.17, Apache-2.0 (models keep their own licences).
- sky130 PDK decks run as written; BSIM4 has ngspice-style limiting.
- Verilog-A models see gmin and source stepping through `$simparam`, unlike ngspice.
- `--backend cuda|hip` fails if no usable GPU is found; `auto` falls back to the CPU.
- About 90 hardening fixes; 797 of 799 corpus decks pass.
- `zig build bench` and `bench-suites` compare against ngspice and VACASK.
