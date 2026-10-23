{
  description = "ESPice";

  nixConfig = {
    extra-substituters = [ "https://omarsiwy.cachix.org" ];
    extra-trusted-public-keys = [
      "omarsiwy.cachix.org-1:fE15rGllP0D8ijLsCorvAx66mlXLp+1H1l0lR72iZ3U="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    zig-overlay.url = "github:mitchellh/zig-overlay";
    zig-overlay.inputs.nixpkgs.follows = "nixpkgs";
    nix-bundle.url = "github:nix-community/nix-bundle/eff01593f62794d458ec714090091419194ab64d";
    nix-bundle.inputs.nixpkgs.follows = "nixpkgs";
    nix-bundle.inputs.utils.follows = "flake-utils";
    # The SPICE decks (tests/netlist/), also checked out at tests/conformance:
    # a git flake leaves submodules out of its source.
    conformance.url = "github:OmarSiwy/VAMS-Conformance-Suite";
    conformance.inputs.nixpkgs.follows = "nixpkgs";
    conformance.inputs.flake-utils.follows = "flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      zig-overlay,
      nix-bundle,
      conformance,
      ...
    }:
    {
      overlays.default =
        _: prev: removeAttrs self.packages.${prev.stdenv.hostPlatform.system} [ "default" ];
    }
    // flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        lib = pkgs.lib;
        zig = zig-overlay.packages.${system}.master;

        # Explicit glibc target: in the sandbox Zig falls back to a dynamic musl binary that cannot run.
        zigTarget = "${pkgs.stdenv.hostPlatform.parsed.cpu.name}-${
          if pkgs.stdenv.isDarwin then "macos" else "linux-gnu.2.28"
        }";

        src = lib.fileset.toSource {
          root = ./.;
          fileset = lib.fileset.unions [
            ./build.zig
            ./build.zig.zon
            ./src
            ./models
            ./include
            ./tests
            ./tools
            ./docs/embed/host.c
          ];
        };

        # The sandbox has no network: fetch build.zig.zon's dependencies in a fixed-output derivation.
        zigDeps = pkgs.stdenvNoCC.mkDerivation {
          name = "espice-zig-deps";
          inherit src;
          nativeBuildInputs = [ zig ];
          dontConfigure = true;
          dontInstall = true;
          buildPhase = ''
            export HOME="$TMPDIR" ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zgc"
            zig build --fetch
            cp -r "$TMPDIR/zgc/p" "$out"
          '';
          outputHashMode = "recursive";
          outputHash = "sha256-FxNNSbdKPSyj9kJwD6HzPcCN6MHFR/MRNRMb4U7xY20=";
        };

        gpuPkgs = lib.optionals pkgs.stdenv.isLinux (
          [ pkgs.cudaPackages.cudatoolkit ]
          ++ lib.optionals pkgs.stdenv.hostPlatform.isx86_64 [
            pkgs.rocmPackages.clr
            pkgs.rocmPackages.rocminfo
          ]
        );
        gpuLibPath = lib.makeLibraryPath ([ "/run/opengl-driver" ] ++ gpuPkgs);

        openvafPkg = import ./nix/openvaf.nix { inherit pkgs; };
        vacaskPkg = import ./nix/vacask.nix { inherit pkgs openvafPkg; };

        devShells = rec {
          ci = pkgs.mkShell {
            packages = [
              zig
              pkgs.python3
            ];
            # One global cache per Zig store path: two builds of the same Zig version
            # sharing ~/.cache/zig evict each other's build runner on every `zig build`.
            shellHook = ''export ZIG_GLOBAL_CACHE_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/zig-${builtins.baseNameOf zig}"'';
          };
          default = pkgs.mkShell {
            inputsFrom = [ ci ];
            packages = gpuPkgs;
            LD_LIBRARY_PATH = gpuLibPath;
          };
        }
        // lib.optionalAttrs (system == "x86_64-linux") {
          benchmarking = pkgs.mkShell {
            inputsFrom = [ devShells.default ];
            packages = [
              (pkgs.python3.withPackages (ps: [ ps.matplotlib ]))
              pkgs.ngspice
              pkgs.xyce
              openvafPkg
              vacaskPkg
              pkgs.perf
              pkgs.inferno
            ];
            LD_LIBRARY_PATH = gpuLibPath;
          };
        };

        releaseTargets = builtins.fromJSON (builtins.readFile ./nix/release-targets.json);
        releaseVariants = lib.concatMap (
          host:
          map (
            backend:
            host
            // {
              name = "${host.name}-${backend.name}";
              inherit (backend) cuda hip;
            }
          ) (builtins.filter (backend: builtins.elem system backend.systems) releaseTargets.backends)
        ) (builtins.filter (host: host.system == system) releaseTargets.hosts);

        mkEspice =
          {
            name ? "espice",
            target ? zigTarget,
            cpu ? (if pkgs.stdenv.hostPlatform.isx86_64 then "x86_64" else "generic"),
            cuda ? "none",
            hip ? "none",
          }:
          pkgs.stdenv.mkDerivation {
            pname = name;
            version = "1.2.0";
            inherit src;
            nativeBuildInputs = [
              zig
              pkgs.makeWrapper
            ]
            ++ lib.optional pkgs.stdenv.isLinux pkgs.autoPatchelfHook;
            dontConfigure = true;
            dontInstall = true;
            buildPhase = ''
              export ZIG_GLOBAL_CACHE_DIR="$TMPDIR/zig-global-cache"
              mkdir -p "$ZIG_GLOBAL_CACHE_DIR"
              cp -r --no-preserve=mode,ownership ${zigDeps} "$ZIG_GLOBAL_CACHE_DIR/p"
              zig build install -Doptimize=ReleaseSafe -Dtarget=${target} -Dcpu=${cpu} \
                -Dgpu=${if cuda != "none" || hip != "none" then "true" else "false"} \
                -Dhip-arch=${hip} -Dcuda-arch=${cuda} \
                -j$NIX_BUILD_CORES --prefix "$out"
              wrapProgram $out/bin/espice --set-default ZIG ${zig}/bin/zig \
                ${lib.optionalString (hip != "none")
                  "--prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath [ pkgs.rocmPackages.clr ]}"
                }
            '';
            meta = {
              description = "SPICE circuit simulator (${cpu}, CUDA ${cuda}, HIP ${hip})";
              mainProgram = "espice";
              license = lib.licenses.asl20;
            };
          };
        espice = mkEspice { };
        espiceCuda = mkEspice {
          name = "espice-cuda";
          cuda = "sm_75";
        };
        baselineName = if pkgs.stdenv.hostPlatform.isx86_64 then "linux-x86_64-v1" else "linux-aarch64";
      in
      {
        inherit devShells;

        packages = {
          default = espice;
          inherit espice;
        }
        // builtins.listToAttrs (
          map (variant: {
            name = "espice-${variant.name}";
            value = mkEspice {
              name = "espice-${variant.name}";
              inherit (variant)
                target
                cpu
                cuda
                hip
                ;
            };
          }) releaseVariants
        )
        // lib.optionalAttrs pkgs.stdenv.isLinux {
          espice-cuda = espiceCuda;
          # Reuse CI's baseline packages when bundling these release variants.
          "espice-${baselineName}-cpu" = espice;
          "espice-${baselineName}-cuda-sm_75" = espiceCuda;
        };

        bundlers = lib.optionalAttrs pkgs.stdenv.isLinux {
          default = nix-bundle.bundlers.${system}.default;
        };

        checks.smoke = pkgs.runCommand "espice-smoke" { } ''
          export HOME=$TMPDIR ESPICE_CACHE=$TMPDIR/cache
          check() {
            awk -F, -v col="$2" -v want="$3" '
              NR == 1 { for (i = 1; i <= NF; i++) if ($i == col) c = i }
              NR == 2 { d = $c - want; ok = c && d * d <= 1e-12 * want * want }
              END { exit !ok }' "$1" || { echo "$1: $2 != $3"; cat "$1"; exit 1; }
          }
          cp -r ${conformance}/tests/netlist/hdl/veriloga_res_divider.{sp,assets} .
          ${lib.getExe espice} --format=csv --rawfile=op.csv ${conformance}/tests/netlist/op/balanced_bridge.sp
          check op.csv 'v(a)' 5
          ${lib.getExe espice} --format=csv --rawfile=hdl.csv veriloga_res_divider.sp
          check hdl.csv 'v(out)' 0.35
          ls cache/hdl > /dev/null
          touch $out
        '';
      }
    );
}
