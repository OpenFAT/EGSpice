#!/usr/bin/env bash
# .deb, .rpm and Arch packages of one release tree (after VerA's tools/package.sh),
# with the Zig the dev shell pins for runtime .hdl. release.yml and cross.yml run it.
# Usage: tools/package.sh espice|espice-cuda|espice-hip TREE VERSION amd64|arm64 DIST
# (espice-hip's TREE is tools/hip_bundle.sh's.)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
NAME=$1 TREE=$(realpath "$2") VERSION=$3 ARCH=$4
mkdir -p "$5"
dist=$(realpath "$5")
case $NAME in
  espice) DESCRIPTION="CPU build" WRAPPER=packaging/espice.sh ;;
  espice-cuda) DESCRIPTION="CUDA build: NVIDIA GPUs from Turing (sm_75) on, through the driver's libcuda.so.1" WRAPPER=packaging/espice.sh ;;
  espice-hip) DESCRIPTION="HIP build: one espice per AMD GPU target, chosen at start, through ROCm's libamdhip64.so" WRAPPER=packaging/espice-hip.sh ;;
  *) echo "package.sh: unknown package $NAME" >&2; exit 2 ;;
esac
# shellcheck disable=SC2016 # the dev shell's sh expands it
ZIG_DIR=$(nix develop "$root#ci" -c sh -c 'dirname "$(dirname "$(readlink -f "$(command -v zig)")")"')
export NAME DESCRIPTION WRAPPER TREE VERSION ARCH ZIG_DIR
config=$(mktemp)
trap 'rm -f "$config"' EXIT
# nfpm expands no variables in contents paths.
python3 -c 'import os, sys; sys.stdout.write(os.path.expandvars(open(sys.argv[1]).read()))' \
  "$root/packaging/nfpm.yaml" > "$config"
cd "$root" # WRAPPER is relative to the checkout
for format in deb rpm archlinux; do
  ext=$format
  [ "$format" = archlinux ] && ext=pkg.tar.zst
  nix shell --inputs-from "$root" nixpkgs#nfpm -c nfpm package -f "$config" -p "$format" \
    -t "$dist/$NAME-$VERSION-linux-$ARCH.$ext"
done
