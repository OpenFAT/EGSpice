#!/usr/bin/env bash
# .deb, .rpm and Arch packages of one release tree (after VerA's tools/package.sh),
# with the Zig the dev shell pins for runtime .hdl. release.yml and cross.yml run it.
# Usage: tools/package.sh TREE VERSION amd64|arm64 DIST
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
TREE=$(realpath "$1") VERSION=$2 ARCH=$3
mkdir -p "$4"
dist=$(realpath "$4")
# shellcheck disable=SC2016 # the dev shell's sh expands it
ZIG_DIR=$(nix develop "$root#ci" -c sh -c 'dirname "$(dirname "$(readlink -f "$(command -v zig)")")"')
export TREE VERSION ARCH ZIG_DIR
config=$(mktemp)
trap 'rm -f "$config"' EXIT
# nfpm expands no variables in contents paths.
python3 -c 'import os, sys; sys.stdout.write(os.path.expandvars(open(sys.argv[1]).read()))' \
  "$root/packaging/nfpm.yaml" > "$config"
cd "$root" # nfpm.yaml names packaging/espice.sh relative to the checkout
for format in deb rpm archlinux; do
  ext=$format
  [ "$format" = archlinux ] && ext=pkg.tar.zst
  nix shell --inputs-from "$root" nixpkgs#nfpm -c nfpm package -f "$config" -p "$format" \
    -t "$dist/espice-$VERSION-linux-$ARCH.$ext"
done
