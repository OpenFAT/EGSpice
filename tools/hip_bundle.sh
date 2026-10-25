#!/usr/bin/env bash
# The espice-hip tree: every linux-x86_64-v1-hip-gfx* release tarball's espice
# as bin/espice-gfx*, beside one share/espice (identical in every build, which
# this checks) and the docs. packaging/espice-hip.sh picks the binary at start.
# ponytail: no lib/libespice.a (130 MB per target, 18 targets); C embedders of
# a HIP build take that target's tarball.
# Usage: tools/hip_bundle.sh DIST VERSION OUT
set -euo pipefail
dist=$1 version=$2 out=$3
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
rm -rf "$out"
mkdir -p "$out/bin"
count=0
for tarball in "$dist/espice-$version-linux-x86_64-v1-hip-"gfx*.tar.gz; do
  [ -e "$tarball" ] || break
  variant=$(basename "$tarball" .tar.gz)
  gfx=${variant##*-hip-}
  tar xzf "$tarball" -C "$work"
  tree="$work/$variant"
  if [ "$count" = 0 ]; then
    cp -R "$tree/share" "$out/"
    cp "$tree/README.md" "$tree/LICENSE" "$tree/CHANGELOGS.md" "$out/"
  elif ! diff -r "$out/share" "$tree/share" >/dev/null; then
    echo "hip_bundle: $variant's share/ differs from the first build's" >&2
    exit 1
  fi
  install -m 755 "$tree/bin/espice" "$out/bin/espice-$gfx"
  rm -rf "$tree"
  count=$((count + 1))
done
[ "$count" -gt 0 ] || { echo "hip_bundle: no espice-$version-linux-x86_64-v1-hip-gfx*.tar.gz in $dist" >&2; exit 1; }
[ -x "$out/bin/espice-gfx1100" ] || { echo "hip_bundle: no gfx1100 build, the wrapper's fallback" >&2; exit 1; }
echo "hip_bundle: $count targets"
