#!/bin/sh
# Installs an ESPice release from sources.json, the same catalogue the flake's
# versioned packages install from, without nix (after VerA's tools/install.sh):
#
#   curl -fsSL https://raw.githubusercontent.com/OpenFAT/EGSpice/main/tools/install.sh | sh
#   curl -fsSL .../install.sh | sh -s -- 0.3.0       # a pinned version
#
# Each version lives in $ESPICE_INSTALL_HOME/<version> (the release tree: bin,
# lib, include, share/espice) and $ESPICE_INSTALL_BIN/espice links to the one
# installed last. A tarball is kept in $ESPICE_INSTALL_CACHE and checked against
# sources.json's sha256 before it is unpacked, so installing a version again,
# or switching back to it, downloads nothing.
#
#   ESPICE_INSTALL_SOURCES  sources.json to read (URL or path; default: main's)
#   ESPICE_INSTALL_HOME     default ~/.local/share/espice
#   ESPICE_INSTALL_BIN      default ~/.local/bin
#   ESPICE_INSTALL_CACHE    default ${XDG_CACHE_HOME:-~/.cache}/espice/releases
# (ESPICE_CACHE is espice's own .hdl cache; it is not read here.)
#
# Needs sh, curl, tar and sha256sum or shasum. Runtime .hdl models build with
# the Zig in sources.json's "zig", printed at the end (zig on PATH, or ZIG).
set -eu

sources=${ESPICE_INSTALL_SOURCES:-https://raw.githubusercontent.com/OpenFAT/EGSpice/main/sources.json}
home=${ESPICE_INSTALL_HOME:-$HOME/.local/share/espice}
bindir=${ESPICE_INSTALL_BIN:-$HOME/.local/bin}
cache=${ESPICE_INSTALL_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/espice/releases}

die() { echo "espice install: $*" >&2; exit 1; }

case $(uname -s) in
  Linux) os=linux ;;
  Darwin) os=darwin ;;
  *) die "no release for $(uname -s); build from source (zig build)" ;;
esac
case $(uname -m) in
  x86_64 | amd64) cpu=x86_64 ;;
  aarch64 | arm64) cpu=aarch64 ;;
  *) die "no release for $(uname -m)" ;;
esac
system="$cpu-$os"

fetch() { # fetch URL-or-path OUT
  case $1 in
    http://* | https://* | file://*) curl -fsSL "$1" -o "$2" ;;
    *) cp "$1" "$2" ;;
  esac
}
sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

mkdir -p "$cache"
catalogue="$cache/sources.json"
fetch "$sources" "$catalogue.part" || die "cannot read $sources"
mv "$catalogue.part" "$catalogue"

# sources.json is written by tools/update-sources.py with json.dump(indent=2,
# sort_keys=True): a version's key at two spaces, a system's at four, its
# url and sha256 at six. `field VERSION KEY NAME` prints one value.
field() {
  awk -v ver="$1" -v key="$2" -v name="$3" '
    /^  "[^"]*": \{/      { split($0, a, "\""); v = a[2]; k = "" }
    /^    "[^"]*": \{/    { split($0, a, "\""); k = a[2] }
    /^    "[^"]*": "/     { split($0, a, "\""); if (v == ver && key == "" && a[2] == name) print a[4] }
    /^      "[^"]*": "/   { split($0, a, "\""); if (v == ver && k == key && a[2] == name) print a[4] }
  ' "$catalogue"
}

version=${1:-}
if [ -z "$version" ]; then
  version=$(awk '/^  "[^"]*": \{/ { split($0, a, "\""); print a[2] }' "$catalogue" | sort -t. -k1,1n -k2,2n -k3,3n | tail -n 1)
  [ -n "$version" ] || die "$sources lists no release"
fi
version=${version#v}
url=$(field "$version" "$system" url)
want=$(field "$version" "$system" sha256)
[ -n "$url" ] && [ -n "$want" ] || die "sources.json has no $system build of $version"

tarball="$cache/espice-$version-$system.tar.gz"
if [ ! -f "$tarball" ] || [ "$(sha256 "$tarball")" != "$want" ]; then
  echo "espice install: downloading $version for $system"
  fetch "$url" "$tarball.part" || die "cannot download $url"
  got=$(sha256 "$tarball.part")
  [ "$got" = "$want" ] || { rm -f "$tarball.part"; die "$url: sha256 $got, sources.json says $want"; }
  mv "$tarball.part" "$tarball"
else
  echo "espice install: $version for $system is cached"
fi

# The tree keeps its layout: espice finds share/espice beside its executable.
dest="$home/$version"
rm -rf "$dest.part"
mkdir -p "$dest.part"
tar xzf "$tarball" -C "$dest.part" --strip-components=1
rm -rf "$dest"
mv "$dest.part" "$dest"

mkdir -p "$bindir"
ln -sf "$dest/bin/espice" "$bindir/espice"
echo "espice install: $bindir/espice -> $dest/bin/espice"
zig=$(field "$version" "" zig)
[ -n "$zig" ] && echo "espice install: runtime .hdl models build with Zig $zig (zig on PATH, or ZIG)"
case ":$PATH:" in *":$bindir:"*) ;; *) echo "espice install: add $bindir to PATH" ;; esac
