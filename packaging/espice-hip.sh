#!/bin/sh
# /usr/bin/espice for espice-hip, which installs one espice per AMD GPU target
# (HIP code objects run on the gfx target they were built for, nothing else).
# This runs the build for the first GPU the kernel's KFD topology reports:
# gfx_target_version is major*10000 + minor*100 + stepping, and the target
# name spells the stepping in hex (90010 = gfx90a, 110000 = gfx1100).
# ESPICE_HIP_ARCH=gfx... picks one instead. With no AMD GPU it runs the
# gfx1100 build, whose CPU backend is the same in every build.
: "${ZIG:=/usr/lib/espice/zig/zig}"
export ZIG
bin=${ESPICE_HIP_BIN:-/usr/lib/espice/bin}
topology=${ESPICE_KFD_TOPOLOGY:-/sys/class/kfd/kfd/topology/nodes}
arch=${ESPICE_HIP_ARCH:-}
if [ -z "$arch" ]; then
  for properties in "$topology"/*/properties; do
    [ -r "$properties" ] || continue
    while read -r key value; do
      [ "$key" = gfx_target_version ] && [ "$value" != 0 ] && break
      value=
    done < "$properties"
    [ -n "$value" ] || continue # the CPU node reports 0
    arch=$(printf 'gfx%d%d%x' $((value / 10000)) $((value / 100 % 100)) $((value % 100)))
    break
  done
fi
if [ -n "$arch" ] && [ -x "$bin/espice-$arch" ]; then
  exec "$bin/espice-$arch" "$@"
fi
if [ -n "$arch" ]; then
  echo "espice: this package has no build for $arch; running the gfx1100 build, whose HIP kernels will not load on it (ESPICE_HIP_ARCH picks another)" >&2
fi
exec "$bin/espice-gfx1100" "$@"
