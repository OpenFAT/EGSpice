#!/bin/sh
# /usr/bin/espice. Runtime .hdl needs the exact Zig this espice was built
# with; a distro `zig` on PATH is usually another version, so the packaged one
# is the default. Set ZIG to override it.
: "${ZIG:=/usr/lib/espice/zig/zig}"
export ZIG
exec /usr/lib/espice/bin/espice "$@"
