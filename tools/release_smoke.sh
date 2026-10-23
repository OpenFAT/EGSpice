#!/usr/bin/env bash
# Run both a built-in model and runtime Verilog-A from a clean working directory.
set -euo pipefail
exe=$1
fixtures=$2
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export ESPICE_CACHE="$work/cache"
"$exe" --version
check() {
  awk -F, -v col="$2" -v want="$3" '
    NR == 1 { for (i = 1; i <= NF; i++) if ($i == col) c = i }
    NR == 2 { d = $c - want; ok = c && d * d <= 1e-12 * want * want }
    END { exit !ok }' "$1"
}
"$exe" --backend cpu --format=csv --rawfile=op.csv "$fixtures/op/balanced_bridge.sp"
check op.csv 'v(a)' 5
cp "$fixtures/hdl/veriloga_res_divider.sp" .
cp -R "$fixtures/hdl/veriloga_res_divider.assets" .
"$exe" --backend cpu --format=csv --rawfile=hdl.csv veriloga_res_divider.sp
check hdl.csv 'v(out)' 0.35
