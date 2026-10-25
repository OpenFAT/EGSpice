#!/usr/bin/env bash
# Determinism oracle: every corpus deck, run with one device and solver thread
# and again with N, must exit the same way and write byte-identical raw output.
# A difference is a reduction-order bug: fix the order, never this check.
#   tools/determinism.sh zig-out/bin/espice tests/conformance/tests/netlist [N]
set -uo pipefail
exe=$(realpath "$1")
fixtures=$2
n=${3:-8}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
export ESPICE_CACHE="$work/cache" exe n work

one() {
  deck=$1
  key=$(printf %s "$deck" | tr / _)
  tok=$(sed -n 's/.*"tokenizer": *"\([^"]*\)".*/\1/p' "${deck%.sp}.expected.json")
  for t in 1 "$n"; do
    (cd "$(dirname "$deck")" && ESPICE_THREADS=$t ESPICE_SOLVER_THREADS=$t timeout 900 \
      "$exe" --backend cpu --format=binary -b -r "$work/$key.$t.raw" --tokenizer "${tok:-ngspice}" \
      "$(basename "$deck")" >/dev/null 2>&1)
    echo $? >"$work/$key.$t.rc"
  done
  if ! cmp -s "$work/$key.1.rc" "$work/$key.$n.rc"; then
    echo "DIFF exit $(cat "$work/$key.1.rc") vs $(cat "$work/$key.$n.rc"): $deck"
  elif [ -e "$work/$key.1.raw" ] && ! cmp -s "$work/$key.1.raw" "$work/$key.$n.raw"; then
    echo "DIFF raw: $deck"
  fi
  rm -f "$work/$key".*
}
export -f one

find "$fixtures" -name '*.sp' | while read -r d; do [ -e "${d%.sp}.expected.json" ] && echo "$d"; done |
  sort | xargs -P "$(nproc)" -I{} bash -c 'one "$1"' _ {} | sort | tee "$work/diffs"
total=$(find "$fixtures" -name '*.expected.json' | wc -l)
bad=$(wc -l <"$work/diffs")
echo "determinism: $bad of $total decks differ between 1 and $n threads"
[ "$bad" -eq 0 ]
