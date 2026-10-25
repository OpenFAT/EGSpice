#!/usr/bin/env bash
# Documentation that executes: every deck under docs/examples/ and every inline
# ```spice block in the docs (the bench reports excepted) must run to exit 0.
# A page whose deck no longer runs fails here, before a reader finds it.
#   tools/doc_decks.sh zig-out/bin/espice
set -uo pipefail
exe=$(realpath "$1")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
export ESPICE_CACHE="$work/cache"
fail=0 total=0

run() { # run LABEL DIR DECK [espice args...]
  local label=$1 dir=$2 deck=$3
  shift 3
  total=$((total + 1))
  if ! (cd "$dir" && timeout 300 "$exe" --backend cpu -b -r "$work/out.raw" "$@" "$deck" >"$work/log" 2>&1); then
    fail=$((fail + 1))
    echo "FAIL $label"
    tail -5 "$work/log"
  fi
}

for deck in docs/examples/*.sp; do run "$deck" docs/examples "$(basename "$deck")"; done
for deck in docs/examples/*.scs; do run "$deck" docs/examples "$(basename "$deck")" --tokenizer spectre; done

# Inline blocks; a block that is one `--8<--` include is an example run above.
while IFS= read -r page; do
  rm -f "$work"/block.*
  awk -v out="$work/block" '
    /^```spice/ { n++; f = out "." n; inb = 1; next }
    inb && /^```/ { inb = 0; next }
    inb { print > f }' "$page"
  for block in "$work"/block.*; do
    [ -e "$block" ] || continue
    grep -q -- '--8<--' "$block" && continue
    cp "$block" "$work/inline.sp"
    run "$page block ${block##*.}" "$work" inline.sp
  done
done < <(find docs -name '*.md' | sort)

echo "doc decks: $((total - fail)) of $total run"
[ "$fail" -eq 0 ]
