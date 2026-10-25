# Contributing

**Status: maintained** by one person. Issues are read, and nothing closes them on a timer: an open issue stays open until it is fixed or answered.

Open to contributors, but please dont push slop.

## Branches

`main` is the only long-lived branch, and it stays green. Every change lands through a pull request whose merge result passed the required checks (`test`, `cross`, `nix`). Work happens on a topic branch that is deleted after its merge. There is no `dev` branch.

## Which ESPice to follow

- **Pinned**: a commit and hash in your `build.zig.zon` (`zig fetch --save`). Nothing on `main` can move it.
- **Released**: a `v*` tag's archives with `SHA256SUMS` and `BUILD-MATRIX.json`, built and smoked by `release.yml`.
- **Head**: `nix run github:OmarSiwy/ESPice`. That is `main` at that moment: gated, but not a release.

Versions are 0.y.z: a minor bump may break, a patch may not. ESPice pins VerA and Gompute by commit and hash, and Zig through `flake.lock`. The Zig an `.hdl` deck compiles with at run time is the release's Zig (`espice --version`).

## A change

1. Keep `nix develop .#ci -c zig build check --watch -fincremental` running while you edit.
2. A fix comes with the deck or test that reproduces it, and with the oracle that would have caught its whole class (see [TESTING.md](TESTING.md)). New or fixed decks go to the [VAMS Conformance Suite](https://github.com/OmarSiwy/VAMS-Conformance-Suite), then a submodule bump.
3. `nix develop .#ci -c zig build test -Doptimize=ReleaseSafe -Dgpu=false` passes locally. A change to a GPU path also passes `zig build test -- --backend cuda --jobs 1` on a card.
4. Add one line to `CHANGELOGS.md`, citing the issue. Measure every number in it; never type one from memory.

## Reporting a bug

Include the deck that triggers it, the exact command, what `espice` printed, and `espice --version`. A minimal deck gets fixed first. For a wrong number, also give the reference result (ngspice, Xyce, VACASK or a derivation) and its version.
