#!/usr/bin/env python3
"""Regenerates sources.json, the release binaries the flake's versioned
packages install (`nix run '.#"0.3.0"'`, `nix run .#latest`), from the
GitHub releases. After VerA's tools/update-sources.py.

Shape:

    { "<version>": { "date": "YYYY-MM-DD", "zig": "<the Zig .hdl builds with>",
                     "<nix system>": { "url": ..., "sha256": <hex> }, ... } }

The hashes are the release's own SHA256SUMS, so a tarball that does not match
what release.yml uploaded fails the Nix fetch. `zig` is the tag's build.zig.zon
`minimum_zig_version`: espice refuses any other Zig for runtime `.hdl`.

Usage: tools/update-sources.py [OUT]   (default: sources.json beside tools/)
GITHUB_TOKEN, when set, lifts the API's anonymous rate limit.
"""

import json
import os
import re
import sys
import urllib.request

REPO = "OmarSiwy/ESPice"
# Nix system -> the CPU-only release variant built for that system's baseline.
SYSTEMS = {
    "x86_64-linux": "linux-x86_64-v1-cpu",
    "aarch64-linux": "linux-aarch64-cpu",
    "x86_64-darwin": "macos-x86_64-cpu",
    "aarch64-darwin": "macos-arm64-cpu",
}


def get(url):
    req = urllib.request.Request(url)
    if "api.github.com" in url and os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["GITHUB_TOKEN"])
    with urllib.request.urlopen(req) as r:
        return r.read().decode()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "sources.json")
    sources = {}
    for rel in json.loads(get(f"https://api.github.com/repos/{REPO}/releases?per_page=100")):
        tag = rel["tag_name"]
        if rel["draft"] or rel["prerelease"] or not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
            continue
        assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
        if "SHA256SUMS" not in assets:
            continue
        sums = dict(reversed(line.split()) for line in get(assets["SHA256SUMS"]).splitlines() if line.strip())
        zon = get(f"https://raw.githubusercontent.com/{REPO}/{tag}/build.zig.zon")
        entry = {
            "date": rel["published_at"][:10],
            "zig": re.search(r'minimum_zig_version\s*=\s*"([^"]+)"', zon).group(1),
        }
        for system, variant in SYSTEMS.items():
            name = f"espice-{tag[1:]}-{variant}.tar.gz"
            if name in assets and name in sums:
                entry[system] = {"url": assets[name], "sha256": sums[name]}
        sources[tag[1:]] = entry
    if not sources:
        # No release yet: the flake then has no versioned package.
        print("update-sources: no release with SHA256SUMS", file=sys.stderr)
    with open(out, "w") as f:
        json.dump(sources, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
