#!/usr/bin/env python3
"""Expand the release catalogue; stdout is a GitHub Actions include matrix."""

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def variants(catalogue):
    rows = []
    for host in catalogue["hosts"]:
        for backend in catalogue["backends"]:
            if host["system"] not in backend["systems"]:
                continue
            cuda, hip = backend["cuda"], backend["hip"]
            if cuda != "none" and hip != "none":
                raise ValueError("a release must contain only one GPU backend")
            name = f'{host["name"]}-{backend["name"]}'
            row = {**host, "name": name, "cuda": cuda, "hip": hip,
                   "gpu": cuda != "none" or hip != "none",
                   "bundle": host["system"].endswith("-linux"),
                   "package": "espice-" + name,
                   # .deb/.rpm/Arch packages: one CPU build per Linux architecture.
                   "distro": host.get("distro", False) and not (cuda != "none" or hip != "none"),
                   "deb_arch": {"x86_64-linux": "amd64", "aarch64-linux": "arm64"}.get(host["system"], "")}
            if not re.fullmatch(r"[a-z0-9_-]+", name):
                raise ValueError(f"invalid artifact name: {name}")
            rows.append(row)
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate release artifact names")
    if not 0 < len(rows) <= 256:
        raise ValueError("release matrix must fit GitHub's 256-job limit")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="all", help="exact variant name, or all")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    rows = variants(json.loads((ROOT / "tools/release-targets.json").read_text()))
    if args.variant != "all":
        rows = [row for row in rows if row["name"] == args.variant]
        if not rows:
            parser.error(f"unknown variant: {args.variant}")
    if args.list:
        print("\n".join(row["name"] for row in rows))
    else:
        print(json.dumps({"include": rows}, separators=(",", ":")))


if __name__ == "__main__":
    main()
