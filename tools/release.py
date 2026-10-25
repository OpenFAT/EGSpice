#!/usr/bin/env python3
"""Plan and verify release assets without building or publishing anything."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

from release_matrix import ROOT, variants


def catalogue():
    return variants(json.loads((ROOT / 'tools/release-targets.json').read_text()))


def version():
    match = re.search(r'\.version\s*=\s*"([^"]+)"', (ROOT / 'build.zig.zon').read_text())
    if not match or not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+(?:-[a-zA-Z0-9.-]+)?', match[1]):
        raise ValueError('missing or invalid package version')
    return match[1]


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plan(selected='all', tag=''):
    release_version = version()
    if tag and tag != 'v' + release_version:
        raise ValueError(f'tag {tag!r} does not match build.zig.zon v{release_version}')
    rows = catalogue()
    if selected != 'all':
        if tag:
            raise ValueError('tag releases require the entire matrix')
        rows = [row for row in rows if row['name'] == selected]
        if not rows:
            raise ValueError(f'unknown release variant: {selected}')
    return release_version, rows


def metadata(name, release_version, runnable):
    row = next((row for row in catalogue() if row['name'] == name), None)
    if row is None or release_version != version():
        raise ValueError('variant or version does not match the checkout')
    return {**row, 'version': release_version, 'revision': revision(),
            'flake_lock_sha256': digest(ROOT / 'flake.lock'),
            'cpu_smoke': 'passed' if runnable else 'unavailable-on-runner',
            'gpu_runtime_tested': False}


def distro_assets(release_version, row):
    """The .deb, .rpm and Arch package release.yml builds for a `distro` row."""
    if not row['distro']:
        return []
    stem = f'espice-{release_version}-linux-{row["deb_arch"]}'
    return [stem + '.deb', stem + '.rpm', stem + '.pkg.tar.zst']


def pkgbuild(directory, release_version):
    """packaging/PKGBUILD.in with each .deb's checksum; SKIP where the .deb is absent."""
    text = (ROOT / 'packaging/PKGBUILD.in').read_text().replace('@VERSION@', release_version)
    for arch in ('amd64', 'arm64'):
        deb = directory / f'espice-{release_version}-linux-{arch}.deb'
        text = text.replace(f'@SHA256_{arch.upper()}@', digest(deb) if deb.exists() else 'SKIP')
    return text


# Formula placeholder -> the CPU variant whose tarball it installs.
HOMEBREW = {'MACOS_ARM64': 'macos-arm64-cpu', 'MACOS_X86_64': 'macos-x86_64-cpu',
            'LINUX_AARCH64': 'linux-aarch64-cpu', 'LINUX_X86_64': 'linux-x86_64-v1-cpu'}


def homebrew(directory, release_version):
    """packaging/homebrew/espice.rb.in with each tarball's checksum and the pinned Zig."""
    zig = re.search(r'minimum_zig_version\s*=\s*"([^"]+)"', (ROOT / 'build.zig.zon').read_text())[1]
    text = (ROOT / 'packaging/homebrew/espice.rb.in').read_text()
    text = text.replace('@VERSION@', release_version).replace('@ZIG@', zig)
    for key, variant in HOMEBREW.items():
        text = text.replace(f'@SHA256_{key}@', digest(directory / f'espice-{release_version}-{variant}.tar.gz'))
    return text


def verify(directory, release_version):
    if release_version != version():
        raise ValueError('release version differs from checkout')
    expected = []
    builds = []
    for row in catalogue():
        stem = f'espice-{release_version}-{row["name"]}'
        archive = directory / (stem + '.tar.gz')
        expected.append(archive)
        if row['bundle']:
            expected.append(directory / (stem + '.run'))
        expected += [directory / name for name in distro_assets(release_version, row)]
        with tarfile.open(archive, 'r:gz') as tar:
            stream = tar.extractfile(stem + '/BUILD.json')
            if stream is None:
                raise ValueError(f'{archive.name}: missing BUILD.json')
            build = json.load(stream)
        want = metadata(row['name'], release_version, build.get('cpu_smoke') == 'passed')
        if build != want:
            raise ValueError(f'{archive.name}: build metadata differs from the release plan')
        builds.append(build)
    actual = {path for pattern in ('*.run', '*.tar.gz', '*.deb', '*.rpm', '*.pkg.tar.zst')
              for path in directory.glob(pattern)}
    if actual != set(expected):
        raise ValueError('release assets do not exactly match the complete matrix')
    assets = []
    for path in sorted(expected):
        size = path.stat().st_size
        if not 0 < size < 2 * 1024**3:
            raise ValueError(f'{path.name}: empty or exceeds GitHub\'s 2 GiB asset limit')
        assets.append({'name': path.name, 'bytes': size, 'sha256': digest(path)})
    matrix = {'version': release_version, 'revision': revision(), 'builds': builds, 'assets': assets}
    manifest = directory / 'BUILD-MATRIX.json'
    manifest.write_text(json.dumps(matrix, indent=2) + '\n')
    checksums = [f'{asset["sha256"]}  {asset["name"]}\n' for asset in assets]
    checksums.append(f'{digest(manifest)}  {manifest.name}\n')
    if any(row['distro'] for row in catalogue()):
        (directory / 'PKGBUILD').write_text(pkgbuild(directory, release_version))
        checksums.append(f'{digest(directory / "PKGBUILD")}  PKGBUILD\n')
    if {row['name'] for row in catalogue()} >= set(HOMEBREW.values()):
        (directory / 'espice.rb').write_text(homebrew(directory, release_version))
        checksums.append(f'{digest(directory / "espice.rb")}  espice.rb\n')
    (directory / 'SHA256SUMS').write_text(''.join(checksums))
    print(f'Validated {len(builds)} variants and {len(assets)} assets')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('plan')
    meta = commands.add_parser('metadata')
    meta.add_argument('variant')
    meta.add_argument('version')
    meta.add_argument('runnable', choices=['true', 'false'])
    check = commands.add_parser('verify')
    check.add_argument('directory', type=Path)
    check.add_argument('version')
    arch = commands.add_parser('pkgbuild')
    arch.add_argument('directory', type=Path)
    arch.add_argument('version')
    args = parser.parse_args()
    if args.command == 'plan':
        release_version, rows = plan(os.environ.get('VARIANT', 'all'), os.environ.get('RELEASE_TAG', ''))
        print('version=' + release_version)
        print('matrix=' + json.dumps({'include': rows}, separators=(',', ':')))
    elif args.command == 'metadata':
        print(json.dumps(metadata(args.variant, args.version, args.runnable == 'true'), indent=2))
    elif args.command == 'pkgbuild':
        print(pkgbuild(args.directory, args.version), end='')
    else:
        verify(args.directory, args.version)


if __name__ == '__main__':
    main()
