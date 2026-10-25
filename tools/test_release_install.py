"""tools/install.sh against a sources.json written as update-sources.py writes it."""

import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CPU = {'x86_64': 'x86_64', 'amd64': 'x86_64', 'aarch64': 'aarch64', 'arm64': 'aarch64'}[platform.machine().lower()]
SYSTEM = f'{CPU}-{"darwin" if platform.system() == "Darwin" else "linux"}'


def release_tarball(path, version):
    """A release-shaped tree whose bin/espice prints its version and finds share/espice beside it."""
    stem = f'espice-{version}-test'
    script = (b'#!/bin/sh\nd=$(dirname "$(readlink -f "$0")")\n'
              b'test -f "$d/../share/espice/device/eval.zig" || exit 3\n'
              b'echo "espice ' + version.encode() + b'"\n')
    with tarfile.open(path, 'w:gz') as tar:
        for name, data, mode in [(f'{stem}/bin/espice', script, 0o755),
                                 (f'{stem}/share/espice/device/eval.zig', b'', 0o644)]:
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InstallScriptTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.env = {**os.environ, 'HOME': str(self.dir / 'home'),
                    'ESPICE_INSTALL_SOURCES': str(self.dir / 'sources.json'),
                    'ESPICE_INSTALL_HOME': str(self.dir / 'versions'),
                    'ESPICE_INSTALL_BIN': str(self.dir / 'bin'),
                    'ESPICE_INSTALL_CACHE': str(self.dir / 'cache')}
        sources = {}
        for version in ['0.9.0', '0.10.0']:
            tarball = self.dir / f'espice-{version}.tar.gz'
            sources[version] = {'date': '2026-10-25', 'zig': '0.18.0-dev.349+d924f83bf',
                                SYSTEM: {'url': str(tarball), 'sha256': release_tarball(tarball, version)}}
        (self.dir / 'sources.json').write_text(json.dumps(sources, indent=2, sort_keys=True) + '\n')

    def install(self, *args):
        return subprocess.run(['sh', str(ROOT / 'tools/install.sh'), *args], env=self.env,
                              capture_output=True, text=True)

    def espice(self):
        return subprocess.run([str(self.dir / 'bin/espice')], capture_output=True, text=True).stdout.strip()

    def test_newest_by_version_order_then_cached_then_pinned(self):
        first = self.install()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn('downloading 0.10.0', first.stdout)  # 0.10 sorts after 0.9
        self.assertIn('Zig 0.18.0-dev.349+d924f83bf', first.stdout)
        self.assertEqual(self.espice(), 'espice 0.10.0')
        again = self.install()
        self.assertIn('0.10.0 for', again.stdout)
        self.assertIn('is cached', again.stdout)
        pinned = self.install('v0.9.0')
        self.assertEqual(pinned.returncode, 0, pinned.stderr)
        self.assertEqual(self.espice(), 'espice 0.9.0')
        self.assertTrue((self.dir / 'versions/0.10.0/bin/espice').exists())

    def test_tampered_tarball_is_refused_and_not_installed(self):
        tarball = self.dir / 'espice-0.10.0.tar.gz'
        release_tarball(tarball, '6.6.6')
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('sha256', result.stderr)
        self.assertFalse((self.dir / 'bin/espice').exists())
        self.assertFalse((self.dir / 'versions/0.10.0').exists())

    def test_unknown_version_fails(self):
        result = self.install('1.2.3')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('has no', result.stderr)


if __name__ == '__main__':
    unittest.main()
