"""packaging/espice-hip.sh picks the espice built for the GPU KFD reports."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['gfx90a', 'gfx942', 'gfx1030', 'gfx1100', 'gfx1151', 'gfx1201']


class HipWrapperTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / 'bin').mkdir()
        for gfx in TARGETS:
            stand_in = self.dir / 'bin' / f'espice-{gfx}'
            stand_in.write_text(f'#!/bin/sh\necho "{gfx} ZIG=$ZIG $*"\n')
            stand_in.chmod(0o755)

    def run_wrapper(self, *versions, **env):
        topology = self.dir / 'nodes'
        shutil.rmtree(topology, ignore_errors=True)
        for node, version in enumerate(versions):
            (topology / str(node)).mkdir(parents=True)
            (topology / str(node) / 'properties').write_text(
                f'cpu_cores_count 0\ngfx_target_version {version}\nmax_waves_per_simd 16\n')
        environ = {k: v for k, v in os.environ.items() if k not in ('ZIG', 'ESPICE_HIP_ARCH')}
        environ.update(ESPICE_HIP_BIN=str(self.dir / 'bin'), ESPICE_KFD_TOPOLOGY=str(topology), **env)
        return subprocess.run(['sh', str(ROOT / 'packaging/espice-hip.sh'), '--version', 'deck.sp'],
                              env=environ, capture_output=True, text=True)

    def test_kfd_version_names_the_target(self):
        # The stepping is hex in the name: 90010 is gfx90a, not gfx9010.
        for version, gfx in [(90010, 'gfx90a'), (90402, 'gfx942'), (100300, 'gfx1030'),
                             (110000, 'gfx1100'), (110501, 'gfx1151'), (120001, 'gfx1201')]:
            with self.subTest(version=version):
                result = self.run_wrapper(0, version)  # node 0 is the CPU agent
                self.assertEqual(result.stdout.split()[0], gfx)
                self.assertEqual(result.stderr, '')

    def test_arguments_and_the_bundled_zig_pass_through(self):
        result = self.run_wrapper(0, 110000)
        self.assertEqual(result.stdout.strip(), 'gfx1100 ZIG=/usr/lib/espice/zig/zig --version deck.sp')
        result = self.run_wrapper(0, 110000, ZIG='/opt/zig')
        self.assertIn('ZIG=/opt/zig ', result.stdout)

    def test_override_and_fallbacks(self):
        self.assertEqual(self.run_wrapper(0, 110000, ESPICE_HIP_ARCH='gfx942').stdout.split()[0], 'gfx942')
        unknown = self.run_wrapper(0, 100100)  # gfx1010: not in the package
        self.assertEqual(unknown.stdout.split()[0], 'gfx1100')
        self.assertIn('no build for gfx1010', unknown.stderr)
        quiet = self.run_wrapper()  # no AMD GPU: the CPU backend, no warning
        self.assertEqual(quiet.stdout.split()[0], 'gfx1100')
        self.assertEqual(quiet.stderr, '')


if __name__ == '__main__':
    unittest.main()
