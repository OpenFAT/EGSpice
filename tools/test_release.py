import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import release
from release_matrix import variants


class ReleaseTests(unittest.TestCase):
    def test_gpu_backends_are_exclusive_and_platforms_are_explicit(self):
        rows = release.catalogue()
        self.assertEqual(len(rows), len({row['name'] for row in rows}))
        self.assertLessEqual(len(rows), 256)
        for row in rows:
            self.assertNotIn('windows', row['system'])
            self.assertFalse(row['cuda'] != 'none' and row['hip'] != 'none')
            if row['system'].endswith('-darwin'):
                self.assertFalse(row['gpu'])
                self.assertFalse(row['bundle'])
            if row['hip'] != 'none':
                self.assertEqual(row['system'], 'x86_64-linux')
        for system in {row['system'] for row in rows}:
            self.assertTrue(any(row['system'] == system and not row['gpu'] for row in rows))

    def test_refuses_mixed_backend_and_duplicate_artifacts(self):
        data = json.loads((release.ROOT / 'tools/release-targets.json').read_text())
        mixed = copy.deepcopy(data)
        mixed['backends'][0].update(cuda='sm_75', hip='gfx1100')
        with self.assertRaises(ValueError):
            variants(mixed)
        data['hosts'].append(data['hosts'][0])
        with self.assertRaises(ValueError):
            variants(data)

    def test_version_and_selection_fail_closed(self):
        with self.assertRaises(ValueError):
            release.plan(tag='v0.0.0')
        with self.assertRaises(ValueError):
            release.plan(selected='does-not-exist')
        with self.assertRaises(ValueError):
            release.plan(selected=release.catalogue()[0]['name'], tag='v' + release.version())
        _, rows = release.plan(selected=release.catalogue()[0]['name'])
        self.assertEqual(len(rows), 1)

    def test_manifest_preserves_unavailable_runtime_coverage(self):
        row = release.catalogue()[0]
        meta = release.metadata(row['name'], release.version(), False)
        self.assertEqual(meta['cpu_smoke'], 'unavailable-on-runner')
        self.assertIs(meta['gpu_runtime_tested'], False)

    def make_assets(self, directory, row, alter=None):
        stem = f'espice-{release.version()}-{row["name"]}'
        data = release.metadata(row['name'], release.version(), True)
        if alter:
            data.update(alter)
        payload = json.dumps(data).encode()
        with tarfile.open(directory / (stem + '.tar.gz'), 'w:gz') as tar:
            info = tarfile.TarInfo(stem + '/BUILD.json')
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        bundle = directory / (stem + '.run')
        bundle.write_bytes(b'example bundle')
        return bundle

    def test_complete_assets_have_verifiable_checksums(self):
        row = release.catalogue()[0]
        with tempfile.TemporaryDirectory() as temp, patch.object(release, 'catalogue', return_value=[row]):
            directory = Path(temp)
            self.make_assets(directory, row)
            release.verify(directory, release.version())
            for line in (directory / 'SHA256SUMS').read_text().splitlines():
                checksum, name = line.split('  ')
                self.assertEqual(checksum, release.digest(directory / name))
            manifest = json.loads((directory / 'BUILD-MATRIX.json').read_text())
            self.assertEqual(len(manifest['assets']), 2)

    def test_missing_empty_extra_or_mismatched_assets_block_publication(self):
        row = release.catalogue()[0]
        for failure in ['missing', 'empty', 'extra', 'revision']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp, patch.object(release, 'catalogue', return_value=[row]):
                directory = Path(temp)
                bundle = self.make_assets(directory, row, {'revision': 'wrong'} if failure == 'revision' else None)
                if failure == 'missing':
                    bundle.unlink()
                elif failure == 'empty':
                    bundle.write_bytes(b'')
                elif failure == 'extra':
                    (directory / 'unexpected.run').write_bytes(b'wrong')
                with self.assertRaises((ValueError, FileNotFoundError)):
                    release.verify(directory, release.version())


if __name__ == '__main__':
    unittest.main()
