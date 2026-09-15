"""Authored BRID controls; CI needs no corpus, network, audio or model."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

import brid_training_inventory as inventory


def archive_bytes(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return stream.getvalue()


def entries():
    return [('Annotations/beats/[0001] M4-01-SA.beats', '0.0\t2\n0.5\t1\n1.0\t2\n'),
            ('Annotations/tempo/[0001] M4-01-SA.bpm', '120\n')]


class BridInventoryTests(unittest.TestCase):
    def test_beat_positions_need_not_start_on_downbeat(self):
        times, positions = inventory.parse_beats(b'0.0 2\n0.5 1\n')
        self.assertEqual(times, [0, .5])
        self.assertEqual(positions, [2, 1])

    def test_invalid_beat_rows_fail_closed(self):
        for value in (b'', b'0 1', b'0 1\n0 2', b'1 1\n0 2', b'-1 1\n0 2',
                      b'0 1\nnan 2', b'0 1\ninf 2', b'0 1\n1 3', b'0 1 x\n1 2',
                      b'0 1\n\n1 2', b'0 1\n1 1.0'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                inventory.parse_beats(value)

    def test_summary_and_hashes_are_derived(self):
        row, = inventory.parse_annotations(archive_bytes(entries()))
        self.assertEqual(row['beat_count'], 3)
        self.assertEqual(row['downbeat_count'], 1)
        self.assertEqual(row['annotated_span_seconds'], 1)
        self.assertEqual(row['descriptive_median_interval_bpm'], 120)
        self.assertEqual(row['annotation_assets'][0]['sha256'],
                         hashlib.sha256(entries()[0][1].encode()).hexdigest())
        self.assertEqual(row['intended_role'], 'training_candidate_only')

    def test_discontinuity_is_retained_not_repaired(self):
        changed = entries()
        changed[0] = (changed[0][0], '0 1\n0.5 1\n1 2\n')
        row, = inventory.parse_annotations(archive_bytes(changed))
        self.assertEqual(row['position_repeat_count'], 1)
        self.assertEqual(row['beat_count'], 3)

    def test_missing_pairs_and_unexpected_paths_are_rejected(self):
        cases = [entries()[:1], entries()[1:], [],
                 entries() + [('Annotations/beats/../escape.beats', '0 1\n1 2')],
                 entries() + [('Annotations/beats/[0002] M2-01-SA.bpm', '120')],
                 entries() + [('Annotations/beats/[0094] S1-PD1-01-SA.beats', '0 1\n1 2')]]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                inventory.parse_annotations(archive_bytes(case))

    def test_invalid_global_tempi_are_rejected(self):
        for bpm in ('0', '-1', 'nan', 'inf', '120 121', ''):
            changed = entries()
            changed[1] = (changed[1][0], bpm)
            with self.subTest(bpm=bpm), self.assertRaises(ValueError):
                inventory.parse_annotations(archive_bytes(changed))

    def test_duplicate_members_are_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            value = archive_bytes(entries() + entries()[:1])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            inventory.parse_annotations(value)

    def test_duplicate_ids_are_rejected(self):
        changed = entries() + [(name.replace('M4-01-SA', 'M2-01-SA'), payload)
                               for name, payload in entries()]
        with self.assertRaisesRegex(ValueError, 'duplicate recording ID'):
            inventory.parse_annotations(archive_bytes(changed))

    def test_symlink_members_are_rejected(self):
        link = zipfile.ZipInfo(entries()[0][0])
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        with self.assertRaisesRegex(ValueError, 'regular'):
            inventory.parse_annotations(archive_bytes([(link, entries()[0][1]), entries()[1]]))

    def test_locked_bytes_reject_truncation_extension_and_mutation(self):
        data = b'authored metadata'
        lock = dict(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                    md5=hashlib.md5(data).hexdigest())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metadata.bin'
            path.write_bytes(data)
            self.assertEqual(inventory.read_locked(path, lock), data)
            for value in (data[:-1], data + b'x', b'X' + data[1:]):
                path.write_bytes(value)
                with self.assertRaises(ValueError):
                    inventory.read_locked(path, lock)

    def test_small_fixture_cannot_be_reported_as_full_corpus(self):
        with self.assertRaisesRegex(ValueError, 'exact 93'):
            inventory.build_report(archive_bytes(entries()), b'')

    def test_directory_does_not_silently_accept_missing_audio(self):
        rows = inventory.parse_annotations(archive_bytes(entries()))
        with self.assertRaisesRegex(ValueError, 'audio inventory'):
            inventory.attach_audio_directory(rows, archive_bytes([]))

    def test_retained_inventory_has_exact_counts_and_no_admission(self):
        path = Path(__file__).resolve().parents[1] / 'datasets/brid-training-candidate-v1.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        rows = report['records']
        self.assertEqual([row['recording_id'] for row in rows], [f'{i:04d}' for i in range(1, 94)])
        self.assertEqual(report['beat_count'], sum(row['beat_count'] for row in rows))
        self.assertEqual(report['beat_count'], 4342)
        self.assertEqual(report['downbeat_count'], 2205)
        self.assertAlmostEqual(report['annotated_span_seconds'], 2527.192)
        self.assertEqual(report['style_counts'], dict(MA=3, PA=28, SA=41, SE=21))
        self.assertEqual(report['audio_without_released_beat_files'], 274)
        self.assertEqual(report['position_repeat_recordings'], [])
        self.assertIsNone(report['independent_work_count'])
        self.assertEqual({row['leakage_group'] for row in rows}, {'brid-corpus-v1'})
        for key in ('audio_payloads_acquired', 'admitted_training_recordings', 'optimizer_steps'):
            self.assertEqual(report[key], 0)
        for key in ('feature_access', 'project_holdout_access', 'public_default_changed'):
            self.assertIs(report[key], False)
        for row in rows:
            self.assertIsNone(row['audio_candidate']['sha256'])
            self.assertIs(row['audio_candidate']['payload_verified'], False)


if __name__ == '__main__':
    unittest.main()
