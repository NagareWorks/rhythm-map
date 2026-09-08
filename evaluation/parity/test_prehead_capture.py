"""Model-free contract, geometry, identity and private-archive controls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import prehead_capture as capture
import prehead_capture_audit as audit


def chunks_with_identity(frames):
    chunks = []
    for owner, row in enumerate(capture.layout(frames)):
        # Every channel identifies its original chunk and local coordinate.
        scalar = (owner * 10000 + np.arange(row['size'])).astype(np.float32)
        chunks.append(dict(hidden=np.repeat(scalar[:, None], 512, axis=1),
                           beat=scalar.copy(), downbeat=-scalar))
    return chunks


class PreheadCaptureControls(unittest.TestCase):
    def test_frozen_success_report_tracks_sources_and_every_comparison(self):
        report = audit.read_verified(audit.HERE / 'prehead-capture-v1.json',
            'f2102630d9e5b36d14cea43ea8a0121df2dfabf5c2ccfd57b7df0b8cdaf4d90a')
        self.assertEqual(report['lock_sha256'], audit.sha(audit.LOCK_PATH.read_bytes()))
        for name, digest in report['source_sha256'].items():
            self.assertEqual(digest, audit.sha((audit.HERE / name).read_bytes()))
        self.assertEqual(len(report['authored']), 4)
        self.assertEqual([r['case_id'] for r in report['references']], list(audit.CASE_IDS))
        self.assertTrue(report['complete'] and report['model_state_unchanged'])
        self.assertEqual(report['hook_failure_control'], dict(injected_exception_observed=True, hook_removed=True))
        self.assertEqual(report['decision'], 'capture_fidelity_only_no_discrimination_or_promotion')
        self.assertFalse(any(report[k] for k in ('training', 'holdout_access', 'production_changed', 'full_cohort_capture')))
        rows = report['authored'] + report['references']
        self.assertEqual(sum(r['chunk_count'] for r in rows), 10)
        for row in rows:
            self.assertEqual(row['hidden_shape'], [row['frames'], 512])
            self.assertEqual(sum(row['owner_frame_counts']), row['frames'])
            self.assertTrue(row['private_archive']['roundtrip_bit_exact'])
            self.assertTrue(all(c['passed'] for c in row['checks'].values()))
            self.assertEqual(len(row['chunk_checks']), row['chunk_count'])
            for chunk in row['chunk_checks']:
                self.assertEqual(len(chunk['checks']), 4)
                self.assertTrue(all(c == dict(passed=True, bit_exact=True, max_abs=0.0)
                                    for c in chunk['checks'].values()))

    def test_public_report_has_hashes_not_feature_arrays_or_machine_paths(self):
        report = json.loads((audit.HERE / 'prehead-capture-v1.json').read_bytes())
        for row in report['authored'] + report['references']:
            self.assertEqual(set(row['array_sha256']), {'hidden', 'beat', 'downbeat', 'owner'})
            for value in row['array_sha256'].values():
                self.assertRegex(value, r'^[0-9a-f]{64}$')
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {'mono_samples', 'mel_values', 'time_s',
                                               'beat_logits', 'downbeat_logits', 'upstream_beats'})
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, str):
                self.assertFalse(value.startswith(('C:/', 'D:/', 'C:\\', 'D:\\', '/home/', '/Users/')))
        check(report)

    def test_frozen_reference_population_and_budgets(self):
        lock, reference, baseline = audit.verified_plan()
        self.assertEqual([c['case_id'] for c in baseline['cases']], list(audit.CASE_IDS))
        self.assertEqual(reference['checkpoint']['name'], 'final0')
        self.assertEqual(lock['budgets']['tap_vs_untapped'], 'bit_exact')
        self.assertEqual(lock['versions']['numpy'], '2.2.6')

    def test_modified_tap_population_or_budget_is_rejected(self):
        lock = json.loads(audit.LOCK_PATH.read_bytes())
        for key, value in [('feature_width', 1024), ('threads', 4), ('ownership', 'keep_last'),
                           ('feature', 'all_layers'), ('authored_controls', []),
                           ('budgets', dict(lock['budgets'], head_reconstruction_atol=0.1))]:
            with self.subTest(key=key), patch.object(audit, 'LOCK_PATH',
                    Mock(read_bytes=lambda: json.dumps(dict(lock, **{key: value})).encode())):
                with self.assertRaises(ValueError):
                    audit.verified_plan()

    def test_byte_hash_rejected_before_deserialization(self):
        with patch.object(audit, 'Path') as path, patch.object(audit.json, 'loads') as loads:
            path.return_value.read_bytes.return_value = b'not json'
            with self.assertRaisesRegex(ValueError, 'byte identity'):
                audit.read_verified('unused', '0' * 64)
            loads.assert_not_called()

    def test_short_padding_and_exact_boundary_geometry(self):
        for n in (1, 17, 1487, 1488, 1489, 1500, 1751, 2976, 2977, 2980, 4096):
            with self.subTest(frames=n):
                mel = audit.authored_mel(n, 'modular')
                original = mel.copy()
                parts = capture.split(mel)
                for row, part in zip(capture.layout(n), parts):
                    self.assertEqual(part.shape, (row['size'], 128))
                    for local in range(len(part)):
                        global_frame = row['start'] + local
                        expected = original[global_frame] if 0 <= global_frame < n else np.zeros(128)
                        np.testing.assert_array_equal(part[local], expected)
                np.testing.assert_array_equal(mel, original)
        self.assertEqual([r['start'] for r in capture.layout(1488)], [-6])
        self.assertEqual([r['start'] for r in capture.layout(1489)], [-6, -5])
        self.assertEqual([r['start'] for r in capture.layout(1751)], [-6, 257])
        self.assertEqual([r['start'] for r in capture.layout(2980)], [-6, 1482, 1486])

    def test_all_channels_share_first_valid_owner_without_border_leakage(self):
        for n in (1, 17, 1488, 1489, 1751, 2980, 4096):
            result = capture.stitch(n, chunks_with_identity(n))
            rows = capture.layout(n)
            for t in range(n):
                owner = next(i for i, r in enumerate(rows) if r['write_start'] <= t < r['write_end'])
                local = t - rows[owner]['start']
                value = owner * 10000 + local
                self.assertEqual(result['owner'][t], owner)
                self.assertEqual(result['beat'][t], value)
                self.assertEqual(result['downbeat'][t], -value)
                self.assertTrue(np.all(result['hidden'][t] == value))

    def test_large_tail_overlap_keeps_earlier_context(self):
        result = capture.stitch(1751, chunks_with_identity(1751))
        self.assertTrue(np.all(result['owner'][:1488] == 0))
        self.assertTrue(np.all(result['owner'][1488:] == 1))
        # The late window also covers this point; central-owner selection is forbidden.
        self.assertEqual(result['beat'][1200], 1206)

    def test_missing_extra_wrong_shape_and_nonfinite_chunks_fail(self):
        for mode in ('missing', 'extra', 'shape', 'dtype', 'nonfinite', 'field'):
            chunks = chunks_with_identity(1751)
            if mode == 'missing':
                chunks.pop()
            elif mode == 'extra':
                chunks.append(chunks[0])
            elif mode == 'shape':
                chunks[0]['hidden'] = chunks[0]['hidden'].T
            elif mode == 'dtype':
                chunks[0]['beat'] = chunks[0]['beat'].astype(np.float64)
            elif mode == 'nonfinite':
                chunks[0]['hidden'][0, 0] = np.nan
            else:
                chunks[0]['confidence'] = chunks[0]['beat']
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                capture.stitch(1751, chunks)

    def test_invalid_frames_and_mel_fail_without_padding_into_availability(self):
        for n in (0, -1, True, 1.0, 4097):
            with self.assertRaises(ValueError):
                capture.layout(n)
        for mel in (np.zeros((2, 127), np.float32), np.zeros((2, 128), np.float64),
                    np.zeros((1, 2, 128), np.float32), np.full((2, 128), np.nan, np.float32)):
            with self.assertRaises(ValueError):
                capture.split(mel)

    def test_authored_inputs_are_fixed_and_not_random_music(self):
        self.assertFalse(audit.authored_mel(17, 'zero').any())
        mel = audit.authored_mel(17, 'modular')
        self.assertEqual(mel.dtype, np.float32)
        self.assertEqual(mel[3, 7], ((17 * 3 + 29 * 7) % 257) / 64)
        np.testing.assert_array_equal(mel, audit.authored_mel(17, 'modular'))
        with self.assertRaises(ValueError):
            audit.authored_mel(17, 'random')

    def test_numerical_budget_is_separate_from_bit_identity(self):
        a = np.array([0, 1], np.float32)
        b = np.array([0, 1.00001], np.float32)
        self.assertFalse(capture.difference(a, b, atol=2e-5)['bit_exact'])
        with self.assertRaises(ValueError):
            capture.difference(a, b, exact=True)
        with self.assertRaises(ValueError):
            capture.difference(a, b, atol=1e-7)
        with self.assertRaises(ValueError):
            capture.difference(np.array([0.0], np.float32), np.array([-0.0], np.float32), exact=True)

    def test_comparison_rejects_broadcast_nonfinite_dtype_and_invalid_budget(self):
        for a, b in [([1], [[1]]), ([1.0], [float('nan')]), ([float('inf')], [float('inf')]),
                     (np.ones(1, np.float32), np.ones(1, np.float64)), ([], [])]:
            with self.assertRaises(ValueError):
                capture.difference(a, b)
        for options in (dict(atol=-1), dict(rtol=float('inf'))):
            with self.assertRaises(ValueError):
                capture.difference([1.], [1.], **options)

    def test_event_identity_is_absolute_ordered_and_not_a_matching_search(self):
        self.assertTrue(capture.event_identity([1., 2.01], [1., np.float32(2.01)])['passed'])
        self.assertEqual(capture.event_identity([], [])['count'], 0)
        for a, b in [([1000.0], [1000.01]), ([1.], [1.02]), ([1., 2.], [2., 1.]),
                     ([1., 1.], [1., 1.]), ([1.], []), ([-1.], [-1.]),
                     ([np.nan], [np.nan]), ([1.], [[1.]])]:
            with self.assertRaises(ValueError):
                capture.event_identity(a, b)
        with self.assertRaises(ValueError):
            capture.event_identity([1.004999], [1.005001])

    def test_private_archive_is_exclusive_exact_and_pickle_free(self):
        arrays = capture.stitch(17, chunks_with_identity(17))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'capture.npz'
            report = capture.save_capture(path, arrays)
            self.assertTrue(report['roundtrip_bit_exact'])
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                capture.save_capture(path, arrays)
            self.assertEqual(path.read_bytes(), before)
            with np.load(path, allow_pickle=False) as values:
                self.assertEqual(set(values.files), {'hidden', 'beat', 'downbeat', 'owner'})

    def test_archive_rejects_wrong_owner_before_creating_a_file(self):
        arrays = capture.stitch(17, chunks_with_identity(17))
        arrays['owner'][5] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.npz'
            with self.assertRaises(ValueError):
                capture.save_capture(path, arrays)
            self.assertFalse(path.exists())

    def test_private_output_cannot_overwrite_or_enter_repository(self):
        with self.assertRaises(ValueError):
            audit.private_directory(audit.ROOT / 'uncreated-prehead-output')
        with self.assertRaises(ValueError):
            audit.private_directory(audit.ROOT)


if __name__ == '__main__':
    unittest.main()
