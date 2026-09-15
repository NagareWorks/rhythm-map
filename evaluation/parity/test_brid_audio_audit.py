"""Authored PCM and integrity controls; no corpus, network or neural inference."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np

import brid_audio_audit as audit


def wav_bytes(samples=None, rate=1000, width=2):
    if samples is None:
        samples = np.ones((1000, 2), dtype='<i2') * 1000
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as audio:
        audio.setnchannels(samples.shape[1])
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(samples.tobytes())
    return stream.getvalue()


class BridAudioAuditTests(unittest.TestCase):
    def test_pcm_identity_duration_channels_and_coverage(self):
        samples = np.ones((1000, 2), dtype='<i2') * 1000
        report = audit.analyse_wav(wav_bytes(samples), [.1, .6])
        self.assertEqual(report['pcm_sha256'], hashlib.sha256(samples.tobytes()).hexdigest())
        self.assertEqual(report['duration_seconds'], 1)
        self.assertEqual(report['channels'], 2)
        self.assertEqual(report['unannotated_leading_seconds'], .1)
        self.assertEqual(report['unannotated_trailing_seconds'], .4)
        self.assertTrue(report['structural_gate_passed'])

    def test_out_of_bounds_references_are_retained_not_clamped(self):
        report = audit.analyse_wav(wav_bytes(), [-.1, .5, 1.0, 1.1])
        self.assertEqual(report['out_of_bounds_beat_indices'], [0, 2, 3])
        self.assertEqual(report['unannotated_leading_seconds'], -.1)
        self.assertLess(report['unannotated_trailing_seconds'], 0)
        self.assertFalse(report['structural_gate_passed'])

    def test_entire_silence_blocks_structure_but_has_no_invented_lag(self):
        report = audit.analyse_wav(wav_bytes(np.zeros((1000, 2), dtype='<i2')), [.1, .6])
        self.assertEqual(report['zero_channels'], [0, 1])
        self.assertFalse(report['structural_gate_passed'])
        self.assertIn('entire_audio_is_zero', report['structural_findings'])
        self.assertIsNone(report['acoustic_diagnostics']['best_diagnostic_offset_seconds'])
        self.assertIsNone(report['acoustic_diagnostics']['nearest_rms_rise_peak_distance_ms_quantiles'])

    def test_one_silent_channel_and_clipping_are_diagnostics(self):
        samples = np.zeros((1000, 2), dtype='<i2')
        samples[0, 1], samples[1, 1] = -32768, 32767
        report = audit.analyse_wav(wav_bytes(samples), [.1, .6])
        self.assertEqual(report['zero_channels'], [0])
        self.assertEqual(report['full_scale_sample_fraction_by_channel'], [0, .002])
        self.assertEqual(report['peak_absolute_by_channel'], [0, 1])
        self.assertTrue(report['structural_gate_passed'])

    def test_encoding_and_truncation_fail_closed(self):
        for payload in (b'not a WAV', wav_bytes()[:-2],
                        wav_bytes(np.zeros((1000, 1), dtype='u1'), width=1),
                        wav_bytes(np.zeros((1000, 3), dtype='<i2'))):
            with self.subTest(length=len(payload)), self.assertRaises((ValueError, wave.Error, EOFError)):
                audit.analyse_wav(payload, [.1, .6])

    def test_empty_or_invalid_reference_sequences_fail_closed(self):
        for beats in ([], [0], [0, 0], [1, 0], [0, float('nan')], [0, float('inf')]):
            with self.subTest(beats=beats), self.assertRaises(ValueError):
                audit.analyse_wav(wav_bytes(), beats)

    def test_diagnostic_profile_never_changes_beats_or_certifies_alignment(self):
        samples = np.zeros((1003, 1), dtype='<i2')
        samples[210:220] = 16000
        samples[610:620] = 16000
        beats = [.2, .6]
        report = audit.analyse_wav(wav_bytes(samples), beats)['acoustic_diagnostics']
        self.assertEqual(beats, [.2, .6])
        self.assertEqual(len(report['offset_profile_mean_rise']), 61)
        self.assertEqual(report['discarded_tail_samples'], 3)
        self.assertGreater(report['best_diagnostic_offset_seconds'], 0)
        self.assertFalse(report['offset_is_correction'])
        self.assertFalse(report['musical_alignment_certified'])

    def test_no_common_interior_beats_has_no_profile(self):
        report = audit.analyse_wav(wav_bytes(), [0, .999])['acoustic_diagnostics']
        self.assertEqual(report['offset_profile_common_beats'], 0)
        self.assertEqual(report['offset_profile_mean_rise'], [])
        self.assertIsNone(report['best_diagnostic_offset_seconds'])

    def test_decoder_requires_same_bytes_and_one_output_sample_tolerance(self):
        asset = dict(sha256='a' * 64, size_bytes=100)
        report = dict(schema_version=1, **asset, decoded_sample_rate_hz=22050, duration_s=1)
        audit.check_decoder(report, asset, 1)
        audit.check_decoder(dict(report, duration_s=1 + .5/22050), asset, 1)
        for change in (dict(sha256='b' * 64), dict(size_bytes=99), dict(schema_version=2),
                       dict(decoded_sample_rate_hz=44100), dict(duration_s=1 + 2/22050),
                       dict(duration_s=float('nan')), dict(duration_s=0)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.check_decoder(dict(report, **change), asset, 1)

    def test_verified_asset_rejects_byte_and_size_changes(self):
        data = b'authored source'
        asset = dict(path='source.bin', size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / asset['path']).write_bytes(data)
            self.assertEqual(audit.verified_bytes(root, asset), data)
            for changed in (data[:-1], data + b'x', b'X' + data[1:]):
                (root / asset['path']).write_bytes(changed)
                with self.assertRaises(ValueError):
                    audit.verified_bytes(root, asset)

    def test_asset_paths_and_links_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'source.bin').write_bytes(b'x')
            for relative in ('../source.bin', '/source.bin', 'C:/source.bin', 'a\\b', '', 'missing'):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    audit.safe_file(root, relative)
            with mock.patch.object(Path, 'is_symlink', return_value=True):
                with self.assertRaisesRegex(ValueError, 'linked'):
                    audit.safe_file(root, 'source.bin')

    def test_frozen_evidence_is_source_only_and_precommitted(self):
        inventory, lock = audit.load_pins()
        self.assertEqual(len(inventory['records']), 93)
        self.assertEqual(len(lock['assets']), 12)
        self.assertEqual(lock['license'], 'CC-BY-4.0')
        report = json.loads((audit.ROOT / 'evaluation/datasets/brid-audio-smoke-v1.json').read_text())
        self.assertEqual(report['recording_ids'], list(audit.SMOKE))
        self.assertEqual(report['fetch_lock_sha256'], audit.LOCK_SHA)
        self.assertEqual(report['beat_count'], 209)
        self.assertEqual(report['downbeat_count'], 106)
        self.assertTrue(report['structural_gate_passed'])
        self.assertEqual(report['admitted_training_recordings'], 0)
        self.assertEqual(report['optimizer_steps'], 0)
        self.assertIsNone(report['independent_work_count'])
        self.assertEqual({row['leakage_group'] for row in report['records']}, {'brid-corpus-v1'})
        for key in ('framewise_tempo_targets_created', 'tempo_change_labels_created',
                    'feature_access', 'model_inference', 'project_holdout_access', 'public_default_changed'):
            self.assertFalse(report[key])


if __name__ == '__main__':
    unittest.main()
