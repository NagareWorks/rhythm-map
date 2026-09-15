"""Authored reference/diagnostic controls; no corpus or model download."""
import io
import json
import unittest
import wave

import numpy as np

import brid_reference as adapter


class BridReferenceTests(unittest.TestCase):
    def make(self, beats=(.1, .6, 1.1), positions=(1, 2, 1), frames=101, duration=2., support=None):
        return adapter.reference_arrays(beats, positions, frames, duration, support)

    def test_constant_native_period_and_half_open_support(self):
        a = self.make()
        self.assertEqual(np.flatnonzero(a['valid']).tolist(), list(range(5, 55)))
        self.assertEqual(int(a['cell_valid'].sum()), 49)
        np.testing.assert_allclose(adapter.tempo_target(a)[a['cell_valid']], -1, atol=1e-13)

    def test_unknown_is_nan_not_zero_loss(self):
        a = self.make()
        self.assertTrue(np.isnan(a['reference'][~a['valid']]).all())
        self.assertTrue(np.isnan(adapter.tempo_target(a)[~a['cell_valid']]).all())

    def test_boundary_cell_uses_integrated_rate_not_midpoint_interval(self):
        a = self.make((.1, .51, .8))
        rate = (0.01/.41 + .01/.29)*50
        self.assertAlmostEqual(adapter.tempo_target(a)[25], -np.log2(rate), places=12)

    def test_support_hole_cannot_be_bridged(self):
        support = np.ones(101, bool)
        support[20:25] = False
        a = self.make(support=support)
        self.assertFalse(a['cell_valid'][19:25].any())
        self.assertTrue(a['cell_valid'][18] and a['cell_valid'][25])

    def test_first_position_two_is_preserved_not_rephased(self):
        a = self.make(positions=(2, 1, 2))
        self.assertEqual(a['positions'].tolist(), [2, 1, 2])
        self.assertEqual(a['reference'][5], 0.)  # arbitrary count origin, not downbeat phase

    def test_frame_extent_masks_audio_end(self):
        a = self.make((0., .1), (1, 2), duration=.15)
        self.assertFalse(a['valid'][5:].any())

    def test_bad_beats_fail(self):
        for beats in ((-.1, .6, 1.1), (.1, .1, 1.1), (.6, .1, 1.1), (.1, np.nan, 1.1)):
            with self.subTest(beats=beats), self.assertRaises(ValueError):
                self.make(beats)

    def test_beat_at_audio_end_fails(self):
        with self.assertRaisesRegex(ValueError, 'outside audio'):
            self.make(duration=1.1)

    def test_bad_positions_and_repeats_fail_not_repair(self):
        for positions in ((1, 1, 2), (1, 3, 1), (1., 2., 1.), (True, False, True), (1, 2)):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                self.make(positions=positions)

    def test_bad_frame_count_fails(self):
        for frames in (True, 1, 4097, 10.):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                self.make(frames=frames)

    def test_bad_support_fails(self):
        for support in (np.ones(100, bool), np.ones(101)):
            with self.assertRaises(ValueError):
                self.make(support=support)

    def test_no_cells_fails(self):
        with self.assertRaisesRegex(ValueError, 'no supported'):
            self.make(support=np.zeros(101, bool))

    def test_manifest_hashes_nan_arrays_and_masks(self):
        a = self.make()
        before = adapter.array_manifest(a)
        a['valid'][0] = True
        self.assertNotEqual(before['valid'], adapter.array_manifest(a)['valid'])
        self.assertEqual(before['reference']['dtype'], 'float64')
        self.assertNotIn('log2_period', before)  # transcendental result is not a byte identity

    def test_fixed_pins_and_membership(self):
        inventory, assets, audit = adapter.load_sources()
        self.assertEqual(len(assets), 279)
        self.assertEqual(list(audit), list(adapter.IDS))
        self.assertEqual(len(inventory['records']), 93)

    def test_diagnostic_never_applies_shift(self):
        rate = 44100
        pcm = np.zeros((rate*3, 2), '<i2')
        for beat in (.5, 1., 1.5, 2.):
            start = round((beat+.025)*rate)
            pcm[start:start+220] = 10000
        payload = io.BytesIO()
        with wave.open(payload, 'wb') as audio:
            audio.setparams((2, 2, rate, len(pcm), 'NONE', 'not compressed'))
            audio.writeframes(pcm.tobytes())
        result = adapter.alignment_review(payload.getvalue(), [.5, 1., 1.5, 2.])
        self.assertEqual(result['common_beats'], 4)
        self.assertGreater(result['mean_rise_best_offset_s'], 0)
        self.assertEqual(result['applied_shift_seconds'], 0)
        self.assertFalse(result['musical_alignment_certified'])

    def test_retained_reference_has_complete_coverage_and_current_source(self):
        report = json.loads((adapter.ROOT/'evaluation/datasets/brid-reference-v1.json').read_bytes())
        self.assertEqual(report['source_sha256'], adapter.source_hashes())
        self.assertEqual([r['recording_id'] for r in report['records']], list(adapter.IDS))
        self.assertEqual(report['reference_frames'], 126363)
        self.assertEqual(report['tempo_cells'], 126270)
        self.assertEqual(report['supported_cell_seconds'], 2525.4)
        self.assertFalse(report['training_ready'])
        self.assertEqual(report['timestamp_shifts_applied'], 0)
        row = next(r for r in report['records'] if r['recording_id'] == '0092')
        for key in ('mean_rise_best_offset_s', 'equal_beat_best_offset_s',
                    'first_half_best_offset_s', 'second_half_best_offset_s'):
            self.assertEqual(row['alignment_review'][key], .025)


if __name__ == '__main__':
    unittest.main()
