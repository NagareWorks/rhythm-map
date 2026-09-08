"""Model-free checks for the separately frozen availability composition."""
from contextlib import contextmanager
from fractions import Fraction
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

import musicfm_availability as availability
import musicfm_loader as loader
import musicfm_temporal as temporal


def native_packet():
    plan, _ = availability.protocol()
    return dict(schema='rhythm-map.native-activity.v1', samples=288240, sample_rate=24000, pcm_sha256='authored',
                source_sha256={k: v['sha256'] for k, v in plan['native_sources'].items()},
                method='one-sentinel-membership-query-per-native-center-not-beat-evidence',
                pretrained_inference=False, production_change=False, silence_threshold_db=-40.0, minimum_silence_s=0.8,
                activity=[dict(time_s=i / 20, rms=0.125, relative_db=0) for i in range(241)],
                native_token_retained=[True] * 301)


class MusicFMAvailabilityControls(unittest.TestCase):
    def test_protocol_keeps_previous_failure_and_no_music_or_new_inference(self):
        plan, old = availability.protocol()
        self.assertFalse(old['discrimination']['silence_abstains'])
        self.assertEqual(old['decision'], 'close_fixed_rule_before_music')
        self.assertEqual(len(plan['new_activity_controls']), 9)
        for key in ('new_pretrained_inference', 'music_access', 'holdout_access', 'training',
                    'production_change', 'new_user_parameters'):
            self.assertFalse(plan[key])

    def test_protocol_mutation_fails_before_previous_inputs(self):
        with patch.object(Path, 'read_bytes', return_value=b'{}'), patch.object(availability.previous, 'protocol') as prior:
            with self.assertRaisesRegex(ValueError, 'availability protocol changed'):
                availability.protocol()
            prior.assert_not_called()

    def test_helper_mutation_is_refused(self):
        raw = (availability.HERE / 'musicfm-availability-lock-v1.json').read_bytes()
        with patch.object(Path, 'read_bytes', side_effect=[raw, b'changed']):
            with self.assertRaisesRegex(ValueError, 'previous helper changed'):
                availability.protocol()

    def test_native_default_and_source_drift_are_refused(self):
        plan, _ = availability.protocol()
        for key, value in (('silence_threshold_db', -39.0), ('minimum_silence_s', 0.7),
                           ('source_sha256', {}), ('pcm_sha256', 'other'), ('pretrained_inference', True)):
            packet = native_packet()
            packet[key] = value
            with self.assertRaises(ValueError):
                availability.validate_native(packet, 'authored', 288240, plan)

    def test_complete_native_envelope_and_mask(self):
        plan, _ = availability.protocol()
        self.assertEqual(availability.validate_native(native_packet(), 'authored', 288240, plan), [True] * 301)
        for key in ('activity', 'native_token_retained'):
            packet = native_packet()
            packet[key].pop()
            with self.assertRaises(ValueError):
                availability.validate_native(packet, 'authored', 288240, plan)

    def test_nonfinite_shifted_and_gapped_native_cells_fail(self):
        plan, _ = availability.protocol()
        for key, value in (('rms', -1), ('relative_db', float('nan')), ('time_s', 0.501), ('rms', True)):
            packet = native_packet()
            packet['activity'][10][key] = value
            with self.assertRaises(ValueError):
                availability.validate_native(packet, 'authored', 288240, plan)

    def test_unknown_is_distinct_from_false_and_integer_is_not_bool(self):
        plan, _ = availability.protocol()
        packet = native_packet()
        packet['native_token_retained'][0] = None
        packet['native_token_retained'][1] = False
        self.assertEqual(availability.validate_native(packet, 'authored', 288240, plan)[:2], [None, False])
        packet['native_token_retained'][0] = 1
        with self.assertRaises(ValueError):
            availability.validate_native(packet, 'authored', 288240, plan)

    def test_full_shared_support_is_kept_for_both_clock_families(self):
        for post in (Fraction(25, 4), Fraction(20)):
            result = availability.availability([True] * 301, 100, post)
            self.assertEqual(result['status'], 'available_for_periodic_test')
            self.assertEqual(result['target_frames'], 75)
            self.assertGreaterEqual(result['required_native_centers'], 75)
            self.assertFalse(result['score_support_modified'])

    def test_target_or_interpolation_endpoint_vetoes_entire_window(self):
        for index in (100, 150, 199):
            for value, status in ((False, 'low_activity'), (None, 'unknown_activity')):
                mask = [True] * 301
                mask[index] = value
                result = availability.availability(mask, 100, Fraction(20))
                self.assertEqual(result['status'], status, (index, value))
                self.assertEqual(result['target_frames'], 75)

    def test_outside_query_support_does_not_modify_gate(self):
        mask = [True] * 301
        expected = availability.availability(mask, 100, Fraction(20))
        mask[99], mask[200] = None, False
        self.assertEqual(availability.availability(mask, 100, Fraction(20)), expected)

    def test_unavailable_never_opens_or_scores_features(self):
        for status in ('low_activity', 'unknown_activity'):
            callback = Mock(side_effect=AssertionError('features must not be opened'))
            result = availability.score_if_available(dict(status=status), callback)
            self.assertIsNone(result['clocks'])
            self.assertFalse(result['feature_access'])
            callback.assert_not_called()

    def test_available_only_passes_to_existing_score_not_rhythm_truth(self):
        callback = Mock(return_value=dict(status='zero_norm_support', clocks=None))
        result = availability.score_if_available(dict(status='available_for_periodic_test'), callback)
        self.assertEqual(result['status'], 'zero_norm_support')
        callback.assert_called_once_with()
        with self.assertRaises(ValueError):
            availability.score_if_available(dict(status='known_beats'), callback)

    def test_partial_missing_and_invalid_mask_refused(self):
        for mask, start in (([True] * 100, 1), ([True] * 100, -1), ([1] * 100, 0), ([], 0)):
            with self.assertRaises(ValueError):
                availability.availability(mask, start, Fraction(20))

    def test_new_pcm_control_population_and_original_hashes(self):
        plan, old = availability.protocol()
        cases = availability.controls()
        self.assertEqual(list(cases), plan['original_cases'] + [c['id'] for c in plan['new_activity_controls']])
        for name, pcm in cases.items():
            self.assertEqual(pcm.shape, (288240,))
            self.assertEqual(pcm.dtype, np.float32)
            self.assertTrue(np.isfinite(pcm).all())
            self.assertLessEqual(np.max(np.abs(pcm)), 1)
        for case in old['cases']:
            self.assertEqual(loader.digest(cases[case['id']].tobytes()), case['pcm_sha256'])

    def test_rests_gain_and_nonrhythmic_controls_not_feature_substitutions(self):
        cases = availability.controls()
        clean = cases['constant-clean']
        np.testing.assert_array_equal(cases['quiet-gain'], clean * np.float32(2 ** -16))
        np.testing.assert_array_equal(cases['below-floor'], clean * np.float32(2 ** -60))
        self.assertFalse(np.any(cases['short-rest'][138000:150000]))
        np.testing.assert_array_equal(cases['short-rest'][:138000], clean[:138000])
        self.assertTrue(np.all(cases['dc'] == 0.125))
        again = availability.controls()
        np.testing.assert_array_equal(cases['noise'], again['noise'])

    def test_bad_archive_hash_refused_before_numpy_parser(self):
        plan, old = availability.protocol()
        with patch.object(loader, 'verified_file', side_effect=ValueError('hash mismatch')), patch.object(np, 'load') as parser:
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                availability.cached_features(Path('/private'), 'constant-clean', plan, old['cases'][0])
            parser.assert_not_called()

    def test_authored_archive_shape_hash_and_owner_validation(self):
        plan, _ = availability.protocol()
        hidden = np.zeros((301, 1024), np.float32)
        owners = np.array([0] * 150 + [1] * 100 + [2] * 51, np.int32)
        arrays = [hidden, owners]
        baseline = dict(feature_sha256=loader.digest(hidden.tobytes()), owner_sha256=loader.digest(owners.tobytes()))

        @contextmanager
        def verified(*_args):
            stream = io.BytesIO()
            np.save(stream, arrays.pop(0), allow_pickle=False)
            stream.seek(0)
            yield stream

        with patch.object(loader, 'verified_file', verified):
            actual = availability.cached_features(Path('/private'), 'constant-clean', plan, baseline)
        np.testing.assert_array_equal(actual[0], hidden)
        np.testing.assert_array_equal(actual[1], owners)

    def test_unavailable_member_preserves_failed_pair_denominator(self):
        unavailable = dict(status='low_activity', clocks=None)
        result = temporal.verdict(unavailable, dict(status='eligible', clocks={}))
        self.assertEqual(result['status'], 'pair_rejected')
        self.assertFalse(result['shape_supported'])
        self.assertFalse(result['density_resolved'])

    def test_retained_report_source_identity_and_complete_population(self):
        raw = (availability.HERE / 'musicfm-availability-v1.json').read_bytes()
        self.assertEqual(loader.digest(raw), '1c6141918bdae1836fb220bb47d676c7e11cc18419c9cc5cdc12ccf6b48d807c')
        report = json.loads(raw)
        plan, _ = availability.protocol()
        self.assertTrue(report['complete'])
        self.assertEqual(report['protocol_sha256'], availability.LOCK_SHA)
        self.assertEqual(report['source_sha256'], loader.digest(Path(availability.__file__).read_bytes()))
        self.assertEqual(report['native_sources'], plan['native_sources'])
        self.assertEqual(report['activity_case_denominator'], 15)
        self.assertEqual(len(report['cases']), 15)
        self.assertTrue(all(c['activity_contract_pass'] for c in report['cases']))

    def test_retained_silence_is_not_scored_and_old_scores_are_unchanged(self):
        report = json.loads((availability.HERE / 'musicfm-availability-v1.json').read_bytes())
        _, old = availability.protocol()
        self.assertEqual((report['paired_denominator'], len(report['pairs'])), (6, 6))
        self.assertEqual(report['pairs'], old['discrimination']['pairs'])
        for case in report['cases'][:6]:
            for speed, row in case['scores'].items():
                if case['id'] == 'silence':
                    self.assertIsNone(row['clocks'])
                    self.assertFalse(row['feature_access'])
                else:
                    self.assertEqual({k: v for k, v in row.items() if k != 'feature_access'},
                                     old['discrimination']['scores'][case['id']][speed])
        self.assertEqual(report['feature_cases_replayed'], 5)
        self.assertTrue(report['previous_failure_preserved'])

    def test_retained_new_energy_controls_do_not_claim_neural_acceptance(self):
        report = json.loads((availability.HERE / 'musicfm-availability-v1.json').read_bytes())
        for case in report['cases'][6:]:
            self.assertFalse(case['feature_access'])
            self.assertFalse(case['periodicity_evaluated'])
        for key in ('independent_acceptance', 'new_pretrained_inference', 'music_access', 'holdout_access',
                    'production_change', 'new_user_parameters', 'training'):
            self.assertFalse(report[key])
        self.assertEqual(report['decision'], 'known_counterexample_composition_pass_not_independent_acceptance')


if __name__ == '__main__':
    unittest.main()
