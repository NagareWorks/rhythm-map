"""Algebraic witnesses must not be misreported as audio/model outcomes."""
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

import numpy as np

import rhythm_support_null as audit


class RhythmSupportNullTests(unittest.TestCase):
    def test_retained_report_reconstructs_without_network(self):
        with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')):
            report = audit.build_report()
        path = Path(__file__).with_name('rhythm-support-null-v1.json')
        self.assertEqual(report, json.loads(path.read_text(encoding='utf-8')))

    def test_fft_construction_matches_independent_direct_lags(self):
        for n in (3, 8, 17, 64):
            x = ((np.arange(n) ** 2) % 11 - 5).astype(float)
            phases = np.arange(n // 2 + 1) / 3
            phases[0] = 0
            if n % 2 == 0:
                phases[-1] = 0
            y = audit.phase_surrogate(x, phases)
            with self.subTest(n=n):
                np.testing.assert_allclose(audit.circular_autocorrelation(x),
                                           audit.circular_autocorrelation(y), rtol=0, atol=audit.ABS_TOL)
                np.testing.assert_allclose(abs(np.fft.rfft(x)) ** 2, abs(np.fft.rfft(y)) ** 2,
                                           rtol=0, atol=audit.ABS_TOL)
                self.assertAlmostEqual(x.mean(), y.mean(), places=12)

    def test_every_circular_feature_shift_retains_every_lag(self):
        x = np.arange(48).reshape(16, 3) % 7
        expected = audit.circular_feature_lags(x)
        for shift in range(16):
            with self.subTest(shift=shift):
                np.testing.assert_allclose(audit.circular_feature_lags(np.roll(x, shift, axis=0)),
                                           expected, rtol=0, atol=audit.ABS_TOL)

    def test_linear_edges_not_confused_with_circular_invariance(self):
        x = np.array([1, 0, 2, 3, -1, 4])
        y = np.roll(x, 2)
        self.assertFalse(np.array_equal(np.correlate(x, x, 'full'), np.correlate(y, y, 'full')))
        np.testing.assert_array_equal(audit.circular_autocorrelation(x), audit.circular_autocorrelation(y))

    def test_stationary_rotation_process_is_not_exchangeable(self):
        witness = audit.build_report()['permutation']
        x = np.array(witness['original'])
        orbit = {tuple(np.roll(x, k)) for k in range(len(x))}
        for rotation in range(len(x)):
            self.assertEqual({tuple(np.roll(row, rotation)) for row in orbit}, orbit)
        self.assertNotIn(tuple(witness['permuted']), orbit)
        self.assertEqual(witness['original_orbit_probability'], 0.125)
        self.assertEqual(witness['permuted_orbit_probability'], 0)
        self.assertEqual(witness['original_lag_one'], 0.5)
        self.assertEqual(witness['permuted_lag_one'], -1)

    def test_nonlinear_or_nonstationary_is_not_a_support_label(self):
        row = audit.build_report()['nonstationary_reference']
        np.testing.assert_array_equal(np.square(row['amplitude']), row['variance'])
        self.assertGreater(max(row['variance']), min(row['variance']))
        self.assertEqual(row['distinct_time_covariances'], 0)
        self.assertIsNone(row['listener_label'])
        self.assertFalse(row['repeated_pulse_imposed'])

    def test_rank_includes_observed_and_conservative_ties(self):
        self.assertEqual(audit.rank_tail(1.0, [0.0] * 99), 0.01)
        self.assertEqual(audit.rank_tail(1.0, [1.0] * 99), 1)
        self.assertEqual(audit.rank_tail(1.0, [0.0, 1.0, 2.0]), 0.75)

    def test_search_must_be_applied_to_each_surrogate(self):
        # An authored table, not exchangeable samples or a significance claim.
        draws = [[1, 10]] * 9
        original_max = 5
        self.assertEqual(audit.rank_tail(original_max, [row[0] for row in draws]), 0.1)
        self.assertEqual(audit.rank_tail(original_max, [max(row) for row in draws]), 1)

    def test_invalid_rank_inputs_refused(self):
        for original, draws in ((True, [1]), (float('nan'), [1]), (1, []), (1, [float('inf')]),
                                (1, [True]), (1, ['2']), (1, [0] * 10000)):
            with self.subTest(original=original), self.assertRaises(ValueError):
                audit.rank_tail(original, draws)

    def test_invalid_vector_inputs_refused(self):
        for values in ([], [1], [[1, 2]], [True, False], [1j, 2j], ['1', '2'],
                       [1, float('nan')], [float('inf'), 1], [1e7, 0], [0] * 2049):
            with self.subTest(values=str(values)[:40]), self.assertRaises(ValueError):
                audit.vector(values)

    def test_invalid_feature_matrices_refused(self):
        for values in ([1, 2], [[]] * 2, [[1, float('nan')]] * 2, [[1j], [2j]],
                       [[1e7], [0]], np.zeros((2049, 1)), np.zeros((2, 33))):
            with self.subTest(shape=np.shape(values)), self.assertRaises(ValueError):
                audit.circular_feature_lags(values)

    def test_real_endpoint_and_phase_geometry_checks(self):
        for phases in ([1, 0, 0], [0, 0, 1], [0, 0]):
            with self.subTest(phases=phases), self.assertRaises(ValueError):
                audit.phase_surrogate([1, 2, 3, 4], phases)

    def test_absence_of_model_gate_and_training_claims(self):
        report = audit.build_report()
        self.assertEqual(report['producer_gate'], dict(status='not_run', passes=None,
                                                      case_count=11, unknown_count=None))
        for flag in ('new_audio', 'new_inference', 'feature_access', 'holdout_access', 'training',
                     'training_necessity_proven', 'production_change',
                     'valid_no_support_reference_established'):
            self.assertIs(report[flag], False)
        self.assertFalse(report['rank_arithmetic']['actual_test_performed'])
        self.assertIsNone(report['rank_arithmetic']['rhythm_probability'])

    def test_finite_serialization_and_fixed_numerical_budget(self):
        report = audit.build_report()
        self.assertEqual(report['numerical_abs_budget'], 1e-10)
        self.assertEqual(report['reported_decimal_places'], 12)
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
