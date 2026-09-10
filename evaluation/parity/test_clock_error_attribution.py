"""Authored controls for oracle-only attribution, not repaired beat outputs."""
import unittest
import numpy as np

from experiments.trajectory_clock.attribution import circular_median, attribute, wrap


class AttributionTests(unittest.TestCase):
    def test_circular_median_matches_exhaustive_breakpoints(self):
        for n in (1, 2, 3, 41, 301):
            errors = np.random.default_rng(n).normal(size=n) * 7
            expected = min(np.abs(wrap(errors - shift)).mean() for shift in errors % 1)
            self.assertAlmostEqual(circular_median(errors)['mean_absolute_cycles'], expected, places=12)

    def test_antipodes_duplicates_and_integer_gauge(self):
        self.assertAlmostEqual(circular_median([0., .5])['mean_absolute_cycles'], .25)
        self.assertEqual(circular_median([.25, 1.25, -2.75])['mean_absolute_cycles'], 0.)
        self.assertEqual(circular_median([.25])['mean_absolute_cycles'], 0.)

    def test_constant_offset_is_fully_repairable_by_one_oracle_anchor(self):
        reference = np.arange(501) / 25
        result = attribute(reference + .25, reference, np.ones(501, bool))
        self.assertAlmostEqual(result['observed_phase_error_cycles'], .25)
        self.assertAlmostEqual(result['best_constant_phase_error_cycles'], 0.)
        self.assertAlmostEqual(result['centered_clock_rmse_cycles'], 0.)

    def test_constant_rate_error_is_not_an_origin_error(self):
        t = np.arange(3001) / 50
        result = attribute(2 * t + .4 + t / 60, 2 * t, np.ones(3001, bool))
        self.assertAlmostEqual(result['reference_spans'][0]['native_rate_bias_bpm'], 1.)
        self.assertAlmostEqual(result['reference_spans'][0]['terminal_drift_cycles'], 1.)
        self.assertGreater(result['best_constant_phase_error_cycles'], .249)
        self.assertAlmostEqual(result['nonlinear_residual_rmse_cycles'], 0., places=12)

    def test_local_change_remains_after_constant_rate_detrending(self):
        t = np.arange(1001) / 50
        result = attribute(2 * t + .01 * (t - 10) ** 2, 2 * t, np.ones(1001, bool))
        self.assertGreater(result['nonlinear_residual_rmse_cycles'], .29)
        self.assertAlmostEqual(result['constant_rate_component_rmse_cycles'], 0., places=12)

    def test_holes_do_not_create_drift_from_unrelated_count_origins(self):
        reference = np.arange(501) / 25
        prediction = reference.copy()
        valid = np.ones(501, bool)
        valid[200:300] = False
        prediction[~valid] = np.nan
        prediction[300:] += 100
        result = attribute(prediction, reference, valid)
        self.assertEqual(len(result['reference_spans']), 2)
        self.assertAlmostEqual(result['centered_clock_rmse_cycles'], 0., places=12)
        self.assertEqual(result['points'], 401)

    def test_missing_required_output_is_rejected(self):
        with self.assertRaises(ValueError):
            attribute(np.array([0., np.nan]), np.array([0., 1.]), np.ones(2, bool))
        with self.assertRaises(ValueError):
            circular_median([])

    def test_singleton_support_cannot_establish_drift(self):
        result = attribute(np.array([.2, .3, .4]), np.array([0., 1., 2.]), np.array([True, False, True]))
        self.assertIsNone(result['centered_clock_rmse_cycles'])


if __name__ == '__main__':
    unittest.main()
