"""Authored witnesses for descriptive decomposition, masks and nulls."""
import copy
import unittest

import numpy as np

from experiments.tempo_attribution import diagnostic as d


def fixture(rates):
    rates = np.asarray(rates, np.float64)
    return np.r_[0., np.cumsum(rates / 50)], np.ones(len(rates) + 1, bool)


class TempoDiagnosticTests(unittest.TestCase):
    def test_perfect_variable_tempo_has_zero_error_and_unit_response(self):
        rate = np.exp2(np.linspace(0., 2., 700))
        reference, mask = fixture(rate)
        out = d.analyze(rate, reference, mask)
        self.assertAlmostEqual(out['offset_shape']['log2_mse'], 0.)
        self.assertAlmostEqual(out['offset_shape']['centered_response_slope'], 1.)
        for lag in ('50', '200'):
            self.assertAlmostEqual(out['changes'][lag]['all']['response_slope'], 1.)
            self.assertAlmostEqual(out['changes'][lag]['all']['error_rmse_log2'], 0.)

    def test_pure_double_time_has_bias_not_shape_error(self):
        rate = np.exp2(np.linspace(0., 1., 700))
        reference, mask = fixture(rate)
        out = d.analyze(rate * 2, reference, mask)
        self.assertAlmostEqual(out['offset_shape']['signed_bias_log2'], 1.)
        self.assertAlmostEqual(out['offset_shape']['squared_bias'], 1.)
        self.assertAlmostEqual(out['offset_shape']['residual_log2_mse'], 0.)
        self.assertAlmostEqual(out['octave']['double_fraction'], 1.)
        self.assertAlmostEqual(out['changes']['200']['all']['response_slope'], 1.)

    def test_constant_predictor_misses_changes_even_with_zero_mean_bias(self):
        rate = np.exp2(np.linspace(0., 2., 700))
        reference, mask = fixture(rate)
        out = d.analyze(np.full(700, 2.), reference, mask)
        self.assertAlmostEqual(out['offset_shape']['signed_bias_log2'], 0.)
        self.assertGreater(out['offset_shape']['residual_log2_mse'], .3)
        self.assertAlmostEqual(out['changes']['200']['all']['response_slope'], 0.)
        self.assertAlmostEqual(out['changes']['200']['all']['reference_rms_log2'],
                               out['changes']['200']['all']['error_rmse_log2'])
        self.assertEqual(out['changes']['200']['large']['large_change_direction_agreement'], 0.)

    def test_wrong_direction_and_wrong_amplitude_are_distinct(self):
        dx = np.array([.5, -.5])
        self.assertEqual(d.response(dx, -dx)['large_change_direction_agreement'], 0.)
        out = d.response(dx, .1 * dx)
        self.assertEqual(out['large_change_direction_agreement'], 1.)
        self.assertAlmostEqual(out['response_slope'], .1)

    def test_constant_reference_has_null_slope_not_perfect_tracking(self):
        reference, mask = fixture(np.full(501, 2.))
        out = d.analyze(np.full(501, 3.), reference, mask)
        self.assertIsNone(out['offset_shape']['centered_response_slope'])
        self.assertIsNone(out['changes']['200']['all']['response_slope'])
        self.assertEqual(out['changes']['200']['large']['pairs'], 0)
        self.assertIsNone(out['changes']['200']['large']['error_rmse_log2'])

    def test_internal_gap_is_not_bridged_and_all_pair_counts_remain(self):
        reference, valid = fixture(np.full(251, 2.))
        valid[125] = False
        out = d.analyze(np.full(251, 2.), reference, valid)
        cells = valid[:-1] & valid[1:]
        for lag in d.LAGS:
            expected = sum(cells[i:i + lag + 1].all() for i in range(len(cells) - lag))
            self.assertEqual(out['changes'][str(lag)]['all']['pairs'], expected)
        self.assertEqual(out['changes']['200']['all']['pairs'], 0)

    def test_empty_and_short_support_is_null_not_zero_accuracy(self):
        reference, valid = fixture(np.full(20, 2.))
        out = d.analyze(np.full(20, 2.), reference, valid)
        self.assertEqual(out['changes']['50']['all']['pairs'], 0)
        out = d.analyze(np.full(20, np.nan), reference, np.zeros(21, bool))
        self.assertIsNone(out['offset_shape']['log2_mse'])
        self.assertEqual(out['reference']['seconds'], 0.)
        self.assertIsNone(out['reference']['bpm_p50'])

    def test_bad_required_prediction_is_rejected_not_filtered(self):
        reference, valid = fixture(np.full(251, 2.))
        for value in (np.nan, np.inf, 0., -1.):
            rate = np.full(251, 2.)
            rate[125] = value
            with self.assertRaisesRegex(ValueError, 'supported rate'):
                d.analyze(rate, reference, valid)

    def test_octave_bias_and_fluctuation_sum_to_total_error(self):
        reference, valid = fixture(np.full(501, 2.))
        predicted = np.exp2(1.4 + np.sin(np.arange(501) / 20) * .2)
        arrays = [a.copy() for a in (predicted, reference, valid)]
        out = d.analyze(predicted, reference, valid)['offset_shape']
        self.assertAlmostEqual(out['log2_mse'], out['squared_bias'] + out['residual_log2_mse'])
        for old, new in zip(arrays, (predicted, reference, valid)):
            np.testing.assert_array_equal(old, new)

    def test_fixed_reference_bands_partition_every_cell(self):
        bpm = np.array([30., 60., 90., 120., 180., 240.])
        out = d.coverage(bpm / 60, np.ones(6, bool))
        self.assertAlmostEqual(sum(v for k, v in out.items() if k.startswith('fraction_')), 1.)
        self.assertEqual(out['fraction_180_inf'], 2 / 6)

    def test_work_macro_is_not_record_pooling_and_does_not_drop_nulls(self):
        rows = [dict(work=w, common={'a': {'field': v}}) for w, v in (('x', 1.), ('x', 3.), ('y', 8.))]
        out = d.macro(rows, 'common', 'a')['field']
        self.assertEqual(out['work_macro'], 5.)
        altered = copy.deepcopy(rows)
        altered[0]['common']['a']['field'] = None
        out = d.macro(altered, 'common', 'a')['field']
        self.assertIsNone(out['work_macro'])
        self.assertEqual(out['missing_recordings'], 1)


if __name__ == '__main__':
    unittest.main()
