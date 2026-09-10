"""Offline synthetic checks for the fixed fitter and honest comparison masks."""
import unittest

import numpy as np
import torch

from experiments.coupled_clock.measurement import clock_fields, independent_fields, intervals, measure, macro, gates, KEYS
from experiments.coupled_clock.run import work_weights, work_mean, input_tensor
from experiments.coupled_clock.prepare import replay_v1
from experiments.clock_readout.model import ClockReadout, HALO
from experiments.coupled_clock.supervision import reference_clock


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.q = np.arange(501, dtype=np.float64) / 25
        self.valid = np.ones(501, bool)

    def test_perfect_denominators(self):
        result = measure(clock_fields(self.q), self.q, self.valid)
        self.assertEqual([result[k] for k in KEYS], [0., 0., 0., 0.])
        self.assertEqual(result['reference_cells'], 500)
        self.assertEqual(result['reference_points'], 501)
        self.assertEqual(result['reference_4s_intervals'], 301)

    def test_native_double_count_does_not_alias(self):
        result = measure(clock_fields(self.q * 2), self.q, self.valid)
        self.assertAlmostEqual(result[KEYS[0]], 100)
        self.assertAlmostEqual(result[KEYS[3]], 8)

    def test_phase_wrap(self):
        rate, phase = clock_fields(self.q)
        result = measure((rate, phase + .99), self.q, self.valid)
        self.assertAlmostEqual(result[KEYS[2]], .01)

    def test_missing_prediction_does_not_shrink_denominator(self):
        rate, phase = clock_fields(self.q)
        phase = phase.copy()
        rate[250] = np.nan
        phase[300] = np.nan
        result = measure((rate, phase), self.q, self.valid)
        self.assertTrue(all(result[k] is None for k in KEYS))
        self.assertEqual(result['reference_cells'], 500)
        self.assertEqual(result['available_cells'], 499)

    def test_hole_blocks_every_crossing_interval(self):
        valid = self.valid.copy()
        valid[250] = False
        support = intervals(valid, 200)
        self.assertFalse(support[50:251].any())
        self.assertEqual(int(support.sum()), 100)

    def test_unsupported_prefix_nan_does_not_poison_counts(self):
        q, valid = self.q.copy(), self.valid.copy()
        q[:10], valid[:10] = np.nan, False
        result = measure(clock_fields(q), self.q, valid)
        self.assertLess(result[KEYS[3]], 1e-12)
        self.assertEqual(result['reference_cells'], 490)

    def test_short_crop_reports_absent_four_second_metric(self):
        result = measure(clock_fields(self.q[:10]), self.q[:10], self.valid[:10])
        self.assertEqual(result['reference_4s_intervals'], 0)
        self.assertIsNone(result[KEYS[3]])

    def test_trapezoid_averages_rates_not_periods(self):
        prediction = np.array([[0., 1., 0.], [-1., 1., 0.], [0., 1., 0.]])
        rate, phase = independent_fields(prediction)
        np.testing.assert_allclose(rate, [1.5, 1.5])
        np.testing.assert_array_equal(phase, [0., 0., 0.])

    def test_zero_phase_amplitude_is_missing(self):
        prediction = np.zeros((501, 3))
        result = measure(independent_fields(prediction), self.q, self.valid)
        self.assertIsNone(result[KEYS[2]])
        self.assertEqual(result['available_points'], 0)

    def test_reference_cell_straddles_subframe_transition(self):
        ref = reference_clock([0., .513, 1.513, 2.513], 101, 2.)
        rate, _ = clock_fields(ref.cycles)
        expected = ((.513 - .5) / .513 + (.52 - .513) / 1.) * 50
        self.assertAlmostEqual(rate[25], expected)

    def test_work_macro_not_recording_or_frame_micro(self):
        rows = [dict(work=w, score={k: v for k in KEYS}) for w, v in [('a', 0.), ('a', 2.), ('b', 9.)]]
        self.assertEqual(macro(rows, 'score')[KEYS[0]], 5.)
        rows[0]['score'][KEYS[0]] = None
        self.assertIsNone(macro(rows, 'score')[KEYS[0]])

    def test_gate_fails_missing_metrics_budget_and_breadth(self):
        names = ('audio', 'zero', 'audio_common', 'zero_common', 'raw_common', 'v1_common')
        rows = [dict(role=role, work=str(i), **{name: {k: (2. if name == 'zero' else 1.) for k in KEYS} for name in names})
                for role in ('development', 'diagnostic') for i in range(3)]
        summaries = {role: {name: macro([r for r in rows if r['role'] == role], name) for name in names}
                     for role in ('development', 'diagnostic')}
        fits = [dict(epochs=20, budget_exhausted=False)] * 2
        self.assertTrue(gates(rows, summaries, fits)['passed'])
        fits = [dict(epochs=19, budget_exhausted=True)] * 2
        self.assertFalse(gates(rows, summaries, fits)['passed'])
        rows[0]['audio'][KEYS[0]] = None
        self.assertFalse(gates(rows, summaries, fits)['complete'])
        for row in rows:
            row['audio_common'][KEYS[2]] = 1.1
        self.assertFalse(gates(rows, summaries, fits)['recording_breadth'])


class FitterContractTests(unittest.TestCase):
    def test_work_weights_sum_equally(self):
        records = [dict(work=w) for w in ['a', 'a', 'b', 'c', 'c', 'c']]
        weights = work_weights(records)
        for work in weights:
            self.assertAlmostEqual(sum(weights[r['work']] / len(records) for r in records if r['work'] == work), 1 / 3)
        self.assertEqual(work_mean(dict(a=[0., 2.], b=[9.])), 5.)

    def test_zero_control_keeps_full_crop_and_does_not_mutate_audio(self):
        row = dict(features=torch.ones(501, 512))
        self.assertEqual(tuple(input_tensor(row, True).shape), (1, 501, 512))
        self.assertEqual(int(input_tensor(row, True).count_nonzero()), 0)
        self.assertTrue(torch.all(input_tensor(row, False) == 1).item())

    def test_v1_replay_preserves_tail_and_physical_edge_context(self):
        torch.set_num_threads(2)
        torch.manual_seed(142)
        model = ClockReadout().eval()
        hidden = np.random.default_rng(142).normal(size=(413, 512)).astype(np.float32)
        actual = replay_v1(model, hidden)
        full = np.pad(hidden, ((HALO, HALO), (0, 0)))
        with torch.inference_mode():
            expected = model(torch.from_numpy(full)[None])[0, HALO:-HALO].numpy()
        self.assertEqual(actual.shape, (413, 3))
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
