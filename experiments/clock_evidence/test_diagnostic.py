import unittest

import numpy as np
import torch

from experiments.clock_readout.model import ClockReadout, HALO
from experiments.coupled_clock.clock import integrate
from experiments.coupled_clock.model import CoupledClockReadout
from experiments.coupled_clock.measurement import KEYS, measure, clock_fields
from experiments.clock_evidence.diagnostic import intervention, interior, field_response, paired_summary
from experiments.clock_evidence.run import predict


class InterventionTests(unittest.TestCase):
    def test_variants_do_not_mutate_features(self):
        x = np.arange(30, dtype=np.float32).reshape(10, 3)
        saved = x.copy()
        for variant in ('natural', 'zero', 'time_mean', 'half_roll'):
            result = intervention(x, variant)
            self.assertEqual(result.dtype, x.dtype)
            self.assertEqual(result.shape, x.shape)
            result[0, 0] = -100
            np.testing.assert_array_equal(x, saved)

    def test_mean_preserves_channel_mean_but_removes_variation(self):
        x = np.arange(35, dtype=np.float32).reshape(7, 5)
        result = intervention(x, 'time_mean')
        np.testing.assert_array_equal(result, np.broadcast_to(x.mean(axis=0), x.shape))
        np.testing.assert_array_equal(intervention(x, 'zero'), np.zeros_like(x))

    def test_roll_is_invertible_and_preserves_frame_vectors(self):
        for n in (6, 7):
            x = np.arange(n * 5, dtype=np.float32).reshape(n, 5)
            shifted = intervention(x, 'half_roll')
            np.testing.assert_array_equal(np.roll(shifted, -(n // 2), axis=0), x)
            np.testing.assert_array_equal(np.sort(shifted, axis=0), np.sort(x, axis=0))

    def test_invalid_variants_and_features_fail(self):
        for x in (np.ones((4, 2)), np.ones((1, 2), np.float32), np.full((4, 2), np.nan, np.float32)):
            with self.assertRaises(ValueError):
                intervention(x, 'natural')
        with self.assertRaises(ValueError):
            intervention(np.ones((4, 2), np.float32), 'best_shift')

    def test_interior_matches_brute_force_contiguous_source_support(self):
        for n in (2, 7, 8, 17, 18):
            for halo in (0, 1, 3):
                expected = []
                for t in range(n):
                    a, b = t - halo, t + halo
                    sources = (np.arange(a, b + 1) - n // 2) % n
                    expected.append(a >= 0 and b < n and np.all(np.diff(sources) == 1))
                np.testing.assert_array_equal(interior(n, halo), expected)

    def test_short_interior_is_missing_not_full_support(self):
        self.assertFalse(interior(20).any())
        with self.assertRaises(ValueError):
            interior(20, -1)

    def test_field_response_respects_missing_phase(self):
        natural = np.tile([0., 1., 0.], (5, 1))
        modified = np.tile([1., 0., 1.], (5, 1))
        result = field_response(natural, modified, np.ones(5, bool), True)
        self.assertEqual(result['log2_period_rms_change'], 1.)
        self.assertEqual(result['dense_phase_mean_absolute_change_cycles'], .25)
        modified[:, 1:] = 0
        self.assertIsNone(field_response(natural, modified, np.ones(5, bool), True)['dense_phase_mean_absolute_change_cycles'])
        self.assertIsNone(field_response(natural, modified, np.zeros(5, bool), True)['log2_period_rms_change'])

    def test_unused_coupled_phase_field_is_not_reported_as_observation(self):
        result = field_response(np.zeros((4, 2)), np.ones((4, 2)), np.ones(4, bool), False)
        self.assertIsNone(result['dense_phase_mean_absolute_change_cycles'])

    def test_paired_macro_is_work_balanced_and_missing_is_not_dropped(self):
        def row(work, delta):
            return dict(work=work, metrics={
                'natural': {k: 1. for k in KEYS},
                'zero': {k: None if delta is None else 1. + delta for k in KEYS}})
        result = paired_summary([row('a', 1.), row('a', 1.), row('b', -2.)], 'zero')[KEYS[0]]
        self.assertEqual(result['intervention_minus_natural_work_macro'], -.5)
        self.assertEqual(result['natural_better_recordings'], 2)
        result = paired_summary([row('a', 1.), row('b', None)], 'zero')[KEYS[0]]
        self.assertIsNone(result['intervention_minus_natural_work_macro'])
        self.assertEqual(result['missing_recordings'], 1)

    def test_bad_musical_output_keeps_denominator(self):
        q = np.arange(400, dtype=np.float64) / 25
        bad = q.copy()
        bad[250] = np.nan
        result = measure(clock_fields(bad), q, np.ones(400, bool))
        self.assertEqual(result['reference_points'], 400)
        self.assertEqual(result['available_points'], 399)
        self.assertIsNone(result[KEYS[2]])


class ArchitecturalPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_origin_phase_field_enters_integrated_clock(self):
        fields = torch.zeros((1, 20, 2), dtype=torch.float64, requires_grad=True)
        clock = integrate(fields[:, :-1, 0], fields[:, 0, 1])
        clock.cycles.square().sum().backward()
        self.assertGreater(abs(float(fields.grad[0, 0, 1])), 0.)
        self.assertEqual(torch.count_nonzero(fields.grad[0, 1:, 1]).item(), 0)
        self.assertEqual(torch.count_nonzero(fields.grad[0, :-1, 0]).item(), 19)
        changed = fields.detach().clone()
        changed[:, 1:, 1] = 123.
        torch.testing.assert_close(integrate(changed[:, :-1, 0], changed[:, 0, 1]).cycles, clock.cycles)

    def test_late_feature_change_can_change_rate_but_not_readout_anchor(self):
        torch.manual_seed(142)
        model = CoupledClockReadout().eval()
        x = torch.randn(1, 700, 512)
        changed = x.clone()
        changed[:, 500:550] += 1
        with torch.inference_mode():
            a, b = model.fields(x), model.fields(changed)
        self.assertEqual(float(a[0, 0, 1]), float(b[0, 0, 1]))
        self.assertGreater(float((a[:, 500:550, 0] - b[:, 500:550, 0]).abs().max()), 0.)

    def test_both_local_field_implementations_are_roll_equivariant_away_from_seams(self):
        torch.manual_seed(142)
        x = np.random.default_rng(142).normal(size=(601, 512)).astype(np.float32)
        safe = interior(len(x))
        self.assertTrue(safe.any())
        for model, direct in ((ClockReadout().eval(), True), (CoupledClockReadout().eval(), False)):
            a, _ = predict(model, x, direct)
            b, _ = predict(model, intervention(x, 'half_roll'), direct)
            np.testing.assert_allclose(b[safe], np.roll(a, len(x) // 2, axis=0)[safe], atol=1e-5, rtol=2e-5)
        self.assertEqual(HALO, 126)


if __name__ == '__main__':
    unittest.main()
