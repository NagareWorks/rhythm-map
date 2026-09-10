"""Authored structural witnesses; no audio capture, optimizer run or holdout."""
import copy
import unittest

import numpy as np
import torch

from experiments.coupled_clock.clock import FPS, integrate
from experiments.coupled_clock.model import CoupledClockReadout, HALO, PARAMETERS
from experiments.coupled_clock.supervision import LAGS, clock_loss, reference_clock, supported_pairs

torch.set_num_threads(2)


def from_rates(rates, anchor=0.):
    rates = torch.as_tensor(rates, dtype=torch.float64).reshape(1, -1)
    return integrate(-torch.log2(rates), torch.tensor([anchor], dtype=torch.float64))


def losses(clock, q, valid=None):
    q = torch.as_tensor(q, dtype=torch.float64).reshape(1, -1)
    mask = torch.ones_like(q, dtype=torch.bool) if valid is None else torch.as_tensor(valid).reshape(1, -1)
    return clock_loss(clock, q, mask)


class ClockTests(unittest.TestCase):
    def test_constant_120_has_consistent_phase_and_cell_tempo(self):
        clock = from_rates(np.full(100, 2.))
        torch.testing.assert_close(clock.cycles[0], torch.arange(101, dtype=torch.float64) / 25)
        torch.testing.assert_close(clock.cell_bpm, torch.full((1, 100), 120., dtype=torch.float64))
        torch.testing.assert_close(clock.phase_vector[0, ::25],
                                   torch.tensor([[1., 0.]] * 5, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_step_is_not_smoothed_and_advancement_remains_positive(self):
        rates = np.r_[np.full(50, 2.), np.full(50, 1.), np.full(50, 1.5)]
        clock = from_rates(rates)
        torch.testing.assert_close(clock.cycles.diff(dim=1)[0], torch.from_numpy(rates) / FPS)
        self.assertEqual(clock.cell_bpm[0, 49].item(), 120)
        self.assertEqual(clock.cell_bpm[0, 50].item(), 60)
        self.assertTrue((clock.cycles.diff(dim=1) > 0).all().item())

    def test_linear_rate_ramp_matches_integral_at_grid_points(self):
        # A linear instantaneous rate has the cell midpoint as its exact mean.
        midpoint = (np.arange(300) + .5) / FPS
        clock = from_rates(1. + .2 * midpoint, anchor=.3)
        times = torch.arange(301, dtype=torch.float64) / FPS
        torch.testing.assert_close(clock.cycles[0], .3 + times + .1 * times.square())

    def test_chunked_integration_carries_anchor_instead_of_resetting(self):
        rates = torch.linspace(.8, 4., 3010, dtype=torch.float64)[None]
        period = -torch.log2(rates)
        anchor = torch.tensor([.17], dtype=torch.float64)
        whole = integrate(period, anchor)
        pieces = [anchor[:, None]]
        for start in range(0, period.shape[1], 200):
            part = integrate(period[:, start:start + 200], anchor)
            pieces.append(part.cycles[:, 1:])
            anchor = part.cycles[:, -1]
        torch.testing.assert_close(torch.cat(pieces, dim=1), whole.cycles, rtol=0, atol=1e-11)

    def test_multiple_beats_per_cell_are_not_wrapped_into_backwards_motion(self):
        clock = from_rates([100., 125., 150.])
        torch.testing.assert_close(clock.cycles[0], torch.tensor([0., 2., 4.5, 7.5], dtype=torch.float64))

    def test_float32_fields_accumulate_in_float64(self):
        clock = integrate(torch.zeros(1, 3000), torch.zeros(1))
        self.assertEqual(clock.cycles.dtype, torch.float64)
        self.assertEqual(clock.log2_period.dtype, torch.float64)
        self.assertAlmostEqual(clock.cycles[0, -1].item(), 60., places=10)

    def test_invalid_numerics_do_not_get_clipped_into_a_clock(self):
        for value in (float('nan'), float('inf'), -2000., 2000., -1020.):
            with self.subTest(value=value), self.assertRaises(ValueError):
                integrate(torch.tensor([[value]], dtype=torch.float64), torch.zeros(1))
        for anchor in (float('nan'), 1e20, 1e9):
            with self.subTest(anchor=anchor), self.assertRaises(ValueError):
                integrate(torch.zeros(1, 2), torch.tensor([anchor], dtype=torch.float64))

    def test_invalid_geometry_and_frame_rate_fail(self):
        for period, anchor in ((torch.zeros(2), torch.zeros(1)),
                               (torch.zeros(1, 0), torch.zeros(1)),
                               (torch.zeros(2, 4), torch.zeros(1)),
                               (torch.zeros(1, 4, dtype=torch.int64), torch.zeros(1))):
            with self.assertRaises(ValueError):
                integrate(period, anchor)
        for fps in (0, -1, True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                integrate(torch.zeros(1, 2), torch.zeros(1), fps)

    def test_integration_gradient_matches_numerical_derivative(self):
        periods = torch.tensor([[-1., -.8, -.6]], dtype=torch.float64, requires_grad=True)
        anchor = torch.tensor([.2], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda p, a: integrate(p, a).cycles,
                                                (periods, anchor)))


class SupervisionTests(unittest.TestCase):
    def test_native_beats_define_unwrapped_counts(self):
        ref = reference_clock([0., .5, 1., 1.5, 2.], 101, 2.)
        np.testing.assert_allclose(ref.cycles[:100], np.arange(100) / 25)
        self.assertFalse(ref.valid[-1])
        self.assertTrue(np.isnan(ref.cycles[-1]))

    def test_track_edges_and_physical_tail_remain_unknown(self):
        ref = reference_clock([.5, 1., 1.5, 2.], 101, 1.7)
        self.assertEqual(np.where(ref.valid)[0].tolist(), list(range(25, 85)))
        self.assertTrue(np.isnan(ref.cycles[~ref.valid]).all())

    def test_oracle_grid_can_represent_a_subframe_tempo_change(self):
        # The change at .513 s lies INSIDE a 20 ms cell. Integrate exact cell
        # advancement, not the left endpoint's instantaneous period across it.
        ref = reference_clock([0., .513, 1.513, 2.513], 125, 2.5)
        self.assertTrue(ref.valid.all())
        rate = np.diff(ref.cycles) * FPS
        clock = from_rates(rate, ref.cycles[0])
        torch.testing.assert_close(clock.cycles[0], torch.from_numpy(ref.cycles))
        self.assertLess(losses(clock, ref.cycles)['total'].item(), 1e-20)
        self.assertNotAlmostEqual(rate[25], rate[24])
        self.assertNotAlmostEqual(rate[25], rate[26])

    def test_weak_or_missing_observation_is_not_missing_reference(self):
        # The same sparse detected events could support either reference. The
        # loss has NO detector-event input and must distinguish their counts.
        events = np.array([0., .5, 1., 2., 3.])
        continued = reference_clock(np.arange(0., 3.51, .5), 151, 3.02)
        changed = reference_clock([0., .5, 1., 2., 3., 4.], 151, 3.02)
        self.assertEqual(len(events), 5)
        self.assertAlmostEqual(continued.cycles[125], 5.)
        self.assertAlmostEqual(changed.cycles[125], 3.5)
        clock = from_rates(np.full(150, 2.))
        self.assertLess(losses(clock, continued.cycles)['total'].item(), 1e-20)
        self.assertGreater(losses(clock, changed.cycles)['advance'].item(), 0)

    def test_half_and_double_time_cannot_hide_at_coincident_phase_ticks(self):
        for native, predicted in ((50., 100.), (100., 50.)):
            with self.subTest(native=native):
                q = np.arange(6) * native / FPS
                terms = losses(from_rates(np.full(5, predicted)), q)
                self.assertAlmostEqual(terms['phase'].item(), 0., places=12)
                self.assertAlmostEqual(terms['advance'].item(), 1., places=12)

    def test_integer_origin_is_irrelevant_but_fractional_phase_is_not(self):
        q = np.arange(101) / 25
        clock = from_rates(np.full(100, 2.), anchor=3.)
        self.assertLess(losses(clock, q)['total'].item(), 1e-20)
        shifted = from_rates(np.full(100, 2.), anchor=.25)
        self.assertAlmostEqual(losses(shifted, q)['phase'].item(), 1., places=12)

    def test_holes_do_not_bridge_even_when_endpoints_are_valid(self):
        valid = torch.ones(1, 251, dtype=torch.bool)
        valid[0, 100] = False
        for lag in LAGS:
            pairs = supported_pairs(valid, lag)[0].numpy()
            expected = np.array([valid[0, i:i + lag + 1].all().item()
                                 for i in range(251 - lag)])
            np.testing.assert_array_equal(pairs, expected)
        self.assertFalse(supported_pairs(valid, 200).any().item())

    def test_unknown_nan_targets_are_not_zero_labels_or_nan_gradients(self):
        support = np.ones(251, dtype=bool)
        support[100] = False
        ref = reference_clock(np.arange(0., 6., .5), 251, 5.02, support)
        periods = torch.full((1, 250), -.9, dtype=torch.float64, requires_grad=True)
        anchor = torch.tensor([.1], dtype=torch.float64, requires_grad=True)
        terms = losses(integrate(periods, anchor), ref.cycles, ref.valid)
        self.assertEqual(terms['reference_frames'], 250)
        self.assertEqual(terms['pairs_by_lag'][200], 0)
        terms['total'].backward()
        self.assertTrue(torch.isfinite(periods.grad).all().item())
        self.assertTrue(torch.isfinite(anchor.grad).all().item())

    def test_short_tail_keeps_available_lag_instead_of_adding_zero_terms(self):
        terms = losses(from_rates([4., 4.]), [0., .04, .08])
        self.assertAlmostEqual(terms['advance'].item(), 1., places=12)
        self.assertEqual(terms['pairs_by_lag'], {1: 2, 25: 0, 100: 0, 200: 0})

    def test_invalid_annotations_and_support_fail(self):
        for beats in ([0.], [0., 0.], [1., 0.], [-1., 0.], [0., np.nan]):
            with self.assertRaises(ValueError):
                reference_clock(beats, 100, 2.)
        for support in (np.ones(100), np.ones(99, dtype=bool)):
            with self.assertRaises(ValueError):
                reference_clock([0., 1.], 100, 2., support)

    def test_missing_or_nonmonotone_supervision_is_not_a_perfect_score(self):
        clock = from_rates([2., 2.])
        for q, valid in (([0., 1., 2.], [False, False, False]),
                         ([0., 1., 2.], [True, False, True]),
                         ([0., float('nan'), 2.], [True, True, True]),
                         ([0., 1., .5], [True, True, True])):
            with self.assertRaises(ValueError):
                losses(clock, q, valid)

    def test_label_origin_and_mask_do_not_change_model_input(self):
        q = np.arange(101) / 25
        clock = from_rates(np.full(100, 2.))
        a = losses(clock, q)
        b = losses(clock, q + 7.)
        self.assertAlmostEqual(a['total'].item(), b['total'].item(), places=20)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(142)
        self.model = CoupledClockReadout().eval()

    def test_parameter_budget_and_point_cell_shapes(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), PARAMETERS)
        self.assertEqual(PARAMETERS, 23970)
        self.assertEqual(HALO, 126)
        clock = self.model(torch.zeros(2, 251, 512))
        self.assertEqual(clock.cycles.shape, (2, 251))
        self.assertEqual(clock.cell_bpm.shape, (2, 250))
        self.assertEqual(clock.phase_vector.shape, (2, 251, 2))

    def test_full_and_chunked_fields_and_global_clock_match(self):
        x = torch.randn(1, 615, 512)
        with torch.inference_mode():
            full_fields, full = self.model.fields(x), self.model(x)
            for size in (1, 127, 200, 614, 1000):
                fields, clock = self.model.fields(x, size), self.model(x, size)
                torch.testing.assert_close(fields, full_fields, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(clock.cycles, full.cycles, atol=2e-5, rtol=1e-6)
                torch.testing.assert_close(clock.phase_vector, full.phase_vector, atol=2e-5, rtol=1e-5)

    def test_two_frame_input_and_nondivisible_tail_are_retained(self):
        with torch.inference_mode():
            for length in (2, 199, 200, 201, 401):
                clock = self.model(torch.randn(1, length, 512), 200)
                self.assertEqual(clock.cycles.shape[1], length)
                self.assertEqual(clock.log2_period.shape[1], length - 1)

    def test_random_model_is_coherent_but_not_claimed_correct(self):
        with torch.inference_mode():
            clock = self.model(torch.randn(1, 601, 512))
        torch.testing.assert_close(clock.cycles.diff(dim=1), clock.cell_bpm / (60 * FPS),
                                   atol=1e-12, rtol=1e-9)
        torch.testing.assert_close(clock.phase_vector.square().sum(dim=-1),
                                   torch.ones_like(clock.cycles))

    def test_backward_reaches_rate_anchor_and_temporal_layers_without_update(self):
        before = copy.deepcopy(self.model.state_dict())
        clock = self.model(torch.randn(1, 251, 512), 100)
        terms = losses(clock, np.arange(251) / 25 + .13)
        terms['total'].backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        for row in range(2):
            self.assertGreater(self.model.output.weight.grad[row].abs().sum().item(), 0)
        self.assertGreater(self.model.blocks[-1].depthwise.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.equal(value, self.model.state_dict()[key])
                            for key, value in before.items()))

    def test_forward_preserves_input_and_weights(self):
        x = torch.randn(1, 203, 512)
        old, state = x.clone(), copy.deepcopy(self.model.state_dict())
        self.model(x, 100)
        self.assertTrue(torch.equal(x, old))
        self.assertTrue(all(torch.equal(value, self.model.state_dict()[key])
                            for key, value in state.items()))

    def test_invalid_feature_shape_or_chunk_fails(self):
        for x in (torch.zeros(1, 1, 512), torch.zeros(1, 10, 511),
                  torch.zeros(10, 512), torch.full((1, 2, 512), float('nan'))):
            with self.assertRaises(ValueError):
                self.model(x)
        for size in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                self.model(torch.zeros(1, 2, 512), size)


if __name__ == '__main__':
    unittest.main()
