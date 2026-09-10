"""Native trajectory supervision and a source-checked single-factor fitter."""
import inspect
import unittest

import torch

from experiments.coupled_clock.clock import Clock, integrate
from experiments.coupled_clock import run as previous
from experiments.trajectory_clock import run
from experiments.trajectory_clock.supervision import trajectory_loss


def measured(q, reference, valid=None):
    if q.ndim == 1:
        q, reference = q[None], reference[None]
    if valid is None:
        valid = torch.ones_like(q, dtype=torch.bool)
    return trajectory_loss(Clock(q, torch.zeros_like(q[:, :-1]), 50), reference, valid)


class TrajectoryLossTests(unittest.TestCase):
    def test_identical_trajectory_and_integer_gauge_have_zero_loss(self):
        truth = torch.arange(31, dtype=torch.float64) / 25
        for offset in (-100., 0., 7.):
            self.assertAlmostEqual(float(measured(truth + offset, truth)['total']), 0.)

    def test_fractional_origin_is_not_removed_as_a_free_float_offset(self):
        truth = torch.arange(31, dtype=torch.float64) / 25
        result = measured(truth + .25, truth)
        self.assertAlmostEqual(float(result['total']), .0625)
        self.assertAlmostEqual(float(result['centered']), 0.)
        self.assertAlmostEqual(float(result['origin']), .0625)

    def test_one_integer_for_span_not_independent_wrapping_each_frame(self):
        truth = torch.arange(11, dtype=torch.float64)
        result = measured(truth * 2, truth)
        self.assertAlmostEqual(float(result['total']), 10.)
        self.assertAlmostEqual(float(result['centered']), 10.)

    def test_longer_extent_exposes_accumulated_rate_error(self):
        short = torch.arange(101, dtype=torch.float64) / 50
        long = torch.arange(1001, dtype=torch.float64) / 50
        a = measured(2 * short + short / 60, 2 * short)['centered']
        b = measured(2 * long + long / 60, 2 * long)['centered']
        self.assertGreater(float(b / a), 98.)

    def test_loss_decomposes_into_drift_and_registered_mean(self):
        truth = torch.arange(301, dtype=torch.float64) / 25
        result = measured(truth * 1.07 + .31, truth)
        torch.testing.assert_close(result['total'], result['centered'] + result['origin'])

    def test_reference_hole_does_not_require_unknown_cross_gap_count(self):
        truth = torch.arange(21, dtype=torch.float64) / 25
        prediction = truth.clone()
        prediction[11:] += 10
        truth[10] = float('nan')
        mask = torch.ones(1, 21, dtype=torch.bool)
        mask[0, 10] = False
        result = measured(prediction, truth, mask)
        self.assertAlmostEqual(float(result['total']), 0.)
        self.assertEqual(result['reference_frames'], 20)
        self.assertEqual(result['reference_spans'], 2)

    def test_rounding_tie_uses_one_finite_branch(self):
        truth = torch.arange(11, dtype=torch.float64) / 25
        prediction = (truth + .5).requires_grad_()
        result = measured(prediction, truth)
        self.assertAlmostEqual(float(result['total'].detach()), .25)
        result['total'].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all().item())

    def test_gradcheck_through_positive_integral(self):
        period = torch.full((1, 30), -1., dtype=torch.float64, requires_grad=True)
        anchor = torch.tensor([.13], dtype=torch.float64, requires_grad=True)
        truth = torch.arange(31, dtype=torch.float64)[None] / 25
        valid = torch.ones_like(truth, dtype=torch.bool)
        objective = lambda p, a: trajectory_loss(integrate(p, a), truth, valid)['total']
        self.assertTrue(torch.autograd.gradcheck(objective, (period, anchor), eps=1e-6, atol=1e-5))

    def test_gradient_reaches_rate_and_anchor_without_weight_update(self):
        period = torch.full((1, 100), -.98, dtype=torch.float64, requires_grad=True)
        anchor = torch.tensor([.1], dtype=torch.float64, requires_grad=True)
        truth = torch.arange(101, dtype=torch.float64)[None] / 25
        result = trajectory_loss(integrate(period, anchor), truth, torch.ones_like(truth, dtype=torch.bool))
        result['total'].backward()
        self.assertGreater(float(period.grad.abs().sum()), 0.)
        self.assertGreater(float(anchor.grad.abs().sum()), 0.)

    def test_missing_and_nonmonotone_reference_fail(self):
        truth = torch.arange(11, dtype=torch.float64) / 25
        with self.assertRaises(ValueError):
            measured(truth, truth, torch.zeros(1, 11, dtype=torch.bool))
        truth[5] = -1
        with self.assertRaises(ValueError):
            measured(truth, truth)

    def test_optimizer_and_development_loops_change_only_loss_call(self):
        for name in ('fit', 'development_loss'):
            expected = inspect.getsource(getattr(previous, name)).replace('clock_loss(', 'trajectory_loss(')
            self.assertEqual(inspect.getsource(getattr(run, name)), expected)


if __name__ == '__main__':
    unittest.main()
