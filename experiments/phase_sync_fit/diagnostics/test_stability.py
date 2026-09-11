import unittest

import torch

from experiments.phase_sync_fit.diagnostics.stability import check, opposed_observations
from experiments.phase_sync.scan import MAX_GAIN, synchronize
from experiments.phase_sync.test_phase_sync import reference


class StabilityWitnessTests(unittest.TestCase):
    def test_positive_full_minute_can_overflow_float32_backward(self):
        result = check()
        for row in result['evidence'].values():
            self.assertTrue(row['forward_finite'])
            self.assertTrue(row['clock_strictly_increasing'])
            self.assertAlmostEqual(row['min_bpm'], 120., places=5)
            self.assertAlmostEqual(row['max_bpm'], 120., places=5)
            self.assertGreater(row['local_state_jacobian_min'], 1.)
            self.assertGreater(row['analytical_origin_gradient_log10'], result['float32_max_log10'])
        a = result['evidence']['float64']
        self.assertIsNone(a['backward_error'])
        self.assertAlmostEqual(a['observed_origin_gradient_log10'], a['analytical_origin_gradient_log10'], places=8)
        self.assertEqual(result['evidence']['float32']['backward_error'], 'nonfinite clock gradient')
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertFalse(result['failed_music_state_reproduced'])

    def test_short_opposed_case_agrees_with_plain_torch_autograd(self):
        args = (torch.full((1, 8), -1., dtype=torch.float64, requires_grad=True),
                torch.from_numpy(opposed_observations(8)).double().requires_grad_(),
                torch.zeros(1, dtype=torch.float64, requires_grad=True),
                torch.tensor(MAX_GAIN / 2, dtype=torch.float64, requires_grad=True))
        custom, plain = synchronize(*args).cycles, reference(*args)
        torch.testing.assert_close(custom, plain, atol=1e-13, rtol=1e-13)
        a = torch.autograd.grad(custom[:, -1].sum(), args)
        b = torch.autograd.grad(plain[:, -1].sum(), args)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
