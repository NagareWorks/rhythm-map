import math
import unittest

import torch
from torch.autograd import gradcheck

from experiments.coupled_clock.clock import integrate
from experiments.coupled_clock.supervision import clock_loss
from experiments.phase_sync.model import PARAMETERS, PhaseSyncReadout
from experiments.phase_sync.scan import MAX_GAIN, synchronize

torch.set_num_threads(1)


def reference(period, observation, initial, gain, fps=50.):
    """Independent ordinary-Torch unroll: no analytic Jacobians reused."""
    q = initial.double()
    states = [q]
    for t in range(period.shape[1]):
        advance = torch.exp2(-period[:, t].double()) / fps
        predicted = q + advance
        vector = observation[:, t].double()
        normalized = vector / torch.sqrt(1 + vector.square().sum(-1, keepdim=True))
        angle = 2 * math.pi * torch.remainder(predicted, 1.)
        error = normalized[:, 1] * angle.cos() - normalized[:, 0] * angle.sin()
        q = q + advance * torch.exp(gain.double() * error)
        states.append(q)
    return torch.stack(states, dim=1)


def phase(cycles, amplitude=3.):
    return amplitude * torch.stack(((2 * math.pi * cycles).cos(),
                                    (2 * math.pi * cycles).sin()), -1)


def circular_error(actual, expected):
    return torch.abs(torch.remainder(actual - expected + .5, 1.) - .5)


def fixture(cells=12, batch=2, dtype=torch.float64, gradients=False):
    generator = torch.Generator().manual_seed(321)
    values = (torch.randn(batch, cells, generator=generator, dtype=dtype) * .2 - .7,
              torch.randn(batch, cells, 2, generator=generator, dtype=dtype),
              torch.full((batch,), .137, dtype=dtype),
              torch.tensor(.4, dtype=dtype))
    return tuple(v.requires_grad_(gradients) for v in values)


class PhaseScanTests(unittest.TestCase):
    def test_finite_difference_all_inputs(self):
        self.assertTrue(gradcheck(lambda *x: synchronize(*x).cycles,
                                 fixture(4, gradients=True), atol=2e-6, rtol=2e-5))

    def test_reference_forward_and_backward(self):
        args = fixture(31, gradients=True)
        actual = synchronize(*args).cycles
        expected = reference(*args)
        weights = torch.arange(actual.numel(), dtype=torch.float64).reshape(actual.shape) / 10
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual_grad = torch.autograd.grad((actual * weights).sum(), args)
        expected_grad = torch.autograd.grad((expected * weights).sum(), args)
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-11)

    def test_incoming_gradient_is_not_mutated(self):
        args = fixture(7, gradients=True)
        result = synchronize(*args).cycles
        upstream = torch.randn_like(result)
        saved = upstream.clone()
        torch.autograd.grad(result, args, upstream)
        torch.testing.assert_close(upstream, saved, atol=0, rtol=0)

    def test_second_derivative_explicitly_unsupported(self):
        args = fixture(3, gradients=True)
        result = synchronize(*args).cycles
        upstream = torch.ones_like(result, requires_grad=True)
        first, = torch.autograd.grad(result, args[0], upstream, create_graph=True)
        with self.assertRaisesRegex(RuntimeError, 'once_differentiable'):
            first.sum().backward()

    def test_exact_constant_clock_and_returned_tempo(self):
        cells = 3000
        truth = torch.arange(cells + 1, dtype=torch.float64)[None] * .04 + .1
        period = torch.full((1, cells), -1., dtype=torch.float64)
        result = synchronize(period, phase(truth[:, 1:]), truth[:, 0], torch.tensor(.5))
        torch.testing.assert_close(result.cycles, truth, atol=5e-12, rtol=0)
        torch.testing.assert_close(result.cell_bpm, torch.full_like(period, 120.),
                                   atol=5e-8, rtol=0)

    def test_exact_gradual_change(self):
        prior = torch.linspace(-.8, -1.5, 600, dtype=torch.float64)[None]
        truth = integrate(prior, torch.tensor([.17])).cycles
        result = synchronize(prior, phase(truth[:, 1:]), truth[:, 0], torch.tensor(.5))
        torch.testing.assert_close(result.cycles, truth, atol=2e-12, rtol=0)

    def test_abrupt_change_is_not_forced_smooth(self):
        prior = torch.tensor([[-1.] * 150 + [0.] * 150 + [-1.] * 150], dtype=torch.float64)
        truth = integrate(prior, torch.tensor([.17])).cycles
        result = synchronize(prior, phase(truth[:, 1:]), truth[:, 0], torch.tensor(.5))
        torch.testing.assert_close(result.log2_period, prior, atol=3e-12, rtol=0)

    def test_dense_evidence_bounds_drift_from_biased_rate(self):
        cells = 3000
        truth = torch.arange(cells + 1, dtype=torch.float64)[None] * .04
        prior = torch.full((1, cells), -math.log2(1.8), dtype=torch.float64)
        result = synchronize(prior, phase(truth[:, 1:]), truth[:, 0], torch.tensor(.5))
        naive = integrate(prior, truth[:, 0])
        self.assertLess((result.cycles - truth).abs().max().item(), .1)
        self.assertGreater((naive.cycles - truth).abs().max().item(), 11.)
        # Feedback bounds, but does not eliminate, the steady-state phase error.
        self.assertGreater(abs((result.cycles - truth)[0, -1].item()), .01)

    def test_exact_rate_coasts_through_zero_observation_gap(self):
        cells = 1000
        prior = torch.full((1, cells), -1., dtype=torch.float64)
        truth = integrate(prior, torch.tensor([.13])).cycles
        observation = phase(truth[:, 1:])
        observation[:, 100:800] = 0
        result = synchronize(prior, observation, truth[:, 0], torch.tensor(.5))
        torch.testing.assert_close(result.cycles, truth, atol=2e-12, rtol=0)

    def test_wrong_rate_drifts_in_gap_and_phase_cannot_restore_count(self):
        cells = 1800
        prior = torch.full((1, cells), -math.log2(1.8), dtype=torch.float64)
        truth = torch.arange(cells + 1, dtype=torch.float64)[None] * .04
        observation = phase(truth[:, 1:])
        observation[:, 300:750] = 0  # 9 s * 0.2 cycles/s = 1.8 lost cycles.
        result = synchronize(prior, observation, truth[:, 0], torch.tensor(.5))
        delta = result.cycles.diff(dim=1)
        torch.testing.assert_close(delta[:, 300:750], torch.full_like(delta[:, 300:750], .036))
        error = result.cycles - truth
        self.assertGreater(abs(error[0, 750].item()), 1.7)
        self.assertLess(circular_error(result.cycles[:, -1], truth[:, -1]).item(), .1)
        self.assertGreater(abs(error[0, -1].item()), 1.8)  # No oracle cycle repair.
        self.assertLessEqual(delta.max().item(), .036 * 2)

    def test_correct_clock_with_phase_only_octave_witness(self):
        # Only coincident whole-second phase observations: both 60 and 120 BPM
        # have phase zero there. A reference count/rate is needed to distinguish.
        observations = torch.zeros(1, 200, 2, dtype=torch.float64)
        observations[:, 49::50, 0] = 3
        clocks = [synchronize(torch.full((1, 200), p, dtype=torch.float64), observations,
                              torch.zeros(1), torch.tensor(.5)) for p in (0., -1.)]
        self.assertAlmostEqual(clocks[0].cycles[0, -1].item(), 4., places=10)
        self.assertAlmostEqual(clocks[1].cycles[0, -1].item(), 8., places=10)
        torch.testing.assert_close(clocks[0].phase_vector[:, 50::50],
                                   clocks[1].phase_vector[:, 50::50], atol=2e-11, rtol=0)

    def test_late_evidence_changes_future_not_past(self):
        args = fixture(40, batch=1, gradients=True)
        result = synchronize(*args).cycles
        modified = args[1].detach().clone()
        modified[:, 23, 0] += .7
        altered = synchronize(args[0], modified, args[2], args[3]).cycles
        torch.testing.assert_close(result[:, :24], altered[:, :24], atol=0, rtol=0)
        self.assertGreater((result[:, 24:] - altered[:, 24:]).abs().max().item(), 1e-5)
        derivative, = torch.autograd.grad(result[:, 24].sum(), args[1])
        self.assertGreater(derivative[:, 23].abs().sum().item(), 1e-6)
        self.assertEqual(derivative[:, 24:].abs().sum().item(), 0.)

    def test_zero_gain_is_integration_and_zero_observation_gradient(self):
        period, obs, initial, _ = fixture(40, gradients=True)
        gain = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        result = synchronize(period, obs, initial, gain)
        torch.testing.assert_close(result.cycles, integrate(period, initial).cycles,
                                   atol=3e-14, rtol=0)
        gradient, = torch.autograd.grad(result.cycles.sum(), obs)
        self.assertEqual(gradient.abs().sum().item(), 0.)

    def test_zero_vectors_are_neutral_but_learnable(self):
        period, obs, initial, gain = fixture(40)
        obs = torch.zeros_like(obs, requires_grad=True)
        result = synchronize(period, obs, initial, gain)
        torch.testing.assert_close(result.cycles, integrate(period, initial).cycles)
        gradient, = torch.autograd.grad(result.cycles.sum(), obs)
        self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_corrected_rate_is_derived_from_clock(self):
        args = fixture(80)
        result = synchronize(*args)
        delta = result.cycles.diff(dim=1)
        torch.testing.assert_close(result.cell_bpm, 60 * 50 * delta, atol=1e-11, rtol=1e-12)
        prior_delta = torch.exp2(-args[0]) / 50
        self.assertTrue((delta > .5 * prior_delta).all())
        self.assertTrue((delta < 2 * prior_delta).all())
        self.assertGreater((result.log2_period - args[0]).abs().max().item(), .01)

    def test_state_carry_chunk_forward_and_gradients(self):
        args = fixture(87, gradients=True)
        whole = synchronize(*args).cycles
        origin, pieces = args[2], []
        for start, end in ((0, 13), (13, 52), (52, 87)):
            chunk = synchronize(args[0][:, start:end], args[1][:, start:end], origin, args[3])
            pieces.append(chunk.cycles if start == 0 else chunk.cycles[:, 1:])
            origin = chunk.cycles[:, -1]
        joined = torch.cat(pieces, 1)
        torch.testing.assert_close(joined, whole, atol=0, rtol=0)
        weights = torch.arange(whole.shape[1], dtype=torch.float64)
        a = torch.autograd.grad((whole * weights).sum(), args)
        b = torch.autograd.grad((joined * weights).sum(), args)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=2e-10, rtol=2e-12)

    def test_single_cell_noncontiguous_and_mixed_dtype(self):
        p, o, q, k = fixture(2, dtype=torch.float32)
        p, o = p[:, ::2].requires_grad_(), o[:, ::2].double().requires_grad_()
        q.requires_grad_()
        k.requires_grad_()
        result = synchronize(p, o, q, k)
        self.assertEqual(result.cycles.dtype, torch.float64)
        result.cycles.sum().backward()
        for value in (p, o, q, k):
            self.assertEqual(value.grad.dtype, value.dtype)
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_large_phase_vectors_normalize_without_squaring_overflow(self):
        p, o, q, k = fixture(5)
        o[:] = 1e200
        result = synchronize(p, o, q, k)
        self.assertTrue(torch.isfinite(result.cycles).all())

    def test_invalid_geometry_and_dtype(self):
        p, o, q, k = fixture()
        for args in ((p[0], o, q, k), (p[:, :0], o[:, :0], q, k),
                     (p, o[..., 0], q, k), (p, o, q[:, None], k),
                     (p, o, q, k[None]), (p.int(), o, q, k), (None, o, q, k)):
            with self.subTest(args=[getattr(v, 'shape', None) for v in args]):
                with self.assertRaises(ValueError):
                    synchronize(*args)

    def test_invalid_fps_gain_and_nonfinite_values(self):
        args = fixture()
        for fps in (0, -1, True, float('nan'), float('inf'), '50'):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                synchronize(*args, fps=fps)
        for gain in (-.01, MAX_GAIN + .001):
            with self.assertRaisesRegex(ValueError, 'gain outside'):
                synchronize(*args[:3], torch.tensor(gain))
        for index in range(4):
            damaged = [v.clone() for v in args]
            damaged[index].fill_(float('nan'))
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                synchronize(*damaged)

    def test_unrepresentable_numerics_fail_without_clamping(self):
        p, o, q, k = fixture(2)
        for period in (-2000., 2000.):
            with self.assertRaisesRegex(ValueError, 'unrepresentable'):
                synchronize(torch.full_like(p, period), o, q, k)
        with self.assertRaisesRegex(ValueError, 'accumulated|precision'):
            synchronize(p, o, torch.full_like(q, 1e17), k)
        with self.assertRaisesRegex(ValueError, 'observation norm'):
            synchronize(p, torch.full_like(o, 1.7e308), q, k)

    def test_nonfinite_gradient_rejected(self):
        args = fixture(3, gradients=True)
        output = synchronize(*args).cycles
        with self.assertRaisesRegex(ValueError, 'nonfinite clock gradient'):
            output.backward(torch.full_like(output, float('inf')))

    def test_native_count_loss_can_reject_phase_compatible_octave(self):
        reference_cycles = torch.arange(251, dtype=torch.float64)[None] * .02
        period = torch.full((1, 250), -1., dtype=torch.float64, requires_grad=True)
        # No phase evidence: two advances per second versus one native beat/s.
        predicted = synchronize(period, torch.zeros(1, 250, 2), torch.zeros(1), torch.tensor(.5))
        valid = torch.ones_like(reference_cycles, dtype=torch.bool)
        valid[:, 100:150] = False  # Missing annotation is not an observation gap.
        masked_reference = reference_cycles.clone()
        masked_reference[~valid] = float('nan')
        losses = clock_loss(predicted, masked_reference, valid)
        self.assertAlmostEqual(losses['advance'].item(), 1., places=10)
        self.assertEqual(losses['pairs_by_lag'][200], 0)
        gradient, = torch.autograd.grad(losses['advance'], period)
        self.assertLess(gradient.sum().item(), 0.)  # Descent increases period.

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_forward_backward_and_device_mismatch(self):
        cpu = fixture(23, gradients=True)
        gpu = tuple(v.detach().cuda().requires_grad_() for v in cpu)
        a, b = synchronize(*cpu).cycles, synchronize(*gpu).cycles
        torch.testing.assert_close(a, b.cpu(), atol=0, rtol=0)
        a.sum().backward()
        b.sum().backward()
        for x, y in zip(cpu, gpu):
            torch.testing.assert_close(x.grad, y.grad.cpu(), atol=1e-12, rtol=1e-12)
        with self.assertRaisesRegex(ValueError, 'devices differ'):
            synchronize(gpu[0], cpu[1], gpu[2], gpu[3])


class PhaseModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(777)
        self.model = PhaseSyncReadout().double()
        self.features = torch.randn(1, 53, 512, dtype=torch.float64) * .1

    def test_parameter_count_and_origin_initialization(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), PARAMETERS)
        self.assertEqual(PARAMETERS, 24037)
        self.assertEqual(self.model.origin.weight.abs().sum().item(), 0.)
        self.assertEqual(self.model.origin.bias.abs().sum().item(), 0.)

    def test_convolution_chunk_fields_outputs_and_gradients(self):
        whole = self.model(self.features)
        chunk = self.model(self.features, chunk_frames=17)
        torch.testing.assert_close(self.model.fields(self.features),
                                   self.model.fields(self.features, 17), atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(whole.cycles, chunk.cycles, atol=2e-13, rtol=1e-13)
        loss_a = whole.phase_vector.square().sum() + whole.log2_period.square().mean()
        loss_b = chunk.phase_vector.square().sum() + chunk.log2_period.square().mean()
        a = torch.autograd.grad(loss_a, tuple(self.model.parameters()))
        b = torch.autograd.grad(loss_b, tuple(self.model.parameters()))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=2e-11, rtol=2e-10)

    def test_all_neural_evidence_heads_receive_clock_gradient(self):
        clock = self.model(self.features)
        (clock.cycles.square().mean() + clock.log2_period.square().mean()).backward()
        for channel in range(3):
            self.assertGreater(self.model.output.weight.grad[channel].abs().sum().item(), 1e-10)
        self.assertGreater(self.model.origin.weight.grad.abs().sum().item(), 1e-10)
        self.assertGreater(self.model.gain_logit.grad.abs().item(), 1e-10)
        for parameter in self.model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_existing_native_supervision_reaches_distributed_phase(self):
        clock = self.model(self.features)
        reference_cycles = torch.arange(clock.cycles.shape[1], dtype=torch.float64)[None] * .04
        valid = torch.ones_like(reference_cycles, dtype=torch.bool)
        clock_loss(clock, reference_cycles, valid)['total'].backward()
        self.assertGreater(self.model.output.weight.grad[1:3].abs().sum().item(), 1e-10)
        self.assertGreater(self.model.gain_logit.grad.abs().item(), 1e-10)

    def test_invalid_feature_and_chunk_geometry(self):
        for features in (self.features[:, :1], self.features[..., :511], self.features[0],
                         torch.full_like(self.features, float('nan'))):
            with self.assertRaises(ValueError):
                self.model(features)
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.model(self.features, size)


if __name__ == '__main__':
    unittest.main()
