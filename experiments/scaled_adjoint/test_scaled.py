"""Numerical contracts only: authored inputs and tiny synthetic optimizer steps."""
import copy
from decimal import Decimal, localcontext
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.coupled_clock.supervision import clock_loss
from experiments.phase_sync.model import PhaseSyncReadout
from experiments.phase_sync.test_phase_sync import fixture, reference
from experiments.phase_sync_fit.diagnostics.stability import opposed_observations
from experiments.phase_sync.scan import synchronize
from experiments.training_observer.phase_trace import traced_synchronize
from experiments.scaled_adjoint.scaled import Gradient, accumulate
from experiments.scaled_adjoint.scan import scaled_vjp
from experiments.scaled_adjoint.transport import record_gradient, install_clipped

torch.set_num_threads(1)


def materialize(gradient):
    return {name: np.ldexp(x, gradient.exponent) for name, x in gradient.blocks.items()}


class ScaleTests(unittest.TestCase):
    def test_global_clip_matches_torch_including_epsilon_and_unclipped_region(self):
        for scale in (0., 1e-12, .01, 1., 1000., 1e100):
            with self.subTest(scale=scale):
                arrays = dict(a=np.array([1., -2., 3.]) * scale, b=np.array(.123) * scale)
                tensors = [torch.nn.Parameter(torch.zeros(x.shape, dtype=torch.float64)) for x in arrays.values()]
                for p, x in zip(tensors, arrays.values()):
                    p.grad = torch.from_numpy(np.asarray(x).copy())
                torch.nn.utils.clip_grad_norm_(tensors, 1., error_if_nonfinite=True)
                clipped = Gradient.create(arrays).clipped()
                for value, p in zip(clipped.values(), tensors):
                    np.testing.assert_allclose(value, p.grad.numpy(), rtol=3e-15, atol=1e-300)

    def test_huge_exponents_clip_without_materializing_norm(self):
        gradient = Gradient.create(dict(a=np.array([3., 4.])), exponent=10000)
        np.testing.assert_allclose(gradient.clipped()['a'], [.6, .8], rtol=1e-15)
        self.assertAlmostEqual(gradient.norm_log2(), 10000 + math.log2(5))

    def test_weighted_records_are_combined_before_single_clip(self):
        a = Gradient.create(dict(p=np.array([10., 0.]))).multiply(.2)
        b = Gradient.create(dict(p=np.array([-1., 2.]))).multiply(.8)
        combined = accumulate([a, b]).clipped()['p']
        expected = np.array([1.2, 1.6]) / (2. + 1e-6)
        np.testing.assert_allclose(combined, expected, rtol=1e-15)
        self.assertGreater(np.linalg.norm(combined - (a.clipped()['p'] + b.clipped()['p'])), .1)

    def test_cancellation_zero_subnormal_and_alignment_loss_are_explicit(self):
        positive = Gradient.create(dict(p=np.array([.75, -.5])), exponent=1000)
        negative = positive.multiply(-1.)
        zero = accumulate([positive, negative])
        self.assertIsNone(zero.norm_log2())
        np.testing.assert_array_equal(zero.clipped()['p'], [0., 0.])
        small = Gradient.create(dict(p=np.array([np.nextafter(0., 1.)])))
        np.testing.assert_array_equal(small.clipped()['p'], [np.nextafter(0., 1.)])
        lost = accumulate([Gradient.create(dict(p=np.array([1.])), exponent=2000), small])
        self.assertGreater(lost.alignment_underflows, 0)

    def test_invalid_shapes_values_weights_and_clip_fail(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                Gradient.create(dict(p=np.array([value])))
            with self.assertRaises(ValueError):
                Gradient.create(dict(p=np.array([1.]))).multiply(value)
        with self.assertRaises(ValueError):
            accumulate([Gradient.create(dict(p=np.ones(2))), Gradient.create(dict(p=np.ones(3)))])
        with self.assertRaises(ValueError):
            Gradient.create(dict(p=np.ones(2))).clipped(-1)

    def test_ownership_normalization_and_weight_underflow(self):
        original = np.array([.75, np.nextafter(0., 1.)])
        packet = Gradient.create(dict(p=original))
        original[:] = 0
        self.assertEqual(packet.blocks['p'][0], .75)
        with self.assertRaises(ValueError):
            packet.blocks['p'][0] = 0
        with self.assertRaises(TypeError):
            packet.blocks['q'] = np.ones(2)
        for kwargs in (dict(exponent=0), dict(exponent=1, alignment_underflows=-1)):
            with self.assertRaises(ValueError):
                Gradient(dict(p=np.array([2.])), **kwargs)
        self.assertEqual(packet.multiply(1.).alignment_underflows, 0)
        np.testing.assert_array_equal(packet.multiply(1.).blocks['p'], packet.blocks['p'])

    def test_extreme_finite_clip_contract_does_not_overflow_denominator(self):
        largest = np.finfo(np.float64).max
        value = Gradient.create(dict(p=np.array([largest])))
        np.testing.assert_allclose(value.clipped(largest, largest)['p'], [largest / 2], rtol=1e-15)
        tiny = np.nextafter(0., 1.)
        np.testing.assert_array_equal(Gradient.create(dict(p=np.array([1.]))).clipped(tiny, 0)['p'], [tiny])
        np.testing.assert_array_equal(Gradient.create(dict(p=np.array([tiny]))).clipped(tiny, largest)['p'], [0.])
        small = Gradient.create(dict(p=np.array([.1]))).clipped(.01, 1e308)['p'][0]
        expected = .1 * (.01 / (.1 + 1e308))
        self.assertGreater(small, 0)
        self.assertLess(abs(small / expected - 1), 2e-12)


class AdjointTests(unittest.TestCase):
    def test_scaled_vjp_matches_independent_plain_torch_unroll(self):
        args = fixture(31, gradients=True)
        owner = SimpleNamespace(last_trace=None)
        actual = traced_synchronize(owner, *args).cycles
        upstream = torch.linspace(-2., 3., actual.numel(), dtype=torch.float64).reshape(actual.shape)
        expected = torch.autograd.grad(reference(*args), args, upstream)
        jacobian = owner.last_trace['jacobian'].numpy()
        before, original = jacobian.copy(), upstream.clone()
        gradient = scaled_vjp(jacobian, upstream.numpy())
        self.assertEqual(gradient.alignment_underflows, 0)
        for value, reference_value in zip(materialize(gradient).values(), expected):
            np.testing.assert_allclose(value, reference_value.numpy(), rtol=3e-12, atol=3e-12)
        np.testing.assert_array_equal(jacobian, before)
        torch.testing.assert_close(upstream, original, rtol=0, atol=0)

    def test_long_finite_opposed_clock_matches_decimal_origin_gradient(self):
        cells = 3000
        owner = SimpleNamespace(last_trace=None)
        args = (torch.full((1, cells), -1., requires_grad=True),
                torch.from_numpy(opposed_observations()).requires_grad_(),
                torch.zeros(1, requires_grad=True), torch.tensor(math.log(2.) / 2, dtype=torch.float64, requires_grad=True))
        clock = traced_synchronize(owner, *args)
        upstream = np.zeros((1, cells + 1)); upstream[:, -1] = 1.
        with self.assertRaisesRegex(ValueError, 'nonfinite clock gradient'):
            clock.cycles[:, -1].sum().backward()
        jac = owner.last_trace['jacobian'].numpy()
        gradient = scaled_vjp(jac, upstream)
        self.assertEqual(gradient.alignment_underflows, 0)
        self.assertGreater(gradient.norm_log2(), 128)
        with localcontext() as context:
            context.prec = 80
            exact = Decimal(1)
            for value in jac[0, 0]:
                exact *= Decimal.from_float(float(value))
            recovered = Decimal.from_float(float(gradient.blocks['initial'][0])) * Decimal(2) ** gradient.exponent
            self.assertLess(abs((recovered - exact) / exact), Decimal('1e-12'))
        clipped = gradient.clipped()
        self.assertTrue(all(np.isfinite(x).all() for x in clipped.values()))
        self.assertAlmostEqual(math.sqrt(sum(float(np.sum(x*x)) for x in clipped.values())), 1., places=13)

    def test_adjoint_range_can_exceed_float64_without_changing_the_recurrence(self):
        cells = 1500
        jac = np.zeros((5, 1, cells)); jac[0] = 2.; jac[1, 0, 0] = 1.
        upstream = np.zeros((1, cells + 1)); upstream[:, -1] = 1.
        gradient = scaled_vjp(jac, upstream)
        self.assertGreater(gradient.exponent, 1024)
        self.assertEqual(gradient.alignment_underflows, 0)
        clipped = gradient.clipped()
        self.assertAlmostEqual(clipped['initial'][0], 2 / math.sqrt(5), places=14)
        self.assertAlmostEqual(clipped['periods'][0, 0], 1 / math.sqrt(5), places=14)

    def test_zero_and_nonfinite_upstream_and_geometry(self):
        jac = np.ones((5, 2, 9))
        zero = scaled_vjp(jac, np.zeros((2, 10)))
        self.assertIsNone(zero.norm_log2())
        for upstream in (np.zeros((2, 9)), np.full((2, 10), float('nan'))):
            with self.assertRaises(ValueError):
                scaled_vjp(jac, upstream)


class NeuralTransportTests(unittest.TestCase):
    def compare(self, device, dtype):
        torch.manual_seed(142)
        model = PhaseSyncReadout().to(device=device, dtype=dtype)
        plain = copy.deepcopy(model)
        features = torch.randn(1, 101, 512, device=device, dtype=dtype) * .1
        target = torch.arange(101, device=device, dtype=torch.float64)[None] * .04
        valid = torch.ones_like(target, dtype=torch.bool)
        transported = []
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        plain.zero_grad(set_to_none=True)
        for weight, x in ((.3, features), (.7, features * .7)):
            prediction = plain(x)
            loss = clock_loss(prediction, target, valid)['total']
            (loss * weight).backward()
            packet = record_gradient(model, x, target, valid)
            torch.testing.assert_close(packet['clock'], prediction.cycles.detach(), rtol=0, atol=0)
            self.assertEqual(packet['loss'], float(loss.detach()))
            transported.append(packet['gradient'].multiply(weight))
        torch.nn.utils.clip_grad_norm_(plain.parameters(), 1., error_if_nonfinite=True)
        install_clipped(model, accumulate(transported))
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        for a, b in zip(model.parameters(), plain.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5 if dtype == torch.float32 else 2e-11,
                                       atol=2e-8 if dtype == torch.float32 else 2e-12)
        expected_updates = []
        for network in (model, plain):
            # First AdamW step with zero moments has this closed form. A fixed
            # cross-model weight tolerance hides the g/(abs(g)+eps) sensitivity
            # near zero: compare each update to its OWN unchanged input gradient.
            expected_updates.append([(p.detach().double() * (1 - .001 * .0001)
                                      - .001 * p.grad.double() / (p.grad.double().abs() + 1e-8))
                                     for p in network.parameters()])
        aopt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        bopt = torch.optim.AdamW(plain.parameters(), lr=.001, weight_decay=.0001)
        aopt.step(); bopt.step()  # One authored optimizer check, never a musical fit.
        for network, expected in zip((model, plain), expected_updates):
            for (name, parameter), value in zip(network.named_parameters(), expected):
                # Eight dtype epsilons of the update's operand magnitude cover
                # native moment/division/weight-rounding, not gradient mismatch.
                bound = 8 * torch.finfo(dtype).eps * (before[name].double().abs() + .001)
                self.assertTrue(torch.all((parameter.detach().double() - value).abs() <= bound).item())

    def test_native_cnn_batch_clipping_and_adamw_match_reference_float32(self):
        self.compare('cpu', torch.float32)

    def test_native_cnn_batch_clipping_and_adamw_match_reference_float64(self):
        self.compare('cpu', torch.float64)

    def compare_overflow(self, device):
        class AffineFields(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(4, dtype=dtype))
                self.gain_logit = torch.nn.Parameter(torch.tensor(0., dtype=dtype))
                fields = torch.zeros(1, 3001, 4, dtype=torch.float32)
                fields[:, :, 0] = -1
                fields[:, 1:, 1:3] = torch.from_numpy(opposed_observations())
                self.register_buffer('authored', fields.to(dtype))

            def fields(self, unused):
                return self.authored + self.bias

        def last_cycle(clock, unused_reference, unused_valid):
            return dict(total=clock.cycles[:, -1].sum())

        def forward(model):
            fields = model.fields(None)
            return synchronize(fields[:, :-1, 0], fields[:, 1:, 1:3], fields[:, 0, 3],
                               math.log(2.) * torch.sigmoid(model.gain_logit.double()))

        model, plain = AffineFields(torch.float32).to(device), AffineFields(torch.float64).to(device)
        with self.assertRaisesRegex(ValueError, 'nonfinite clock gradient'):
            forward(model).cycles[:, -1].sum().backward()
        model.zero_grad(set_to_none=True)
        expected = forward(plain)
        expected.cycles[:, -1].sum().backward()
        self.assertGreater(float(plain.bias.grad.abs().max()), 1e70)
        torch.nn.utils.clip_grad_norm_(plain.parameters(), 1., error_if_nonfinite=True)
        packet = record_gradient(model, None, None, None, loss_fn=last_cycle)
        self.assertGreaterEqual(packet['neural_vjp_passes'], 4)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        torch.testing.assert_close(packet['clock'], expected.cycles.detach(), rtol=0, atol=0)
        install_clipped(model, packet['gradient'])
        for actual, reference_parameter in zip(model.parameters(), plain.parameters()):
            torch.testing.assert_close(actual.grad.double(), reference_parameter.grad, rtol=2e-6, atol=2e-8)

    def test_native_parameter_vjp_survives_opposed_clock_overflow(self):
        self.compare_overflow('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_native_cuda_parameter_vjp_survives_opposed_clock_overflow(self):
        self.compare_overflow('cuda')

    def test_zero_clock_loss_installs_zero_gradients_without_a_neural_vjp(self):
        torch.manual_seed(142)
        model = PhaseSyncReadout()
        features = torch.randn(1, 21, 512) * .1
        packet = record_gradient(model, features, None, None,
                                 loss_fn=lambda clock, *_: dict(total=clock.cycles.sum() * 0))
        self.assertEqual(packet['neural_vjp_passes'], 0)
        self.assertIsNone(packet['gradient'].norm_log2())
        install_clipped(model, packet['gradient'])
        self.assertTrue(all(torch.count_nonzero(p.grad).item() == 0 for p in model.parameters()))

    def test_transport_budget_and_failed_install_leave_parameters_and_grads_untouched(self):
        torch.manual_seed(142)
        model = PhaseSyncReadout()
        features = torch.randn(1, 21, 512) * .1
        target = torch.arange(21, dtype=torch.float64)[None] * .04
        valid = torch.ones_like(target, dtype=torch.bool)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        for p in model.parameters():
            p.grad = torch.full_like(p, .125)
        with patch('experiments.scaled_adjoint.transport.MAX_BANDS', 0):
            with self.assertRaisesRegex(ValueError, 'band budget'):
                record_gradient(model, features, target, valid)
        values = {name: np.ones(tuple(p.shape)) for name, p in model.named_parameters()}
        last = next(reversed(values))
        values[last] = np.ones(3)  # Fail after earlier parameter conversions succeeded.
        with self.assertRaisesRegex(ValueError, 'shape differs'):
            install_clipped(model, Gradient.create(values))
        with self.assertRaisesRegex(ValueError, 'lost nonzero'):
            install_clipped(model, Gradient.create(values, alignment_underflows=1))
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            torch.testing.assert_close(p.grad, torch.full_like(p, .125), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_native_cuda_cnn_and_adamw_match_reference(self):
        self.compare('cuda', torch.float32)


if __name__ == '__main__':
    unittest.main()
