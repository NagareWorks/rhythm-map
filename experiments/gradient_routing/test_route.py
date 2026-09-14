import math
import unittest

import numpy as np
import torch

from experiments.coupled_clock.clock import integrate
from experiments.coupled_clock.run import state_hash
from experiments.gradient_routing.authored import adamw_witness, learnability, record_terms, teacher_case
from experiments.gradient_routing.route import GROUPS, displacement_check, model_shapes, partition, project, route, route_batch
from experiments.phase_sync.scan import synchronize
from experiments.scaled_adjoint.scaled import Gradient, accumulate

torch.set_num_threads(2)


def vector(values, exponent=0):
    return Gradient.create({'x': np.asarray(values, dtype=np.float64)}, exponent)


def raw(gradient):
    return {n: np.ldexp(v, gradient.exponent) for n, v in gradient.blocks.items()}


def model_gradient(seed):
    generator = np.random.default_rng(seed)
    return Gradient.create({n: generator.normal(size=s) for n, s in model_shapes().items()})


class ProjectionTests(unittest.TestCase):
    def test_dense_independent_projection(self):
        generator = np.random.default_rng(223)
        for _ in range(20):
            p, c = generator.normal(size=(2, 9))
            expected = c + p - min(0., p @ c / (c @ c)) * c
            actual, _, _ = project(vector(p), vector(c))
            np.testing.assert_allclose(raw(actual)['x'], expected, atol=2e-14, rtol=2e-14)

    def test_aligned_orthogonal_and_complete_opposition(self):
        for p, expected in (([2, 0], [3, 0]), ([0, 2], [1, 2]), ([-4, 0], [1, 0])):
            actual, retained, _ = project(vector(p), vector([1, 0]))
            np.testing.assert_array_equal(raw(actual)['x'], expected)
            if p == [-4, 0]:
                self.assertIsNone(retained.norm_log2())

    def test_zero_count_and_phase_are_explicit(self):
        for p, c in (([2, 0], [0, 0]), ([0, 0], [1, 0]), ([0, 0], [0, 0])):
            actual, _, receipt = project(vector(p), vector(c))
            np.testing.assert_array_equal(raw(actual)['x'], np.array(p) + c)
            self.assertEqual(receipt['count_direction_defined'], any(c))
            if not any(c):
                self.assertIsNone(receipt['routed_count_cosine'])

    def test_exponents_beyond_native_range(self):
        for exponent in (-2000, 2000):
            actual, _, receipt = project(vector([-4, 2], exponent), vector([1, 0], exponent))
            self.assertAlmostEqual(actual.norm_log2() - exponent, .5 * math.log2(5))
            self.assertAlmostEqual(receipt['count_projection_ratio_log2'], 0.)

    def test_lost_input_and_output_alignment_fail_closed(self):
        bad = Gradient.create({'x': np.array([1., 0.])}, alignment_underflows=1)
        with self.assertRaisesRegex(ValueError, 'input alignment'):
            project(bad, vector([1, 0]))
        with self.assertRaisesRegex(ValueError, 'output alignment'):
            project(vector([0, 1], 2000), vector([1, 0]))

    def test_no_mutation_and_geometry_rejection(self):
        phase, count = vector([-4, 2]), vector([1, 0])
        saved = {n: x.copy() for n, x in phase.blocks.items()}
        project(phase, count)
        np.testing.assert_array_equal(phase.blocks['x'], saved['x'])
        with self.assertRaisesRegex(ValueError, 'geometry'):
            project(vector([1, 2, 3]), count)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            route(phase, count)

    def test_partition_exhaustive_disjoint_and_specific_output_rows(self):
        original = model_gradient(2)
        parts = partition(original)
        recovered = accumulate(list(parts.values()))
        for n, value in raw(original).items():
            np.testing.assert_array_equal(raw(recovered)[n], value)
            self.assertTrue((sum((g.blocks[n] != 0).astype(int) for g in parts.values()) <= 1).all())
        self.assertTrue((parts['period_head'].blocks['output.weight'][1:] == 0).all())
        self.assertTrue((parts['phase_parameters'].blocks['output.weight'][0] == 0).all())
        self.assertEqual(parts['shared_cnn'].blocks['origin.bias'][0], 0.)

    def test_all_groups_protected_and_clipping_preserves_descent(self):
        count = model_gradient(3)
        phase = count.multiply(-4)
        actual, receipt = route(phase, count)
        self.assertEqual(set(receipt), set(GROUPS))
        for n, x in raw(count).items():
            np.testing.assert_allclose(raw(actual)[n], x, atol=1e-12)
        clipped = Gradient.create(actual.clipped(.1))
        for name, group in partition(clipped).items():
            c = partition(count)[name]
            self.assertGreater(sum(float(np.sum(group.blocks[n] * c.blocks[n])) for n in c.blocks), 0.)

    def test_aggregate_before_nonlinear_routing(self):
        a, b = model_gradient(4), model_gradient(5)
        records = [(.3, a.multiply(-3), a), (.7, b, b.multiply(-2))]
        actual, _ = route_batch(records)
        expected, _ = route(accumulate([p.multiply(w) for w, p, c in records]),
                            accumulate([c.multiply(w) for w, p, c in records]))
        wrong = accumulate([route(p, c)[0].multiply(w) for w, p, c in records])
        for n, x in raw(actual).items():
            np.testing.assert_array_equal(x, raw(expected)[n])
        self.assertGreater(np.linalg.norm(raw(actual)['output.weight'] - raw(wrong)['output.weight']), .1)
        with self.assertRaisesRegex(ValueError, 'weights'):
            route_batch([(0., a, b)])


class ClockMechanismTests(unittest.TestCase):
    def check_native(self, device):
        model, payload = teacher_case(device)
        before = state_hash(model)
        with torch.no_grad():
            forward = model(payload['features']).cycles.cpu()
        phase, count = record_terms(model, payload)
        actual, receipt = route(phase, count)
        self.assertEqual(before, state_hash(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with torch.no_grad():
            torch.testing.assert_close(model(payload['features']).cycles.cpu(), forward, atol=0, rtol=0)
        self.assertEqual(set(actual.blocks), set(dict(model.named_parameters())))
        for row in receipt.values():
            self.assertGreaterEqual(row['routed_count_cosine'], 0.)

    def test_native_cpu_full_cnn_clock_unchanged(self):
        self.check_native('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_native_cuda_full_cnn_clock_unchanged(self):
        self.check_native('cuda')

    def test_constant_step_ramp_gap_representability_unchanged(self):
        constant = torch.full((1, 401), -1., dtype=torch.float64)
        step = constant.clone()
        step[:, 200:] = -.4
        ramp = torch.linspace(-1., -.4, 401, dtype=torch.float64)[None]
        for name, period in (('constant', constant), ('step', step), ('ramp', ramp), ('gap', constant)):
            truth = integrate(period, torch.tensor([.137])).cycles
            angle = 2 * math.pi * truth[:, 1:]
            obs = 3 * torch.stack((angle.cos(), angle.sin()), -1)
            if name == 'gap':
                obs[:, 50:350] = 0
            clock = synchronize(period, obs, truth[:, 0], torch.tensor(.4))
            torch.testing.assert_close(clock.cycles, truth, atol=2e-11, rtol=0)
            torch.testing.assert_close(clock.log2_period, period, atol=2e-11, rtol=0)

    def test_fixed_authored_learning_reports_both_losses_without_gate_rewrite(self):
        result = learnability()
        self.assertEqual(result['updates'], 40)
        self.assertEqual(result['both_losses_reduced'], all(result['after'][k] < result['before'][k]
                                                          for k in ('phase', 'advance')))
        self.assertTrue(all(math.isfinite(x) for x in result['after'].values()))


class OptimizerBoundaryTests(unittest.TestCase):
    def test_adamw_first_step_preconditioner_can_reverse_count_descent(self):
        result = adamw_witness()
        self.assertGreater(result['raw_gradient']['routed_count_cosine'], 0)
        self.assertFalse(result['check']['first_order_nonincrease'])
        self.assertGreater(result['quadratic_count_loss_change'], 0)

    def test_adamw_retained_momentum_can_reverse_count_descent(self):
        result = adamw_witness(momentum=True)
        self.assertGreater(result['raw_gradient']['routed_count_cosine'], 0)
        self.assertFalse(result['check']['first_order_nonincrease'])
        self.assertGreater(result['quadratic_count_loss_change'], 0)

    def test_displacement_zero_count_and_native_rounding_distinct(self):
        before = {'x': np.array([1e8], dtype=np.float32)}
        after = {'x': before['x'] + np.float32(.1)}
        receipt = displacement_check(vector([1]), before, after)
        self.assertFalse(receipt['moved'])
        self.assertTrue(receipt['first_order_nonincrease'])
        receipt = displacement_check(vector([0]), before, {'x': before['x'] + 16})
        self.assertTrue(receipt['moved'])
        self.assertIsNone(receipt['first_order_nonincrease'])

    def test_zero_first_order_does_not_guarantee_finite_loss(self):
        before, after = {'x': np.array([0., 0.])}, {'x': np.array([0., 1.])}
        self.assertTrue(displacement_check(vector([1, 0]), before, after)['first_order_nonincrease'])
        self.assertGreater(float(after['x'] @ after['x']) / 2 + after['x'][0], 0.)


if __name__ == '__main__':
    unittest.main()
