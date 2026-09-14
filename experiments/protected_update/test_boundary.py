"""Authored optimizer and native-endpoint checks; no fitted checkpoints."""
import copy
from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.gradient_routing.route import GROUPS, model_shapes, partition
from experiments.phase_sync.model import PhaseSyncReadout
from experiments.protected_update.boundary import ProtectedAdamW, arrays, exact_sign, protect
from experiments.scaled_adjoint.scaled import Gradient

torch.set_num_threads(1)


def vector(values, name='projection.bias', exponent=0):
    blocks = {n: np.zeros(s) for n, s in model_shapes().items()}
    blocks[name].reshape(-1)[:len(values)] = values
    return Gradient.create(blocks, exponent)


def tree(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            tree(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            tree(test, x, y)
    else:
        test.assertEqual(a, b)


def setup(device='cpu', dtype=torch.float64, **options):
    model = PhaseSyncReadout().to(device=device, dtype=dtype)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    raw = torch.optim.AdamW(model.parameters(), **(dict(lr=.01, weight_decay=0.) | options))
    return model, raw, ProtectedAdamW(model, raw)


def install(model, values, name='projection.bias'):
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    dict(model.named_parameters())[name].grad.reshape(-1)[:len(values)] = torch.tensor(values)


class BoundaryTests(unittest.TestCase):
    def test_real_preconditioner_is_corrected_after_adamw(self):
        model, raw, boundary = setup()
        install(model, [-1., 7.])
        before = arrays(boundary.parameters)
        boundary.arm(vector([2., 1.]))
        boundary.step()
        r = boundary.receipt['shared_cnn']
        self.assertEqual(r['proposal_sign'], 1)
        self.assertLessEqual(r['final_sign'], 0)
        self.assertNotEqual(r['action'], 'unchanged_proposal')
        self.assertEqual(exact_sign(vector([2., 1.]), before, arrays(boundary.parameters)), r['final_sign'])
        self.assertEqual(boundary.committed_steps, 1)
        self.assertTrue(all(float(s['step']) == 1 for s in raw.state.values()))

    def test_retained_momentum_and_decay_are_not_ignored(self):
        for mechanism in ('momentum', 'decay'):
            with self.subTest(mechanism=mechanism):
                model, raw, boundary = setup(weight_decay=1. if mechanism == 'decay' else 0.)
                if mechanism == 'momentum':
                    for _ in range(10):
                        install(model, [-1.])
                        raw.step()
                else:
                    with torch.no_grad():
                        model.projection.bias[0] = -10.
                install(model, [1.])
                boundary.arm(vector([1.]))
                boundary.step()
                self.assertEqual(boundary.receipt['shared_cnn']['proposal_sign'], 1)
                self.assertLessEqual(boundary.receipt['shared_cnn']['final_sign'], 0)
                self.assertEqual(float(raw.state[model.projection.bias]['step']), 11 if mechanism == 'momentum' else 1)

    def test_safe_proposal_exactly_matches_native_adamw_state_across_steps(self):
        model, raw, boundary = setup(amsgrad=True, weight_decay=.0001)
        other, reference = copy.deepcopy((model, raw))
        for step in range(3):
            install(model, [1., 2.])
            install(other, [1., 2.])
            boundary.reset()
            boundary.arm(vector([1., 2.]))
            boundary.step()
            reference.step()
            tree(self, model.state_dict(), other.state_dict())
            tree(self, raw.state_dict(), reference.state_dict())
            self.assertEqual(boundary.moved_steps, step + 1)

    def test_undefined_count_can_move_but_rounded_zero_does_not(self):
        model, raw, boundary = setup(dtype=torch.float32)
        install(model, [1.])
        with torch.no_grad():
            model.projection.bias.fill_(1e8)
        boundary.arm(vector([1.]))
        boundary.step()
        self.assertEqual((boundary.committed_steps, boundary.moved_steps), (1, 0))
        self.assertEqual(boundary.receipt['shared_cnn']['final_sign'], 0)
        boundary.reset()
        boundary.arm(vector([0.]))
        with torch.no_grad():
            model.projection.bias.zero_()
        boundary.step()
        self.assertFalse(boundary.receipt['shared_cnn']['count_direction_defined'])
        self.assertIsNone(boundary.receipt['shared_cnn']['final_sign'])
        self.assertEqual(boundary.moved_steps, 1)

    def test_rounded_projected_group_is_vetoed_not_accepted_by_tolerance(self):
        before = {n: np.zeros(s, np.float32) for n, s in model_shapes().items()}
        before['projection.bias'][:2] = [1e8, 0.]
        proposal = {n: x.copy() for n, x in before.items()}
        proposal['projection.bias'][:2] += [0., 1.]
        # Projected delta (-.4, .8): first coordinate rounds to zero displacement;
        # the second remains positive, which would violate the count halfspace.
        after, receipt = protect(vector([2., 1.]), before, proposal)
        self.assertEqual(receipt['shared_cnn']['action'], 'rounded_projection_veto')
        self.assertFalse(receipt['shared_cnn']['moved'])
        self.assertTrue(all(np.array_equal(x, after[n]) for n, x in before.items()))

    def test_exact_sign_handles_underflow_and_cancellation(self):
        for c, y in (([.5, 2.**-1074], [0., 2.**-1074]),
                     ([.5, .5, .5], [1e308, 1., -1e308])):
            count = Gradient.create({'x': np.array(c)})
            sign = exact_sign(count, {'x': np.zeros(len(c))}, {'x': np.array(y)})
            self.assertEqual(sign, 1)
            oracle = sum(Fraction(float(a)) * Fraction(float(b)) for a, b in zip(c, y))
            self.assertGreater(oracle, 0)

    def test_positive_common_exponent_and_group_independence(self):
        before = {n: np.zeros(s, np.float64) for n, s in model_shapes().items()}
        proposal = {n: x.copy() for n, x in before.items()}
        proposal['projection.bias'][0], proposal['output.bias'][0], proposal['origin.bias'][0] = 1., -2., 3.
        count = vector([1.], exponent=10000)
        after, receipt = protect(count, before, proposal)
        self.assertEqual(receipt['shared_cnn']['final_sign'], 0)
        self.assertEqual(after['output.bias'][0], -2.)
        self.assertEqual(after['origin.bias'][0], 3.)
        self.assertEqual(set(receipt), set(GROUPS))

    def test_first_order_nonincrease_is_not_a_finite_loss_certificate(self):
        before = {n: np.zeros(s, np.float64) for n, s in model_shapes().items()}
        proposal = {n: x.copy() for n, x in before.items()}
        proposal['projection.bias'][:2] = [0., 1.]
        after, receipt = protect(vector([1., 0.]), before, proposal)
        self.assertEqual(receipt['shared_cnn']['final_sign'], 0)
        delta = after['projection.bias'][:2]
        self.assertGreater(float(delta[0] + .5 * delta @ delta), 0.)

    def test_partial_copy_failure_restores_weights_and_optimizer_and_closes(self):
        model, raw, boundary = setup()
        install(model, [1.])
        old_model, old_state = copy.deepcopy((model.state_dict(), raw.state_dict()))
        boundary.arm(vector([1.]))
        def fail(values):
            model.projection.bias.copy_(values['projection.bias'])
            raise ValueError('authored partial copy')
        with patch.object(boundary, '_copy_parameters', side_effect=fail):
            with self.assertRaisesRegex(ValueError, 'partial copy'):
                boundary.step()
        tree(self, old_model, model.state_dict())
        tree(self, old_state, raw.state_dict())
        self.assertEqual((boundary.committed_steps, boundary.rollback), (0, 'restored'))
        with self.assertRaisesRegex(ValueError, 'closed'):
            boundary.reset()

    def test_nonfinite_proposal_and_prepared_fault_leave_live_state_unchanged(self):
        for fault in ('nonfinite', 'prepared'):
            model, raw, boundary = setup()
            install(model, [1.])
            before = copy.deepcopy((model.state_dict(), raw.state_dict()))
            boundary.arm(vector([1.]))
            real = boundary._propose
            def proposal():
                p, o = real()
                p['projection.bias'].fill_(float('inf'))
                return p, o
            def notify(stage):
                if stage == 'prepared':
                    raise OSError('authored persistence')
            boundary.notify = notify if fault == 'prepared' else None
            with patch.object(boundary, '_propose', side_effect=proposal if fault == 'nonfinite' else real):
                with self.assertRaises((ValueError, OSError)):
                    boundary.step()
            tree(self, before, (model.state_dict(), raw.state_dict()))
            self.assertEqual(boundary.committed_steps, 0)

    def test_failed_rollback_is_explicitly_unknown_and_never_retried(self):
        model, raw, boundary = setup()
        install(model, [1.])
        boundary.arm(vector([1.]))
        patcher = patch.object(model.projection.bias, 'copy_', side_effect=OSError('authored device unavailable'))
        def fail(candidate):
            model.projection.bias.copy_(candidate['projection.bias'])
            patcher.start()
            raise OSError('authored interrupted commit')
        self.addCleanup(patcher.stop)
        with patch.object(boundary, '_copy_parameters', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'interrupted commit') as raised:
                boundary.step()
        self.assertEqual(boundary.rollback, 'failed_unknown')
        self.assertEqual(boundary.committed_steps, 0)
        self.assertTrue(boundary.failed)
        self.assertIn('Rollback failed', raised.exception.__notes__[0])

    def test_ownership_gradients_and_execution_modes_fail_closed(self):
        for fault in ('alias', 'replace', 'foreign', 'hooks', 'missing_grad', 'nonfinite_grad', 'capturable'):
            model, raw, boundary = setup()
            install(model, [1.])
            boundary.arm(vector([1.]))
            if fault == 'alias':
                model.alias = model.projection.bias
            elif fault == 'replace':
                model.projection.bias = torch.nn.Parameter(model.projection.bias.clone())
            elif fault == 'foreign':
                raw.param_groups[0]['params'].pop()
            elif fault == 'hooks':
                raw.register_step_pre_hook(lambda *_: None)
            elif fault == 'missing_grad':
                model.gain_logit.grad = None
            elif fault == 'nonfinite_grad':
                model.gain_logit.grad.fill_(float('nan'))
            else:
                raw.param_groups[0]['capturable'] = True
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                boundary.step()
            self.assertEqual(boundary.committed_steps, 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_actual_adamw_proposal_and_native_commit(self):
        model, raw, boundary = setup(device='cuda', dtype=torch.float32)
        install(model, [-1., 7.])
        before = arrays(boundary.parameters)
        boundary.arm(vector([2., 1.]))
        boundary.step()
        self.assertEqual(boundary.receipt['shared_cnn']['proposal_sign'], 1)
        self.assertLessEqual(exact_sign(vector([2., 1.]), before, arrays(boundary.parameters)), 0)
        self.assertEqual(boundary.committed_steps, 1)


if __name__ == '__main__':
    unittest.main()
