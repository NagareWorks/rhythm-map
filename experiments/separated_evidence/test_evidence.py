import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.separated_evidence.model import SharedEvidence, SeparatedEvidence, SHARED_PARAMETERS, SEPARATED_PARAMETERS
from experiments.separated_evidence.supervision import losses, task_loss
from experiments.separated_evidence.training import TaskTrainer


def case(device='cpu', length=96):
    # Explicit authored teaching channels, NOT cached music or encoder evidence.
    generator = torch.Generator().manual_seed(713)
    features = torch.randn(1, length, 512, generator=generator) * .01
    rate = torch.linspace(1.5, 2.5, length - 1, dtype=torch.float64)
    rate[(length - 1) // 2:] = 1.25
    reference = torch.cat((torch.tensor([.13], dtype=torch.float64), .13 + torch.cumsum(rate / 50, 0)))[None]
    features[0, :-1, 0] = -torch.log2(rate)
    features[0, -1, 0] = features[0, -2, 0]
    features[0, :, 1] = torch.cos(2 * math.pi * reference[0])
    features[0, :, 2] = torch.sin(2 * math.pi * reference[0])
    features[0, length // 3:length // 3 + 4, 1:3] = 0
    return dict(id='authored-ramp-step-gap', features=features.to(device), reference=reference.to(device),
                valid=torch.ones_like(reference, dtype=torch.bool, device=device), scale=1.)


def equal_tree(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            equal_tree(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for left, right in zip(a, b):
            equal_tree(test, left, right)
    else:
        test.assertEqual(a, b)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(142)
        torch.set_num_threads(2)

    def test_matched_initial_functions_without_rng_or_parameter_aliasing(self):
        shared = SharedEvidence()
        rng = torch.get_rng_state().clone()
        model = SeparatedEvidence(shared)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in shared.parameters()), SHARED_PARAMETERS)
        self.assertEqual(sum(p.numel() for p in model.parameters()), SEPARATED_PARAMETERS)
        groups = [{p.data_ptr() for p in module.parameters()} for module in (shared, model.tempo, model.phase)]
        self.assertTrue(all(not groups[i] & groups[j] for i in range(3) for j in range(i)))
        x = case()['features']
        for before, after in zip(vars(shared(x)).values(), vars(model(x)).values()):
            self.assertTrue(torch.equal(before, after))

    def test_task_outputs_have_frame_and_cell_geometry_not_clock_semantics(self):
        evidence = SeparatedEvidence(SharedEvidence())(case()['features'])
        self.assertEqual(evidence.log2_cell_period.shape, (1, 95))
        self.assertEqual(evidence.phase_vector.shape, (1, 96, 2))
        for name in ('cycles', 'beats', 'confidence', 'changes'):
            self.assertFalse(hasattr(evidence, name))

    def test_complete_context_chunks_and_gradients_retain_physical_edges(self):
        model = SeparatedEvidence(SharedEvidence()).double()
        x = case(length=309)['features'].double().requires_grad_()
        before = x.detach().clone()
        full = model(x)
        full_grad = torch.autograd.grad(full.log2_cell_period.square().sum() + full.phase_vector.square().sum(),
                                        (x, *model.parameters()))
        chunked = model(x, chunk_frames=83)
        chunk_grad = torch.autograd.grad(chunked.log2_cell_period.square().sum() + chunked.phase_vector.square().sum(),
                                         (x, *model.parameters()))
        for a, b in zip(vars(full).values(), vars(chunked).values()):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
        for a, b in zip(full_grad, chunk_grad):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
        self.assertTrue(torch.equal(before, x))

    def test_tempo_and_phase_gradients_have_disjoint_full_trunk_ownership(self):
        model, row = SeparatedEvidence(SharedEvidence()), case()
        for task in ('tempo', 'phase'):
            model.zero_grad(set_to_none=True)
            loss = losses(model(row['features']), row['reference'], row['valid'])[task]
            loss.backward()
            own = getattr(model, task)
            other = model.phase if task == 'tempo' else model.tempo
            self.assertTrue(all(p.grad is not None for p in own.parameters()))
            self.assertTrue(all(p.grad is None for p in other.parameters()))

    def test_native_period_targets_reject_half_double_and_do_not_smooth_change(self):
        row = case()
        target = -torch.log2(row['reference'].diff(dim=1) * 50)
        self.assertEqual(float(task_loss('tempo', target, row['reference'], row['valid'])), 0.)
        for shift in (-1., 1.):
            self.assertAlmostEqual(float(task_loss('tempo', target + shift, row['reference'], row['valid'])), 1.)

    def test_phase_target_wraps_integer_counts_without_claiming_tempo(self):
        row = case()
        angle = 2 * math.pi * torch.remainder(row['reference'], 1.)
        target = torch.stack((angle.cos(), angle.sin()), dim=-1)
        self.assertEqual(float(task_loss('phase', target, row['reference'], row['valid'])), 0.)
        self.assertLess(float(task_loss('phase', target, row['reference'] + 3, row['valid'])), 1e-26)
        self.assertGreater(float(task_loss('phase', target, row['reference'] + .5, row['valid'])), 1.9)

    def test_missing_reference_does_not_bridge_cells_or_become_no_rhythm(self):
        row = case()
        row['valid'][:, 30:40] = False
        row['reference'][:, 30:40] = float('nan')
        prediction = SharedEvidence()(row['features'])
        value = losses(prediction, row['reference'], row['valid'])
        changed = prediction.log2_cell_period.clone()
        changed[:, 29:40] += 100
        self.assertEqual(float(value['tempo'].detach()),
                         float(task_loss('tempo', changed, row['reference'], row['valid']).detach()))
        changed[:, 20] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite task'):
            task_loss('tempo', changed, row['reference'], row['valid'])

    def test_invalid_geometry_dtype_and_support_fail(self):
        model, row = SharedEvidence(), case()
        for features in (row['features'][:, :1], row['features'].double(), row['features'][:, :, :3]):
            with self.assertRaises(ValueError):
                model(features)
        for chunk in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                model(row['features'], chunk)
        with self.assertRaises(ValueError):
            losses(model(row['features']), row['reference'], torch.zeros_like(row['valid']))
        with self.assertRaisesRegex(ValueError, 'must advance'):
            losses(model(row['features']), -row['reference'], row['valid'])

    def test_nonfinite_features_rejected(self):
        row = case()
        row['features'][0, 1, 3] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite features'):
            SeparatedEvidence(SharedEvidence())(row['features'])

    def isolation(self, device):
        model = SeparatedEvidence(SharedEvidence()).to(device)
        reference = copy.deepcopy(model)
        row = case(device)
        trainers = {k: TaskTrainer(getattr(model, k), max_updates=3) for k in ('tempo', 'phase')}
        independent = {k: TaskTrainer(getattr(reference, k), max_updates=3) for k in ('tempo', 'phase')}
        # Real momentum and decay histories exist before checking isolation.
        for step in range(3):
            phase_before = copy.deepcopy(model.phase.state_dict())
            phase_optimizer = copy.deepcopy(trainers['phase'].optimizer.state_dict())
            trainers['tempo'].update([row])
            independent['tempo'].update([row])
            equal_tree(self, model.phase.state_dict(), phase_before)
            equal_tree(self, trainers['phase'].optimizer.state_dict(), phase_optimizer)
            independent['phase'].update([dict(row, scale=1000.)])
            trainers['phase'].update([dict(row, scale=1000.)])
            for kind in trainers:
                equal_tree(self, getattr(model, kind).state_dict(), getattr(reference, kind).state_dict())
                equal_tree(self, trainers[kind].optimizer.state_dict(), independent[kind].optimizer.state_dict())
                self.assertEqual(trainers[kind].returned_updates, step + 1)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_separate_task_updates_equal_standalone_native_adamw_with_momentum(self):
        self.isolation('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_initial_parity_and_task_isolation(self):
        shared = SharedEvidence().cuda()
        split = SeparatedEvidence(shared)
        row = case('cuda')
        for a, b in zip(vars(shared(row['features'])).values(), vars(split(row['features'])).values()):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        self.isolation('cuda')

    def test_shared_control_really_has_a_cross_task_learning_path(self):
        shared, row = SharedEvidence(), case()
        before = shared(row['features']).log2_cell_period.detach().clone()
        optimizer = torch.optim.AdamW(shared.parameters(), lr=.001, weight_decay=.0001)
        losses(shared(row['features']), row['reference'], row['valid'])['phase'].backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, shared(row['features']).log2_cell_period))

    def test_record_weighting_matches_independent_full_batch_gradient(self):
        branch = SeparatedEvidence(SharedEvidence()).tempo
        manual = copy.deepcopy(branch)
        trainer = TaskTrainer(branch, max_updates=1)
        optimizer = torch.optim.AdamW(manual.parameters(), lr=.001, weight_decay=.0001)
        rows = [dict(case(length=n), id=str(n), scale=s) for n, s in ((73, .2), (97, .8))]
        optimizer.zero_grad(set_to_none=True)
        for row in rows:
            (task_loss('tempo', manual(row['features']), row['reference'], row['valid']) * row['scale']).backward()
        torch.nn.utils.clip_grad_norm_(manual.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        result = trainer.update(rows)
        equal_tree(self, manual.state_dict(), branch.state_dict())
        equal_tree(self, optimizer.state_dict(), trainer.optimizer.state_dict())
        self.assertEqual(result['weighted_loss'], math.fsum(r['scale'] * v for r, v in zip(rows, result['record_losses'])))

    def test_failed_optimizer_entry_is_unknown_and_cannot_retry(self):
        trainer = TaskTrainer(SeparatedEvidence(SharedEvidence()).tempo, max_updates=2)
        with patch.object(trainer.optimizer, 'step', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                trainer.update([case()])
        self.assertEqual(trainer.stage, 'optimizer_entered')
        self.assertEqual(trainer.returned_updates, 0)
        self.assertTrue(trainer.failed)
        with self.assertRaisesRegex(ValueError, 'closed'):
            trainer.update([case()])

    def test_nonfinite_returned_optimizer_state_is_counted_then_closed(self):
        trainer = TaskTrainer(SeparatedEvidence(SharedEvidence()).tempo, max_updates=2)
        native = trainer.optimizer.step
        def corrupted():
            native()
            next(iter(trainer.optimizer.state.values()))['exp_avg'].fill_(float('inf'))
        with patch.object(trainer.optimizer, 'step', side_effect=corrupted):
            with self.assertRaisesRegex(ValueError, 'optimizer state'):
                trainer.update([case()])
        self.assertEqual(trainer.stage, 'optimizer_returned')
        self.assertEqual(trainer.returned_updates, 1)
        self.assertTrue(trainer.failed)

    def test_fixed_budget_and_bad_input_stop_before_optimizer(self):
        trainer = TaskTrainer(SeparatedEvidence(SharedEvidence()).phase, max_updates=1)
        trainer.update([case()])
        with patch.object(trainer.optimizer, 'step', side_effect=AssertionError('must not step')):
            with self.assertRaisesRegex(ValueError, 'budget'):
                trainer.update([case()])
        other = TaskTrainer(SeparatedEvidence(SharedEvidence()).phase, max_updates=1)
        row = case()
        with self.assertRaisesRegex(ValueError, 'distinct'):
            other.update([row, row])
        self.assertEqual(other.returned_updates, 0)

    def test_fixed_authored_learning_and_snapshot_budgets(self):
        from experiments.separated_evidence.authored import run
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'authored'
            report = run(directory, 'cpu')
            self.assertEqual(report['initial']['shared'], report['initial']['separated'])
            self.assertTrue(report['gate_passed'])
            self.assertEqual(report['shared_updates'], 40)
            self.assertEqual(report['task_updates'], {'tempo': 40, 'phase': 40})
            self.assertEqual(len(report['steps']), 40)
            self.assertEqual(report['musical_updates'], 0)
            self.assertFalse(report['admission'])
            self.assertFalse(report['coherent_clock'])
            before = torch.load(directory / 'before-update.pt', weights_only=True, map_location='cpu')
            after = torch.load(directory / 'after-update.pt', weights_only=True, map_location='cpu')
            self.assertEqual((before['shared_steps'], after['shared_steps']), (39, 40))
            self.assertEqual(after['tasks']['tempo']['returned_updates'], 40)
            self.assertEqual(after['tasks']['phase']['returned_updates'], 40)
            self.assertIn('features', after['input'])
            self.assertIn('rng', after)

    def test_second_task_failure_preserves_first_task_commit_and_primary_error(self):
        from experiments.separated_evidence.authored import run
        native = TaskTrainer.update
        def injected(trainer, records):
            if trainer.branch.task == 'phase':
                raise RuntimeError('authored phase fault')
            return native(trainer, records)
        with tempfile.TemporaryDirectory() as temporary, patch.object(TaskTrainer, 'update', injected):
            directory = Path(temporary) / 'fault'
            with self.assertRaisesRegex(RuntimeError, 'authored phase fault'):
                run(directory, 'cpu')
            saved = torch.load(directory / 'failure.pt', weights_only=True, map_location='cpu')
            self.assertEqual(saved['stage'], 'phase')
            self.assertEqual(saved['shared_steps'], 1)
            self.assertEqual(saved['tasks']['tempo']['returned_updates'], 1)
            self.assertEqual(saved['tasks']['phase']['returned_updates'], 0)
            self.assertFalse((directory / 'report.json').exists())


if __name__ == '__main__':
    unittest.main()
