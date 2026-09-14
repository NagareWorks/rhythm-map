"""Live CNN/VJP integration and durable failure boundaries, not musical fitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.gradient_routing.authored import record_terms, teacher_case
from experiments.gradient_routing.route import partition, route
from experiments.observed_adjoint.transport import pack_gradient, unpack_gradient
from experiments.protected_update.boundary import arrays, exact_sign
from experiments.protected_update.recorder import ProtectedFitRecorder
from experiments.protected_update.test_boundary import tree
from experiments.scaled_adjoint.scaled import accumulate
from experiments.training_observer.recorder import cpu_copy

torch.set_num_threads(1)


class RecorderTests(unittest.TestCase):
    def make(self, device='cpu'):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        model, payload = teacher_case(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        observer = ProtectedFitRecorder(Path(temporary.name) / 'authored', model, optimizer,
            dict(protocol='protected-update-authored-v1'), max_updates=2)
        other = dict(payload, features=payload['features'] * .7)
        rows = [dict(id='first', scale=.3, payload=payload), dict(id='second', scale=.7, payload=other)]
        return observer, rows

    def load(self, observer, name='failure.pt'):
        return torch.load(observer.output / name, weights_only=True)

    def events(self, observer):
        return [json.loads(line) for line in (observer.output / 'journal.jsonl').read_text().splitlines()]

    def live(self, device):
        observer, rows = self.make(device)
        expected = [record_terms(observer.model, r['payload']) for r in rows]
        p, c = [accumulate([g[index].multiply(r['scale']) for r, g in zip(rows, expected)]) for index in (0, 1)]
        routed, _ = route(p, c)
        before = arrays(observer.optimizer.parameters)
        observer.update(1, 0, rows)
        tree(self, pack_gradient(p), pack_gradient(observer.phase))
        tree(self, pack_gradient(c), pack_gradient(observer.count))
        tree(self, pack_gradient(routed), pack_gradient(observer.routed))
        saved = self.load(observer, 'after-update.pt')
        boundary = saved['protected_update']['boundary']
        self.assertEqual((saved['completed_updates'], boundary['committed_steps'], boundary['moved_steps']), (1, 1, 1))
        self.assertEqual(saved['cursor']['optimizer_step_status'], 'returned')
        self.assertEqual(boundary['stage'], 'committed')
        for group, count in partition(c).items():
            sign = exact_sign(count, before, arrays(observer.optimizer.parameters))
            self.assertLessEqual(sign, 0)
            self.assertEqual(sign, boundary['receipt'][group]['final_sign'])
        # Durable proposal is pre-commit, with OLD live optimizer and all new moments.
        proposed = self.load(observer, 'proposal.pt')
        self.assertEqual(proposed['protected_update']['boundary']['stage'], 'prepared')
        self.assertEqual(proposed['optimizer']['state'], {})
        self.assertTrue(proposed['protected_update']['boundary']['proposal_state']['state'])
        self.assertEqual(proposed['completed_updates'], 0)
        self.assertTrue(all(len(r['terms']) == 2 for r in saved['protected_update']['weighted_records']))
        for index, retained in enumerate(proposed['protected_update']['weighted_records']):
            tree(self, cpu_copy(rows[index]), retained['record'])
            self.assertEqual(retained['record']['payload']['features'].device.type, 'cpu')
            self.assertIn('torch_cpu', retained['before_record_rng'])
            self.assertIn('torch_cuda', retained['before_record_rng'])
            for term, gradient in zip(('phase', 'advance'), expected[index]):
                tree(self, retained['terms'][term]['gradient'], pack_gradient(gradient.multiply(rows[index]['scale'])))
        observer.update(1, 1, rows)
        self.assertEqual(observer.completed_updates, 2)
        self.assertEqual(observer.optimizer.committed_steps, 2)
        self.assertTrue(all(float(s['step']) == 2 for s in observer.optimizer.raw.state.values()))
        with self.assertRaisesRegex(ValueError, 'budget'):
            observer.update(1, 2, rows)

    def test_complete_weighted_batch_real_transport_and_persistence(self):
        self.live('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_complete_batch_real_transport_and_persistence(self):
        self.live('cuda')

    def test_loss_term_failure_retains_previous_record_and_current_phase(self):
        observer, rows = self.make()
        original = observer.terms['advance'].run
        calls = 0
        def fail(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError('authored second count transport')
            return original(*args)
        with patch.object(observer.terms['advance'], 'run', side_effect=fail):
            with self.assertRaisesRegex(ValueError, 'second count'):
                observer.update(1, 0, rows)
        saved = self.load(observer)
        state = saved['protected_update']
        self.assertEqual(saved['cursor']['recording'], 'second')
        self.assertEqual(len(state['weighted_records']), 1)
        self.assertIsNotNone(state['terms']['phase']['parameter_gradient'])
        self.assertIsNone(state['terms']['advance']['parameter_gradient'])
        self.assertEqual(state['boundary']['committed_steps'], 0)
        self.assertEqual(saved['cursor']['optimizer_step_status'], 'not_started')
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            observer.update(1, 0, rows)

    def test_proposal_save_fault_is_unchanged_live_state_with_closed_recorder(self):
        observer, rows = self.make()
        old = copy.deepcopy((observer.model.state_dict(), observer.optimizer.state_dict()))
        real_save = observer._save
        def fail(name, **extra):
            if name == 'proposal.pt':
                raise OSError('authored proposal disk fault')
            return real_save(name, **extra)
        with patch.object(observer, '_save', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'proposal disk'):
                observer.update(1, 0, rows)
        tree(self, old, (observer.model.state_dict(), observer.optimizer.state_dict()))
        saved = self.load(observer)
        self.assertEqual(saved['protected_update']['boundary']['stage'], 'prepared')
        self.assertEqual(saved['protected_update']['boundary']['committed_steps'], 0)
        self.assertFalse(self.events(observer)[-1]['update_count_exact'])
        self.assertTrue((observer.output / 'before-update.pt').exists())
        # The saved OLD model plus EVERY input can reproduce every captured
        # record gradient, even though the failed owner cannot resume fitting.
        for retained in saved['protected_update']['weighted_records']:
            phase, count = record_terms(observer.model, retained['record']['payload'])
            for term, gradient in (('phase', phase), ('advance', count)):
                tree(self, retained['terms'][term]['gradient'], pack_gradient(gradient.multiply(retained['scale'])))

    def test_retained_records_own_inputs_after_caller_mutation(self):
        observer, rows = self.make()
        observer.update(1, 0, rows)
        proposed = self.load(observer, 'proposal.pt')
        rows[0]['payload']['features'].add_(1.)
        tree(self, observer.records[0]['record'], proposed['protected_update']['weighted_records'][0]['record'])

    def test_partial_commit_failure_has_rollback_evidence_and_unchanged_state(self):
        observer, rows = self.make()
        old = copy.deepcopy((observer.model.state_dict(), observer.optimizer.state_dict()))
        def fail(candidate):
            observer.model.projection.bias.copy_(candidate['projection.bias'])
            raise OSError('authored partial copy')
        with patch.object(observer.optimizer, '_copy_parameters', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'partial copy'):
                observer.update(1, 0, rows)
        tree(self, old, (observer.model.state_dict(), observer.optimizer.state_dict()))
        saved = self.load(observer)
        self.assertEqual(saved['protected_update']['boundary']['rollback'], 'restored')
        self.assertEqual(saved['protected_update']['boundary']['committed_steps'], 0)
        self.assertFalse(self.events(observer)[-1]['update_count_exact'])

    def test_postcommit_journal_fault_does_not_disguise_a_committed_proposal(self):
        observer, rows = self.make()
        real = observer._event
        def fail(event, **fields):
            if event == 'optimizer_boundary' and observer.optimizer.stage == 'committed':
                raise OSError('authored committed journal fault')
            return real(event, **fields)
        with patch.object(observer, '_event', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'committed journal'):
                observer.update(1, 0, rows)
        saved = self.load(observer)
        self.assertEqual(saved['completed_updates'], 0)  # step did not return.
        self.assertEqual(saved['protected_update']['boundary']['committed_steps'], 1)
        self.assertEqual(saved['protected_update']['boundary']['stage'], 'committed')
        self.assertTrue(all(float(s['step']) == 1 for s in saved['optimizer']['state'].values()))
        self.assertFalse(self.events(observer)[-1]['update_count_exact'])

    def test_after_snapshot_fault_retains_exact_returned_and_committed_counts(self):
        observer, rows = self.make()
        real = observer._save
        def fail(name, **extra):
            if name == 'after-update.pt':
                raise OSError('authored after snapshot')
            return real(name, **extra)
        with patch.object(observer, '_save', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'after snapshot'):
                observer.update(1, 0, rows)
        saved = self.load(observer)
        self.assertEqual(saved['completed_updates'], 1)
        self.assertEqual(saved['protected_update']['boundary']['committed_steps'], 1)
        self.assertTrue(self.events(observer)[-1]['update_count_exact'])

    def test_captured_proposal_sign_replay_needs_no_optimizer_or_forward(self):
        observer, rows = self.make()
        observer.update(1, 0, rows)
        saved = self.load(observer, 'proposal.pt')
        captured = saved['protected_update']['boundary']
        before = {n: t.numpy() for n, t in saved['model'].items()}
        candidate = {n: t.numpy() for n, t in captured['candidate'].items()}
        with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('no optimizer')):
            for group, c in partition(unpack_gradient(captured['count'])).items():
                self.assertEqual(exact_sign(c, before, candidate), captured['receipt'][group]['final_sign'])


if __name__ == '__main__':
    unittest.main()
