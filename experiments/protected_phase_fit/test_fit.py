"""Authored admission/failure tests; no music packet or learned weight access."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.coupled_clock.run import state_hash, work_weights, work_mean
from experiments.phase_sync.model import PhaseSyncReadout
from experiments.phase_sync_fit import run as original
from experiments.phase_sync_fit.test_fit import passing_gate
from experiments.protected_update.recorder import ProtectedFitRecorder
from experiments.protected_update.test_boundary import tree
from experiments.scaled_phase_fit.test_fit import authored
from experiments.protected_phase_fit import run
from experiments.protected_phase_fit.recorder import MusicalRecorder, conditioning
from experiments.protected_phase_fit.register import make_plan, source_hashes


def tiny_plan(device='cpu'):
    # Only a unit-test harness. The CLI rejects these budgets for real input.
    return dict(make_plan(device), epochs=1, updates=5, complete_records=20,
                complete_record_gradients=40, protected_step_receipts=5)


class FitContracts(unittest.TestCase):
    def test_registration_freezes_population_runtime_budget_and_mechanism(self):
        a, b = make_plan('cpu'), make_plan('cpu')
        self.assertEqual(a, b)
        self.assertEqual((a['epochs'], a['updates'], a['seconds_per_fit'], a['watchdog_seconds']), (20, 100, 1800, 3900))
        self.assertEqual((a['complete_records'], a['complete_record_gradients'], a['batch_recordings']), (400, 800, 4))
        self.assertEqual(a['optimizer'], dict(name='ProtectedAdamW', learning_rate=.001, weight_decay=.0001, max_norm=1.))
        self.assertEqual([sum(r['role'] == role for r in a['population']) for role in ('fit', 'development', 'diagnostic')], [20, 5, 15])
        self.assertFalse(any(a[k] for k in ('old_weights_loaded', 'automatic_retry', 'training', 'holdout_access', 'production_change')))
        for source in ('protected_update/boundary.py', 'protected_phase_fit/recorder.py', 'protected_phase_fit/run.py'):
            self.assertIn('experiments/' + source, a['source_sha256'])

    def test_closed_prerequisite_drift_is_rejected(self):
        with patch('experiments.protected_phase_fit.register.sha', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'prerequisite identity'):
                source_hashes()

    def test_metrics_and_every_old_gate_are_imported_unchanged(self):
        self.assertIs(run.evaluate, original.evaluate)
        self.assertIs(run.comparison_gate, original.comparison_gate)
        rows, summary, fits = passing_gate()
        self.assertTrue(run.comparison_gate(rows, summary, fits)['passed'])
        summary['diagnostic']['raw_common']['phase_mean_absolute_cycles'] = .1
        self.assertFalse(run.comparison_gate(rows, summary, fits)['passed'])

    def test_modified_cli_plan_is_rejected_before_loading_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = tiny_plan()
            run.write_json(root / 'plan.json', plan)
            args = SimpleNamespace(inputs=root / 'absent', output=root / 'run', device='cpu',
                plan=root / 'plan.json', expected_plan_sha256=run.sha(root / 'plan.json'))
            with patch('argparse.ArgumentParser.parse_args', return_value=args), \
                    patch.object(run, 'load_records', side_effect=AssertionError('must not open music')):
                with self.assertRaisesRegex(ValueError, 'registered runtime'):
                    run.main()
            self.assertFalse(args.output.exists())

    def test_wrong_initialization_fails_before_forward_or_step(self):
        plan = dict(tiny_plan(), initial_state_sha256='wrong')
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(PhaseSyncReadout, 'fields', side_effect=AssertionError('no forward')):
            with self.assertRaisesRegex(ValueError, 'initialization differs'):
                run.fit(authored(), 'cpu', False, Path(temporary), plan, 'authored')
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def manual(self, rows, device, root, plan):
        torch.manual_seed(plan['seed'])
        model = PhaseSyncReadout().to(device)
        observer = ProtectedFitRecorder(root / 'manual', model,
            torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001),
            {'protocol': 'authored-manual-parity'}, max_updates=5)
        weights, training = work_weights(rows[:20]), {}
        order = np.random.default_rng(142).permutation(20)
        for offset in range(0, 20, 4):
            batch = [rows[i] for i in order[offset:offset + 4]]
            losses = observer.update(1, offset // 4, [run.packet(r, False, weights[r['work']] / 4) for r in batch])
            for row, loss in zip(batch, losses):
                training.setdefault(row['work'], []).append(loss)
        return model, original.development_loss(model, rows, False), work_mean(training)

    def full(self, device):
        plan, rows = tiny_plan(device), authored(device)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected, development, training = self.manual(rows, device, root, plan)
            model, report = run.fit(rows, device, False, root, plan, 'authored')
            self.assertEqual(state_hash(model), state_hash(expected))
            self.assertEqual(report['development_loss'], development)
            self.assertEqual(report['history'][0]['training_work_macro_loss'], training)
            info = report['conditioning']
            self.assertEqual((info['record_gradients'], info['complete_records'], info['protected_steps']), (40, 20, 5))
            self.assertEqual((info['post_step_records'], info['group_receipts'], info['adverse_defined_groups']), (20, 15, 0))
            self.assertEqual(info['moved_steps'] + info['stalled_steps'], 5)
            events = run.read_events(root / 'audio/journal.jsonl')
            self.assertFalse(any('diagnostic-must-not-run' in str(e) for e in events))
            self.assertEqual(info, conditioning(events))
            paired = dict(report, kind='zero-audio')
            self.assertTrue(run.execution_gate([report, paired], plan))
            self.assertFalse(run.execution_gate([report, report], plan))
            for field in ('complete_records', 'record_gradients', 'protected_steps', 'post_step_records',
                          'committed_steps', 'group_receipts'):
                bad = copy.deepcopy(paired)
                bad['conditioning'][field] -= 1
                self.assertFalse(run.execution_gate([report, bad], plan), field)
            best = torch.load(root / 'audio/best.pt', weights_only=True)
            self.assertEqual(best['best']['completed_updates'], 5)
            self.assertEqual(best['protected_update']['weighted_records'], [])
            self.assertTrue(all(v['clock_trace'] is None for v in best['protected_update']['terms'].values()))
            self.assertIsNone(best['finite_loss_audit']['before'])

    def test_authored_epoch_audits_do_not_change_weights_or_selection(self):
        self.full('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_authored_epoch_and_same_native_weight_parity(self):
        self.full('cuda')

    def test_mid_batch_timeout_retains_inputs_without_optimizer_entry(self):
        real = MusicalRecorder.check_deadline
        def timeout(observer):
            if len(observer.records) == 2:
                raise TimeoutError('authored middle batch')
            return real(observer)
        with tempfile.TemporaryDirectory() as temporary, patch.object(MusicalRecorder, 'check_deadline', timeout):
            root = Path(temporary)
            with self.assertRaisesRegex(TimeoutError, 'middle batch'):
                run.fit(authored(), 'cpu', False, root, tiny_plan(), 'authored')
            saved = torch.load(root / 'audio/failure.pt', weights_only=True)
            self.assertEqual(saved['completed_updates'], 0)
            self.assertEqual(saved['cursor']['optimizer_step_status'], 'not_started')
            self.assertEqual(len(saved['protected_update']['weighted_records']), 2)
            self.assertTrue(all('record' in r for r in saved['protected_update']['weighted_records']))
            self.assertFalse((root / 'zero-audio').exists())

    def test_post_step_timeout_retains_committed_state_without_stale_backward(self):
        real = MusicalRecorder.check_deadline
        def timeout(observer):
            if observer.cursor['stage'] == 'post_step_loss':
                raise TimeoutError('authored post-step budget')
            return real(observer)
        with tempfile.TemporaryDirectory() as temporary, patch.object(MusicalRecorder, 'check_deadline', timeout):
            root = Path(temporary)
            with self.assertRaisesRegex(TimeoutError, 'post-step budget'):
                run.fit(authored(), 'cpu', False, root, tiny_plan(), 'authored')
            saved = torch.load(root / 'audio/failure.pt', weights_only=True)
            self.assertEqual(saved['completed_updates'], 1)
            self.assertEqual(saved['protected_update']['boundary']['committed_steps'], 1)
            self.assertEqual(saved['cursor']['optimizer_step_status'], 'returned')
            self.assertTrue(all(v['clock_trace'] is None for v in saved['protected_update']['terms'].values()))
            self.assertIsNotNone(saved['finite_loss_audit']['before'])
            self.assertIsNone(saved['finite_loss_audit']['after'])
            self.assertFalse((root / 'audio.weights.npz').exists())

    def test_development_fault_cannot_select_incomplete_epoch(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(run, 'score', return_value=float('nan')):
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'finite development'):
                run.fit(authored(), 'cpu', False, root, tiny_plan(), 'authored')
            saved = torch.load(root / 'audio/failure.pt', weights_only=True)
            self.assertEqual(saved['completed_updates'], 5)
            self.assertIsNone(saved['best'])
            self.assertEqual(saved['protected_update']['weighted_records'], [])
            self.assertTrue(all(v['clock_trace'] is None for v in saved['protected_update']['terms'].values()))

    def test_export_hashing_and_write_remain_inside_wall_budget(self):
        for operation in ('hash', 'write'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                root, late = Path(temporary), False
                real_time, real_sha, real_write = run.time.monotonic, run.sha, run.write_json
                def clock():
                    return real_time() + (1801 if late else 0)
                def hash_file(path):
                    nonlocal late
                    value = real_sha(path)
                    if operation == 'hash' and Path(path).suffix == '.pt':
                        late = True
                    return value
                def write_file(path, value):
                    nonlocal late
                    real_write(path, value)
                    if operation == 'write' and Path(path).name == 'audio.fit.json':
                        late = True
                with patch.object(run.time, 'monotonic', clock), patch.object(run, 'sha', hash_file), \
                        patch.object(run, 'write_json', write_file):
                    with self.assertRaisesRegex(TimeoutError, 'wall budget'):
                        run.fit(authored(), 'cpu', False, root, tiny_plan(), 'authored')
                self.assertTrue(late)
                saved = torch.load(root / 'audio/failure.pt', weights_only=True)
                self.assertEqual(saved['completed_updates'], 5)
                self.assertEqual(saved['cursor']['stage'], 'selected_export')


if __name__ == '__main__':
    unittest.main()
