"""Authored pre-fit tests only; no musical packets or checkpoints are opened."""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.coupled_clock.run import state_hash, work_weights, work_mean, input_tensor
from experiments.phase_sync.model import PhaseSyncReadout
from experiments.phase_sync_fit import run as previous
from experiments.phase_sync_fit.test_fit import passing_gate
from experiments.scaled_adjoint.transport import record_gradient, install_clipped
from experiments.scaled_adjoint.scaled import accumulate
from experiments.scaled_phase_fit import run
from experiments.scaled_phase_fit.register import make_plan, prerequisites, PINNED
from experiments.scaled_phase_fit.recorder import MusicalRecorder, conditioning, score


def authored(device='cpu'):
    rng = np.random.default_rng(803)
    rows = []
    for role, count, works in (('fit', 20, 9), ('development', 5, 3)):
        for i in range(count):
            frames = 211 + i
            rows.append(dict(id=f'{role}-{i}', role=role, work=f'{role}-work-{i % works}',
                features=torch.from_numpy(rng.normal(0, .1, (frames, 512)).astype(np.float32)).to(device),
                target=torch.arange(frames, dtype=torch.float64, device=device) * .04 + .17,
                mask=torch.ones(frames, dtype=torch.bool, device=device)))
    # A sentinel that would fail if diagnostic inputs entered fitting/selection.
    rows.append(dict(id='diagnostic-must-not-run', role='diagnostic', work='diagnostic',
                     features=torch.full((251, 512), float('nan')), target=None, mask=None))
    return rows


def tiny_plan():
    # An authored one-epoch test harness only; CLI exact-plan equality forbids
    # passing these changed budgets to a musical execution.
    plan = make_plan('cpu')
    plan.update(epochs=1, updates=5, complete_record_gradients=20)
    return plan


class FitContracts(unittest.TestCase):
    def test_plan_pins_closed_outcome_full_runtime_and_unchanged_budget(self):
        a, b = make_plan('cpu'), make_plan('cpu')
        self.assertEqual(a, b)
        self.assertEqual(a['prerequisite_sha256'], PINNED)
        self.assertEqual((a['epochs'], a['updates'], a['seconds_per_fit'], a['batch_recordings']), (20, 100, 1800, 4))
        self.assertEqual(a['complete_record_gradients'], 400)
        self.assertEqual((a['runtime']['cpu_threads'], a['runtime']['interop_threads']), (2, 1))
        self.assertFalse(a['runtime']['matmul_tf32'])
        self.assertEqual(a['optimizer'], dict(name='AdamW', learning_rate=.001, weight_decay=.0001, max_norm=1.))
        self.assertEqual([sum(r['role'] == role for r in a['population'])
                          for role in ('fit', 'development', 'diagnostic')], [20, 5, 15])
        for flag in ('training', 'automatic_retry', 'holdout_access', 'production_change', 'encoder_inference'):
            self.assertFalse(a[flag])
        for path in ('experiments/scaled_phase_fit/run.py', 'experiments/scaled_phase_fit/recorder.py',
                     'experiments/observed_adjoint/transport.py', 'experiments/training_observer/recorder.py'):
            self.assertIn(path, a['source_sha256'])

    def test_prerequisite_identity_drift_fails_closed(self):
        with patch('experiments.scaled_phase_fit.register.sha', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'prerequisite changed'):
                prerequisites()

    def test_metrics_and_all_old_accuracy_gates_are_reused_not_rewritten(self):
        self.assertIs(run.evaluate, previous.evaluate)
        self.assertIs(run.comparison_gate, previous.comparison_gate)
        self.assertEqual(run.ALL_KINDS, previous.ALL_KINDS)
        rows, summary, fits = passing_gate()
        self.assertTrue(run.comparison_gate(rows, summary, fits)['passed'])
        summary['diagnostic']['raw_common']['phase_mean_absolute_cycles'] = .1
        self.assertFalse(run.comparison_gate(rows, summary, fits)['passed'])

    def test_packets_preserve_full_inputs_and_exact_work_batch_weights(self):
        rows = authored()
        weights = work_weights(rows[:20])
        for row in rows[:20]:
            item = run.packet(row, False, weights[row['work']] / 4)
            self.assertEqual(item['payload']['features'].shape[1], len(row['features']))
            self.assertTrue(torch.equal(item['payload']['reference'][0], row['target']))
            self.assertEqual(item['scale'], 20 / (9 * sum(r['work'] == row['work'] for r in rows[:20])) / 4)
            self.assertEqual(torch.count_nonzero(run.packet(row, True)['payload']['features']), 0)
            self.assertGreater(torch.count_nonzero(row['features']), 0)

    def test_changed_initialization_fails_before_first_forward_or_optimizer(self):
        plan = tiny_plan()
        plan['initial_state_sha256'] = 'different'
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(PhaseSyncReadout, 'fields', side_effect=AssertionError('must not forward')):
                with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('must not step')):
                    with self.assertRaisesRegex(ValueError, 'initialization differs before fit'):
                        run.fit(authored(), 'cpu', False, Path(temporary), plan, 'authored')
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_cli_rejects_changed_budget_before_loading_music_or_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = make_plan('cpu')
            plan['epochs'] = 21
            plan_path = directory / 'changed-plan.json'
            run.write_json(plan_path, plan)
            args = SimpleNamespace(plan=plan_path, expected_plan_sha256=run.sha(plan_path),
                                   device='cpu', inputs=directory / 'absent-inputs', output=directory / 'run')
            with patch('argparse.ArgumentParser.parse_args', return_value=args), \
                    patch.object(run, 'load_records', side_effect=AssertionError('must not load music')):
                with self.assertRaisesRegex(ValueError, 'registered runtime'):
                    run.main()
            self.assertFalse(args.output.exists())

    def manual(self, rows, device, plan):
        torch.manual_seed(plan['seed'])
        model = PhaseSyncReadout().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        population = rows[:20]
        weights = work_weights(population)
        order = np.random.default_rng(plan['seed']).permutation(20)
        training = {}
        for offset in range(0, 20, 4):
            batch = [population[i] for i in order[offset:offset + 4]]
            optimizer.zero_grad(set_to_none=True)
            gradients = []
            for row in batch:
                item = run.packet(row, False, weights[row['work']] / 4)
                info = record_gradient(model, item['payload']['features'],
                                       item['payload']['reference'], item['payload']['valid'])
                gradients.append(info['gradient'].multiply(item['scale']))
                training.setdefault(row['work'], []).append(info['loss'])
            install_clipped(model, accumulate(gradients), 1.)
            optimizer.step()
        return model, previous.development_loss(model, rows, False), work_mean(training)

    def assert_full(self, device):
        rows, plan = authored(device), tiny_plan()
        expected, development, training = self.manual(rows, device, plan)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            actual, report = run.fit(rows, device, False, output, plan, 'authored')
            self.assertEqual(state_hash(actual), state_hash(expected))
            self.assertEqual(report['development_loss'], development)
            self.assertEqual(report['history'][0]['training_work_macro_loss'], training)
            self.assertEqual((report['updates'], report['epochs'], report['selected_epoch']), (5, 1, 1))
            self.assertEqual(report['conditioning']['record_gradients'], 20)
            self.assertEqual(report['conditioning']['update_receipts'], 5)
            self.assertEqual(report['conditioning']['alignment_underflows'], 0)
            self.assertIn('best.pt', report['snapshot_sha256'])
            best = torch.load(output / 'audio/best.pt', weights_only=True)
            self.assertEqual(best['best']['completed_updates'], 5)
            self.assertIsNone(best['scaled_accumulation']['transport']['clock_trace'])
            events = run.read_events(output / 'audio/journal.jsonl')
            self.assertEqual(report['conditioning'], conditioning(events))
            self.assertEqual(len([e for e in events if e['event'] == 'development_started']), 5)
            self.assertFalse(any('diagnostic-must-not-run' in json.dumps(e) for e in events))
            paired = copy.deepcopy(report)
            paired['kind'] = 'zero-audio'
            self.assertTrue(run.execution_gate([report, paired], plan))
            self.assertFalse(run.execution_gate([report, report], plan))
            incomplete = copy.deepcopy(report)
            incomplete['conditioning']['record_gradients'] -= 1
            self.assertFalse(run.execution_gate([report, incomplete], plan))
            self.assertFalse(run.execution_gate([report], plan))
            with self.assertRaises(FileExistsError):
                run.fit(rows, device, False, output, plan, 'authored')

    def test_full_authored_fit_matches_frozen_scaled_math_and_complete_epoch_selection(self):
        self.assert_full('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_full_cuda_authored_fit_matches_frozen_scaled_math(self):
        self.assert_full('cuda')

    def test_mid_batch_timeout_keeps_records_without_optimizer_entry(self):
        plan = tiny_plan()
        real = MusicalRecorder.check_deadline
        def timeout(observer):
            if len(observer.records) == 2:
                raise TimeoutError('authored deadline')
            return real(observer)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(MusicalRecorder, 'check_deadline', timeout):
                with self.assertRaisesRegex(TimeoutError, 'authored deadline'):
                    run.fit(authored(), 'cpu', False, output, plan, 'authored')
            saved = torch.load(output / 'audio/failure.pt', weights_only=True)
            self.assertEqual(saved['completed_updates'], 0)
            self.assertEqual(saved['cursor']['optimizer_step_status'], 'not_started')
            self.assertEqual(len(saved['scaled_accumulation']['weighted_records']), 2)
            self.assertIsNotNone(saved['scaled_accumulation']['transport']['clock_trace'])
            self.assertIsNone(saved['best'])
            self.assertFalse((output / 'audio.weights.npz').exists())
            evidence = run.failure_evidence(output)['audio']
            self.assertTrue(evidence['last_failed']['update_count_exact'])
            self.assertEqual(evidence['conditioning']['record_gradients'], 2)
            self.assertEqual(evidence['conditioning']['update_receipts'], 0)

    def test_development_failure_clears_training_trace_and_cannot_select_partial_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(run, 'score', return_value=float('nan')):
                with self.assertRaisesRegex(ValueError, 'finite development'):
                    run.fit(authored(), 'cpu', False, output, tiny_plan(), 'authored')
            saved = torch.load(output / 'audio/failure.pt', weights_only=True)
            self.assertEqual(saved['completed_updates'], 5)
            self.assertEqual(saved['cursor']['stage'], 'development')
            self.assertEqual(saved['current_record']['id'], 'development-0')
            self.assertIsNone(saved['scaled_accumulation']['transport']['clock_trace'])
            self.assertEqual(saved['scaled_accumulation']['weighted_records'], [])
            self.assertIsNone(saved['best'])
            self.assertFalse((output / 'audio.weights.npz').exists())

    def test_external_checkpoint_fault_keeps_current_weights_and_known_update_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(MusicalRecorder, 'checkpoint', side_effect=OSError('authored checkpoint fault')):
                with self.assertRaisesRegex(OSError, 'authored checkpoint fault'):
                    run.fit(authored(), 'cpu', False, output, tiny_plan(), 'authored')
            saved = torch.load(output / 'audio/failure.pt', weights_only=True)
            self.assertEqual(saved['completed_updates'], 5)
            self.assertEqual(saved['cursor']['stage'], 'checkpoint')
            self.assertIsNone(saved['cursor']['recording'])
            self.assertIsNone(saved['best'])

    def test_zero_norms_are_not_omitted_from_conditioning_denominators(self):
        events = [dict(event='record_gradient', field_norm_log2=None, parameter_norm_log2=None,
                       neural_vjp_passes=0, alignment_underflows=0),
                  dict(event='update_completed', gradient_norm_log2=None, max_norm=1., alignment_underflows=0)]
        result = conditioning(events)
        self.assertEqual(result['record_gradients'], 1)
        self.assertEqual(result['zero_parameter_gradients'], 1)
        self.assertEqual(result['batches_norm_above_clip'], 0)
        self.assertEqual(result['field_norm_log2'], dict(min=None, max=None))
        events[0]['field_norm_log2'] = float('inf')
        with self.assertRaisesRegex(ValueError, 'nonfinite conditioning'):
            conditioning(events)

    def test_final_hashing_and_report_persistence_cannot_escape_wall_budget(self):
        for delayed_operation in ('snapshot_hash', 'report_write'):
            with self.subTest(operation=delayed_operation), tempfile.TemporaryDirectory() as temporary:
                output, plan = Path(temporary), tiny_plan()
                delayed = False
                real_time, real_sha, real_write = time.monotonic, run.sha, run.write_json
                def clock():
                    return real_time() + (1801 if delayed else 0)
                def hash_file(path):
                    nonlocal delayed
                    value = real_sha(path)
                    if delayed_operation == 'snapshot_hash' and Path(path).suffix == '.pt':
                        delayed = True
                    return value
                def write_file(path, value):
                    nonlocal delayed
                    real_write(path, value)
                    if delayed_operation == 'report_write' and Path(path).name == 'audio.fit.json':
                        delayed = True
                with patch.object(run.time, 'monotonic', clock), patch.object(run, 'sha', hash_file), \
                        patch.object(run, 'write_json', write_file):
                    with self.assertRaisesRegex(TimeoutError, 'wall budget exhausted'):
                        run.fit(authored(), 'cpu', False, output, plan, 'authored')
                self.assertTrue(delayed)
                saved = torch.load(output / 'audio/failure.pt', weights_only=True)
                self.assertEqual(saved['cursor']['stage'], 'selected_export')
                self.assertEqual(saved['completed_updates'], 5)
                self.assertIsNotNone(saved['best'])
                self.assertFalse((output / 'zero-audio').exists())


if __name__ == '__main__':
    unittest.main()
