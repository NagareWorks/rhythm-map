"""Authored protocol/optimizer/failure tests; never open musical packets."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.coupled_clock.run import state_hash, work_weights
from experiments.scaled_phase_fit.test_fit import authored
from experiments.separated_evidence.model import SeparatedEvidence, SharedEvidence
from experiments.separated_evidence.supervision import losses
from experiments.separated_evidence.training import TaskTrainer
from experiments.separated_evidence_fit import run, register
from experiments.separated_evidence_fit.recorder import Recorder, Selection, SharedTrainer, branch


def tiny_plan(device='cpu'):
    # Only generated-input unit tests; the public CLI rejects this alteration.
    return dict(register.make_plan(device), epochs=1, paired_updates=5)


class FitTests(unittest.TestCase):
    def test_registration_pins_four_arms_population_runtime_and_selection(self):
        a, b = register.make_plan('cpu'), register.make_plan('cpu')
        self.assertEqual(a, b)
        self.assertEqual(a['arms'], list(register.ARMS))
        self.assertEqual([sum(r['role'] == role for r in a['population'])
                          for role in ('fit', 'development', 'diagnostic')], [20, 5, 15])
        self.assertEqual((a['epochs'], a['paired_updates'], a['seconds_per_arm'], a['watchdog_seconds']), (20, 100, 1800, 7500))
        self.assertEqual(a['parameters'], dict(shared=24003, separated=47907))
        self.assertEqual(a['optimizer_calls'], dict(shared=100, separated=200))
        self.assertEqual((a['runtime']['cpu_threads'], a['runtime']['interop_threads']), (2, 1))
        self.assertEqual(a['source_sha256'], register.sources())
        self.assertFalse(any(a[k] for k in ('old_weights_loaded', 'training', 'automatic_retry', 'encoder_inference',
                                           'holdout_access', 'production_change', 'coherent_clock')))

    def test_changed_prerequisite_is_rejected(self):
        with patch.object(register, 'sha', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'prerequisite changed'):
                register.sources()

    def test_modified_plan_fails_before_input_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run.write_json(root / 'plan.json', tiny_plan())
            args = SimpleNamespace(inputs=root / 'absent', output=root / 'output', device='cpu',
                plan=root / 'plan.json', expected_plan_sha256=run.sha(root / 'plan.json'))
            with patch('argparse.ArgumentParser.parse_args', return_value=args), \
                    patch.object(run, 'load_records', side_effect=AssertionError('no musical access')):
                with self.assertRaisesRegex(ValueError, 'registered runtime'):
                    run.main()
            self.assertFalse(args.output.exists())

    def test_wrong_initialization_stops_before_any_forward(self):
        plan = tiny_plan()
        plan['initial_state_sha256']['shared'] = 'wrong'
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(SharedEvidence, 'forward', side_effect=AssertionError('no forward')):
            with self.assertRaisesRegex(ValueError, 'initialization differs'):
                run.fit(authored(), 'cpu', 'shared-natural', Path(temporary), plan, 'authored')
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def manual(self, records, architecture, zero, device):
        shared, separated = register.initial_models(device)
        model = shared if architecture == 'shared' else separated
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        trainers = {k: TaskTrainer(getattr(separated, k), max_updates=5) for k in ('tempo', 'phase')}
        population = [r for r in records if r['role'] == 'fit']
        order, weights = np.random.default_rng(142).permutation(20), work_weights(population)
        for offset in range(0, 20, 4):
            batch = [population[i] for i in order[offset:offset + 4]]
            weighted = [run.packet(r, zero, weights[r['work']] / len(batch)) for r in batch]
            if architecture == 'shared':
                optimizer.zero_grad(set_to_none=True)
                for row in weighted:
                    (sum(losses(model(row['features']), row['reference'], row['valid']).values()) * row['scale']).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            else:
                for trainer in trainers.values():
                    trainer.update(weighted)
        scores = run.development(model, records, zero, SimpleNamespace(inference_stage=lambda *a: None, check=lambda: None))
        return (SeparatedEvidence(model) if architecture == 'shared' else model), scores

    def full(self, device):
        records, plan, fits = authored(device), tiny_plan(device), []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm in register.ARMS:
                architecture, variant = arm.split('-')
                expected, scores = self.manual(records, architecture, variant == 'zero', device)
                model, report = run.fit(records, device, arm, root, plan, 'authored')
                self.assertEqual(state_hash(model), state_hash(expected), arm)
                self.assertEqual(report['history'][0]['development'], scores)
                self.assertEqual(report['receipts']['task_records'], dict(tempo=20, phase=20))
                self.assertEqual(report['optimizer_calls'], {'shared': 5} if architecture == 'shared' else {'tempo': 5, 'phase': 5})
                self.assertFalse(report['coherent_clock'])
                events = (root / arm / 'journal.jsonl').read_text()
                self.assertNotIn('diagnostic-must-not-run', events)
                for task in ('tempo', 'phase'):
                    saved = torch.load(root / arm / ('best-' + task + '.pt'), map_location='cpu', weights_only=True)
                    self.assertEqual(saved['paired_updates'], 5)
                    self.assertEqual(saved['weighted_records'], [])
                    self.assertEqual(saved['selection_epoch'], 1)
                fits.append(report)
            self.assertTrue(run.execution_gate(fits, plan))
            for mutate in (lambda f: f['optimizer_calls'].update(shared=4),
                           lambda f: f['selected']['tempo'].update(epoch=2),
                           lambda f: f['history'][0].update(epoch=2),
                           lambda f: f.update(elapsed_s=float('nan'))):
                bad = copy.deepcopy(fits)
                mutate(bad[0])
                self.assertFalse(run.execution_gate(bad, plan))
            self.assertFalse(run.execution_gate(fits[::-1], plan))

    def test_all_four_authored_arms_match_standalone_updates_and_selection(self):
        self.full('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_four_arm_native_update_and_selection_parity(self):
        self.full('cuda')

    def test_both_architectures_get_identical_task_specific_selection_privileges(self):
        shared, separated = register.initial_models('cpu')
        for model in (shared, separated):
            select = Selection()
            select.observe(1, dict(tempo=.2, phase=.4), model)
            tempo = state_hash(branch(model, 'tempo'))
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(.01)
            select.observe(2, dict(tempo=.3, phase=.1), model)
            phase = state_hash(branch(model, 'phase'))
            select.observe(3, dict(tempo=.2, phase=.1), model)
            selected = select.export(model)
            self.assertEqual([select.best[k]['epoch'] for k in ('tempo', 'phase')], [1, 2])
            self.assertEqual(state_hash(selected.tempo), tempo)
            self.assertEqual(state_hash(selected.phase), phase)
            with self.assertRaisesRegex(ValueError, 'complete finite'):
                select.observe(4, dict(tempo=.01, phase=float('nan')), model)
            self.assertEqual(select.epoch, 3)

    def test_second_task_entry_failure_preserves_first_commit_and_unknown_boundary(self):
        original, calls = torch.optim.AdamW.step, 0
        def step(optimizer, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyError('authored optimizer entry failure')
            return original(optimizer, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(torch.optim.AdamW, 'step', step):
                with self.assertRaisesRegex(KeyError, 'entry failure'):
                    run.fit(authored(), 'cpu', 'separated-natural', root, tiny_plan(), 'authored')
            failed = torch.load(root / 'separated-natural/failure.pt', weights_only=True)
            self.assertEqual(failed['paired_updates'], 0)
            self.assertEqual(failed['trainers']['tempo']['returned_updates'], 1)
            self.assertEqual(failed['trainers']['phase']['returned_updates'], 0)
            self.assertEqual(failed['trainers']['phase']['stage'], 'optimizer_entered')
            self.assertEqual(len(failed['weighted_records']), 4)
            self.assertTrue(failed['failed'])
            self.assertFalse((root / 'separated-natural.weights.npz').exists())

    def test_timeout_after_returned_call_cannot_erase_it(self):
        original = Recorder.check
        def check(recorder):
            if recorder.cursor['stage'] == 'update_returned':
                raise TimeoutError('authored after return')
            return original(recorder)
        with tempfile.TemporaryDirectory() as temporary, patch.object(Recorder, 'check', check):
            root = Path(temporary)
            with self.assertRaisesRegex(TimeoutError, 'after return'):
                run.fit(authored(), 'cpu', 'shared-natural', root, tiny_plan(), 'authored')
            failed = torch.load(root / 'shared-natural/failure.pt', weights_only=True)
            self.assertEqual(failed['trainers']['shared']['returned_updates'], 1)
            self.assertEqual(failed['trainers']['shared']['stage'], 'optimizer_returned')

    def test_incomplete_development_cannot_select_or_export(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(run, 'development', return_value=dict(tempo=.1, phase=float('nan'))):
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'complete finite'):
                run.fit(authored(), 'cpu', 'shared-natural', root, tiny_plan(), 'authored')
            failed = torch.load(root / 'shared-natural/failure.pt', weights_only=True)
            self.assertEqual(failed['selection'], {})
            self.assertEqual(failed['paired_updates'], 5)
            self.assertEqual(failed['weighted_records'], [])

    def test_failed_arm_stops_later_fits_and_all_diagnostic_forwards(self):
        with patch.object(run, 'fit', side_effect=[(None, {'arm': 'shared-natural'}), RuntimeError('second arm')]) as fit, \
                patch.object(run, 'evaluate', side_effect=AssertionError('must not evaluate')) as evaluate:
            with self.assertRaisesRegex(RuntimeError, 'second arm'):
                run.execute([], 'cpu', Path('unused'), {}, 'authored', {}, lambda: None)
            self.assertEqual(fit.call_count, 2)
            evaluate.assert_not_called()

    def test_export_persistence_is_inside_wall_budget(self):
        late, original, original_time = False, run.write_json, run.time.monotonic
        def write(path, value):
            nonlocal late
            original(path, value)
            if path.name.endswith('.fit.json'):
                late = True
        def clock():
            return original_time() + (1801 if late else 0)
        with tempfile.TemporaryDirectory() as temporary, patch.object(run, 'write_json', write), \
                patch.object(run.time, 'monotonic', clock):
            root = Path(temporary)
            with self.assertRaisesRegex(TimeoutError, 'arm wall budget'):
                run.fit(authored(), 'cpu', 'shared-natural', root, tiny_plan(), 'authored')
            failed = torch.load(root / 'shared-natural/failure.pt', weights_only=True)
            self.assertEqual(failed['paired_updates'], 5)
            self.assertEqual(failed['cursor']['stage'], 'selected_export')

    def test_failed_journal_does_not_prevent_failure_snapshot_or_mask_primary(self):
        shared, _ = register.initial_models('cpu')
        with tempfile.TemporaryDirectory() as temporary:
            recorder = Recorder(Path(temporary) / 'run', shared, {}, updates=1, deadline=run.time.monotonic() + 60)
            original = RuntimeError('primary')
            with patch.object(recorder, 'event', side_effect=OSError('journal')):
                recorder.abort(original)
            self.assertTrue((recorder.output / 'failure.pt').exists())
            self.assertIn('OSError', original.__notes__[0])

    def test_returned_nonfinite_shared_optimizer_state_is_counted_and_closed(self):
        shared, _ = register.initial_models('cpu')
        trainer = SharedTrainer(shared, 1)
        original = trainer.optimizer.step
        def step():
            original()
            next(iter(trainer.optimizer.state.values()))['exp_avg'].fill_(float('nan'))
        with patch.object(trainer.optimizer, 'step', step):
            with self.assertRaisesRegex(ValueError, 'nonfinite returned'):
                trainer.update([run.packet(authored()[0], False)])
        self.assertEqual(trainer.returned_updates, 1)
        self.assertTrue(trainer.failed)
        with self.assertRaisesRegex(ValueError, 'closed'):
            trainer.update([])


if __name__ == '__main__':
    unittest.main()
