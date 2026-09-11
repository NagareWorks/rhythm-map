"""Authored tiny updates/faults only; no music, fitted checkpoint or dataset I/O."""
import copy
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.training_observer import recorder as mod
from experiments.training_observer.recorder import FitRecorder, atomic_save, rng_state
from experiments.training_observer.phase_trace import TracedPhaseSyncReadout, traced_synchronize, replay_backward
from experiments.phase_sync.model import PhaseSyncReadout
from experiments.phase_sync.scan import MAX_GAIN
from experiments.phase_sync_fit.diagnostics.stability import opposed_observations

torch.set_num_threads(1)


def record(name, fail=False, scale=.5):
    return dict(id=name, packet_sha256='authored-not-a-music-packet', scale=scale,
                payload=dict(x=torch.tensor([[1., 2.]], dtype=torch.float64), fail=fail))


class BadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        raise ValueError('authored backward failure')


def loss_fn(model, payload):
    value = model(payload['x']).square().mean()
    return BadBackward.apply(value) if payload['fail'] else value


class PartialStep(torch.optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        self.param_groups[0]['params'][0].add_(1.)
        raise ValueError('authored partial step failure')


class InfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float('inf'))


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'new-fit'
        torch.manual_seed(142)
        self.model = torch.nn.Linear(2, 1, dtype=torch.float64)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=.0001)
        self.recorder = FitRecorder(self.path, self.model, self.optimizer,
                                    dict(protocol='authored-test-only'), max_updates=4)

    def load(self, name):
        return torch.load(self.path / name, weights_only=True)

    def events(self):
        return [json.loads(line) for line in (self.path / 'journal.jsonl').read_text().splitlines()]

    def assertTree(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assertTree(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assertTree(x, y)
        else:
            self.assertEqual(a, b)

    def test_observer_preserves_actual_optimizer_math_and_rng(self):
        plain = copy.deepcopy(self.model)
        optimizer = torch.optim.AdamW(plain.parameters(), lr=.001, weight_decay=.0001)
        rows = [record('first', scale=.2), record('second', scale=.8)]
        before = rng_state(self.model)
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            for row in rows:
                (loss_fn(plain, row['payload']) * row['scale']).backward()
            torch.nn.utils.clip_grad_norm_(plain.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            self.recorder.update(1, update, rows, loss_fn)
        self.assertTree(self.model.state_dict(), plain.state_dict())
        self.assertTree(self.optimizer.state_dict(), optimizer.state_dict())
        self.assertTree(before, rng_state(self.model))
        self.assertEqual(self.load('after-update.pt')['completed_updates'], 2)
        self.assertEqual([e['completed_updates'] for e in self.events() if e['event'] == 'update_completed'], [1, 2])

    def test_backward_failure_keeps_current_best_optimizer_partial_gradients_and_exact_record(self):
        self.recorder.update(1, 0, [record('good')], loss_fn)
        self.assertTrue(self.recorder.checkpoint(1, 3.))
        best = copy.deepcopy(self.model.state_dict())
        self.recorder.update(2, 0, [record('good')], loss_fn)
        current = copy.deepcopy(self.model.state_dict())
        with self.assertRaisesRegex(ValueError, 'authored backward failure'):
            self.recorder.update(3, 7, [record('partial-good'), record('bad', fail=True)], loss_fn)
        failed = self.load('failure.pt')
        self.assertEqual(failed['completed_updates'], 2)
        self.assertEqual(failed['cursor'], dict(stage='backward', epoch=3, batch=7, recording='bad',
            record_index=1, batch_recordings=['partial-good', 'bad'], optimizer_step_status='not_started'))
        self.assertTree(failed['model'], current)
        self.assertTree(failed['best']['model'], best)
        self.assertTree(self.load('best.pt')['model'], best)
        self.assertTree(failed['optimizer'], self.optimizer.state_dict())
        self.assertTrue(all(g is not None for g in failed['gradients'].values()))
        self.assertTrue(all(g['nonfinite'] == 0 for g in failed['gradient_summaries'].values()))
        self.assertTrue(failed['current_record']['payload']['fail'])
        self.assertTree(failed['before_record_rng'], failed['rng'])
        self.assertTrue(self.events()[-1]['update_count_exact'])
        self.assertEqual(self.events()[-1]['snapshot_sha256'], mod.hashlib.sha256((self.path / 'failure.pt').read_bytes()).hexdigest())
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.recorder.update(3, 7, [record('retry')], loss_fn)

    def test_partial_optimizer_step_is_unknown_not_counted_as_success(self):
        model = torch.nn.Linear(2, 1, dtype=torch.float64)
        optimizer = PartialStep(model.parameters(), lr=.1)
        observer = FitRecorder(Path(self.temporary.name) / 'partial', model, optimizer,
                               dict(protocol='authored'), max_updates=1)
        with self.assertRaisesRegex(ValueError, 'partial step'):
            observer.update(1, 0, [record('good')], loss_fn)
        failed = torch.load(observer.output / 'failure.pt', weights_only=True)
        before = torch.load(observer.output / 'before-update.pt', weights_only=True)
        self.assertEqual(failed['completed_updates'], 0)
        self.assertEqual(failed['cursor']['optimizer_step_status'], 'entered')
        self.assertFalse(torch.equal(failed['model']['weight'], before['model']['weight']))
        event = json.loads((observer.output / 'journal.jsonl').read_text().splitlines()[-1])
        self.assertFalse(event['update_count_exact'])

    def test_pre_step_journal_failure_does_not_claim_optimizer_entry(self):
        before = copy.deepcopy(self.model.state_dict())
        real_event = self.recorder._event
        def fail_intent(event, **fields):
            if event == 'optimizer_step_prepared':
                raise OSError('authored intent journal failure')
            return real_event(event, **fields)
        with patch.object(self.recorder, '_event', side_effect=fail_intent):
            with self.assertRaisesRegex(OSError, 'intent journal failure'):
                self.recorder.update(1, 0, [record('good')], loss_fn)
        failed = self.load('failure.pt')
        self.assertEqual(failed['cursor']['stage'], 'optimizer_step_prepare')
        self.assertEqual(failed['cursor']['optimizer_step_status'], 'not_started')
        self.assertEqual(failed['completed_updates'], 0)
        self.assertTree(failed['model'], before)
        self.assertTrue(self.events()[-1]['update_count_exact'])

    def test_checkpoint_ties_keep_earlier_and_disk_failure_keeps_old_complete_file(self):
        self.recorder.checkpoint(1, 2.)
        before = (self.path / 'best.pt').read_bytes()
        self.assertFalse(self.recorder.checkpoint(2, 2.))
        self.assertEqual((self.path / 'best.pt').read_bytes(), before)
        with patch.object(torch, 'save', side_effect=OSError('authored full disk')):
            with self.assertRaises(OSError):
                atomic_save(self.path / 'best.pt', {'unfinished': True})
        self.assertEqual((self.path / 'best.pt').read_bytes(), before)
        self.assertFalse(list(self.path.glob('*.tmp')))

    def test_secondary_snapshot_or_trace_failure_does_not_hide_original(self):
        real_save = atomic_save
        def fail_snapshot(path, payload):
            if path.name == 'failure.pt':
                raise OSError('authored full disk')
            return real_save(path, payload)
        def fail_trace():
            raise RuntimeError('authored trace failure')
        self.recorder.trace = fail_trace
        with patch.object(mod, 'atomic_save', side_effect=fail_snapshot):
            with self.assertRaisesRegex(ValueError, 'authored backward failure') as caught:
                self.recorder.update(1, 0, [record('bad', fail=True)], loss_fn)
        self.assertIn('snapshot: OSError', caught.exception.__notes__[0])
        self.assertEqual(self.events()[-1]['capture_errors'], ['trace: RuntimeError', 'snapshot: OSError'])
        self.assertIsNone(self.events()[-1]['snapshot_sha256'])

    def test_forward_and_development_failures_keep_record_identity(self):
        def fail(model, payload):
            raise ValueError('authored forward failure')
        with self.assertRaisesRegex(ValueError, 'forward failure'):
            self.recorder.development_record(1, record('development-case'), fail)
        failure = self.load('failure.pt')
        self.assertEqual(failure['cursor']['recording'], 'development-case')
        self.assertEqual(failure['cursor']['stage'], 'development')
        self.assertEqual(failure['completed_updates'], 0)
        self.assertIsNone(failure['best'])

    def test_clipping_failure_saves_nonfinite_gradients_without_json_nan(self):
        def bad(model, payload):
            return InfiniteBackward.apply(loss_fn(model, payload))
        with self.assertRaises(RuntimeError):
            self.recorder.update(1, 0, [record('bad-gradient')], bad)
        failure = self.load('failure.pt')
        self.assertEqual(failure['cursor']['stage'], 'clip')
        self.assertEqual(failure['completed_updates'], 0)
        self.assertTrue(any(x['nonfinite'] for x in failure['gradient_summaries'].values()))
        self.assertTrue(self.events()[-1]['update_count_exact'])

    def test_budget_and_existing_output_fail_closed(self):
        self.recorder.max_updates = 1
        self.recorder.update(1, 0, [record('good')], loss_fn)
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.recorder.update(1, 1, [record('extra')], loss_fn)
        self.assertEqual(self.recorder.completed_updates, 1)
        with self.assertRaises(FileExistsError):
            FitRecorder(self.path, self.model, self.optimizer, dict(protocol='authored'), max_updates=1)
        with self.assertRaisesRegex(ValueError, 'private output'):
            FitRecorder(mod.ROOT / 'forbidden-observer-output', self.model, self.optimizer,
                        dict(protocol='authored'), max_updates=1)

    def test_snapshot_retains_rng_states_without_consuming_them(self):
        random.seed(6)
        np.random.seed(7)
        torch.manual_seed(8)
        before = rng_state(self.model)
        self.recorder._save('rng.pt')
        self.assertTree(before, self.load('rng.pt')['rng'])
        self.assertTree(before, rng_state(self.model))

    def test_private_output_remains_anchored_after_caller_changes_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.temporary.name)
            observer = FitRecorder('relative-fit', self.model, self.optimizer,
                                   dict(protocol='authored'), max_updates=1)
        finally:
            os.chdir(previous)
        observer.update(1, 0, [record('good')], loss_fn)
        self.assertEqual(observer.output, Path(self.temporary.name).resolve() / 'relative-fit')
        self.assertTrue((observer.output / 'after-update.pt').is_file())


class PhaseTraceTests(unittest.TestCase):
    def test_adapter_preserves_forward_gradients_and_inference(self):
        torch.manual_seed(142)
        plain, traced = PhaseSyncReadout(), TracedPhaseSyncReadout()
        traced.load_state_dict(plain.state_dict())
        inputs = torch.randn(1, 81, 512) * .1
        a, b = plain(inputs), traced(inputs)
        torch.testing.assert_close(a.cycles, b.cycles, rtol=0, atol=0)
        a.cycles.square().mean().backward()
        b.cycles.square().mean().backward()
        for x, y in zip(plain.parameters(), traced.parameters()):
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        self.assertEqual(traced.last_trace['jacobian'].shape, (5, 1, 80))
        self.assertIsNotNone(traced.last_trace['upstream'])
        grads = replay_backward(traced.last_trace)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values()))
        with torch.inference_mode():
            torch.testing.assert_close(plain(inputs).cycles, traced(inputs).cycles, rtol=0, atol=0)
        self.assertIsNone(traced.last_trace['jacobian'])
        with self.assertRaisesRegex(ValueError, 'capture required'):
            replay_backward(traced.last_trace)

    def test_new_authored_failure_is_persisted_and_replayed_without_optimizer_update(self):
        model = torch.nn.Module()
        model.register_parameter('period', torch.nn.Parameter(torch.full((1, 3000), -1.)))
        owner = SimpleNamespace(last_trace=None)
        row = dict(id='authored-opposed-not-music', scale=1., payload=dict(
            observation=torch.from_numpy(opposed_observations()), initial=torch.zeros(1),
            gain=torch.tensor(MAX_GAIN / 2, dtype=torch.float64)))
        def loss(model, payload):
            return traced_synchronize(owner, model.period, payload['observation'],
                                      payload['initial'], payload['gain']).cycles[:, -1].sum()
        with tempfile.TemporaryDirectory() as temporary:
            observer = FitRecorder(Path(temporary) / 'failed', model, torch.optim.AdamW(model.parameters()),
                                   dict(protocol='authored-no-music'), max_updates=1, trace=lambda: owner.last_trace)
            with self.assertRaisesRegex(ValueError, 'nonfinite clock gradient'):
                observer.update(1, 0, [row], loss)
            failed = torch.load(observer.output / 'failure.pt', weights_only=True)
            self.assertEqual(failed['completed_updates'], 0)
            self.assertEqual(failed['cursor']['recording'], row['id'])
            self.assertEqual(failed['diagnostic']['upstream'].shape, (1, 3001))
            self.assertGreater(failed['diagnostic']['state_jacobian_summary']['finite_min'], 1.)
            with self.assertRaisesRegex(ValueError, 'nonfinite clock gradient'):
                replay_backward(failed['diagnostic'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_capture_keeps_device_identity_without_mutating_gradients(self):
        torch.manual_seed(142)
        plain, traced = PhaseSyncReadout().cuda(), TracedPhaseSyncReadout().cuda()
        traced.load_state_dict(plain.state_dict())
        inputs = torch.zeros(1, 81, 512, device='cuda')
        a, b = plain(inputs), traced(inputs)
        a.cycles.sum().backward()
        b.cycles.sum().backward()
        for x, y in zip(plain.parameters(), traced.parameters()):
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        self.assertTrue(all(d.startswith('cuda:') for d in traced.last_trace['input_devices']))
        self.assertEqual(traced.last_trace['upstream'].device.type, 'cpu')
        self.assertIn(str(torch.cuda.current_device()), rng_state(traced)['torch_cuda'])
        self.assertTrue(all(g.is_cuda for g in replay_backward(traced.last_trace, 'cuda').values()))


if __name__ == '__main__':
    unittest.main()
