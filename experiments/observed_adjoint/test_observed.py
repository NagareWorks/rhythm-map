"""Synthetic updates and injected faults; no old weights, music or fit retry."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.coupled_clock.supervision import clock_loss
from experiments.training_observer.recorder import rng_state
from experiments.training_observer.test_recorder import PartialStep
from experiments.scaled_adjoint.scaled import Gradient, accumulate
from experiments.scaled_adjoint.transport import record_gradient, install_clipped
from experiments.observed_adjoint.transport import ObservedTransport, pack_gradient, unpack_gradient, replay_field_adjoint
from experiments.observed_adjoint.recorder import ScaledFitRecorder
from experiments.observed_adjoint.fixtures import fixture

torch.set_num_threads(1)


def assert_tree(test, left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for name in left:
            assert_tree(test, left[name], right[name])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree(test, a, b)
    else:
        test.assertEqual(left, right)


class TransportTests(unittest.TestCase):
    def test_safe_gradient_packet_roundtrip_and_ownership(self):
        gradient = Gradient.create(dict(p=np.array([3., -4.])), exponent=10000)
        packet = pack_gradient(gradient)
        stream = io.BytesIO()
        torch.save(packet, stream)
        stream.seek(0)
        restored = unpack_gradient(torch.load(stream, weights_only=True))
        assert_tree(self, pack_gradient(restored), packet)
        packet['blocks']['p'].zero_()
        np.testing.assert_array_equal(restored.blocks['p'], gradient.blocks['p'])
        for invalid in ({}, dict(schema='wrong'), dict(pack_gradient(gradient), exponent=True),
                        dict(pack_gradient(gradient), blocks={'p': torch.ones(2)})):
            with self.assertRaises(ValueError):
                unpack_gradient(invalid)

    def compare_transport(self, kind, cells, device):
        model, rows = fixture(kind, cells, device=device)
        row = rows[0]
        before = copy.deepcopy(model.state_dict())
        old = record_gradient(model, **row['payload'])
        transport = ObservedTransport()
        gradient, loss = transport.run(model, row['payload'], clock_loss)
        assert_tree(self, pack_gradient(gradient), pack_gradient(old['gradient']))
        self.assertEqual(loss, old['loss'])
        assert_tree(self, transport.last_trace, old['trace'])
        assert_tree(self, before, model.state_dict())
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        if kind == 'opposed':
            self.assertGreaterEqual(len(transport.pieces), 4)
            self.assertGreater(gradient.norm_log2(), 128)
            self.assertGreater(transport.last_trace['state_jacobian_summary']['finite_min'], 1.05)
            with self.assertRaisesRegex(ValueError, 'nonfinite clock gradient'):
                clock_loss(model(row['payload']['features']), row['payload']['reference'], row['payload']['valid'])['total'].backward()
        snapshot = transport.snapshot()
        replay = replay_field_adjoint(snapshot)
        assert_tree(self, pack_gradient(replay), snapshot['field_gradient'])

    def test_staged_transport_exactly_matches_frozen_normal_reference(self):
        self.compare_transport('natural', 61, 'cpu')

    def test_full_cnn_authored_overflow_uses_multiple_bands_without_changing_math(self):
        self.compare_transport('opposed', 3000, 'cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_full_cnn_multi_band_reference_and_replay(self):
        self.compare_transport('opposed', 3000, 'cuda')


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def observer(self, kind='natural', cells=61, device='cpu', optimizer_type=torch.optim.AdamW, name='observed'):
        model, rows = fixture(kind, cells, device=device)
        optimizer = optimizer_type(model.parameters(), lr=.001, weight_decay=.0001)
        observer = ScaledFitRecorder(self.root / name, model, optimizer,
                                     dict(protocol='authored-observed-only'), max_updates=3)
        return observer, rows

    def load(self, observer, name='failure.pt'):
        return torch.load(observer.output / name, weights_only=True)

    def events(self, observer):
        return [json.loads(line) for line in (observer.output / 'journal.jsonl').read_text().splitlines()]

    def compare_updates(self, device):
        observer, rows = self.observer(device=device)
        plain = copy.deepcopy(observer.model)
        optimizer = torch.optim.AdamW(plain.parameters(), lr=.001, weight_decay=.0001)
        before_rng = rng_state(plain)
        for index in range(2):
            optimizer.zero_grad(set_to_none=True)
            gradients = [record_gradient(plain, **row['payload'])['gradient'].multiply(row['scale']) for row in rows]
            install_clipped(plain, accumulate(gradients))
            optimizer.step()
            observer.update(index + 1, 0, rows)
            assert_tree(self, observer.model.state_dict(), plain.state_dict())
            assert_tree(self, observer.optimizer.state_dict(), optimizer.state_dict())
            self.assertTrue(observer.checkpoint(index + 1, 3. - index))
        assert_tree(self, before_rng, rng_state(plain))
        saved = self.load(observer, 'after-update.pt')
        self.assertEqual(saved['completed_updates'], 2)
        self.assertEqual(len(saved['scaled_accumulation']['weighted_records']), 2)
        self.assertIsNotNone(saved['scaled_accumulation']['batch_gradient'])
        self.assertEqual(saved['cursor']['optimizer_step_status'], 'returned')
        self.assertTrue(all(e['gradient_norm_log2'] is not None for e in self.events(observer) if e['event'] == 'update_completed'))

    def test_full_observed_updates_match_frozen_manual_updates_and_rng(self):
        self.compare_updates('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_full_cuda_observed_updates_match_manual_and_rng(self):
        self.compare_updates('cuda')

    def test_later_record_failure_keeps_prior_weighted_records_best_state_and_exact_stage(self):
        observer, rows = self.observer()
        observer.update(1, 0, rows)
        observer.checkpoint(1, 2.)
        before = copy.deepcopy(observer.model.state_dict())
        bad = copy.deepcopy(rows)
        bad[1]['payload']['features'] = bad[1]['payload']['features'].clone()
        bad[1]['payload']['features'][0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite features'):
            observer.update(2, 4, bad)
        saved = self.load(observer)
        self.assertEqual(saved['completed_updates'], 1)
        self.assertEqual(saved['cursor']['recording'], bad[1]['id'])
        self.assertEqual(saved['cursor']['transport_stage'], 'fields')
        self.assertEqual(len(saved['scaled_accumulation']['weighted_records']), 1)
        self.assertIsNone(saved['scaled_accumulation']['transport']['clock_trace'])
        self.assertTrue(torch.isnan(saved['current_record']['payload']['features']).any())
        assert_tree(self, before, saved['model'])
        assert_tree(self, before, saved['best']['model'])
        self.assertTrue(self.events(observer)[-1]['update_count_exact'])
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            observer.update(2, 4, rows)

    def test_middle_band_failure_retains_completed_pieces_and_replayable_clock(self):
        observer, rows = self.observer('opposed', 3000)
        calls = []
        def fail(gradient):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError('authored third VJP failure')
        hook = observer.model.gain_logit.register_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, 'third VJP'):
                observer.update(1, 0, rows)
        finally:
            hook.remove()
        saved = self.load(observer)
        state = saved['scaled_accumulation']['transport']
        self.assertEqual((state['stage'], state['band_index'], len(state['completed_bands'])), ('neural_vjp', 2, 2))
        self.assertEqual(saved['completed_updates'], 0)
        self.assertIsNone(state['parameter_gradient'])
        self.assertEqual(saved['scaled_accumulation']['weighted_records'], [])
        self.assertTrue(all(g is None for g in saved['gradients'].values()))
        assert_tree(self, pack_gradient(replay_field_adjoint(state)), state['field_gradient'])

    def test_clip_failure_keeps_entire_batch_before_optimizer_entry(self):
        observer, rows = self.observer()
        with patch('experiments.observed_adjoint.recorder.install_clipped', side_effect=ValueError('authored clip fault')):
            with self.assertRaisesRegex(ValueError, 'clip fault'):
                observer.update(1, 0, rows)
        saved = self.load(observer)
        self.assertEqual(saved['cursor']['transport_stage'], 'gradient_install')
        self.assertEqual(saved['cursor']['stage'], 'clip')
        self.assertEqual(saved['cursor']['optimizer_step_status'], 'not_started')
        state = saved['scaled_accumulation']
        expected = accumulate([unpack_gradient(r['gradient']) for r in state['weighted_records']])
        assert_tree(self, pack_gradient(expected), state['batch_gradient'])

    def test_partial_step_remains_unknown_and_intent_write_failure_remains_pre_step(self):
        observer, rows = self.observer(optimizer_type=PartialStep)
        with self.assertRaisesRegex(ValueError, 'partial step'):
            observer.update(1, 0, rows)
        saved = self.load(observer)
        self.assertEqual(saved['cursor']['optimizer_step_status'], 'entered')
        self.assertEqual(saved['completed_updates'], 0)
        self.assertIsNotNone(saved['scaled_accumulation']['batch_gradient'])
        self.assertFalse(self.events(observer)[-1]['update_count_exact'])

    def test_intent_journal_failure_is_not_an_entered_optimizer(self):
        observer, rows = self.observer()
        before = copy.deepcopy(observer.model.state_dict())
        real_event = observer._event
        def fail(event, **fields):
            if event == 'optimizer_step_prepared':
                raise OSError('authored intent fault')
            return real_event(event, **fields)
        with patch.object(observer, '_event', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'intent fault'):
                observer.update(1, 0, rows)
        saved = self.load(observer)
        self.assertEqual(saved['cursor']['optimizer_step_status'], 'not_started')
        assert_tree(self, saved['model'], before)
        self.assertTrue(self.events(observer)[-1]['update_count_exact'])

    def test_second_record_start_failure_does_not_mislabel_the_previous_trace(self):
        observer, rows = self.observer()
        real_event = observer._event
        def fail(event, **fields):
            if event == 'recording_started' and observer.cursor['record_index'] == 1:
                raise OSError('authored second record start fault')
            return real_event(event, **fields)
        with patch.object(observer, '_event', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'second record start fault'):
                observer.update(1, 0, rows)
        saved = self.load(observer)
        state = saved['scaled_accumulation']['transport']
        self.assertEqual(saved['cursor']['recording'], rows[1]['id'])
        self.assertEqual(saved['cursor']['transport_stage'], 'idle')
        self.assertEqual(state['stage'], 'idle')
        self.assertIsNone(state['clock_trace'])
        self.assertIsNone(state['field_gradient'])
        self.assertEqual(state['completed_bands'], [])
        self.assertEqual(len(saved['scaled_accumulation']['weighted_records']), 1)
        assert_tree(self, saved['current_record'], rows[1])
        self.assertEqual(saved['completed_updates'], 0)
        self.assertEqual(saved['cursor']['optimizer_step_status'], 'not_started')

    def test_backward_start_journal_failure_keeps_the_current_upstream_stage(self):
        observer, rows = self.observer()
        real_event = observer._event
        def fail(event, **fields):
            if event == 'backward_started':
                raise OSError('authored backward intent fault')
            return real_event(event, **fields)
        with patch.object(observer, '_event', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'backward intent fault'):
                observer.update(1, 0, rows)
        saved = self.load(observer)
        state = saved['scaled_accumulation']['transport']
        self.assertEqual(saved['cursor']['stage'], 'backward')
        self.assertEqual(saved['cursor']['transport_stage'], 'upstream')
        self.assertIsNone(saved['cursor']['band_index'])
        self.assertEqual(state['stage'], 'upstream')
        self.assertIsNone(state['band_index'])
        self.assertIsNone(state['clock_trace']['upstream'])
        self.assertEqual(saved['completed_updates'], 0)
        self.assertEqual(saved['cursor']['optimizer_step_status'], 'not_started')
        assert_tree(self, saved['current_record'], rows[0])
        self.assertEqual(saved['scaled_accumulation']['weighted_records'], [])
        self.assertFalse(any(e['event'] == 'optimizer_step_prepared' for e in self.events(observer)))

    def test_early_transport_failures_keep_distinct_stage_and_available_evidence(self):
        for stage in ('clock_forward', 'loss', 'upstream', 'field_adjoint', 'band_plan'):
            with self.subTest(stage=stage):
                observer, rows = self.observer(name=stage)
                real_event = observer._event
                def fail(event, **fields):
                    if event == 'transport_stage' and fields.get('transport_stage') == stage:
                        raise ValueError('authored stage fault')
                    return real_event(event, **fields)
                with patch.object(observer, '_event', side_effect=fail):
                    with self.assertRaisesRegex(ValueError, 'stage fault'):
                        observer.update(1, 0, rows)
                saved = self.load(observer)
                state = saved['scaled_accumulation']['transport']
                self.assertEqual(saved['cursor']['transport_stage'], stage)
                self.assertEqual(state['stage'], stage)
                self.assertEqual(saved['completed_updates'], 0)
                if stage in ('field_adjoint', 'band_plan'):
                    self.assertIsNotNone(state['clock_trace']['upstream'])
                    replay_field_adjoint(state)
                else:
                    with self.assertRaisesRegex(ValueError, 'capture|trace/upstream'):
                        replay_field_adjoint(state)

    def test_batch_alignment_loss_is_saved_and_rejected_before_step(self):
        observer, rows = self.observer()
        blocks = {name: np.ones(tuple(p.shape)) for name, p in observer.model.named_parameters()}
        with patch.object(observer.transport, 'run', side_effect=[
                (Gradient.create(blocks, exponent=2000), 1.), (Gradient.create(blocks), 1.)]):
            with self.assertRaisesRegex(ValueError, 'lost nonzero'):
                observer.update(1, 0, rows)
        saved = self.load(observer)
        self.assertGreater(saved['scaled_accumulation']['batch_gradient']['alignment_underflows'], 0)
        self.assertEqual(saved['cursor']['optimizer_step_status'], 'not_started')
        self.assertTrue(all(g is None for g in saved['gradients'].values()))

    def test_zero_gradient_is_json_null_norm_not_an_absent_record(self):
        observer, rows = self.observer()
        observer.update(1, 0, rows, lambda clock, *_: dict(total=clock.cycles.sum() * 0))
        saved = self.load(observer, 'after-update.pt')
        state = saved['scaled_accumulation']
        self.assertEqual(len(state['weighted_records']), 2)
        self.assertIsNone(state['clip_report']['gradient_norm_log2'])
        self.assertEqual(state['transport']['completed_bands'], [])
        self.assertIsNone(self.events(observer)[-1]['gradient_norm_log2'])

    def test_returned_nonfinite_update_is_counted_but_not_accepted_as_complete(self):
        class ReturnedNonfinite(torch.optim.SGD):
            @torch.no_grad()
            def step(self, closure=None):
                self.param_groups[0]['params'][0].fill_(float('inf'))
        observer, rows = self.observer(optimizer_type=ReturnedNonfinite)
        with self.assertRaisesRegex(ValueError, 'nonfinite returned model'):
            observer.update(1, 0, rows)
        saved = self.load(observer)
        self.assertEqual(saved['completed_updates'], 1)  # Exact returned calls, not valid/admitted updates.
        self.assertEqual(saved['cursor']['optimizer_step_status'], 'returned')
        self.assertEqual(saved['cursor']['stage'], 'post_step_validation')
        self.assertTrue(self.events(observer)[-1]['update_count_exact'])
        self.assertFalse(any(e['event'] == 'update_completed' for e in self.events(observer)))
        self.assertFalse((observer.output / 'after-update.pt').exists())


if __name__ == '__main__':
    unittest.main()
