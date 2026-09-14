"""Authored loss, ownership, recording and exact-control tests; no music."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.separated_evidence.supervision import task_loss
from experiments.local_tempo.fit import fit as old_fit
from experiments.local_tempo.test_local_tempo import TinyTempo
from experiments.tempo_variation import data, recorder
from experiments.tempo_variation.loss import components, active_loss
from experiments.tempo_variation.fit import fit, ARMS
from experiments.tempo_variation.run import control_replay, receipts, verify_freeze
from experiments.tempo_variation.evaluate import rename, ALIASES

torch.set_num_threads(2)


def reference():
    return torch.arange(5, dtype=torch.float64)[None]*.04, torch.ones(1, 5, dtype=torch.bool)


class LossTests(unittest.TestCase):
    def test_value_and_gradient_decomposition(self):
        p = torch.tensor([[1., -2., 3., 4.]], dtype=torch.float64, requires_grad=True)
        r, m = reference()
        terms = components(p, r, m)
        torch.testing.assert_close(terms['full'], terms['offset']+terms['variation'], rtol=0, atol=1e-14)
        grads = {k: torch.autograd.grad(v, p, retain_graph=True)[0] for k, v in terms.items()}
        torch.testing.assert_close(grads['full'], grads['offset']+grads['variation'], rtol=0, atol=1e-14)
        self.assertAlmostEqual(float(grads['variation'].sum()), 0.)

    def test_control_is_exact_old_loss_and_gradient(self):
        p = torch.tensor([[.13, -.7, .6, 2.]], requires_grad=True)
        r, m = reference()
        a = active_loss(ARMS[0], components(p, r, m))
        b = task_loss('tempo', p, r, m)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(a, p)[0], torch.autograd.grad(b, p)[0], rtol=0, atol=0)

    def test_additive_shift_does_not_change_variation(self):
        p = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64)
        r, m = reference()
        a, b = components(p, r, m), components(p+17, r, m)
        torch.testing.assert_close(a['variation'], b['variation'])
        self.assertNotEqual(float(a['full']), float(b['full']))

    def test_constant_wrong_absolute_tempo_is_not_fixed(self):
        p = torch.full((1, 4), 2., requires_grad=True)
        t = components(p, *reference())
        self.assertEqual(float(t['variation'].detach()), 0.)
        self.assertGreater(float(t['full'].detach()), 0.)
        torch.testing.assert_close(torch.autograd.grad(t['variation'], p)[0], torch.zeros_like(p))

    def test_mask_excludes_missing_cells_without_filling(self):
        p = torch.tensor([[1., 100., 200., 3.]], requires_grad=True)
        r, _ = reference()
        m = torch.tensor([[True, True, False, True, True]])
        t = components(p, r, m)
        self.assertAlmostEqual(float(t['variation'].detach()), 1.)
        torch.testing.assert_close(torch.autograd.grad(t['variation'], p)[0], torch.tensor([[-1., 0., 0., 1.]]))

    def test_single_supported_cell_has_zero_variation(self):
        p = torch.ones(1, 4, requires_grad=True)
        r, _ = reference()
        t = components(p, r, torch.tensor([[True, True, False, False, False]]))
        self.assertEqual(float(t['variation'].detach()), 0.)

    def test_cannot_center_across_recordings(self):
        r, m = reference()
        with self.assertRaises(ValueError):
            components(torch.ones(2, 4), r.repeat(2, 1), m.repeat(2, 1))

    def test_bad_output_outside_support_fails(self):
        r, _ = reference()
        with self.assertRaises(ValueError):
            components(torch.tensor([[1., 1., 1., float('nan')]]), r, torch.tensor([[True, True, False, False, False]]))

    def test_no_supported_cells_fails(self):
        r, _ = reference()
        with self.assertRaises(ValueError):
            components(torch.ones(1, 4), r, torch.tensor([[True, False, True, False, True]]))

    def test_unknown_arm_fails(self):
        with self.assertRaises(ValueError):
            active_loss('adaptive', {})


class ContractTests(unittest.TestCase):
    def test_prior_closure(self):
        self.assertTrue(data.prior()['completed'])
        self.assertIn('experiments/tempo_variation/loss.py', data.closure())

    def test_exposed_roles_not_fit_or_fresh(self):
        points = data.points()
        self.assertEqual(len(points), 1281)
        self.assertEqual(sum(r['admitted'] and r['pair_role'] == 'fit_exposed' for r in points), 559)
        exposed = [r for r in points if r['pair_role'] == 'exposed_schedule_diagnostic']
        self.assertEqual(len(exposed), 366)
        self.assertEqual(sum(r['admitted'] for r in exposed), 224)
        self.assertTrue(all(r['previous_pair_role'] == 'fresh_schedule_diagnostic' for r in exposed))

    def test_aliases_are_explicit_and_do_not_modify_legacy(self):
        from experiments.local_tempo.evaluate import NAMES
        self.assertEqual(tuple(ALIASES), NAMES)
        original = {'local-retained-selected': {'rmse': 3.}}
        self.assertEqual(rename(original), {'variation-retained-selected': {'rmse': 3.}})
        self.assertIn('local-retained-selected', original)

    def test_control_rejects_one_state_change(self):
        prior = data.prior()
        control = copy.deepcopy(prior['fits'][1])
        self.assertTrue(control_replay(control, prior)['passed'])
        control['history'][1]['state_sha256'] = '0'*64
        with self.assertRaises(ValueError):
            control_replay(control, prior)

    def test_empty_freeze_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            verify_freeze(Path(directory), {})

    def test_snapshot_is_observation_only_and_has_optimizer_rng(self):
        m = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(m.parameters())
        m(torch.ones(1, 2)).sum().backward()
        opt.step()
        before = copy.deepcopy(m.state_dict())
        rng = np.random.default_rng(8)
        state = copy.deepcopy(rng.bit_generator.state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.pt'
            recorder.snapshot(path, m, opt, rng)
            saved = torch.load(path, weights_only=False)
            self.assertTrue(saved['optimizer']['state'])
            self.assertEqual(saved['numpy_rng'], state)
            self.assertEqual(rng.bit_generator.state, state)
            for k, v in m.state_dict().items():
                torch.testing.assert_close(v, before[k], rtol=0, atol=0)
                torch.testing.assert_close(v, saved['model'][k], rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                recorder.snapshot(path, m, opt, rng)

    def test_independent_adamw_formula_and_wrong_step_rejection(self):
        m = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(m.parameters(), lr=.0003, weight_decay=.0001)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            before = copy.deepcopy(dict(model=m.state_dict(), optimizer=opt.state_dict()))
            m(torch.ones(1, 2)).sum().backward()
            g = recorder.gradient_vector(m)
            opt.step()
            after = copy.deepcopy(dict(model=m.state_dict(), optimizer=opt.state_dict()))
            self.assertLessEqual(recorder.adamw_reference(before, after, g), 2e-7)
            after['model']['weight'] += .001
            with self.assertRaises(ValueError):
                recorder.adamw_reference(before, after, g)


class FitTests(unittest.TestCase):
    def test_authored_matched_fit_and_original_control_parity(self):
        torch.manual_seed(5)
        records = []
        for role, count, works in (('fit', 20, 9), ('development', 5, 3)):
            for i in range(count):
                r = np.arange(400, dtype=np.float64)*.04
                mask = np.ones(400, bool)
                records.append(dict(id=f'{role}-{i}', work=f'{role}-{i % works}', role=role,
                                    reference=r, valid=mask, target=torch.from_numpy(r), mask=torch.from_numpy(mask),
                                    features=torch.randn(400, 1)))
        captures = {(i, p): torch.randn(400, 1) for i in data.old.IDS for p in ('source', *data.old.old.PROFILES)}
        points = [dict(source_id=i, profile=p, admitted=True, pair_role='fit_exposed', index=0,
                       source_s=3., output_s=3.5, rate=1.25) for i in data.old.IDS for p in data.old.old.PROFILES]
        initial = TinyTempo()
        with tempfile.TemporaryDirectory() as directory, patch('experiments.tempo_variation.fit.EPOCHS', 1), \
                patch('experiments.tempo_variation.fit.STEPS', 5), patch('experiments.tempo_variation.run.STEPS', 5), \
                patch('experiments.local_tempo.fit.EPOCHS', 1), patch('experiments.local_tempo.fit.STEPS', 5), patch('builtins.print'):
            root = Path(directory)
            new = root/'new'
            old = root/'old'
            new.mkdir()
            old.mkdir()
            _, _, old_report = old_fit(initial, records, captures, points, 'local-retained', old, lambda: None)
            reports = [fit(initial, records, captures, points, a, new, lambda: None)[2] for a in ARMS]
            for a, b in zip(reports[0]['history'], old_report['history']):
                self.assertEqual(a['state_sha256'], b['state_sha256'])
                self.assertEqual(a['development'], b['development'])
            self.assertNotEqual(reports[0]['final']['state_sha256'], reports[1]['final']['state_sha256'])
            self.assertTrue(receipts(new, reports)['matched_draws'])
            for report in reports:
                self.assertTrue(report['teacher_unchanged'])
                self.assertTrue(all(k.startswith('fit-') for k in report['teacher_fit_ids']))
                for row in report['losses']:
                    t = row['losses']
                    self.assertAlmostEqual(t['full'], t['offset']+t['variation'], places=12)
                    self.assertEqual(t['active'], t['full' if report['arm'] == ARMS[0] else 'variation'])
            (new/(ARMS[0]+'.update-001.gradients.npz')).unlink()
            with self.assertRaises(FileNotFoundError):
                receipts(new, reports)


class FrozenOutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((data.HERE/'results-v1.json').read_bytes())
        cls.plan = json.loads((data.HERE/'plan-v1.json').read_bytes())

    def test_hashes_and_source_closure(self):
        self.assertEqual(data.sha(data.HERE/'results-v1.json'), '774b86f5f1ddf7a24758af8f742a14e544e7778eeb0c571af285562c02330dc9')
        self.assertEqual(data.sha(data.HERE/'plan-v1.json'), self.report['plan_sha256'])
        self.assertEqual(self.report['source_sha256'], data.closure())
        self.assertEqual(self.plan['source_sha256'], data.closure())

    def test_completed_fixed_fit_not_promotion(self):
        self.assertTrue(self.report['completed'])
        self.assertEqual(self.report['optimizer_steps'], 200)
        self.assertEqual(self.report['encoder_calls'], 0)
        for key in ('independent_acceptance', 'holdout_access', 'production_change'):
            self.assertFalse(self.report[key])
        self.assertEqual(self.plan['primary'], 'variation-retained')

    def test_exact_control_epoch_replay(self):
        self.assertEqual(control_replay(self.report['fits'][0], data.prior()), self.report['control_replay'])

    def test_every_learned_epoch_rejected(self):
        from experiments.local_tempo import selection
        for fit in self.report['fits']:
            self.assertEqual(fit['updates'], 100)
            self.assertEqual([r['epoch'] for r in fit['history'] if r['admissibility']['passed']], [0])
            self.assertEqual(selection.choose(fit['history'])['epoch'], 0)
            self.assertEqual(fit['selected']['state_sha256'], fit['initial_state_sha256'])
            for row in fit['history']:
                self.assertEqual(selection.admissibility(row['development'], fit['history'][0]['development']), row['admissibility'])

    def test_all_batch_loss_decompositions(self):
        for fit in self.report['fits']:
            self.assertEqual([r['update'] for r in fit['losses']], list(range(1, 101)))
            for row in fit['losses']:
                t = row['losses']
                self.assertAlmostEqual(t['full'], t['offset']+t['variation'], places=12)
                self.assertEqual(t['active'], t['full' if fit['arm'] == ARMS[0] else 'variation'])
                for key in ('before_sha256', 'after_sha256', 'gradient_sha256'):
                    self.assertEqual(len(row[key]), 64)

    def test_all_natural_decompositions_are_absolute_mse(self):
        self.assertEqual(len(self.report['native_decomposition']), 40)
        for row, native in zip(self.report['native_decomposition'], self.report['natural']):
            self.assertEqual(row['id'], native['id'])
            for name, values in row['metrics'].items():
                self.assertAlmostEqual(values['full'], values['offset']+values['variation'], places=12)
                self.assertAlmostEqual(values['full'], native['metrics'][name]['mse'], places=12)

    def test_exposed_not_fresh_or_fit(self):
        self.assertEqual(self.plan['points'], data.points())
        exposed = [p for p in self.report['pairs'] if p['role'] == 'exposed_schedule_diagnostic']
        self.assertEqual(len(exposed), 6)
        self.assertEqual(sum(p['admitted_points'] for p in exposed), 224)
        self.assertNotIn('fresh_transfer', self.report['gate'])

    def test_all_pair_metric_recomputation(self):
        def compare(actual, expected):
            self.assertEqual(set(actual), set(expected))
            for key, value in expected.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(actual[key], value, places=12)
                else:
                    self.assertEqual(actual[key], value)

        for pair in self.report['pairs']:
            target = np.array([r['target_log2_rate'] for r in pair['observations']])
            for name, metrics in pair['metrics'].items():
                p = np.array([r['prediction'][name] for r in pair['observations']])
                compare(data.old.old.relative_metrics(p, target), metrics['all'])
                expected = data.old.old.relative_metrics(p[target != 0], target[target != 0]) if np.any(target != 0) else None
                if expected is None:
                    self.assertIsNone(metrics['changed'])
                else:
                    compare(expected, metrics['changed'])

    def test_frozen_numerical_gate_unchanged(self):
        from experiments.local_tempo import evaluate as old
        reverse = {v: k for k, v in ALIASES.items()}

        def aliases(value):
            if isinstance(value, dict):
                return {reverse.get(k, k): aliases(v) for k, v in value.items()}
            if isinstance(value, list):
                return [aliases(v) for v in value]
            return value

        summary, gate = old.decision(aliases(self.report['natural']), aliases(self.report['pairs']), self.report['fits'][1]['selected_epoch'])
        for key in ('fresh_transfer', 'fresh_pairs_passed', 'fresh_transfer_passed'):
            gate[key.replace('fresh', 'exposed_schedule', 1)] = gate.pop(key)
        gate['independent_acceptance'] = False
        self.assertEqual(rename(summary), self.report['summary'])
        self.assertEqual(gate, self.report['gate'])
        self.assertFalse(gate['passed'])

    def test_improvement_does_not_hide_regressions(self):
        s = self.report['summary']
        metric = 'tempo_median_error_percent'
        self.assertLess(s['development']['variation-retained-final'][metric], s['development']['native-retained-final'][metric])
        self.assertGreater(s['development']['variation-retained-final'][metric], s['development']['initial'][metric])
        self.assertGreater(s['diagnostic']['variation-retained-final'][metric], s['diagnostic']['initial'][metric])
