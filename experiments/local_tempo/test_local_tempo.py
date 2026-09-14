"""Authored tests: no private music, pretrained weights or network access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.local_tempo import data, selection
from experiments.local_tempo.capture import verify_freeze
from experiments.local_tempo.fit import retention_loss, save_weights, read_weights, fit

torch.set_num_threads(2)


def development():
    return [dict(id=str(i), work=str(i % 3), mse=1., tempo_median_error_percent=10.) for i in range(5)]


class SelectionTests(unittest.TestCase):
    def test_initial_admissible(self):
        self.assertTrue(selection.admissibility(development(), development())['passed'])

    def test_work_mse_veto(self):
        rows = development()
        rows[2]['mse'] = 1.051
        self.assertFalse(selection.admissibility(rows, development())['passed'])

    def test_work_bpm_veto(self):
        rows = development()
        rows[2]['tempo_median_error_percent'] = 10.51
        self.assertFalse(selection.admissibility(rows, development())['passed'])

    def test_record_regression_count(self):
        rows = development()
        for i in (0, 1, 2):
            rows[i]['tempo_median_error_percent'] = 11.
        result = selection.admissibility(rows, development())
        self.assertEqual(result['recording_regressions'], 3)
        self.assertFalse(result['passed'])

    def test_population_cannot_reorder(self):
        with self.assertRaises(ValueError):
            selection.admissibility(development()[::-1], development())

    def test_nonfinite_cannot_select(self):
        rows = development()
        rows[1]['mse'] = float('nan')
        with self.assertRaises(ValueError):
            selection.admissibility(rows, development())

    def test_tie_uses_first_and_inadmissible_ignored(self):
        rows = [dict(epoch=i, training_pair_mse=s, admissibility=dict(passed=p))
                for i, (s, p) in enumerate(((4., True), (2., True), (2., True), (0., False)))]
        self.assertEqual(selection.choose(rows)['epoch'], 1)

    def test_epoch_zero_is_explicit_fallback(self):
        rows = [dict(epoch=i, training_pair_mse=s, admissibility=dict(passed=p))
                for i, (s, p) in enumerate(((4., True), (2., False)))]
        self.assertEqual(selection.choose(rows)['epoch'], 0)

    def test_missing_epoch_rejected(self):
        with self.assertRaises(ValueError):
            selection.choose([dict(epoch=1, training_pair_mse=0., admissibility=dict(passed=True))])


class RetentionTests(unittest.TestCase):
    def test_detached_teacher_and_exact_cells(self):
        x = torch.tensor([[1., 20., 30., 4.]], requires_grad=True)
        teacher = torch.zeros_like(x, requires_grad=True)
        mask = torch.tensor([[True, True, False, True, True]])
        loss = retention_loss(x, teacher, mask)
        self.assertEqual(float(loss.detach()), 8.5)
        loss.backward()
        self.assertIsNone(teacher.grad)
        torch.testing.assert_close(x.grad, torch.tensor([[1., 0., 0., 4.]]))

    def test_no_cells_rejected(self):
        with self.assertRaises(ValueError):
            retention_loss(torch.ones(1, 2), torch.ones(1, 2), torch.tensor([[True, False, True]]))

    def test_nonfinite_outside_mask_not_dropped(self):
        with self.assertRaises(ValueError):
            retention_loss(torch.tensor([[1., float('nan')]]), torch.zeros(1, 2), torch.tensor([[True, True, False]]))

    def test_bad_geometry_rejected(self):
        with self.assertRaises(ValueError):
            retention_loss(torch.ones(1, 2), torch.ones(1, 3), torch.ones(1, 3, dtype=torch.bool))


class PopulationTests(unittest.TestCase):
    def test_prior_closure_pinned(self):
        self.assertTrue(data.prior()['completed'])
        self.assertIn('experiments/local_tempo/fit.py', data.closure())

    def test_old_local_is_now_training_exposed(self):
        rows = data.fit_points()
        self.assertEqual(len(rows), 915)
        self.assertEqual(sum(r['admitted'] for r in rows), 559)
        self.assertTrue(all(r['pair_role'] == 'fit_exposed' for r in rows))
        self.assertEqual(sum(r['admitted'] and r['previous_pair_role'] == 'schedule_diagnostic' for r in rows), 225)
        self.assertEqual(len(data.groups(rows, data.old.PROFILES)), 15)

    def test_fresh_schedule_geometry(self):
        for profile, rates in data.FRESH.items():
            timeline = data.Timeline(data.old.SOURCE_EDGES, rates)
            self.assertEqual(timeline.output_edges[-1], 61.)
            self.assertEqual(len(np.arange(.5, 61., 1.)), 61)

    def test_fresh_requires_support(self):
        row = data.fresh_point(data.IDS[0], next(iter(data.FRESH)), 0, dict(center_s=10.), {}, False, 'interior')
        self.assertFalse(row['admitted'])
        self.assertEqual(row['exclusion'], 'alignment_not_supported')

    def test_fresh_requires_context(self):
        row = data.fresh_point(data.IDS[0], next(iter(data.FRESH)), 0, dict(center_s=1.), {}, True, 'interior')
        self.assertFalse(row['admitted'])
        self.assertEqual(row['exclusion'], 'cnn_context_near_seam_or_edge')

    def test_fresh_interior_role(self):
        row = data.fresh_point(data.IDS[0], next(iter(data.FRESH)), 0, dict(center_s=10.), {}, True, 'interior')
        self.assertTrue(row['admitted'])
        self.assertEqual(row['pair_role'], 'fresh_schedule_diagnostic')
        self.assertAlmostEqual(row['source_s'], 8.)


class ArtifactTests(unittest.TestCase):
    def test_export_roundtrip_and_identity_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'weights.npz'
            model = torch.nn.Linear(2, 1)
            report = save_weights(model, path)
            other = read_weights(model, path, report['weight_sha256'])
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                read_weights(model, path, '0'*64)
            with self.assertRaises(FileExistsError):
                save_weights(model, path)

    def test_freeze_requires_all_four_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                verify_freeze(Path(directory), {})

    def test_freeze_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [a+s for a in ('local-only', 'local-retained') for s in ('-selected', '-final')]
            frozen = {k: save_weights(torch.nn.Linear(1, 1), root/(k+'.weights.npz'))['weight_sha256'] for k in names}
            verify_freeze(root, frozen)
            frozen[names[0]] = '0'*64
            with self.assertRaises(ValueError):
                verify_freeze(root, frozen)


class TinyTempo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.1))

    def forward(self, x):
        return self.scale*x[:, :-1, 0]


class FitSmokeTests(unittest.TestCase):
    def test_matched_draws_teacher_and_complete_epoch(self):
        torch.manual_seed(5)
        records = []
        for role, count, works in (('fit', 20, 9), ('development', 5, 3)):
            for i in range(count):
                reference = np.arange(400, dtype=np.float64)*.04
                valid = np.ones(400, bool)
                records.append(dict(id=f'{role}-{i}', work=f'{role}-{i % works}', role=role,
                    reference=reference, valid=valid, target=torch.from_numpy(reference), mask=torch.from_numpy(valid),
                    features=torch.randn(400, 1)))
        captures = {(i, p): torch.randn(400, 1) for i in data.IDS for p in ('source', *data.old.PROFILES)}
        points = [dict(source_id=i, profile=p, admitted=True, pair_role='fit_exposed', index=0,
                       source_s=3., output_s=3.5, rate=1.25) for i in data.IDS for p in data.old.PROFILES]
        initial = TinyTempo()
        with tempfile.TemporaryDirectory() as directory, patch('experiments.local_tempo.fit.EPOCHS', 1), patch('experiments.local_tempo.fit.STEPS', 5), patch('builtins.print'):
            root = Path(directory)
            reports = []
            for arm in ('local-only', 'local-retained'):
                selected, terminal, report = fit(initial, records, captures, points, arm, root, lambda: None)
                reports.append(report)
                self.assertEqual(report['updates'], 5)
                self.assertEqual(len(report['history']), 2)
                self.assertTrue(report['teacher_unchanged'])
                self.assertTrue(all(i.startswith('fit-') for i in report['teacher_fit_ids']))
            events = [json.loads(line) for line in (root/'journal.jsonl').read_text().splitlines()]
            draws = [[(e['natural_ids'], e['pair_samples']) for e in events if e['event'] == 'update_entered' and e['arm'] == a]
                     for a in ('local-only', 'local-retained')]
            self.assertEqual(draws[0], draws[1])
            self.assertTrue(all(r['retention_loss'] == 0 for r in reports[0]['losses']))
            self.assertTrue(any(r['retention_loss'] > 0 for r in reports[1]['losses']))
            self.assertEqual(float(initial.scale.detach()), float(torch.tensor(.1)))


class FrozenOutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((data.HERE/'results-v1.json').read_bytes())
        cls.plan = json.loads((data.HERE/'plan-v1.json').read_bytes())
        cls.prepared = json.loads((data.HERE/'preparation-v1.json').read_bytes())

    def test_report_identities_and_source(self):
        self.assertEqual(data.sha(data.HERE/'results-v1.json'), 'e4c61280383e8c56fd0378eb9d178fc9e4a6eeb3e649ce64fe0683e85f1580ee')
        self.assertEqual(data.sha(data.HERE/'plan-v1.json'), self.report['plan_sha256'])
        self.assertEqual(data.sha(data.HERE/'preparation-v1.json'), self.report['preparation_sha256'])
        self.assertEqual(self.report['source_sha256'], data.closure())
        self.assertEqual(self.prepared['source_sha256'], data.closure())

    def test_fresh_and_exposed_ledgers(self):
        rows = self.plan['points']
        self.assertEqual(len(rows), 915+366)
        self.assertEqual(sum(r['admitted'] and r['pair_role'] == 'fit_exposed' for r in rows), 559)
        fresh = [r for r in rows if r['pair_role'] == 'fresh_schedule_diagnostic']
        self.assertEqual(len(fresh), 366)
        self.assertEqual(sum(r['admitted'] for r in fresh), 224)
        self.assertEqual(sum(r['admitted'] and r['rate'] != 1 for r in fresh), 140)

    def test_preparation_passed_without_model(self):
        from experiments.local_tempo.run import load_preparation
        report, points = load_preparation(data.HERE/'preparation-v1.json', self.report['preparation_sha256'], data.closure())
        self.assertTrue(report['gate_passed'])
        self.assertEqual(report['model_calls'], 0)
        self.assertEqual(len(report['authored']), 6)
        self.assertTrue(all(r['passed'] for r in report['authored']))

    def test_no_learned_checkpoint_admissible(self):
        for fit in self.report['fits']:
            self.assertEqual(fit['updates'], 100)
            self.assertEqual(fit['selected_epoch'], 0)
            self.assertEqual(selection.choose(fit['history'])['epoch'], 0)
            self.assertEqual(fit['selected']['state_sha256'], fit['initial_state_sha256'])
            self.assertEqual([r['epoch'] for r in fit['history'] if r['admissibility']['passed']], [0])
            for row in fit['history']:
                self.assertEqual(selection.admissibility(row['development'], fit['history'][0]['development']), row['admissibility'])

    def test_recompute_all_pair_metrics(self):
        def same(actual, expected):
            self.assertEqual(set(actual), set(expected))
            for key, value in expected.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(actual[key], value, places=12)
                else:
                    self.assertEqual(actual[key], value)

        for pair in self.report['pairs']:
            target = np.asarray([r['target_log2_rate'] for r in pair['observations']])
            changed = target != 0
            for name, metrics in pair['metrics'].items():
                p = np.asarray([r['prediction'][name] for r in pair['observations']])
                same(data.old.relative_metrics(p, target), metrics['all'])
                if changed.any():
                    same(data.old.relative_metrics(p[changed], target[changed]), metrics['changed'])

    def test_failed_primary_and_terminal_verdicts(self):
        from experiments.local_tempo.evaluate import decision, PRIMARY
        summary, gate = decision(self.report['natural'], self.report['pairs'], self.report['fits'][1]['selected_epoch'])
        self.assertEqual(summary, self.report['summary'])
        self.assertEqual(gate, self.report['gate'])
        self.assertFalse(gate['passed'])
        self.assertFalse(gate['learned_checkpoint'])
        self.assertEqual(gate['fresh_pairs_passed'], 1)
        for name in ('local-only-final', 'local-retained-final'):
            pairs = copy.deepcopy(self.report['pairs'])
            for pair in pairs:
                pair['metrics'][PRIMARY] = pair['metrics'][name]
            # Descriptive terminal transfer only, not an alternate selection.
            self.assertEqual(decision(self.report['natural'], pairs, 20)[1]['fresh_pairs_passed'], 0)

    def test_no_promotion_or_holdout(self):
        self.assertTrue(self.report['completed'])
        self.assertEqual(self.report['optimizer_steps'], 200)
        self.assertEqual(len(self.report['capture']['cases']), 6)
        for key in ('independent_acceptance', 'holdout_access', 'production_change'):
            self.assertIs(self.report[key], False)


if __name__ == '__main__':
    unittest.main()
