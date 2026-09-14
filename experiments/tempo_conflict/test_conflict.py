"""Authored gradient, accounting, ownership and attribution-limit witnesses."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.coupled_clock.run import state_hash, work_weights
from experiments.tempo_conflict import data, gradients as g, statistics as s
from experiments.tempo_conflict.run import first_draw, same_metrics

torch.set_num_threads(2)


class Tiny(torch.nn.Module):
    def __init__(self, value=.2):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(value))

    def forward(self, features):
        if torch.any(features == 999):
            raise AssertionError('forbidden diagnostic forward')
        return self.w*features[:, :-1, 0]


def row(identity, work, x, role='fit'):
    reference = np.arange(400, dtype=np.float64)*.04
    valid = np.ones(400, dtype=bool)
    return dict(id=identity, work=work, role=role, reference=reference, valid=valid,
                features=torch.full((400, 1), float(x)), target=torch.from_numpy(reference), mask=torch.from_numpy(valid))


def pair(profile, count, rate):
    return [dict(source_id='source', profile=profile, index=i, source_s=3., output_s=3.5,
                 admitted=True, pair_role='fit_exposed', rate=rate) for i in range(count)]


class StatisticsTests(unittest.TestCase):
    def test_direction_sign(self):
        good, bad = s.descent([1, 0], [2, 0]), s.descent([1, 0], [-2, 0])
        self.assertEqual(good['unit_negative_gradient_derivative'], -1)
        self.assertEqual(bad['unit_negative_gradient_derivative'], 1)

    def test_zero_has_no_unit_direction(self):
        result = s.descent([1, 2], [0, 0])
        self.assertIsNone(result['cosine'])
        self.assertIsNone(result['unit_negative_gradient_derivative'])

    def test_diagonal_scaling_can_reverse_alignment(self):
        c, training = np.array([2., 1.]), np.array([-1., 7.])
        self.assertLess(s.descent(c, training)['negative_gradient_derivative'], 0)
        diagonal = np.array([1., 1/7])
        self.assertGreater(np.dot(c, -diagonal*training), 0)

    def test_curvature_not_certified_by_start_gradient(self):
        result = s.observed([0.], [4.], [2.], 0., 4.)
        self.assertEqual(result['start_linear'], 0.)
        self.assertEqual(result['actual_mse_change'], 4.)

    def test_endpoint_trapezoid_can_have_residual(self):
        result = s.observed([0.], [32.], [2.], 0., 16.)
        self.assertEqual(result['endpoint_trapezoid'], 32.)
        self.assertEqual(result['trapezoid_residual'], -16.)

    def test_vector_geometry_and_finiteness(self):
        for left, right in (([1], [1, 2]), ([float('nan')], [1]), ([], [])):
            with self.assertRaises(ValueError):
                s.relation(left, right)

    def test_work_macro_is_not_record_mean(self):
        vectors = {k: np.ones(1) for k in ('native', 'pair_identity', 'pair_global', 'pair_local', 'retention')}
        vectors.update(dev_0=np.array([2.]), dev_1=np.array([8.]), dev_2=np.array([8.]))
        records = [dict(id='a', work='A', mse=2.), dict(id='b', work='B', mse=8.), dict(id='c', work='B', mse=8.)]
        training, targets, losses = s.aggregates(vectors, records)
        self.assertEqual(losses['macro'], 5.)
        self.assertEqual(targets['macro'][0], 5.)
        self.assertEqual(training['native_pair_retention'][0], 5.)


class GradientTests(unittest.TestCase):
    def setUp(self):
        self.model = Tiny()
        self.rows = [row('a', 'A', 1), row('b', 'B', 2), row('c', 'B', 3)]
        self.targets = {r['id']: Tiny(.1)(r['features'][None]).detach() for r in self.rows}

    def test_native_work_weighted_gradient_matches_analytic(self):
        vectors, losses = g.native(self.model, self.rows, self.targets, work_weights(self.rows), 3, lambda: None)
        x = np.array([1., 2., 3.])
        weights = np.array([.5, .25, .25])
        w = float(self.model.w.detach())
        self.assertAlmostEqual(vectors['native'][0], np.sum(weights*2*(w*x+1)*x), places=6)
        self.assertAlmostEqual(losses['native'], np.sum(weights*(w*x+1)**2), places=6)

    def test_retention_gradient_matches_analytic(self):
        vectors, losses = g.native(self.model, self.rows, self.targets, work_weights(self.rows), 3, lambda: None)
        self.assertAlmostEqual(vectors['retention'][0], .2*(.5+1+2.25), places=6)
        self.assertGreater(losses['retention'], 0)

    def test_initial_teacher_has_exactly_zero_retention(self):
        teacher = {r['id']: self.model(r['features'][None]).detach().clone() for r in self.rows}
        vectors, losses = g.native(self.model, self.rows, teacher, work_weights(self.rows), 3, lambda: None)
        self.assertEqual(losses['retention'], 0.)
        np.testing.assert_array_equal(vectors['retention'], [0.])

    def test_no_parameter_or_gradient_buffer_mutation(self):
        before = state_hash(self.model)
        g.native(self.model, self.rows, self.targets, work_weights(self.rows), 3, lambda: None)
        self.assertEqual(state_hash(self.model), before)
        self.assertIsNone(self.model.w.grad)

    def test_development_cannot_supply_native_training_direction(self):
        with self.assertRaises(ValueError):
            g.native(self.model, [row('dev', 'D', 1, 'development')], {}, {}, 1, lambda: None)

    def test_wrong_denominator_rejected(self):
        with self.assertRaises(ValueError):
            g.native(self.model, self.rows, self.targets, work_weights(self.rows), 2, lambda: None)

    def test_pair_chunk_weight_and_group_mean(self):
        captures = {('source', 'source'): torch.ones(400, 1), ('source', 'identity_seams'): torch.full((400, 1), 2.),
                    ('source', 'slow'): torch.full((400, 1), 4.)}
        groups = [pair('identity_seams', 3, 1.), pair('slow', 17, .8)]
        vectors, losses, details = g.paired(self.model, captures, groups, lambda: None)
        w = float(self.model.w.detach())
        self.assertAlmostEqual(vectors['pair_identity'][0], w, places=6)
        self.assertAlmostEqual(vectors['pair_global'][0], 3*(3*w+np.log2(.8)), places=6)
        self.assertEqual([r['points'] for r in details], [3, 17])
        self.assertAlmostEqual(sum(losses.values()), .5*(w*w+(3*w+np.log2(.8))**2), places=6)

    def test_repeated_draw_indices_keep_multiplicity(self):
        captures = {('source', 'source'): torch.ones(400, 1), ('source', 'slow'): torch.full((400, 1), 4.)}
        group = pair('slow', 1, .8)*4
        vectors, _, detail = g.paired(self.model, captures, [group], lambda: None)
        self.assertEqual(detail[0]['points'], 4)
        self.assertAlmostEqual(vectors['pair_global'][0], 6*(3*.2+np.log2(.8)), places=6)

    def test_unadmitted_pair_rejected(self):
        group = pair('slow', 1, .8)
        group[0]['admitted'] = False
        with self.assertRaises(ValueError):
            g.paired(self.model, {}, [group], lambda: None)

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):
            g.pair_kind('fresh_slow_fast_normal')

    def test_no_art_forward_and_dev_gradient_stays_separate(self):
        records = self.rows+[row('dev', 'D', 2, 'development'), row('art', 'X', 999, 'diagnostic')]
        captures = {('source', 'source'): torch.ones(400, 1), ('source', 'slow'): torch.full((400, 1), 4.)}
        vectors, report = g.inspect(self.model, records, captures, [pair('slow', 3, .8)], self.targets, lambda: None)
        self.assertEqual([r['id'] for r in report['development']], ['dev'])
        self.assertIn('dev_0', vectors)
        self.assertNotIn('dev_1', vectors)
        self.assertIsNone(self.model.w.grad)


class ContractTests(unittest.TestCase):
    def test_all_epochs_with_only_shared_zero_deduplicated(self):
        slots = data.slots(data.prior())
        self.assertEqual(len(slots), 42)
        self.assertEqual(len({r['key'] for r in slots}), 41)

    def test_missing_epoch_rejected(self):
        report = copy.deepcopy(data.prior())
        report['fits'][0]['history'].pop()
        with self.assertRaises(ValueError):
            data.slots(report)

    def test_audit_closure_includes_predecessor(self):
        self.assertIn('experiments/local_tempo/fit.py', data.closure())
        self.assertIn('experiments/tempo_conflict/gradients.py', data.closure())

    def test_metric_tolerance_does_not_drop_discrete_population(self):
        same_metrics(dict(value=1.+1e-9, cells=20), dict(value=1., cells=20))
        with self.assertRaises(ValueError):
            same_metrics(dict(value=1., cells=19), dict(value=1., cells=20))
        with self.assertRaises(ValueError):
            same_metrics(dict(value=float('nan')), dict(value=1.))

    def test_runner_contains_no_backward_or_optimizer_step(self):
        for name in ('run.py', 'gradients.py'):
            tree = ast.parse((data.HERE/name).read_text())
            calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
            self.assertFalse(set(calls) & {'backward', 'step', 'AdamW', 'SGD'})

    def test_first_draw_preserves_records_and_sample_multiplicity(self):
        groups = data.old.groups(data.old.fit_points(), data.old.old.PROFILES)
        samples = [dict(source_id=group[0]['source_id'], profile=group[0]['profile'], indices=[group[0]['index']]*4) for group in groups]
        records = [row(str(i), str(i), 1) for i in range(4)]
        events = []
        for arm in ('local-only', 'local-retained'):
            events.extend([dict(arm=arm, event='update_entered', update=1, natural_ids=[r['id'] for r in records], pair_samples=samples),
                           dict(arm=arm, event='update_returned', update=1, unclipped_grad_norm=1.)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'journal.jsonl'
            path.write_text('\n'.join(json.dumps(e) for e in events), encoding='utf-8')
            previous = dict(journal_sha256=data.sha(path))
            natural, chosen, receipt = first_draw(Path(directory), previous, records, groups)
            self.assertEqual([r['id'] for r in natural], ['0', '1', '2', '3'])
            self.assertEqual(len(chosen), 15)
            self.assertTrue(all(len(group) == 4 and len({r['index'] for r in group}) == 1 for group in chosen))
            records[0]['role'] = 'development'
            with self.assertRaises(ValueError):
                first_draw(Path(directory), previous, records, groups)

    def test_first_draw_rejects_changed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'journal.jsonl'
            path.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                first_draw(Path(directory), dict(journal_sha256='0'*64), [], [])


class FrozenOutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((data.HERE/'results-v1.json').read_bytes())
        cls.plan = json.loads((data.HERE/'plan-v1.json').read_bytes())

    def test_identity_and_code_closure(self):
        self.assertEqual(data.sha(data.HERE/'results-v1.json'), 'ec0321b9992221b2ab7008af50f8516914bd2f001944b5aed0536c4bc0762b99')
        self.assertEqual(self.report['plan_sha256'], data.sha(data.HERE/'plan-v1.json'))
        self.assertEqual(self.report['source_sha256'], data.closure())
        self.assertEqual(self.plan['source_sha256'], data.closure())

    def test_complete_state_and_role_coverage(self):
        expected = ['initial']+[f'{a}.epoch-{e:02}' for a in ('local-only', 'local-retained') for e in range(1, 21)]
        self.assertEqual([r['key'] for r in self.report['states']], expected)
        self.assertEqual(len(self.report['intervals']), 40)
        self.assertEqual(sum(r['elements'] for r in self.plan['parameter_geometry']), 23937)
        for row in self.report['states']:
            self.assertEqual(len(row['development']), 5)
            self.assertEqual(len({r['work'] for r in row['development']}), 3)
            self.assertEqual(sum(r['points'] for r in row['pair_groups']), 559)

    def test_all_sign_counts_and_zero_exception_retained(self):
        expected = dict(native=(34, 7, 0), pair=(17, 24, 0), pair_global=(2, 39, 0),
                        pair_local=(37, 4, 0), retention=(1, 39, 1))
        for term, counts in expected.items():
            rows = [r['geometry']['development']['macro'][term] for r in self.report['states']]
            self.assertEqual((sum(r['dot'] < 0 for r in rows), sum(r['dot'] > 0 for r in rows),
                              sum(r['right_norm'] == 0 for r in rows)), counts)
        zero = self.report['states'][0]['geometry']['development']['macro']['retention']
        self.assertIsNone(zero['cosine'])
        self.assertIsNone(zero['unit_negative_gradient_derivative'])

    def test_scalar_derivative_accounting(self):
        for row in self.report['states']:
            for terms in row['geometry']['development'].values():
                for value in terms.values():
                    self.assertEqual(value['negative_gradient_derivative'], -value['dot'])
                    if value['right_norm']:
                        self.assertAlmostEqual(value['unit_negative_gradient_derivative'], -value['dot']/value['right_norm'], places=12)
        for interval in self.report['intervals']:
            for value in interval['development'].values():
                self.assertAlmostEqual(value['endpoint_trapezoid'], .5*(value['start_linear']+value['end_linear']), places=12)
                self.assertAlmostEqual(value['trapezoid_residual'], value['actual_mse_change']-value['endpoint_trapezoid'], places=12)

    def test_first_batch_reconstruction_and_no_learning(self):
        first = self.report['first_batch']
        self.assertEqual(first['losses']['retention'], 0.)
        self.assertEqual(len(first['receipt']['natural_ids']), 4)
        self.assertEqual(len(first['receipt']['pair_samples']), 15)
        for original in first['receipt']['returned']:
            self.assertTrue(np.isclose(first['recomputed_unclipped_norm'], original['unclipped_grad_norm'], rtol=1e-5, atol=1e-6))
        for key in ('optimizer_steps', 'encoder_forwards', 'art_forwards', 'fresh_schedule_forwards'):
            self.assertEqual(self.report[key], 0)
        for key in ('holdout_access', 'independent_acceptance', 'production_change'):
            self.assertIs(self.report[key], False)

    def test_all_actual_macro_change_signs_preserved(self):
        for row in self.report['intervals']:
            value = row['development']['macro']
            self.assertEqual(np.sign(value['actual_mse_change']), np.sign(value['start_linear']))
            self.assertEqual(np.sign(value['actual_mse_change']), np.sign(value['endpoint_trapezoid']))


if __name__ == '__main__':
    unittest.main()
