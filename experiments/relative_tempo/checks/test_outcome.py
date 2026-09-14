import json
import unittest

import numpy as np

from experiments.relative_tempo import data
from experiments.relative_tempo.run import ARMS, decision


class FixedOutcomeTests(unittest.TestCase):
    def assert_metrics_close(self, actual, expected):
        if expected is None:
            self.assertIsNone(actual)
            return
        self.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            if isinstance(value, float):
                self.assertAlmostEqual(actual[key], value, delta=1e-12, msg=key)
            else:
                self.assertEqual(actual[key], value)

    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((data.HERE/'results-v1.json').read_bytes())
        cls.plan = json.loads((data.HERE/'plan-v1.json').read_bytes())

    def test_immutable_report_and_plan(self):
        self.assertEqual(data.sha(data.HERE/'results-v1.json'), '599086fdbdb0134b9fe758814add0b785884f6d2eb2e038a54340ef763950ddc')
        self.assertEqual(data.sha(data.HERE/'plan-v1.json'), '95f1aea338a585a691453cfeaadc7d809e24e7cd1b9cf9888d2cd0077b531677')
        self.assertEqual(self.report['plan_sha256'], data.sha(data.HERE/'plan-v1.json'))
        self.assertEqual(self.report['source_sha256'], data.closure())
        self.assertEqual(self.plan['source_sha256'], data.closure())

    def test_exact_sparse_ownership_and_exclusions(self):
        rows = self.plan['points']
        self.assertEqual(rows, data.population())
        self.assertEqual(len(rows), 915)
        self.assertEqual(sum(r['admitted'] for r in rows), 559)
        self.assertEqual(sum(r['admitted'] and r['pair_role'] == 'fit' for r in rows), 334)
        self.assertEqual(sum(r['admitted'] and r['pair_role'] == 'schedule_diagnostic' for r in rows), 225)

    def test_capture_from_every_actual_pcm(self):
        report = self.report['capture']
        self.assertTrue(report['encoder_unchanged'])
        self.assertEqual([(c['source_id'], c['profile']) for c in report['cases']],
                         [(i, p) for i in data.IDS for p in ('source', *data.PROFILES)])
        self.assertTrue(all(c['encoder_chunks'] > 0 and c['max_head_reconstruction_error'] <= 2e-5 for c in report['cases']))

    def test_two_complete_identically_initialized_fits(self):
        fits = self.report['fits']
        self.assertEqual([f['arm'] for f in fits], list(ARMS))
        self.assertEqual(fits[0]['initial_state_sha256'], fits[1]['initial_state_sha256'])
        self.assertEqual(self.report['optimizer_steps'], 200)
        for fit in fits:
            self.assertEqual((fit['epochs'], fit['updates'], fit['selected_update']), (20, 100, 100))
            self.assertEqual([r['update'] for r in fit['history']], list(range(1, 101)))
            self.assertNotEqual(fit['initial_state_sha256'], fit['final_state_sha256'])
            self.assertLess(fit['elapsed_s'], 1800)
        self.assertTrue(all(r['relative_loss'] == 0 for r in fits[0]['history']))

    def test_scalar_pair_metrics_recomputed(self):
        for pair in self.report['pairs']:
            observed = pair['observations']
            expected_rows = [r for r in self.plan['points'] if r['admitted'] and
                             (r['source_id'], r['profile']) == (pair['source_id'], pair['profile'])]
            self.assertEqual([r['index'] for r in observed], [r['index'] for r in expected_rows])
            target = np.array([r['target_log2_rate'] for r in observed])
            np.testing.assert_allclose(target, np.log2([r['rate'] for r in expected_rows]), atol=0, rtol=0)
            for name, metrics in pair['metrics'].items():
                prediction = np.array([r['prediction'][name] for r in observed])
                self.assert_metrics_close(metrics['all'], data.relative_metrics(prediction, target))
                changed = target != 0
                self.assert_metrics_close(metrics['changed'], data.relative_metrics(prediction[changed], target[changed]) if changed.any() else None)

    def test_fixed_failure_is_not_promoted(self):
        summary, gate = decision(self.report['natural'], self.report['pairs'])
        self.assertEqual(summary, self.report['summary'])
        self.assertEqual(gate, self.report['gate'])
        self.assertTrue(self.report['completed'])
        self.assertTrue(gate['identity_passed'])
        self.assertEqual(gate['schedule_pairs_passed'], 1)
        self.assertFalse(gate['passed'])
        self.assertFalse(gate['natural_retention_passed'])
        self.assertFalse(self.report['holdout_access'])
        self.assertFalse(self.report['production_change'])
        self.assertFalse(self.report['independent_acceptance'])

    def test_all_natural_records_and_original_roles_remain(self):
        old = json.loads((data.ROOT/'experiments/coupled_clock/inputs-v1.json').read_bytes())['cases']
        self.assertEqual([(r['id'], r['role'], r['work']) for r in self.report['natural']],
                         [(r['id'], r['role'], r['work']) for r in old])
        self.assertEqual(len(self.report['natural']), 40)

    def test_changed_only_regression_is_preserved(self):
        row = next(r for r in self.report['pairs'] if r['source_id'] == data.IDS[2] and r['profile'] == 'local_slow_fast')
        a, b = row['metrics']['native-plus-relative'], row['metrics']['native-only']
        self.assertGreater(a['changed']['rmse'], b['changed']['rmse'])
        self.assertLess(a['all']['rmse'], b['all']['rmse'])


if __name__ == '__main__':
    unittest.main()
