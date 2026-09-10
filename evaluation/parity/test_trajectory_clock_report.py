"""Immutable evidence and non-promotion checks; no Torch or private arrays."""
import hashlib
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'experiments/trajectory_clock'
PARENT = ROOT / 'experiments/coupled_clock'
KEYS = ('tempo_median_error_percent', 'tempo_p95_error_percent',
        'phase_mean_absolute_cycles', 'count_4s_mean_absolute_cycles')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrajectoryReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((HERE / 'plan-v1.json').read_bytes())
        cls.execution = json.loads((HERE / 'execution-v1.json').read_bytes())
        cls.report = json.loads((HERE / 'results-v1.json').read_bytes())
        cls.parent = json.loads((PARENT / 'results-v1.json').read_bytes())
        cls.attribution = json.loads((HERE / 'attribution-v1.json').read_bytes())

    def test_all_frozen_identities_and_sources(self):
        expected = {'plan-v1.json': 'f822174d6140df7e3eef3ab38aaa9bbe99a38b4faaeeee2c9355b4aaf3fb5ee4',
                    'execution-v1.json': '9a0b8614cdf11e97efc9a259bfb97048f7d95b772da516b7173a766500d8b59a',
                    'results-v1.json': 'a20e960c8fa7d3ffffdfab24e9e2d73fb2a0ab78069cc8a33d7e1cfd42d5a8e1'}
        for name, value in expected.items():
            self.assertEqual(sha(HERE / name), value)
        self.assertEqual(self.report['plan_sha256'], expected['plan-v1.json'])
        self.assertEqual(self.report['execution_sha256'], expected['execution-v1.json'])
        self.assertEqual(self.execution['source_sha256'], self.plan['source_sha256'])
        self.assertEqual(sha(HERE / 'attribution-v1.json'), self.plan['attribution_sha256'])
        self.assertEqual(sha(HERE / 'attribution.py'), self.attribution['source_sha256'])
        for name, value in self.plan['source_sha256'].items():
            self.assertEqual(sha(ROOT / name), value, name)

    def test_same_population_initialization_and_fixed_optimization(self):
        self.assertEqual(self.plan['population'], json.loads((PARENT / 'inputs-v1.json').read_bytes())['cases'])
        self.assertEqual(len(self.report['cases']), 40)
        self.assertEqual(len(self.report['fits']), 2)
        for fit, selected in zip(self.report['fits'], [6, 7]):
            self.assertEqual((fit['epochs'], fit['updates'], fit['seed'], fit['parameters']), (20, 100, 142, 23970))
            self.assertFalse(fit['budget_exhausted'])
            self.assertLess(fit['elapsed_s'], 1800)
            self.assertEqual(fit['initial_state_sha256'], self.parent['fits'][0]['initial_state_sha256'])
            self.assertEqual(fit['selected_epoch'], selected)
            self.assertEqual(min(fit['history'], key=lambda r: r['development_loss'])['epoch'], selected)

    def test_unchanged_comparators_and_complete_common_denominators(self):
        for row, old in zip(self.report['cases'], self.parent['cases']):
            self.assertEqual((row['id'], row['role'], row['work']), (old['id'], old['role'], old['work']))
            self.assertEqual(row['raw_common'], old['raw_common'])
            self.assertEqual(row['v1_common'], old['v1_common'])
            self.assertEqual(row['coupled_common'], old['audio_common'])
            for name in ('audio_common', 'zero_common', 'raw_common', 'v1_common', 'coupled_common'):
                for ref, available in [('reference_points', 'available_points'), ('reference_cells', 'available_cells'),
                                       ('reference_4s_intervals', 'available_4s_intervals')]:
                    self.assertEqual(row[name][ref], row['raw_common'][ref])
                    self.assertEqual(row[name][ref], row[name][available])
                    self.assertGreater(row[name][ref], 0)

    def test_macro_metrics_recompute_without_dropping_cases(self):
        for role, summary in self.report['summary'].items():
            rows = [r for r in self.report['cases'] if r['role'] == role]
            for name, metrics in summary.items():
                for key in KEYS:
                    groups = {}
                    for row in rows:
                        self.assertTrue(math.isfinite(row[name][key]))
                        groups.setdefault(row['work'], []).append(row[name][key])
                    expected = sum(sum(v) / len(v) for v in groups.values()) / len(groups)
                    self.assertAlmostEqual(metrics[key], expected, places=10)

    def test_failed_gate_and_regressions_cannot_be_relabelled_as_success(self):
        self.assertEqual(self.report['gate'], dict(complete=True, finished=True, audio_signal=False,
            common_nonregression=False, recording_breadth=False, passed=False, previous_coupled_nonregression=False))
        for role, raw_counts, parent_counts in [('development', (4, 4), (4, 3)), ('diagnostic', (11, 15), (3, 8))]:
            rows = [r for r in self.report['cases'] if r['role'] == role]
            for name, counts in [('raw_common', raw_counts), ('coupled_common', parent_counts)]:
                for key, expected in zip((KEYS[0], KEYS[2]), counts):
                    self.assertEqual(sum(r['regressions_vs_' + name][key] for r in rows), expected)
        for row in self.report['cases']:
            for baseline in ('raw_common', 'v1_common', 'coupled_common'):
                for key in KEYS:
                    self.assertEqual(row['regressions_vs_' + baseline][key], row['audio_common'][key] > row[baseline][key])
        for flag in ('independent_acceptance', 'holdout_access', 'production_change'):
            self.assertFalse(self.report[flag])

    def test_attribution_is_oracle_only_not_a_corrected_prediction(self):
        self.assertTrue(self.attribution['post_fit_truth_assisted'])
        for flag in ('training', 'holdout_access', 'product_change', 'independent_acceptance'):
            self.assertFalse(self.attribution[flag])
        for row, old in zip(self.attribution['cases'], self.parent['cases']):
            self.assertEqual(row['id'], old['id'])
            self.assertAlmostEqual(row['audio']['observed_phase_error_cycles'], old['audio_common'][KEYS[2]], places=12)
            self.assertLessEqual(row['audio']['best_constant_phase_error_cycles'], row['audio']['observed_phase_error_cycles'] + 1e-12)


if __name__ == '__main__':
    unittest.main()
