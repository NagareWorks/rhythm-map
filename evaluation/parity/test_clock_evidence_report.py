"""Public diagnostic audit without PyTorch, private features or optimizer access."""
import hashlib
import json
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'experiments/clock_evidence'
KEYS = ('tempo_median_error_percent', 'tempo_p95_error_percent',
        'phase_mean_absolute_cycles', 'count_4s_mean_absolute_cycles')
VARIANTS = ('natural', 'zero', 'time_mean', 'half_roll', 'fitted_zero', 'raw')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvidenceReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((HERE / 'results-v1.json').read_bytes())
        cls.execution = json.loads((HERE / 'execution-v1.json').read_bytes())

    def test_sources_and_parents_are_pinned(self):
        self.assertEqual(self.report['execution_sha256'], sha(HERE / 'execution-v1.json'))
        self.assertEqual(self.report['parent_report_sha256'], self.execution['parent_report_sha256'])
        for path, expected in self.execution['source_sha256'].items():
            self.assertEqual(sha(ROOT / path), expected, path)
        for name, folder in (('direct', 'clock_readout'), ('coupled', 'coupled_clock'), ('trajectory', 'trajectory_clock')):
            self.assertEqual(sha(ROOT / 'experiments' / folder / 'results-v1.json'), self.report['parent_report_sha256'][name])
        self.assertEqual(sha(ROOT / 'experiments/coupled_clock/inputs-v1.json'), self.report['inputs_sha256'])

    def test_population_support_and_no_promotion(self):
        manifest = json.loads((ROOT / 'experiments/coupled_clock/inputs-v1.json').read_bytes())
        self.assertEqual(len(self.report['cases']), 120)
        self.assertEqual(self.report['audio_checkpoint_sequence_evaluations'], 480)
        self.assertEqual(self.report['fitted_zero_sequence_evaluations'], 120)
        for name in ('direct', 'coupled', 'trajectory'):
            rows = [r for r in self.report['cases'] if r['model'] == name]
            self.assertEqual([(r['id'], r['role'], r['work'], r['packet_sha256']) for r in rows],
                             [(r['id'], r['role'], r['work'], r['packet_sha256']) for r in manifest['cases']])
            for row in rows:
                for variant in VARIANTS:
                    for key in ('reference_points', 'reference_cells', 'reference_4s_intervals'):
                        self.assertEqual(row['metrics'][variant][key], row['metrics']['natural'][key])
                    for key in KEYS:
                        self.assertIsNotNone(row['metrics'][variant][key])
        for value in (self.report, self.execution):
            for key in ('training', 'encoder_inference', 'holdout_access', 'independent_acceptance', 'production_change'):
                self.assertFalse(value[key])
            self.assertTrue(value['weights_fixed'])
        self.assertFalse(self.report['promotion_gate'])

    def test_work_macros_and_paired_breadth_recompute(self):
        for role, models in self.report['summary'].items():
            for name, summary in models.items():
                rows = [r for r in self.report['cases'] if r['role'] == role and r['model'] == name]
                for variant in VARIANTS:
                    for key in KEYS:
                        groups = {}
                        for row in rows:
                            groups.setdefault(row['work'], []).append(row['metrics'][variant][key])
                        expected = statistics.mean(statistics.mean(v) for v in groups.values())
                        self.assertAlmostEqual(summary['metrics'][variant][key], expected, places=10)
                        if variant not in ('natural', 'raw'):
                            paired = summary['paired'][variant][key]
                            self.assertAlmostEqual(paired['intervention_minus_natural_work_macro'],
                                expected - summary['metrics']['natural'][key], places=10)
                            self.assertEqual(paired['natural_better_recordings'], sum(
                                r['metrics']['natural'][key] < r['metrics'][variant][key] for r in rows))

    def test_phase_signal_does_not_erase_musical_regressions(self):
        for role, count in (('development', 5), ('diagnostic', 15)):
            summary = self.report['summary'][role]['direct']
            for variant in ('zero', 'time_mean', 'half_roll', 'fitted_zero'):
                self.assertEqual(summary['paired'][variant][KEYS[2]]['natural_better_recordings'], count)
        art = self.report['summary']['diagnostic']['direct']['metrics']
        self.assertGreater(art['natural'][KEYS[2]], art['raw'][KEYS[2]])
        self.assertGreater(art['natural'][KEYS[0]], art['raw'][KEYS[0]])
        dev = self.report['summary']['development']['trajectory']['metrics']
        self.assertGreater(dev['natural'][KEYS[2]], dev['time_mean'][KEYS[2]])

    def test_original_predictions_and_local_shift_contract(self):
        for name, folder, metric in (('direct', 'coupled_clock', 'v1_common'),
                                    ('coupled', 'coupled_clock', 'audio_common'),
                                    ('trajectory', 'trajectory_clock', 'audio_common')):
            parent = json.loads((ROOT / 'experiments' / folder / 'results-v1.json').read_bytes())
            rows = [r for r in self.report['cases'] if r['model'] == name]
            for row, old in zip(rows, parent['cases'], strict=True):
                self.assertEqual(row['id'], old['id'])
                for key in KEYS:
                    self.assertAlmostEqual(row['metrics']['natural'][key], old[metric][key], delta=5e-4)
                self.assertLessEqual(row['replay_max_absolute_difference'], 1e-5)
                self.assertLessEqual(row['local_roll_max_absolute_difference'], 1e-5)
                for response in row['field_response'].values():
                    if name != 'direct':
                        self.assertIsNone(response['dense_phase_mean_absolute_change_cycles'])


if __name__ == '__main__':
    unittest.main()
