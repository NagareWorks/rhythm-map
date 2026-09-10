"""Public evidence stays immutable; these checks do not fit or load a model."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'experiments/coupled_clock'
KEYS = ('tempo_median_error_percent', 'tempo_p95_error_percent',
        'phase_mean_absolute_cycles', 'count_4s_mean_absolute_cycles')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CoupledClockReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = json.loads((EXPERIMENT / 'inputs-v1.json').read_bytes())
        cls.execution = json.loads((EXPERIMENT / 'execution-v1.json').read_bytes())
        cls.report = json.loads((EXPERIMENT / 'results-v1.json').read_bytes())

    def test_report_and_protocol_identities(self):
        expected = {'inputs-v1.json': 'b4f05c10ebede84c394b0b8e06f6723b096d1f9a0a48e63024eb81d913fffd9c',
                    'execution-v1.json': '63a82d56c464827c38296d810f47587daef6f88910b1802c59eaada39c3bdb0f',
                    'results-v1.json': 'e33a44db4ce20a35f0676aa3f4fc47806da83f4a2ab80c1cfb555ce5343cda9a'}
        for name, digest in expected.items():
            self.assertEqual(sha(EXPERIMENT / name), digest)
        self.assertEqual(self.report['inputs_sha256'], expected['inputs-v1.json'])
        self.assertEqual(self.report['execution_sha256'], expected['execution-v1.json'])
        self.assertEqual(self.execution['source_sha256'], self.inputs['source_sha256'])
        for name, digest in self.inputs['source_sha256'].items():
            self.assertEqual(sha(ROOT / name), digest, name)

    def test_unchanged_population_no_diagnostic_fitting(self):
        old = json.loads((ROOT / 'experiments/clock_readout/inputs-v1.json').read_bytes())
        extract = lambda rows: [(r['id'], r['role'], r['work'], r['frames']) for r in rows]
        self.assertEqual(extract(self.inputs['cases']), extract(old['cases']))
        self.assertEqual(extract(self.report['cases']), extract(old['cases']))
        for role, recordings, works in [('fit', 20, 9), ('development', 5, 3), ('diagnostic', 15, 1)]:
            rows = [r for r in self.inputs['cases'] if r['role'] == role]
            self.assertEqual(len(rows), recordings)
            self.assertEqual(len({r['work'] for r in rows}), works)

    def test_both_complete_fits_and_checkpoint_selection(self):
        self.assertEqual(len(self.report['fits']), 2)
        for fit, selected in zip(self.report['fits'], [9, 20]):
            self.assertEqual((fit['seed'], fit['epochs'], fit['updates'], fit['parameters']), (142, 20, 100, 23970))
            self.assertFalse(fit['budget_exhausted'])
            self.assertLess(fit['elapsed_s'], 1800)
            self.assertEqual(fit['selected_epoch'], selected)
            self.assertEqual(min(fit['history'], key=lambda h: h['development_loss'])['epoch'], selected)
        self.assertEqual(self.report['fits'][0]['initial_state_sha256'], self.report['fits'][1]['initial_state_sha256'])

    def test_every_comparison_has_complete_fixed_support(self):
        for row in self.report['cases']:
            for name in ('audio', 'zero', 'audio_common', 'zero_common', 'raw_common', 'v1_common'):
                value = row[name]
                for reference, available in [('reference_points', 'available_points'), ('reference_cells', 'available_cells'),
                                              ('reference_4s_intervals', 'available_4s_intervals')]:
                    self.assertGreater(value[reference], 0)
                    self.assertEqual(value[reference], value[available])
                    if name.endswith('_common'):
                        self.assertEqual(value[reference], row['raw_common'][reference])
                self.assertTrue(all(value[k] is not None for k in KEYS))

    def test_retained_failure_and_recording_regressions(self):
        self.assertEqual(self.report['gate'], dict(complete=True, finished=True, audio_signal=False,
                                                  common_nonregression=False, recording_breadth=False, passed=False))
        self.assertFalse(self.report['holdout_access'])
        self.assertFalse(self.report['independent_acceptance'])
        self.assertFalse(self.report['production_change'])
        for role, counts in [('development', [3, 3]), ('diagnostic', [13, 15])]:
            rows = [r for r in self.report['cases'] if r['role'] == role]
            for key, count in zip((KEYS[0], KEYS[2]), counts):
                self.assertEqual(sum(r['regressions_vs_raw_common'][key] for r in rows), count)
        for row in self.report['cases']:
            for baseline in ('raw_common', 'v1_common'):
                for key in KEYS:
                    self.assertEqual(row['regressions_vs_' + baseline][key], row['audio_common'][key] > row[baseline][key])


if __name__ == '__main__':
    unittest.main()
