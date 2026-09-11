"""Audit an incomplete numerical experiment without Torch or private data."""
import hashlib
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'experiments/phase_sync_fit'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PhaseSyncFailureReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((HERE / 'plan-v1.json').read_bytes())
        cls.execution = json.loads((HERE / 'execution-v1.json').read_bytes())
        cls.failure = json.loads((HERE / 'failure-v1.json').read_bytes())
        cls.report = json.loads((HERE / 'results-v1.json').read_bytes())
        cls.witness = json.loads((HERE / 'stability-v1.json').read_bytes())
        cls.journal = [json.loads(line) for line in (HERE / 'journal-v1.jsonl').read_text().splitlines()]

    def test_frozen_sources_plan_and_failure_artifacts_match(self):
        for filename, key in (('plan-v1.json', 'plan_sha256'), ('execution-v1.json', 'execution_sha256'),
                              ('failure-v1.json', 'failure_sha256'), ('journal-v1.jsonl', 'journal_sha256'),
                              ('stability-v1.json', 'stability_witness_sha256')):
            self.assertEqual(sha(HERE / filename), self.report[key])
        self.assertEqual(self.execution['plan_sha256'], self.report['plan_sha256'])
        self.assertEqual(self.execution['source_sha256'], self.plan['source_sha256'])
        self.assertEqual(len(self.plan['source_sha256']), 24)
        for path, expected in self.plan['source_sha256'].items():
            self.assertEqual(sha(ROOT / path), expected, path)
            self.assertNotIn('/diagnostics/', path)
        for path, expected in self.witness['source_sha256'].items():
            self.assertEqual(sha(ROOT / path), expected, path)
        self.assertEqual(self.plan['inputs_sha256'], sha(ROOT / 'experiments/coupled_clock/inputs-v1.json'))
        for name, folder in (('coupled', 'coupled_clock'), ('trajectory', 'trajectory_clock')):
            self.assertEqual(sha(ROOT / 'experiments' / folder / 'results-v1.json'),
                             self.report['parent_report_sha256'][name])

    def test_preflight_change_is_initial_identity_only_and_not_training(self):
        preflight = self.report['initialization_preflight']
        self.assertEqual(preflight['optimizer_steps'], 0)
        self.assertEqual(preflight['only_changed_plan_key'], 'initial_state_sha256')
        old = dict(self.plan, initial_state_sha256='1d6562c263588c7aa89fb54170246a8e3a091f130e67061b00a84f63ab2b753d')
        old_hash = hashlib.sha256((json.dumps(old, indent=2) + '\n').encode()).hexdigest()
        self.assertEqual(old_hash, preflight['vdi_plan_sha256'])
        self.assertEqual(self.report['plan_sha256'], preflight['target_plan_sha256'])
        self.assertEqual(self.execution['initial_state_sha256'], self.plan['initial_state_sha256'])
        self.assertEqual(self.journal[0]['initial_state_sha256'], self.plan['initial_state_sha256'])

    def test_journal_bounds_updates_without_inventing_exact_count(self):
        self.assertEqual(len([row for row in self.journal if row['event'] == 'fit_started']), 1)
        self.assertEqual(self.journal[0]['kind'], 'audio')
        epochs = [row for row in self.journal if row['event'] == 'epoch']
        self.assertEqual([row['epoch'] for row in epochs], list(range(1, 17)))
        self.assertEqual([row['updates'] for row in epochs], list(range(5, 81, 5)))
        self.assertEqual(self.report['completed_audio_epochs'], len(epochs))
        self.assertEqual(self.report['first_incomplete_epoch'], 17)
        self.assertEqual(self.report['confirmed_optimizer_updates'], epochs[-1]['updates'])
        self.assertIsNone(self.report['exact_optimizer_updates'])
        self.assertEqual(self.report['possible_unjournaled_updates_max'], 4)
        best = min(epochs, key=lambda row: row['development_loss'])
        self.assertEqual(best['epoch'], 11)
        self.assertEqual(self.report['best_logged_development_epoch'], best['epoch'])
        self.assertEqual(self.report['best_logged_development_loss'], best['development_loss'])
        self.assertEqual(self.report['last_logged_development_loss'], epochs[-1]['development_loss'])
        self.assertTrue(all(math.isfinite(row['development_loss']) for row in epochs))

    def test_incomplete_fit_is_not_a_musical_comparison_or_selected_model(self):
        self.assertTrue(self.report['training'])
        self.assertEqual(self.report['planned_fits'], 2)
        self.assertEqual(self.report['started_fits'], 1)
        self.assertEqual(self.report['completed_fits'], 0)
        self.assertEqual(self.report['checkpoint_sequence_evaluations'], 0)
        for key in ('fitted_zero_started', 'checkpoint_saved', 'failed_model_state_saved',
                    'failed_recording_known', 'research_gate_passed', 'musical_accuracy_measured',
                    'independent_acceptance', 'encoder_inference', 'holdout_access',
                    'production_change', 'automatic_retry', 'container_oom_killed'):
            self.assertFalse(self.report[key], key)
        self.assertEqual(self.report['container_exit_code'], 1)
        self.assertEqual(self.failure['exception_type'], 'ValueError')
        self.assertEqual(self.report['failure_site']['message'], 'nonfinite clock gradient')
        self.assertNotIn('cases', self.report)
        self.assertNotIn('summary', self.report)
        self.assertEqual(self.report['decision'], 'close_fixed_phase_sync_fit_on_nonfinite_adjoint_no_automatic_retry')

    def test_opposed_witness_is_authored_not_failed_music_replay(self):
        self.assertEqual((self.witness['cells'], self.witness['fps'], self.witness['prior_bpm']), (3000, 50, 120))
        self.assertTrue(self.witness['inputs_finite'])
        self.assertEqual(self.witness['optimizer_steps'], 0)
        self.assertFalse(self.witness['musical_data'])
        self.assertFalse(self.witness['failed_music_state_reproduced'])
        for row in self.witness['evidence'].values():
            self.assertTrue(row['forward_finite'])
            self.assertTrue(row['clock_strictly_increasing'])
            self.assertGreater(row['min_bpm'], 119.999998)
            self.assertLess(row['max_bpm'], 120.000002)
            self.assertGreater(row['local_state_jacobian_min'], 1.)
            self.assertGreater(row['analytical_origin_gradient_log10'], self.witness['float32_max_log10'])
        row = self.witness['evidence']['float64']
        self.assertAlmostEqual(row['observed_origin_gradient_log10'], row['analytical_origin_gradient_log10'])
        self.assertIsNone(row['backward_error'])
        self.assertEqual(self.witness['evidence']['float32']['backward_error'], 'nonfinite clock gradient')


if __name__ == '__main__':
    unittest.main()
