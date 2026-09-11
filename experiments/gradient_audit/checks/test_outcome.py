import json
from pathlib import Path
import unittest

from experiments.gradient_audit.register import sources, sha
from experiments.gradient_audit.run import summarize

HERE = Path(__file__).resolve().parents[1]


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((HERE / 'plan-v1.json').read_bytes())
        cls.report = json.loads((HERE / 'results-v1.json').read_bytes())
        cls.events = [json.loads(line) for line in (HERE / 'journal-v1.jsonl').read_text().splitlines()]

    def test_actual_plan_and_closed_sources_are_unchanged(self):
        self.assertEqual(sha(HERE / 'plan-v1.json'), self.report['plan_sha256'])
        self.assertEqual(self.plan['source_sha256'], sources())
        self.assertEqual(self.plan['source_sha256'], self.report['source_sha256'])
        self.assertEqual(self.plan['prerequisite_sha256'], self.report['prerequisite_sha256'])
        self.assertEqual(self.plan['runtime']['torch'], '2.10.0+cu129')
        self.assertEqual(self.plan['selected_epoch'], {'audio': 20, 'zero': 20})

    def test_every_population_member_and_durable_completion_receipt(self):
        expected = [(r['id'], k) for r in self.plan['population'] for k in ('audio', 'zero')]
        self.assertEqual([(r['id'], r['checkpoint']) for r in self.report['cases']], expected)
        self.assertEqual(len(self.events), 81)
        for index, (row, event) in enumerate(zip(self.report['cases'], self.events[:-1])):
            original = self.plan['population'][index // 2]
            for key in original:
                self.assertEqual(row[key], original[key])
            self.assertEqual(event['completed_cases'], index + 1)
            self.assertEqual((event['id'], event['checkpoint']), expected[index])
            self.assertEqual(event['gradient_packet_sha256'], row['gradient_packet_sha256'])
        self.assertEqual(self.events[-1]['event'], 'audit_complete')
        self.assertEqual(self.events[-1]['report_sha256'], self.report['private_report_sha256'])
        self.assertLess(self.events[-1]['elapsed_s'], 1200)

    def test_all_macros_preserve_roles_and_undefined_counts(self):
        for role, kinds in self.report['summary'].items():
            for kind, expected in kinds.items():
                rows = [r for r in self.report['cases'] if r['role'] == role and r['checkpoint'] == kind]
                self.assertEqual(summarize(rows), expected)
        weights = self.report['fit_work_weights']
        fit = [r for r in self.plan['population'] if r['role'] == 'fit']
        self.assertEqual(len(fit), 20)
        self.assertAlmostEqual(sum(weights[r['work']] for r in fit), 1)
        self.assertEqual(len(weights), 9)

    def test_numeric_audit_does_not_change_state_or_admit_training(self):
        self.assertEqual(self.report['model_state_before'], self.report['model_state_after'])
        self.assertEqual(self.report['actual_cnn_field_forwards'], 320)
        self.assertEqual(self.report['actual_clock_forwards'], 240)
        self.assertEqual(self.report['parameter_updates'], 0)
        for flag in ('training', 'optimizer_constructed', 'encoder_inference', 'holdout_access',
                     'production_change', 'independent_acceptance'):
            self.assertIs(self.report[flag], False)
        for row in self.report['cases']:
            self.assertLessEqual(row['metrics']['field_linearity.residual_relative_to_component_norm_sum'], 1e-10)
            self.assertLessEqual(row['metrics']['parameter_linearity.residual_relative_to_component_norm_sum'], 5e-5)
        self.assertEqual(self.report['decision'], 'fixed_checkpoint_gradient_audit_only_no_automatic_fit')


if __name__ == '__main__':
    unittest.main()
