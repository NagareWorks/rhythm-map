import json
import math
from pathlib import Path
import unittest

from experiments.gradient_routing.packet_audit import make_plan, sha, sources
from experiments.gradient_routing.authored import learnability

HERE = Path(__file__).resolve().parents[1]


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((HERE / 'plan-v1.json').read_bytes())
        cls.result = json.loads((HERE / 'results-v1.json').read_bytes())
        cls.authored = json.loads((HERE / 'authored-v1.json').read_bytes())

    def test_live_authored_learning_gate_reduces_both_losses(self):
        # Enforce the live gate outside the already executed source closure.
        # A historical success JSON must not hide a current learning regression.
        current = learnability()
        self.assertTrue(current['both_losses_reduced'])
        self.assertEqual(current['updates'], 40)
        for term in ('phase', 'advance'):
            self.assertLess(current['after'][term], current['before'][term])

    def test_frozen_plan_sources_and_artifact_identity(self):
        self.assertEqual(self.plan, make_plan())
        self.assertEqual(sha(HERE / 'plan-v1.json'), self.result['plan_sha256'])
        self.assertEqual(self.plan['source_sha256'], sources())
        self.assertEqual(self.result['source_sha256'], self.plan['source_sha256'])
        self.assertEqual(self.authored['source_sha256'], self.plan['source_sha256'])
        self.assertEqual(self.plan['archive_sha256'], self.result['archive_sha256'])

    def test_all_eighty_pairs_and_all_parameter_groups_remain_accounted(self):
        self.assertEqual(len(self.result['cases']), 80)
        for plan_row, row in zip(self.plan['population'], self.result['cases']):
            self.assertEqual(plan_row, {k: row[k] for k in plan_row})
            self.assertEqual(set(row['diagnostic']['groups']), set(self.plan['groups']))
            for group in row['diagnostic']['groups'].values():
                if group['count_direction_defined']:
                    self.assertGreater(group['routed_count_cosine'], 0.)
                    self.assertGreaterEqual(group['count_projection_ratio_log2'], math.log2(1 - 1e-10))
                else:
                    self.assertIsNone(group['routed_count_cosine'])
        for kind in ('audio', 'zero'):
            for group in self.result['fit_aggregate'][kind]['groups'].values():
                self.assertGreater(group['routed_count_cosine'], 0.)

    def test_negative_per_record_prior_directions_are_not_erased(self):
        for role, n, negative in (('fit', 20, 4), ('development', 5, 0), ('diagnostic', 15, 1)):
            rows = [r for r in self.result['cases'] if r['role'] == role and r['checkpoint'] == 'audio']
            self.assertEqual(len(rows), n)
            self.assertEqual(sum(r['diagnostic']['groups']['period_head']['routed_prior_cosine'] < 0
                                 for r in rows), negative)
        aggregate = self.result['fit_aggregate']['audio']['groups']
        self.assertAlmostEqual(aggregate['period_head']['routed_prior_cosine'], .6037698208140776)
        self.assertAlmostEqual(aggregate['shared_cnn']['routed_prior_cosine'], .6702951039060064)

    def test_synthetic_learning_does_not_override_failed_optimizer_admission(self):
        learning = self.authored['learning']
        self.assertTrue(learning['both_losses_reduced'])
        self.assertEqual(learning['updates'], 40)
        for term in ('phase', 'advance'):
            self.assertLess(learning['after'][term], learning['before'][term])
        for name in ('preconditioning', 'momentum'):
            witness = self.authored[name]
            self.assertFalse(witness['check']['first_order_nonincrease'])
            self.assertGreater(witness['quadratic_count_loss_change'], 0.)
        self.assertFalse(self.authored['admission'])
        self.assertEqual(self.authored['musical_updates'], 0)
        for key in ('model_forwards', 'clock_forwards', 'optimizer_updates'):
            self.assertEqual(self.result[key], 0)
        self.assertFalse(self.result['prior_used_for_routing'])
        self.assertFalse(self.result['holdout_access'])
        self.assertLess(self.result['elapsed_s'], self.plan['seconds'])
        self.assertEqual(self.result['decision'], 'mechanism_only_adamw_boundary_not_admitted')


if __name__ == '__main__':
    unittest.main()
