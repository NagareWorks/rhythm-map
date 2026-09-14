import itertools
import json
import unittest

import numpy as np

from experiments.tempo_alignment.alignment import LIMITS, paired_support, region, summarize
from experiments.tempo_alignment.run import OLD_SHA, changed_context, closure, interior_gate, negative_summary
from experiments.tempo_pairs import protocol as old
from experiments.tempo_pairs.timeline import Timeline


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((old.ROOT/'experiments/tempo_alignment/results-v1.json').read_bytes())

    def test_closure_and_previous_outcome_frozen(self):
        self.assertEqual(self.report['source_closure'], closure())
        self.assertEqual(self.report['previous_report_sha256'], OLD_SHA)
        prior = json.loads((old.ROOT/'experiments/tempo_pairs/results-v1.json').read_bytes())
        self.assertEqual(self.report['inputs'], [r for r in prior['previews'] if r['backend'] == 'rubberband'])

    def test_population_complete(self):
        cases = self.report['cases']
        self.assertEqual(len(cases), 15)
        self.assertEqual({(c['source_id'], c['profile']) for c in cases}, set(itertools.product(old.IDS, old.PROFILES)))
        self.assertEqual(sum(len(c['points']) for c in cases), 2215)
        for case in cases:
            t = Timeline(old.SOURCE_EDGES, old.PROFILES[case['profile']])
            grid = [r for r in case['points'] if r['kind'] == 'grid']
            self.assertEqual([r['natural']['center_s'] for r in grid], np.arange(.5, t.output_edges[-1], 1.).tolist())

    def test_every_support_and_region_recomputed(self):
        for case in self.report['cases']:
            t = Timeline(old.SOURCE_EDGES, old.PROFILES[case['profile']])
            for r in case['points']:
                center = r['natural']['center_s']
                self.assertEqual(r['region'], region(center, t.output_edges[-1], t.output_edges[1:-1]))
                self.assertEqual(r['feature_capture_supported'], paired_support(r['natural'], r['delayed']))
                if 'wrong_map' in r:
                    self.assertEqual(r['changed_context'], changed_context(t, center))

    def test_all_summaries_and_gates_recomputed(self):
        self.assertEqual(self.report['limits'], LIMITS)
        for case in self.report['cases']:
            grid = [r for r in case['points'] if r['kind'] == 'grid']
            self.assertEqual(case['grid'], summarize(grid))
            self.assertEqual(case['beats'], summarize([r for r in case['points'] if r['kind'] == 'beat']))
            self.assertEqual(case['interior_gate_passed'], interior_gate(case['grid'], case['beats']))
            if 'negative_control' in case:
                self.assertEqual(case['negative_control'], negative_summary(grid))
        self.assertEqual(self.report['interior_gate_passed'], all(c['interior_gate_passed'] for c in self.report['cases']))
        self.assertEqual(self.report['negative_gate_passed'], all(c['negative_control']['passed'] for c in self.report['cases'] if 'negative_control' in c))
        self.assertEqual(self.report['gate_passed'], self.report['negative_gate_passed'] and self.report['interior_gate_passed'])

    def test_failures_not_promoted_or_silently_restricted(self):
        self.assertTrue(self.report['completed'])
        self.assertFalse(self.report['gate_passed'])
        self.assertTrue(self.report['negative_gate_passed'])
        self.assertEqual(sum(c['interior_gate_passed'] for c in self.report['cases']), 11)
        for key in ('training_eligible', 'holdout_access', 'independent_acceptance'):
            self.assertIs(self.report[key], False)
        for key in ('model_calls', 'optimizer_steps'):
            self.assertEqual(self.report[key], 0)
        self.assertTrue(all(c['training_eligible'] is False for c in self.report['cases']))

    def test_reported_measurement_limit_is_not_asserted_drift(self):
        points = [r for c in self.report['cases'] if 'berlioz' in c['source_id'] for r in c['points']
                  if r['kind'] == 'grid' and r['region'] == 'interior' and r['natural']['status'] == 'mismatch']
        self.assertEqual(len(points), 91)
        self.assertTrue(all(r['natural']['lag_s'] is None for r in points))
        self.assertTrue(all(r['natural']['resolutions'][0]['score'] < LIMITS['min_score'] for r in points))
        self.assertEqual(sum(r['natural']['resolutions'][1]['status'] == 'qualified' for r in points), 74)


if __name__ == '__main__':
    unittest.main()
