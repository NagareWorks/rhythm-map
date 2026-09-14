import itertools
import json
import unittest

import numpy as np

from experiments.tempo_pairs import protocol as p
from experiments.tempo_pairs.render import duration_metrics, timing_metrics
from experiments.tempo_pairs.timeline import Timeline


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((p.ROOT / 'experiments/tempo_pairs/results-v1.json').read_bytes())

    def test_complete_frozen_closure(self):
        self.assertTrue(self.report['completed'])
        self.assertEqual(self.report['source_closure'], p.source_closure())
        self.assertEqual(self.report['sources'], p.selected_sources())

    def test_complete_population(self):
        rows = self.report['authored']
        self.assertEqual(len(rows), 30)
        self.assertEqual({(r['backend'], r['profile'], r['witness']) for r in rows},
                         set(itertools.product(p.BACKENDS, p.PROFILES, p.WITNESSES)))
        rows = self.report['previews']
        self.assertEqual(len(rows), 30)
        self.assertEqual({(r['source_id'], r['backend'], r['profile']) for r in rows},
                         set(itertools.product(p.IDS, p.BACKENDS, p.PROFILES)))

    def test_no_training_or_alignment_promotion(self):
        for key in ('training_eligible', 'music_alignment_verified', 'independent_acceptance', 'holdout_access'):
            self.assertIs(self.report[key], False)
        for key in ('model_calls', 'optimizer_steps'):
            self.assertEqual(self.report[key], 0)
        self.assertTrue(all(r['training_eligible'] is False for r in self.report['previews']))

    def test_all_scalar_timing_recomputed(self):
        events = np.arange(.75, 59.5, .5)
        for row in self.report['authored']:
            if row['witness'] == 'tone':
                self.assertEqual(row['pitch']['passed'], max(abs(v) for v in row['pitch']['cents_per_segment']) <= p.LIMITS['pitch_max_cents'])
                continue
            expected = Timeline(p.SOURCE_EDGES, p.PROFILES[row['profile']]).to_output(events)
            self.assertEqual(row['timing'], timing_metrics(expected, row['observed_events_s']))

    def test_all_durations_recomputed(self):
        for row in self.report['authored'] + self.report['previews']:
            self.assertEqual(row['duration'], duration_metrics(row['duration']['actual_segment_samples'], p.PROFILES[row['profile']]))

    def test_failure_preserved_and_gate_uses_every_witness(self):
        self.assertEqual(self.report['limits'], p.LIMITS)
        self.assertEqual(self.report['authored_gate_by_backend'], {'atempo': False, 'rubberband': True})
        for row in self.report['authored']:
            self.assertEqual(row['passed'], all(row[k]['passed'] for k in ('duration', 'timing', 'pitch') if k in row))
        for backend in p.BACKENDS:
            self.assertEqual(self.report['authored_gate_by_backend'][backend],
                             all(r['passed'] for r in self.report['authored'] if r['backend'] == backend))


if __name__ == '__main__':
    unittest.main()
