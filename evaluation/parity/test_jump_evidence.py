"""Independent combinatorics, frozen-factor reconstruction and retained failures."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import unittest

import numpy as np

import jump_evidence_audit as audit
from clock_boundary_audit import decompose, infer
from test_search_omission import reference, tempo_matrix

ROOT = Path(__file__).resolve().parents[2]


class JumpEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / 'evaluation/parity/jump-evidence-v1.json').read_text())
        cls.frozen = json.loads((ROOT / 'evaluation/parity/clock-boundary-v1.json').read_text())
        cls.rows = {r['case']: r for r in cls.report['cases']}

    def assert_report_close(self, actual, expected):
        """Exact structure/counts; libm-independent tolerance for log weights."""
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_report_close(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_report_close(left, right)
        elif isinstance(expected, float):
            self.assertAlmostEqual(actual, expected, places=11)
        else:
            self.assertEqual(actual, expected)

    def test_provenance_and_scope_are_frozen(self):
        for name, digest in self.report['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        for flag in ('production_output_changed', 'user_parameters_added', 'training_run',
                     'real_music_evaluated', 'holdout_opened', 'calibrated_confidence',
                     'fitted_parameters', 'transition_law_changed', 'boundary_law_changed',
                     'gain_is_calibration', 'labels_are_detected_events', 'full_song_search'):
            self.assertIs(self.report[flag], False)
        self.assertEqual(self.report['domain'], self.frozen['domain'])
        self.assertEqual(self.report['gains'], [1, 2, 4])
        for row in self.frozen['cases']:
            self.assertEqual(self.rows[row['case']]['feature_pairs'], row['feature_pairs'])
            self.assertEqual(self.rows[row['case']]['gain_inferences'][0], audit.compact(row['variants']['stationary']))

    def test_occurrence_and_destination_reconstruct_frozen_transition_law(self):
        controls = self.report['prior_controls']
        base = controls['base']
        independent = tempo_matrix(self.report['domain'])
        np.testing.assert_allclose(base['transition_matrix'],
                                   [[independent[p, q] for q in base['periods']] for p in base['periods']], atol=1e-14)
        for name, law in controls.items():
            rebuilt = audit.prior_accounting(law['periods'], law['frame_duration'])
            self.assert_report_close(rebuilt, law)
            np.testing.assert_allclose(np.sum(law['transition_matrix'], axis=1), 1., atol=1e-14)
            np.testing.assert_allclose(np.array(law['stay_probabilities']) + law['jump_occurrence_probabilities'], 1., atol=1e-14)
            np.testing.assert_allclose(np.sum(law['jump_destination_probabilities'], axis=1),
                                       0. if name == 'singleton' else 1., atol=1e-14)
            self.assertTrue(all(law['jump_destination_probabilities'][i][i] == 0 for i in range(len(law['periods']))))

    def test_time_unit_change_is_not_domain_refinement(self):
        base, units, refined = (self.report['prior_controls'][k] for k in ('base', 'same_atoms_new_units', 'refined_domain'))
        self.assertEqual(base['physical_periods'], units['physical_periods'])
        np.testing.assert_allclose(base['transition_matrix'], units['transition_matrix'], atol=1e-14)
        self.assertAlmostEqual(base['rate_per_time_unit'], units['rate_per_time_unit'], places=14)
        self.assertAlmostEqual(base['rate_per_frame'], 2 * units['rate_per_frame'], places=14)
        self.assertGreater(refined['rate_per_time_unit'] / base['rate_per_time_unit'], 2.5)
        # Destination atoms shrink when new alternatives appear. Conditioning
        # the refined destination back on the same old alternatives recovers it.
        indices = [refined['periods'].index(p) for p in units['periods']]
        for old_i, fine_i in enumerate(indices):
            restricted = np.array(refined['jump_destination_probabilities'][fine_i])[indices]
            np.testing.assert_allclose(restricted / restricted.sum(), base['jump_destination_probabilities'][old_i], atol=1e-14)

    def test_integer_limit_matches_independent_complete_assignment_enumeration(self):
        values = [[2, 1], [0, 2], None, [2, 1], [-1, 0], [0, 0]]
        frames = [i for i, v in enumerate(values) if v is not None]
        groups = {}
        for labels in itertools.product(range(3), repeat=len(frames)):
            counts = labels.count(1), labels.count(2)
            score = sum(values[t][0] + (values[t][1] if label == 2 else 0)
                        for t, label in zip(frames, labels) if label)
            groups.setdefault(counts, []).append((score, labels))
        self.assertEqual(sum(map(len, groups.values())), 3 ** len(frames))
        for entries in groups.values():
            maximum = max(s for s, _ in entries)
            ties = sum(s == maximum for s, _ in entries)
            for score, labels in (min(entries), max(entries)):
                ticks = [dict(frame=t, inferred_label=label) for t, label in zip(frames, labels)]
                limit = audit.assignment_limit(values, ticks)
                self.assertEqual(limit['assignment_count'], len(entries))
                self.assertEqual(limit['maximizing_assignments'], ties)
                self.assertEqual(limit['maximum_feature_score'], maximum)
                self.assertEqual(limit['gain_slope'], score - maximum)
                if score == maximum:
                    self.assertAlmostEqual(limit['limiting_log_feature_ratio'], math.log(len(entries) / ties), places=12)
                else:
                    self.assertIsNone(limit['limiting_log_feature_ratio'])
                # No shared polynomial implementation in this finite-gain check.
                gain = 16
                ratio = len(entries) * math.exp(gain * (score - maximum)) / sum(math.exp(gain * (s - maximum)) for s, _ in entries)
                pairs, norms = reference(audit.scaled(values, gain))
                numerator = math.exp(sum(pairs[t][0] + (pairs[t][1] if label == 2 else 0)
                                         for t, label in zip(frames, labels) if label))
                self.assertAlmostEqual(numerator / norms[labels.count(1), labels.count(2)], ratio, places=10)

    def test_flat_unavailable_and_additive_offsets_do_not_create_feature_evidence(self):
        ticks = [dict(frame=0, inferred_label=1), dict(frame=2, inferred_label=2)]
        for values in ([[0, 0]] * 5, [[7, -3]] * 5):
            limit = audit.assignment_limit(values, ticks)
            self.assertEqual(limit['assignment_count'], limit['maximizing_assignments'])
            self.assertEqual(limit['limiting_log_feature_ratio'], 0.)
        unknown = audit.assignment_limit([None] * 5, [dict(frame=2, inferred_label=None)])
        self.assertEqual(unknown['limiting_log_feature_ratio'], 0.)
        for name in ('half', 'double'):
            row = self.rows[name]
            ticks = row['fixed_paths']['authored']['path']
            original = audit.assignment_limit(row['feature_pairs'], ticks)
            shifted = audit.assignment_limit([[v[0] + 7, v[1] - 3] for v in row['feature_pairs']], ticks)
            for key in ('gain_slope', 'limiting_log_feature_ratio', 'maximizing_assignments'):
                self.assertEqual(original[key], shifted[key])

    def test_fixed_path_limits_preserve_failure_without_calling_them_family_odds(self):
        for name in ('half', 'double'):
            row = self.rows[name]
            paths = row['fixed_paths']
            for item in paths.values():
                rebuilt = audit.fixed_path_audit(row['feature_pairs'], self.report['domain'], item['path'])
                self.assert_report_close(rebuilt, item)
                self.assertEqual(item['feature_limit']['gain_slope'], 0)
                high = decompose(audit.scaled(row['feature_pairs'], 16), self.report['domain'], item['path'], 'stationary')
                self.assertAlmostEqual(high['total'], item['limiting_joint_log_weight'], places=10)
                for score in item['gain_scores']:
                    self.assertAlmostEqual(sum(v for k, v in score.items() if k not in ('total', 'feature_numerator', 'paired_normalizer')),
                                           item['nonfeature_log_weight'], places=12)
            self.assertLess(paths['authored']['limiting_joint_log_weight'], paths['frozen_stationary_map']['limiting_joint_log_weight'])
            for i in range(len(audit.GAINS)):
                self.assertLess(paths['authored']['gain_scores'][i]['total'], paths['frozen_stationary_map']['gain_scores'][i]['total'])

    def test_all_gain_inferences_reproduce_without_feature_or_prior_selection(self):
        for row in self.rows.values():
            for gain, frozen in zip(audit.GAINS[1:], row['gain_inferences'][1:]):
                actual = infer(audit.scaled(row['feature_pairs'], gain), self.report['domain'], 'stationary')
                self.assertAlmostEqual(actual['joint_map_log_weight'], frozen['joint_map_log_weight'], places=10)
                self.assertAlmostEqual(actual['log_ratio'], frozen['log_ratio'], places=10)
                np.testing.assert_allclose(actual['positions'], frozen['positions'], atol=1e-11)
        self.assertEqual(self.rows['half']['gain_inferences'], self.rows['same_features_erased_constant']['gain_inferences'])
        for result in self.rows['half']['gain_inferences']:
            self.assertEqual({t['period_frames'] for t in result['inferred_ticks']}, {6})
        for result in self.rows['double']['gain_inferences']:
            self.assertEqual({t['period_frames'] for t in result['inferred_ticks']}, {3})

    def test_more_context_retains_success_and_both_acceleration_failures(self):
        for row, (name, values, authored) in zip(self.report['context_controls'], audit.context_controls()):
            self.assertEqual(row['case'], name)
            self.assertEqual(row['feature_pairs'], values)
            self.assertEqual(row['posthoc_authored']['path'], authored)
            actual = infer(values, self.report['domain'], 'stationary')
            frozen = row['decoded']
            self.assertAlmostEqual(actual['joint_map_log_weight'], frozen['joint_map_log_weight'], places=10)
            self.assertAlmostEqual(actual['log_ratio'], frozen['log_ratio'], places=10)
            np.testing.assert_allclose(actual['positions'], frozen['positions'], atol=1e-11)
            self.assertEqual(actual['states'], 125844)
            self.assertLess(actual['states'], self.report['domain']['max_states'])
            for key in ('posthoc_authored', 'posthoc_map'):
                self.assert_report_close(audit.fixed_path_audit(values, self.report['domain'], row[key]['path']), row[key])
            authored_score = row['posthoc_authored']['gain_scores'][0]['total']
            if name in ('long_constant', 'long_half'):
                self.assertAlmostEqual(authored_score, frozen['joint_map_log_weight'], places=11)
            else:
                self.assertGreater(frozen['joint_map_log_weight'], authored_score + .2)

    def test_invalid_limit_inputs_and_budget_are_not_silently_accepted(self):
        for periods, dt in (([], 1.), ([3, 3], 1.), ([4, 3], 1.), ([0], 1.), ([3], 0.), ([3], math.inf)):
            with self.assertRaises(ValueError):
                audit.prior_accounting(periods, dt)
        for values in ([], [[1.5, 0]], [[1]], [[math.nan, 0]], [[0, 0]] * 33):
            with self.assertRaises(ValueError):
                audit.assignment_limit(values, [])
        for ticks in ([dict(frame=2, inferred_label=1)], [dict(frame=0, inferred_label=None)],
                      [dict(frame=0, inferred_label=1)] * 2):
            with self.assertRaises(ValueError):
                audit.assignment_limit([[0, 0]], ticks)
        with self.assertRaises(ValueError):
            audit.assignment_limit([None], [dict(frame=0, inferred_label=0)])
        with self.assertRaisesRegex(ValueError, 'budget exceeded'):
            infer([[0, 0]] * 27, dict(self.report['domain'], max_states=1), 'stationary')


if __name__ == '__main__':
    unittest.main()
