"""Joint observation normalization and independent complete-path verification."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import unittest

import numpy as np

import presence_likelihood_audit as audit
from test_clock_boundary import independent_boundary
from test_search_omission import tempo_matrix

ROOT = Path(__file__).resolve().parents[2]
SMALL = dict(min_period=2, max_period=3, min_meter=2, max_meter=3, max_states=250000)


def integral(a, b):
    return math.exp(math.lgamma(a + 1) + math.lgamma(b + 1) - math.lgamma(a + b + 2))


def exhaustive(evidence, domain):
    """No state merging, shared emissions, terminal helper, or log-space search."""
    periods, _, _, initial = independent_boundary(domain)
    meters = list(range(domain['min_meter'], domain['max_meter'] + 1))
    transitions = tempo_matrix(domain)
    total = best = 0.
    paths = 0
    positions = np.zeros((len(evidence), 7))

    def walk(t, p, m, phase, weight, path, n=0, b=0, z=0, d=0, u=0, c=0):
        nonlocal total, best, paths
        value = evidence[t]
        for label in [None] if value is None else range(3 if phase == 0 else 2):
            pulse, accent = label is not None and label > 0, label == 2
            nn, bb, zz, dd = n + (value is not None), b + pulse, z + (pulse and phase == 0), d + accent
            ratio = 1. if label is None else math.exp(value[label] - value[0])
            current = weight * ratio
            following = path + [(t, p, m, label)]
            if t + p >= len(evidence):
                current *= integral(bb, nn - bb) * integral(dd, zz - dd) * integral(c, u - c)
                total += current
                best = max(best, current)
                paths += 1
                for i, (frame, period, meter, state) in enumerate(following):
                    positions[frame, 0] += current
                    positions[frame, 4 if state is None else state + 1] += current
                    if i:
                        positions[frame, 5] += current * (period != following[i - 1][1])
                        positions[frame, 6] += current * (meter != following[i - 1][2])
            else:
                for q in periods:
                    for next_m in meters if phase == m - 1 else [m]:
                        changed = next_m != m
                        probability = transitions[p, q] / (len(meters) - 1 if changed else 1)
                        walk(t + p, q, next_m, (phase + 1) % m, current * probability, following,
                             nn, bb, zz, dd, u + (phase == m - 1 and len(meters) > 1), c + changed)

    for p, m in itertools.product(periods, meters):
        for r, phase in itertools.product(range(max(periods)), range(m)):
            mass = initial[r, p] / sum(meters)
            if mass:
                walk(r, p, m, phase, mass, [])
    return dict(log_ratio=math.log(total), joint_map_log_weight=math.log(best),
                positions=positions / total, paths=paths)


class PresenceLikelihoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / 'evaluation/parity/presence-likelihood-v1.json').read_text())
        cls.previous = json.loads((ROOT / 'evaluation/parity/jump-evidence-v1.json').read_text())
        cls.rows = {r['case']: r for r in cls.report['cases']}

    def test_provenance_unchanged_prior_and_no_audio_calibration_claim(self):
        for name, digest in self.report['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        for flag in ('production_output_changed', 'user_parameters_added', 'training_run', 'real_music_evaluated',
                     'holdout_opened', 'calibrated_audio_confidence', 'fitted_parameters', 'transition_law_changed',
                     'boundary_law_changed', 'omission_law_changed', 'labels_are_detected_events', 'full_song_search'):
            self.assertIs(self.report[flag], False)
        self.assertEqual(self.report['domain'], self.previous['domain'])
        self.assertEqual([r['feature_pairs'] for r in self.report['cases']], [r['feature_pairs'] for r in self.previous['cases']])

    def test_known_sensor_densities_integrate_to_one_not_softmax_over_labels(self):
        nodes, weights = np.polynomial.hermite.hermgauss(24)
        integrals = np.zeros(3)
        for i, j in itertools.product(range(len(nodes)), repeat=2):
            b, d = math.sqrt(2) * nodes[i], math.sqrt(2) * nodes[j]
            logs = audit.gaussian_sensor([[b, d]])[0]
            integrals += weights[i] * weights[j] / math.pi * np.exp(np.array(logs) - logs[0])
        np.testing.assert_allclose(integrals, 1., atol=1e-12)
        for pair in ([0., 0.], [4., 3.], [-4., -4.], [.5, .5]):
            logs = audit.gaussian_sensor([pair])[0]
            self.assertAlmostEqual(logs[1] - logs[0], pair[0] - .5, places=12)
            self.assertAlmostEqual(logs[2] - logs[0], sum(pair) - 1., places=12)
        self.assertEqual(audit.gaussian_sensor([None]), [None])

    def test_independent_complete_paths_match_all_marginals_and_maximum(self):
        for values in ([[0., 0.], [4., 3.], None, [2., 0.], [-4., -4.], [4., 0.], [0., 0.], [1., 1.]],
                       [[.5, .5]] * 8, [None] * 8):
            evidence = audit.gaussian_sensor(values)
            explicit = exhaustive(evidence, SMALL)
            actual = audit.infer(evidence, SMALL)
            self.assertGreater(explicit['paths'], 100)
            for key in ('log_ratio', 'joint_map_log_weight'):
                self.assertAlmostEqual(actual[key], explicit[key], places=11)
            np.testing.assert_allclose(actual['positions'], explicit['positions'], atol=1e-11)
            self.assertAlmostEqual(audit.score_path(evidence, SMALL, actual['inferred_ticks'])['total'], actual['joint_map_log_weight'], places=11)

    def test_all_discrete_observations_normalize_and_missing_frame_marginalizes(self):
        # Rows: latent class. Columns: observed symbol. Every row sums to one.
        channel = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])

        def decode(symbols):
            return audit.infer([None if s is None else np.log(channel[:, s]).tolist() for s in symbols], SMALL)

        mass = sum(math.exp(decode(symbols)['log_evidence']) for symbols in itertools.product(range(3), repeat=4))
        self.assertAlmostEqual(mass, 1., places=12)
        missing = decode([0, None, 1, 2])
        filled = [decode([0, s, 1, 2]) for s in range(3)]
        masses = [math.exp(item['log_evidence']) for item in filled]
        self.assertAlmostEqual(sum(masses), math.exp(missing['log_evidence']), places=12)
        expected = sum(mass * np.array(item['positions']) for mass, item in zip(masses, filled)) / sum(masses)
        np.testing.assert_allclose(expected[:, [0, 5, 6]], np.array(missing['positions'])[:, [0, 5, 6]], atol=1e-12)
        np.testing.assert_allclose(expected[[0, 2, 3], :], np.array(missing['positions'])[[0, 2, 3], :], atol=1e-12)

    def test_unavailable_padding_and_common_density_scale_preserve_posterior(self):
        values = [[0., 0.], [4., 3.], None, [2., 0.], [-4., -4.], [4., 0.], [0., 0.], [1., 1.]]
        evidence = audit.gaussian_sensor(values)
        base = audit.infer(evidence, SMALL)
        padded = audit.infer([None] * 2 + evidence + [None], SMALL)
        for key in ('log_ratio', 'log_evidence'):
            self.assertAlmostEqual(base[key], padded[key], places=11)
        np.testing.assert_allclose(np.array(padded['positions'])[2:-1, :5], np.array(base['positions'])[:, :5], atol=1e-11)
        np.testing.assert_allclose(np.array(padded['positions'])[5:-1, 5:], np.array(base['positions'])[3:, 5:], atol=1e-11)
        # A common log-density scale (e.g. a coordinate Jacobian) affects absolute
        # evidence, not class likelihood ratios. It is not a feature-head shift.
        shifted = [None if v is None else [x + .25 * i for x in v] for i, v in enumerate(evidence)]
        result = audit.infer(shifted, SMALL)
        self.assertAlmostEqual(result['log_ratio'], base['log_ratio'], places=11)
        self.assertAlmostEqual(result['log_evidence'] - base['log_evidence'], sum(.25 * i for i, v in enumerate(evidence) if v is not None), places=11)
        np.testing.assert_allclose(result['positions'], base['positions'], atol=1e-11)

    def test_all_frozen_results_and_fixed_path_scores_reproduce(self):
        groups = ('cases', 'context_controls', 'sensor_controls', 'absence_contrasts')
        for row in itertools.chain.from_iterable(self.report[key] for key in groups):
            evidence = audit.gaussian_sensor(row['feature_pairs'])
            actual = audit.infer(evidence, self.report['domain'])
            expected = row['decoded']
            for key in ('log_ratio', 'log_evidence', 'background_log_density', 'joint_map_log_weight', 'joint_map_probability'):
                self.assertAlmostEqual(actual[key], expected[key], places=10)
            np.testing.assert_allclose(actual['positions'], expected['positions'], atol=1e-10)
            self.assertEqual(actual['states'], expected['states'])
            self.assertEqual(actual['transitions'], expected['transitions'])
            self.assertLess(actual['states'], self.report['domain']['max_states'])
            self.assertAlmostEqual(audit.score_path(evidence, self.report['domain'], expected['inferred_ticks'])['total'], expected['joint_map_log_weight'], places=10)
            for value, probabilities in zip(evidence, actual['emission_positions']):
                if value is None:
                    self.assertEqual(probabilities, [None] * 3)
                else:
                    self.assertAlmostEqual(sum(probabilities), 1., places=11)
                    self.assertTrue(all(-1e-10 <= p <= 1. + 1e-10 for p in probabilities))
        for row in self.report['cases']:
            evidence = audit.gaussian_sensor(row['feature_pairs'])
            for path in row.get('posthoc_fixed_paths', {}).values():
                score = audit.score_path(evidence, self.report['domain'], path['path'])
                np.testing.assert_allclose(list(score.values()), list(path['score'].values()), atol=1e-11)

    def test_presence_and_absence_log_evidence_have_no_count_conditioned_ceiling(self):
        # A fixed assignment, with one ordinary event; compare adding an event
        # to a background frame. All other emission factors cancel, irrespective
        # of event counts elsewhere. No path/count-specific partition remains.
        for coordinate in (-40., -4., 0., 4., 40.):
            densities = audit.gaussian_sensor([[coordinate, 0.]])[0]
            added_event_log_ratio = densities[1] - densities[0]
            self.assertAlmostEqual(added_event_log_ratio, coordinate - .5, places=11)
            self.assertAlmostEqual(densities[0] - densities[1], -added_event_log_ratio, places=11)
        self.assertLess(audit.gaussian_sensor([[-40., 0.]])[0][1] - audit.gaussian_sensor([[-40., 0.]])[0][0], -40.)

    def test_neutral_absence_and_unavailable_are_distinct_not_detected_events(self):
        rows = {r['case']: r['decoded'] for r in self.report['sensor_controls']}
        _, _, mean, _ = independent_boundary(self.report['domain'])
        for name in ('neutral', 'unavailable'):
            self.assertAlmostEqual(rows[name]['log_ratio'], 0., places=11)
            np.testing.assert_allclose(np.array(rows[name]['positions'])[:, 0], 1 / mean, atol=1e-12)
        np.testing.assert_allclose(rows['neutral']['emission_positions'], [[1 - .5 / mean, .4 / mean, .1 / mean]] * 18, atol=1e-12)
        self.assertTrue(all(row == [None] * 3 for row in rows['unavailable']['emission_positions']))
        self.assertLess(rows['absence']['log_ratio'], -1.)
        self.assertLess(sum(row[1] + row[2] for row in rows['absence']['emission_positions']),
                        sum(row[1] + row[2] for row in rows['neutral']['emission_positions']))
        self.assertTrue(all(t['inferred_label'] == 0 for t in rows['absence']['inferred_ticks']))
        self.assertGreater(sum(row[0] for row in rows['absence']['positions']), 0.)  # latent clock still exists

    def test_original_regressions_and_identical_input_ambiguity_are_retained(self):
        self.assertEqual(self.rows['half']['decoded'], self.rows['same_features_erased_constant']['decoded'])
        for name in ('half', 'double'):
            self.assertEqual({t['period_frames'] for t in self.rows[name]['decoded']['inferred_ticks']}, {3})
        # Original flat-middle and longer-half regress relative to the old model.
        self.assertTrue(all(t['inferred_label'] > 0 for t in self.rows['flat_middle']['decoded']['inferred_ticks']))
        longer = {r['case']: r for r in self.report['context_controls']}
        self.assertGreater(longer['long_half']['decoded']['joint_map_log_weight'], longer['long_half']['authored_score']['total'])
        contrast = {r['case']: r for r in self.report['absence_contrasts']}
        for row in self.report['cases'] + self.report['context_controls']:
            self.assertEqual(contrast[row['case']]['feature_pairs'], [[-4., -4.] if p == [0., 0.] else p for p in row['feature_pairs']])
        gap = contrast['flat_middle']['decoded']['inferred_ticks']
        self.assertEqual({t['period_frames'] for t in gap}, {3})
        self.assertEqual([t['frame'] for t in gap if t['inferred_label'] == 0], [7, 10])
        self.assertEqual(contrast['long_half']['decoded']['inferred_ticks'], longer['long_half']['authored_path'])
        self.assertEqual({t['period_frames'] for t in contrast['half']['decoded']['inferred_ticks']}, {6})
        self.assertEqual({t['period_frames'] for t in contrast['double']['decoded']['inferred_ticks']}, {3})

    def test_invalid_contract_budget_and_large_log_ratios_fail_safely(self):
        for values in ([[0, 0]] * 8, [[math.nan, 0, 0]] * 8, [[0, math.inf, 0]] * 8, [[-1e308, 1e308, 0]] * 8, [None], [None] * 33):
            with self.assertRaises(ValueError):
                audit.infer(values, SMALL)
        for domain in (dict(SMALL, max_states=1), dict(SMALL, max_states=250001), dict(SMALL, min_period=2.), dict(SMALL, max_meter=8)):
            with self.assertRaises(ValueError):
                audit.infer([None] * 8, domain)
        for values in ([[math.nan, 0]], [[1.]], [[1e308, 0]]):
            with self.assertRaises(ValueError):
                audit.gaussian_sensor(values)
        result = audit.infer([[0., 1000., -1000.]] * 8, SMALL)
        self.assertTrue(math.isfinite(result['log_ratio']))
        self.assertTrue(np.isfinite(result['positions']).all())
        with self.assertRaises(ValueError):
            audit.score_path([None] * 8, SMALL, [])


if __name__ == '__main__':
    unittest.main()
