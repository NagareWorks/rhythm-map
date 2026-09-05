"""Boundary stationarity, paired frozen comparisons, and exhaustive path audit."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import unittest

import numpy as np

import clock_boundary_audit as audit
from test_search_omission import beta, reference, tempo_matrix

ROOT = Path(__file__).resolve().parents[2]


def independent_boundary(domain):
    """Closed-form reversible tick law, independent of the audit's linear solve."""
    periods = list(range(domain['min_period'], domain['max_period'] + 1))
    t = tempo_matrix(domain)
    if len(periods) == 1:
        pi = [1.]
    else:
        coordinates = [math.log(100) * math.log2(p) for p in periods]
        off = [sum(math.exp(-abs(x - y)) for j, y in enumerate(coordinates) if j != i)
               for i, x in enumerate(coordinates)]
        weights = [a / (1 - t[p, p]) for a, p in zip(off, periods)]
        pi = [w / sum(weights) for w in weights]
    mean = sum(w * p for w, p in zip(pi, periods))
    out = {(r, p): sum(w * t[q, p] for w, q in zip(pi, periods) if r < q) / mean
           for r in range(max(periods)) for p in periods}
    return periods, pi, mean, out


def exhaustive(values, domain):
    """Enumerate every in-window path, marginalizing the unobserved old interval."""
    pairs, norm = reference(values)
    periods, _, _, initial = independent_boundary(domain)
    meters = list(range(domain['min_meter'], domain['max_meter'] + 1))
    transitions = tempo_matrix(domain)
    total, best, paths = 0., 0., 0
    positions = np.zeros((len(values), 7))

    def walk(t, p, m, phase, weight, path, n=0, b=0, z=0, d=0, u=0, c=0):
        nonlocal total, best, paths
        pair = pairs[t]
        for label in [None] if pair is None else range(3 if phase == 0 else 2):
            pulse, accent = label is not None and label > 0, label == 2
            nn, bb, zz, dd = n + (pair is not None), b + pulse, z + (pulse and phase == 0), d + accent
            score = math.exp(pair[0] + (pair[1] if accent else 0.)) if pulse else 1.
            current = weight * score
            following = path + [(t, p, m, label)]
            if t + p >= len(values):
                current *= beta(bb, nn - bb) * beta(dd, zz - dd) * beta(c, u - c) / norm[bb - dd, dd]
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
                    if phase < m - 1:
                        walk(t + p, q, m, phase + 1, current * transitions[p, q], following, nn, bb, zz, dd, u, c)
                    else:
                        for next_m in meters:
                            changed = next_m != m
                            mass = current * transitions[p, q] / (len(meters) - 1 if changed else 1)
                            walk(t + p, q, next_m, 0, mass, following, nn, bb, zz, dd, u + (len(meters) > 1), c + changed)

    for p, m in itertools.product(periods, meters):
        for r, phase in itertools.product(range(max(periods)), range(m)):
            mass = initial[r, p] / sum(meters)
            if mass:
                walk(r, p, m, phase, mass, [])
    return math.log(total), math.log(best), positions / total, paths


class ClockBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / 'evaluation/parity/clock-boundary-v1.json').read_text())
        cls.frozen = json.loads((ROOT / 'evaluation/parity/search-omission-v1.json').read_text())
        cls.rows = {r['case']: r for r in cls.report['cases']}

    def test_frozen_sources_and_only_initial_law_changes(self):
        for name, digest in self.report['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        for flag in ['production_output_changed', 'user_parameters_added', 'training_run', 'real_music_evaluated',
                     'holdout_opened', 'calibrated_confidence', 'feature_pipeline_changed', 'transition_law_changed',
                     'interior_meter_law_changed', 'omission_law_changed', 'terminal_law_changed',
                     'labels_are_detected_events', 'full_song_search', 'fitted_parameters']:
            self.assertIs(self.report[flag], False)
        for old in self.frozen['cases']:
            row = self.rows[old['case']]
            self.assertEqual(row['feature_pairs'], old['feature_pairs'])
            fresh = row['variants']['fresh']
            for field in ('log_ratio', 'joint_map_log_weight', 'joint_map_probability', 'states', 'transitions'):
                self.assertAlmostEqual(fresh[field], old['decoded'][field], places=10)
            positions = [[p['latent_tick_probability'], *p['inferred_label_probabilities'],
                          p['unavailable_tick_probability'], p['tempo_change_probability'], p['meter_change_probability']]
                         for p in old['decoded']['positions']]
            np.testing.assert_allclose(fresh['positions'], positions, atol=1e-11, rtol=1e-10)

    def test_stationary_first_tick_is_not_the_containing_interval(self):
        domain = self.report['domain']
        for d in [domain, dict(domain, min_period=3, max_period=3, min_meter=2, max_meter=2),
                  dict(domain, min_period=2, max_period=4, max_meter=7)]:
            law = audit.stationary_boundary(d)
            periods, pi, mean, initial = independent_boundary(d)
            np.testing.assert_allclose(law['tick_stationary_probabilities'], pi, atol=1e-12)
            np.testing.assert_allclose(law['first_tick_weights'], [[initial[r, p] for p in periods] for r in range(max(periods))], atol=1e-12)
            self.assertAlmostEqual(sum(initial.values()), 1., places=12)
            self.assertAlmostEqual(law['mean_period'], mean, places=12)
            self.assertAlmostEqual(sum(w for _, _, w in audit.roots(d, 'stationary')), 1., places=12)
            # Independent frame-expanded stationary chain. Phase law must work
            # at different meter rates; these checks do not fit or select h.
            transitions = tempo_matrix(d)
            states = list(itertools.chain.from_iterable(
                ((p, r, m, j) for r in range(p) for m in range(d['min_meter'], d['max_meter'] + 1) for j in range(m))
                for p in periods))
            phases = sum(range(d['min_meter'], d['max_meter'] + 1))
            for h in (.2, .5, .8):
                prior = {s: pi[periods.index(s[0])] / mean / phases for s in states}
                following = dict.fromkeys(states, 0.)
                for (p, r, m, j), mass in prior.items():
                    if r:
                        following[p, r - 1, m, j] += mass
                    else:
                        for q in periods:
                            if j < m - 1:
                                following[q, q - 1, m, j + 1] += mass * transitions[p, q]
                            else:
                                for next_m in range(d['min_meter'], d['max_meter'] + 1):
                                    choices = d['max_meter'] - d['min_meter']
                                    prob = 1. if not choices else 1 - h if next_m == m else h / choices
                                    following[q, q - 1, next_m, 0] += mass * transitions[p, q] * prob
                np.testing.assert_allclose(list(prior.values()), list(following.values()), atol=1e-12)
        self.assertGreater(self.report['stationary_boundary']['first_tick_weights'][4][0], 0.)

    def test_complete_path_enumeration_matches_all_marginals_and_maximum(self):
        control = self.report['exhaustive_control']
        ratio, best, probabilities, paths = exhaustive(control['feature_pairs'], control['domain'])
        self.assertGreater(paths, 4532)
        self.assertAlmostEqual(control['decoded']['log_ratio'], ratio, places=11)
        self.assertAlmostEqual(control['decoded']['joint_map_log_weight'], best, places=11)
        np.testing.assert_allclose(control['decoded']['positions'], probabilities, atol=1e-12, rtol=1e-10)
        rebuilt = audit.infer(control['feature_pairs'], control['domain'], 'stationary')
        np.testing.assert_allclose(rebuilt['positions'], probabilities, atol=1e-12, rtol=1e-10)

    def test_all_main_stationary_inferences_and_traceback_scores_reproduce(self):
        domain = self.report['domain']
        for row in self.rows.values():
            actual = audit.infer(row['feature_pairs'], domain, 'stationary')
            frozen = row['variants']['stationary']
            for field in ('log_ratio', 'joint_map_log_weight', 'joint_map_probability'):
                self.assertAlmostEqual(actual[field], frozen[field], places=11)
            np.testing.assert_allclose(actual['positions'], frozen['positions'], atol=1e-12, rtol=1e-10)
            for mode, result in row['variants'].items():
                score = audit.decompose(row['feature_pairs'], domain, result['inferred_ticks'], mode)
                self.assertEqual(score.keys(), result['map_score_decomposition'].keys())
                np.testing.assert_allclose(list(score.values()), [result['map_score_decomposition'][k] for k in score], atol=1e-11, rtol=1e-10)
                self.assertAlmostEqual(score['total'], result['joint_map_log_weight'], places=11)

    def test_unknown_padding_preserves_marginals_not_joint_map_paths(self):
        maximum = self.report['domain']['max_period']
        for padded in self.report['padding_controls']:
            base = self.rows[padded['case']]
            left = padded['left']
            for mode in ('fresh', 'stationary'):
                old = base['variants'][mode]
                new = padded['variants'][mode]
                delta = new['log_ratio'] - old['log_ratio']
                core = np.array(new['positions'])[left:left + len(base['feature_pairs'])]
                if mode == 'stationary' or not left:
                    self.assertAlmostEqual(delta, 0., places=10)
                    np.testing.assert_allclose(core[:, :5], np.array(old['positions'])[:, :5], atol=1e-11)
                    # The original first tick has no in-window predecessor;
                    # compare change events only once a predecessor must exist.
                    np.testing.assert_allclose(core[maximum:, 5:], np.array(old['positions'])[maximum:, 5:], atol=1e-11)
                else:
                    self.assertGreater(abs(delta), .03)
                    self.assertGreater(np.max(np.abs(core[:, :5] - np.array(old['positions'])[:, :5])), .03)
        # Actually rerun an unseen amount of left/right unavailable padding.
        small = self.report['exhaustive_control']
        expanded = audit.infer([None] * 2 + small['feature_pairs'] + [None], small['domain'], 'stationary')
        self.assertAlmostEqual(expanded['log_ratio'], small['decoded']['log_ratio'], places=11)
        np.testing.assert_allclose(np.array(expanded['positions'])[2:-1, :5], np.array(small['decoded']['positions'])[:, :5], atol=1e-11)

    def test_boundaries_do_not_hide_tempo_or_accent_failures(self):
        self.assertEqual(self.rows['half']['variants'], self.rows['same_features_erased_constant']['variants'])
        half = self.rows['half']['variants']['stationary']['inferred_ticks']
        self.assertEqual([t['frame'] for t in half], [1, 7, 13])
        self.assertEqual(half[-1]['inferred_label'], 1)  # accent at 13 is now missed
        double = self.rows['double']['variants']['stationary']['inferred_ticks']
        self.assertEqual([t['period_frames'] for t in double], [3] * 6)
        self.assertEqual(next(t['inferred_label'] for t in double if t['frame'] == 4), 1)
        for name in ('half', 'double'):
            fixed = self.rows[name]['posthoc_fixed_paths']
            for mode, results in fixed['scores'].items():
                for key, value in results.items():
                    rebuilt = audit.decompose(self.rows[name]['feature_pairs'], self.report['domain'], fixed['paths'][key], mode)
                    self.assertEqual(value.keys(), rebuilt.keys())
                    np.testing.assert_allclose(list(value.values()), [rebuilt[k] for k in value], atol=1e-11, rtol=1e-10)
            for key in fixed['paths']:
                old, new = (fixed['scores'][mode][key] for mode in ('fresh', 'stationary'))
                for field in old.keys() - {'clock_initial', 'meter_initial', 'total'}:
                    self.assertEqual(old[field], new[field])
        for name in ('flat', 'unavailable'):
            result = self.rows[name]['variants']['stationary']
            self.assertAlmostEqual(result['log_ratio'], 0., places=11)
            np.testing.assert_allclose(np.array(result['positions'])[:, 0], 1 / self.report['stationary_boundary']['mean_period'], atol=1e-12)

    def test_invalid_input_and_budget_fail_closed(self):
        domain = self.report['domain']
        values = [None] * 18
        for invalid in ([None], [[math.nan, 0.]] * 18, [[1.]] * 18):
            with self.assertRaises(ValueError):
                audit.infer(invalid, domain, 'stationary')
        with self.assertRaisesRegex(ValueError, 'budget exceeded'):
            audit.infer(values, dict(domain, max_states=1), 'stationary')
        with self.assertRaises(ValueError):
            audit.infer(values, domain, 'unknown')
        with self.assertRaises(ValueError):
            audit.infer(values, dict(domain, max_meter=8), 'stationary')


if __name__ == '__main__':
    unittest.main()
