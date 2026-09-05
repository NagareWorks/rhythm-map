"""Probability-space reconstruction plus independent complete-path enumeration."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def beta(a, b):
    return math.factorial(a) * math.factorial(b) / math.factorial(a + b + 1)


def reference(values):
    observed = [v for v in values if v is not None]
    maxima = [max((v[h] for v in observed), default=0.) for h in (0, 1)]
    centered = [None if v is None else [v[h] - maxima[h] for h in (0, 1)] for v in values]
    coefficients = {(0, 0): 1.}
    for pair in centered:
        if pair is None:
            continue
        following = dict(coefficients)
        for (plain, accent), mass in coefficients.items():
            for key, score in [((plain + 1, accent), pair[0]),
                               ((plain, accent + 1), sum(pair))]:
                following[key] = following.get(key, 0.) + mass * math.exp(score)
        coefficients = following
    n = len(observed)
    norm = {key: value / (math.comb(n, key[0]) * math.comb(n - key[0], key[1]))
            for key, value in coefficients.items()}
    return centered, norm


def tempo_matrix(domain):
    periods = list(range(domain['min_period'], domain['max_period'] + 1))
    x = {p: math.log(100) * math.log2(p) for p in periods}
    off = {p: sum(math.exp(-abs(x[p] - x[q])) for q in periods if q != p) for p in periods}
    rate = sum(math.log1p(off[p]) / p for p in periods) / len(periods)
    return {(p, q): math.exp(-rate * p) if p == q else
            -math.expm1(-rate * p) * math.exp(-abs(x[p] - x[q])) / off[p]
            for p in periods for q in periods}


def emitted(pair, phase, counts):
    # Counts: available trials, retained, retained bar ticks, accents, bar decisions, changes.
    if pair is None:
        yield None, counts, 1.
        return
    for label in range(3 if phase == 0 else 2):
        n, b, z, d, u, c = counts
        yield label, (n + 1, b + (label > 0), z + (label > 0 and phase == 0),
                      d + (label == 2), u, c), math.exp(0. if not label else pair[0] + (pair[1] if label == 2 else 0.))


def next_states(p, m, phase, counts, domain, transitions):
    periods = range(domain['min_period'], domain['max_period'] + 1)
    meters = range(domain['min_meter'], domain['max_meter'] + 1)
    for next_m in meters if phase == m - 1 else [m]:
        n, b, z, d, u, c = counts
        decision = phase == m - 1 and len(meters) > 1
        changed = next_m != m
        weight = 1. / (len(meters) - 1) if changed else 1.
        for next_p in periods:
            yield (next_p, next_m, (phase + 1) % m, n, b, z, d, u + decision, c + changed), weight * transitions[p, next_p]


def finish(counts, normalizers):
    n, b, z, d, u, c = counts
    return beta(b, n - b) * beta(d, z - d) * beta(c, u - c) / normalizers[b - d, d]


def initial(domain):
    periods = range(domain['min_period'], domain['max_period'] + 1)
    meters = range(domain['min_meter'], domain['max_meter'] + 1)
    for p, m in itertools.product(periods, meters):
        for t, phase in itertools.product(range(p), range(m)):
            yield t, (p, m, phase, 0, 0, 0, 0, 0, 0), 1. / (len(periods) * p * len(meters) * m)


def reconstruct(values, domain):
    """No Rust tables, log semiring, traceback, or source fixture construction."""
    pairs, normalizers = reference(values)
    transitions = tempo_matrix(domain)
    layers = [{} for _ in pairs]
    for t, key, mass in initial(domain):
        layers[t][key] = [mass, mass]
    partition, best = 0., 0.
    for t, layer in enumerate(layers):
        for key, (mass, maximum) in layer.items():
            p, m, phase, *counts = key
            for _, new_counts, emission in emitted(pairs[t], phase, tuple(counts)):
                if t + p >= len(pairs):
                    terminal = emission * finish(new_counts, normalizers)
                    partition += mass * terminal
                    best = max(best, maximum * terminal)
                else:
                    for next_key, transition in next_states(p, m, phase, new_counts, domain, transitions):
                        target = layers[t + p].setdefault(next_key, [0., 0.])
                        target[0] += mass * emission * transition
                        target[1] = max(target[1], maximum * emission * transition)
    return math.log(partition), math.log(best), sum(map(len, layers))


def enumerate_paths(values, domain):
    """Enumerate actual clock/meter/label paths; never merge states or use DP."""
    pairs, norm = reference(values)
    transitions = tempo_matrix(domain)
    total, maximum, paths = 0., 0., 0
    positions = np.zeros((len(values), 7))

    def walk(t, p, m, phase, weight, path, n=0, b=0, z=0, d=0, u=0, c=0):
        nonlocal total, maximum, paths
        pair = pairs[t]
        choices = [None] if pair is None else range(3 if phase == 0 else 2)
        for label in choices:
            retained = label is not None and label > 0
            accent = label == 2
            nn, bb, zz, dd = n + (pair is not None), b + retained, z + (retained and phase == 0), d + accent
            score = math.exp(pair[0] + (pair[1] if accent else 0.)) if retained else 1.
            next_path = path + [(t, p, m, phase, label)]
            current = weight * score
            if t + p >= len(values):
                current *= beta(bb, nn - bb) * beta(dd, zz - dd) * beta(c, u - c) / norm[bb - dd, dd]
                total += current
                maximum = max(maximum, current)
                paths += 1
                for i, (frame, period, meter, _, state) in enumerate(next_path):
                    positions[frame, 0] += current
                    positions[frame, 4 if state is None else state + 1] += current
                    if i:
                        positions[frame, 5] += current * (period != next_path[i - 1][1])
                        positions[frame, 6] += current * (meter != next_path[i - 1][2])
            else:
                for q in range(domain['min_period'], domain['max_period'] + 1):
                    if phase != m - 1:
                        walk(t + p, q, m, phase + 1, current * transitions[p, q], next_path, nn, bb, zz, dd, u, c)
                    else:
                        meters = range(domain['min_meter'], domain['max_meter'] + 1)
                        for next_m in meters:
                            changed = next_m != m
                            w = current * transitions[p, q] / (len(meters) - 1 if changed else 1)
                            walk(t + p, q, next_m, 0, w, next_path, nn, bb, zz, dd, u + (len(meters) > 1), c + changed)

    # Intentionally independent initial construction, including all edge phases.
    for p in range(domain['min_period'], domain['max_period'] + 1):
        for m in range(domain['min_meter'], domain['max_meter'] + 1):
            weight = 1 / ((domain['max_period'] - domain['min_period'] + 1) * p *
                          (domain['max_meter'] - domain['min_meter'] + 1) * m)
            for t in range(p):
                for phase in range(m):
                    walk(t, p, m, phase, weight, [])
    return math.log(total), math.log(maximum), positions / total, paths


def path_score(values, domain, ticks):
    pairs, norm = reference(values)
    transitions = tempo_matrix(domain)
    p0, m0 = ticks[0]['period_frames'], ticks[0]['meter']
    initial_weight = 1 / ((domain['max_period'] - domain['min_period'] + 1) * p0 *
                          (domain['max_meter'] - domain['min_meter'] + 1) * m0)
    assert ticks[0]['frame'] < p0
    counts = [0] * 6
    emission, tempo, destinations = 0., math.log(initial_weight), 0.
    for i, tick in enumerate(ticks):
        t, p, m, phase, label = [tick[k] for k in ('frame', 'period_frames', 'meter', 'beat_in_bar', 'inferred_label')]
        assert domain['min_period'] <= p <= domain['max_period']
        assert domain['min_meter'] <= m <= domain['max_meter'] and 1 <= phase <= m
        pair = pairs[t]
        assert (label is None) == (pair is None)
        if pair is not None:
            assert label in (0, 1, 2) and (label != 2 or phase == 1)
            counts[0] += 1
            counts[1] += label > 0
            counts[2] += label > 0 and phase == 1
            counts[3] += label == 2
            emission += 0. if label == 0 else pair[0] + (pair[1] if label == 2 else 0.)
        if i:
            old = ticks[i - 1]
            assert t == old['frame'] + old['period_frames']
            tempo += math.log(transitions[old['period_frames'], p])
            wrap = old['beat_in_bar'] == old['meter']
            assert phase == (1 if wrap else old['beat_in_bar'] + 1)
            assert wrap or m == old['meter']
            if wrap and domain['min_meter'] != domain['max_meter']:
                counts[4] += 1
                counts[5] += m != old['meter']
                if m != old['meter']:
                    destinations -= math.log(domain['max_meter'] - domain['min_meter'])
    assert ticks[-1]['frame'] + ticks[-1]['period_frames'] >= len(values)
    n, b, z, d, u, c = counts
    feature = emission - math.log(norm[b - d, d])
    omission = math.log(beta(b, n - b) * beta(d, z - d))
    meter = destinations + math.log(beta(c, u - c))
    return {'feature': feature, 'clock': tempo, 'omission': omission, 'meter': meter,
            'total': feature + tempo + omission + meter}


class SearchOmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / 'evaluation/parity/search-omission-v1.json').read_text())
        cls.rows = {row['case']: row for row in cls.report['cases']}

    def test_source_identity_and_search_boundaries(self):
        for key, name in [('source_sha256', 'search_omission.rs'), ('search_source_sha256', 'support/search_omission.rs'),
                          ('feature_source_sha256', 'support/shared_frames.rs'), ('time_prior_source_sha256', 'support/time_prior.rs')]:
            self.assertEqual(self.report[key], hashlib.sha256((ROOT / 'crates/rhythm-map-eval/examples' / name).read_bytes()).hexdigest())
        for flag in ['production_output_changed', 'user_parameters_added', 'training_run', 'holdout_opened',
                     'real_music_evaluated', 'supplied_clock_templates', 'truth_used_in_search',
                     'rank_pipeline_accuracy_evaluated', 'calibrated_confidence', 'labels_are_detected_events',
                     'full_song_search', 'beam_pruning']:
            self.assertIs(self.report[flag], False)
        self.assertIs(self.report['feature_space_inputs'], True)
        self.assertIs(self.report['tempo_changes_searched'], True)
        self.assertIs(self.report['meter_changes_searched'], True)

    def test_all_partitions_maxima_tracebacks_and_probability_bounds(self):
        for row in self.rows.values():
            with self.subTest(case=row['case']):
                result = row['decoded']
                ratio, best, states = reconstruct(row['feature_pairs'], self.report['domain'])
                self.assertAlmostEqual(result['log_ratio'], ratio, places=10)
                self.assertAlmostEqual(result['joint_map_log_weight'], best, places=10)
                self.assertEqual(result['states'], states)
                score = path_score(row['feature_pairs'], self.report['domain'], result['inferred_ticks'])
                self.assertAlmostEqual(score['total'], best, places=10)
                self.assertAlmostEqual(result['joint_map_probability'], math.exp(best - ratio), places=11)
                for pair, position in zip(row['feature_pairs'], result['positions']):
                    tick = position['latent_tick_probability']
                    parts = sum(position['inferred_label_probabilities']) + position['unavailable_tick_probability']
                    self.assertAlmostEqual(tick, parts, places=11)
                    self.assertTrue(0 <= tick <= 1 + 1e-10)
                    self.assertTrue(0 <= position['tempo_change_probability'] <= tick + 1e-10)
                    self.assertTrue(0 <= position['meter_change_probability'] <= tick + 1e-10)
                    if pair is None:
                        self.assertEqual(position['inferred_label_probabilities'], [0., 0., 0.])

    def test_exhaustive_paths_match_every_event_and_change_marginal(self):
        row = self.report['exhaustive_control']
        ratio, best, probabilities, count = enumerate_paths(row['feature_pairs'], row['domain'])
        actual = row['decoded']
        self.assertGreater(count, 100)
        self.assertAlmostEqual(actual['log_ratio'], ratio, places=11)
        self.assertAlmostEqual(actual['joint_map_log_weight'], best, places=11)
        actual_probabilities = [[p['latent_tick_probability'], *p['inferred_label_probabilities'],
                                p['unavailable_tick_probability'], p['tempo_change_probability'],
                                p['meter_change_probability']] for p in actual['positions']]
        np.testing.assert_allclose(actual_probabilities, probabilities, atol=1e-12, rtol=1e-10)

    def test_retained_failures_missing_semantics_and_prior_only_clock(self):
        self.assertEqual(self.rows['half']['decoded'], self.rows['same_features_erased_constant']['decoded'])
        for name in ['flat', 'unavailable']:
            self.assertAlmostEqual(self.rows[name]['decoded']['log_ratio'], 0., places=11)
        for a, b in zip(self.rows['flat']['decoded']['positions'], self.rows['unavailable']['decoded']['positions']):
            for field in ['latent_tick_probability', 'tempo_change_probability', 'meter_change_probability']:
                self.assertAlmostEqual(a[field], b[field], places=11)
        # Freeze failures rather than silently counting a supported inferred tick as a detection.
        half = self.rows['half']['decoded']['inferred_ticks']
        self.assertEqual([t['frame'] for t in half], [1, 7, 13])
        double = self.rows['double']['decoded']['inferred_ticks']
        self.assertEqual([t['period_frames'] for t in double], [3] * 6)
        self.assertEqual(next(t for t in double if t['frame'] == 4)['inferred_label'], 1)
        self.assertEqual(self.rows['double']['feature_pairs'][4], [0., 0.])
        for name, expected in [('flat_middle', 0), ('unavailable_middle', None)]:
            ticks = self.rows[name]['decoded']['inferred_ticks']
            self.assertEqual([t['period_frames'] for t in ticks], [3] * 6)
            self.assertEqual([t['inferred_label'] for t in ticks if 6 <= t['frame'] < 13], [expected] * 2)

    def test_posthoc_authored_paths_expose_prior_reversal_not_missing_feature_direction(self):
        # Truth is used only AFTER inference to diagnose these development failures.
        paths = {
            'half': ([1, 4, 7, 13], [3, 3, 6, 6], [1, 2, 3, 1], [2, 1, 1, 2]),
            'double': ([1, 7, 10, 13, 16], [6, 3, 3, 3, 3], [1, 2, 3, 1, 2], [2, 1, 1, 2, 1]),
        }
        for name, path in paths.items():
            row = self.rows[name]
            ticks = [dict(frame=t, period_frames=p, meter=3, beat_in_bar=j, inferred_label=l)
                     for t, p, j, l in zip(*path)]
            authored = path_score(row['feature_pairs'], self.report['domain'], ticks)
            selected = path_score(row['feature_pairs'], self.report['domain'], row['decoded']['inferred_ticks'])
            self.assertGreater(authored['feature'], selected['feature'])
            self.assertLess(authored['clock'], selected['clock'])
            self.assertLess(authored['total'], selected['total'])
        probe = self.report['resource_probe']['result']
        self.assertFalse(probe['completed'])
        self.assertFalse(probe['partial_inference_returned'])
        self.assertIn('state budget exceeded', probe['error'])


if __name__ == '__main__':
    unittest.main()
