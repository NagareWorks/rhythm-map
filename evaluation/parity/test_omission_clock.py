"""Independent probability-polynomial omission reconstruction and equivalence."""
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np
from test_rank_clock import reconstruct, independent_features, paired_table

ROOT = Path(__file__).resolve().parents[2]


def beta_log(success, failure):
    return math.lgamma(success+1)+math.lgamma(failure+1)-math.lgamma(success+failure+2)


def factor(values, accented):
    n = len(values)
    shape = (n+1, n+1, n+1 if accented else 1)
    alpha = np.zeros(shape)
    best = np.full(shape, -np.inf)
    alpha[0, 0, 0], best[0, 0, 0] = 1., 0.
    for t, (b, d) in enumerate(values):
        alpha[t+1] = alpha[t]
        alpha[t+1, 1:] += alpha[t, :-1]*math.exp(b)
        best[t+1] = best[t]
        best[t+1, 1:] = np.maximum(best[t+1, 1:], best[t, :-1]+b)
        if accented:
            alpha[t+1, 1:, 1:] += alpha[t, :-1, :-1]*math.exp(b+d)
            best[t+1, 1:, 1:] = np.maximum(best[t+1, 1:, 1:], best[t, :-1, :-1]+b+d)
    return alpha, best


def marginals(values, alpha, terminal, partition, accented):
    backward = terminal.copy()
    out = np.zeros((len(values), 3))
    for t in range(len(values)-1, -1, -1):
        b, d = values[t]
        out[t, 0] = np.sum(alpha[t]*backward)/partition
        out[t, 1] = np.sum(alpha[t, :-1]*backward[1:])*math.exp(b)/partition
        previous = backward.copy()
        previous[:-1] += backward[1:]*math.exp(b)
        if accented:
            out[t, 2] = np.sum(alpha[t, :-1, :-1]*backward[1:, 1:])*math.exp(b+d)/partition
            previous[:-1, :-1] += backward[1:, 1:]*math.exp(b+d)
        backward = previous
    return out


def component(scores, coeff, ticks, available, meter, phase):
    plain = [i for i, t in enumerate(ticks) if available[t] and (i+phase) % meter]
    bars = [i for i, t in enumerate(ticks) if available[t] and (i+phase) % meter == 0]
    pv, bv = scores[np.array(ticks)[plain]], scores[np.array(ticks)[bars]]
    fa, fm = factor(pv, False)
    ga, gm = factor(bv, True)
    o, s = len(plain), len(bars)
    n = o+s
    logw = np.full((o+1, s+1, s+1), -np.inf)
    for k in range(o+1):
        for z in range(s+1):
            for d in range(z+1):
                logw[k, z, d] = beta_log(k+z, n-k-z)+beta_log(d, z-d)-coeff[k+z-d, d]
    shift = np.max(logw)
    w = np.exp(logw-shift)
    mass = fa[-1, :, 0, None, None]*ga[-1][None]*w
    partition = mass.sum()
    count = np.zeros((n+1, (len(ticks)+1)//2+1))
    k, z, d = np.indices(w.shape)
    np.add.at(count, ((k+z).ravel(), d.ravel()), (mass/partition).ravel())
    rate_b = np.sum(mass*(k+z+1)/(n+2))/partition
    rate_d = np.sum(mass*(d+1)/(z+2))/partition
    terminal_f = np.sum(ga[-1][None]*w, axis=(1, 2))[:, None]
    terminal_g = np.sum(fa[-1, :, 0, None, None]*w, axis=0)
    occupancy = np.full((len(ticks), 3), np.nan)
    occupancy[plain] = marginals(pv, fa, terminal_f, partition, False)
    occupancy[bars] = marginals(bv, ga, terminal_g, partition, True)
    best = np.max(fm[-1, :, 0, None, None]+gm[-1][None]+logw)
    return math.log(partition)+shift, rate_b, rate_d, count, occupancy, best


def infer(scores, coeff, ticks, available):
    states, raw, logs, best, intact = [], [], [], [], []
    for meter in range(2, 8):
        for phase in range(meter):
            r = component(scores, coeff, ticks, available, meter, phase)
            prior = -math.log(6*meter)
            raw.append(r)
            states.append((meter, phase))
            logs.append(r[0]+prior)
            best.append(r[-1]+prior)
            ids = [t for t in ticks if available[t]]
            bars = [t for i, t in enumerate(ticks) if available[t] and (i+phase) % meter == 0]
            intact.append(scores[ids, 0].sum()+scores[bars, 1].sum()-coeff[len(ids)-len(bars), len(bars)]+prior)
    total = np.logaddexp.reduce(logs)
    p = np.exp(np.array(logs)-total)
    return dict(log_ratio=total, states=states, components=raw, weights=p,
                rate_b=p@np.array([r[1] for r in raw]), rate_d=p@np.array([r[2] for r in raw]),
                counts=sum(q*r[3] for q, r in zip(p, raw)),
                occupancy=sum(q*r[4] for q, r in zip(p, raw)),
                best=max(best), intact=np.logaddexp.reduce(intact))


def assignment_prior(labels, ticks, available):
    if any(t not in ticks for t, _ in labels):
        return None
    b = len(labels)
    n = sum(available[t] for t in ticks)
    d = sum(label == 2 for _, label in labels)
    logs = []
    for meter in range(2, 8):
        for phase in range(meter):
            positions = [(ticks.index(t), label) for t, label in labels]
            if any(label == 2 and (i+phase) % meter for i, label in positions):
                continue
            z = sum((i+phase) % meter == 0 for i, _ in positions)
            logs.append(-math.log(6*meter)+beta_log(b, n-b)+beta_log(d, z-d))
    return np.logaddexp.reduce(logs) if logs else None


class OmissionClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT/'evaluation/parity/omission-clock-v1.json').read_bytes())
        cls.rows = {r['case']: r for r in cls.report['cases']}

    def test_source_identities_and_constant_meter_diagnostic_scope(self):
        for field, path in (
            ('audit_source_sha256', 'crates/rhythm-map-eval/examples/omission_clock.rs'),
            ('omission_source_sha256', 'crates/rhythm-map-eval/examples/support/omission.rs'),
            ('rank_source_sha256', 'crates/rhythm-map-eval/examples/support/rank_frames.rs'),
            ('feature_source_sha256', 'crates/rhythm-map-eval/examples/support/shared_frames.rs'),
            ('fixture_source_sha256', 'crates/rhythm-map-eval/examples/support/rank_fixtures.rs'),
            ('rank_report_sha256', 'evaluation/parity/rank-clock-v1.json'),
        ):
            self.assertEqual(self.report[field], hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
        for key in ('production_output_changed', 'training_run', 'holdout_opened', 'real_music_evaluated',
                    'unrestricted_clock_search', 'meter_changes_searched', 'calibrated_confidence'):
            self.assertIs(self.report[key], False)
        for key in ('supplied_clock_templates', 'truth_assisted_clock_family', 'static_meter_marginalized',
                    'pulse_and_accent_omissions_marginalized'):
            self.assertIs(self.report[key], True)
        old = json.loads((ROOT/'evaluation/parity/rank-clock-v1.json').read_bytes())
        self.assertEqual(self.report['clock_templates'], old['clock_templates'])
        self.assertEqual(set(self.rows), {r['case'] for r in old['cases']})
        for row in old['cases']:
            for key in ('beat_f64_le_sha256', 'bar_f64_le_sha256', 'availability_u8_sha256', 'authored_clock'):
                self.assertEqual(self.rows[row['case']][key], row[key])

    def test_every_omission_count_marginal_map_and_equivalent_assignment(self):
        clocks = self.report['clock_templates']
        prior = np.array([c['duration_prior_log_weight'] for c in clocks])
        prior -= np.logaddexp.reduce(prior)
        for row in self.rows.values():
            beat, bar, available = reconstruct(row['case'])
            scores = independent_features(beat, bar, available, True)
            coeff = paired_table(scores, available)
            for name, array in (('beat', beat), ('bar', bar)):
                self.assertEqual(row[name+'_f64_le_sha256'], hashlib.sha256(array.astype('<f8').tobytes()).hexdigest())
            self.assertEqual(row['availability_u8_sha256'], hashlib.sha256(available.astype('u1').tobytes()).hexdigest())
            out = row['decoded']
            evidence, intact, maps, paths = [], [], [], []
            for i, (clock, result) in enumerate(zip(clocks, out['clocks'])):
                ticks = [t for t, _ in clock['given_ticks']]
                paths.append(ticks)
                actual = infer(scores, coeff, ticks, available)
                self.assertAlmostEqual(result['log_ratio'], actual['log_ratio'], places=8)
                self.assertAlmostEqual(result['mean_pulse_retention'], actual['rate_b'], places=9)
                self.assertAlmostEqual(result['mean_accent_retention'], actual['rate_d'], places=9)
                np.testing.assert_allclose(result['count_probabilities'], actual['counts'], atol=1e-8, rtol=0)
                obs = np.array([available[t] for t in ticks])
                self.assertEqual([p is not None for p in result['label_probabilities']], obs.tolist())
                reported = np.asarray([p for p in result['label_probabilities'] if p is not None]).reshape(-1, 3)
                np.testing.assert_allclose(reported, actual['occupancy'][obs], atol=1e-8, rtol=0)
                np.testing.assert_allclose(reported.sum(axis=1), 1., atol=1e-8, rtol=0)
                self.assertAlmostEqual(np.sum(result['count_probabilities']), 1., places=8)
                for j, state in enumerate(result['components']):
                    self.assertEqual((state['meter'], state['phase']), actual['states'][j])
                    self.assertAlmostEqual(state['log_ratio'], actual['components'][j][0], places=8)
                    self.assertAlmostEqual(state['probability'], actual['weights'][j], places=9)
                m = result['joint_map']
                self.assertAlmostEqual(m['log_weight'], actual['best'], places=8)
                labels = m['inferred_labels']
                self.assertEqual([l is not None for l in labels], obs.tolist())
                emitted = [(t, l) for t, l in zip(ticks, labels) if l is not None and l > 0]
                self.assertTrue(all(l in (1, 2) for _, l in emitted))
                self.assertTrue(all((j+m['phase']) % m['meter'] == 0 for j, l in enumerate(labels) if l == 2))
                b, d = len(emitted), sum(l == 2 for _, l in emitted)
                z = sum(l is not None and l > 0 and (j+m['phase']) % m['meter'] == 0 for j, l in enumerate(labels))
                feature = sum(scores[t, 0]+(scores[t, 1] if l == 2 else 0) for t, l in emitted)-coeff[b-d, d]
                omission = beta_log(b, sum(obs)-b)+beta_log(d, z-d)
                self.assertAlmostEqual(m['feature_log_ratio'], feature, places=8)
                self.assertAlmostEqual(m['omission_log_prior'], omission, places=8)
                self.assertAlmostEqual(m['log_weight'], feature+omission-math.log(6*m['meter']), places=8)
                evidence.append(actual['log_ratio']+prior[i])
                intact.append(actual['intact']+prior[i])
                maps.append(m['log_weight']+prior[i])
            total = np.logaddexp.reduce(evidence)
            self.assertAlmostEqual(out['clock_family_log_ratio'], total, places=8)
            np.testing.assert_allclose(out['clock_family_probabilities'], np.exp(np.array(evidence)-total), atol=1e-9, rtol=0)
            np.testing.assert_allclose(out['matched_no_omission_log_weights'], intact, atol=1e-8, rtol=0)
            np.testing.assert_allclose(out['matched_no_omission_probabilities'], np.exp(np.array(intact)-np.logaddexp.reduce(intact)), atol=1e-9, rtol=0)
            selected = next(i for i, c in enumerate(clocks) if c['clock'] == out['selected_joint_map_clock'])
            self.assertAlmostEqual(maps[selected], max(maps), places=8)
            eq = out['selected_assignment_equivalence']
            expected_labels = [[t, l] for t, l in zip(paths[selected], out['clocks'][selected]['joint_map']['inferred_labels']) if l is not None and l > 0]
            self.assertEqual(eq['inferred_emitted_labels'], expected_labels)
            class_priors = [assignment_prior(expected_labels, p, available) for p in paths]
            class_priors = [None if p is None else p+prior[i] for i, p in enumerate(class_priors)]
            self.assertEqual([p is None for p in eq['clock_log_priors']], [p is None for p in class_priors])
            np.testing.assert_allclose([p for p in eq['clock_log_priors'] if p is not None], [p for p in class_priors if p is not None], atol=1e-9, rtol=0)
            class_total = np.logaddexp.reduce([p for p in class_priors if p is not None])
            np.testing.assert_allclose(eq['conditional_clock_probabilities'], [0 if p is None else math.exp(p-class_total) for p in class_priors], atol=1e-9, rtol=0)
            feature = out['clocks'][selected]['joint_map']['feature_log_ratio']
            self.assertAlmostEqual(eq['shared_feature_log_ratio'], feature, places=8)
            self.assertAlmostEqual(eq['assignment_probability_in_full_model'], math.exp(feature+class_total-total), places=8)
            self.assertGreaterEqual(eq['assignment_probability_in_full_model'], 0.)
            self.assertLessEqual(eq['assignment_probability_in_full_model'], 1.+1e-9)
            self.assertIs(eq['multiple_compatible_latent_clocks'], sum(p is not None for p in class_priors) > 1)
            self.assertIs(eq['labels_are_detected_events'], False)
            self.assertIs(eq['not_whole_posterior_ambiguity'], True)

    def test_equal_heads_cannot_be_resolved_by_case_labels_or_posterior_preference(self):
        a, b = self.rows['half_speed_intact'], self.rows['constant_erased_beats_and_bars']
        self.assertEqual(a['decoded'], b['decoded'])
        self.assertNotEqual(a['authored_clock'], b['authored_clock'])
        e = a['decoded']['selected_assignment_equivalence']
        self.assertEqual(sum(p is not None for p in e['clock_log_priors']), 3)
        self.assertTrue(e['multiple_compatible_latent_clocks'])
        for name, target in (('constant_intact', 0), ('constant_weak_alternating', 0),
                             ('constant_all_weak', 0), ('half_speed_intact', 1), ('double_speed_intact', 2),
                             ('double_speed_weak_alternating', 2), ('non_octave_intact', 3)):
            for key in ('clock_family_probabilities', 'matched_no_omission_probabilities'):
                self.assertEqual(np.argmax(self.rows[name]['decoded'][key]), target)
        for name in ('constant_intact', 'half_speed_intact', 'double_speed_weak_alternating'):
            self.assertEqual(self.rows[name]['decoded'], self.rows[name+'_tiny_contrast']['decoded'])
            self.assertEqual(np.argmax(self.rows[name]['decoded']['clock_family_probabilities']),
                             np.argmax(self.rows[name+'_middle_offset']['decoded']['clock_family_probabilities']))

    def test_inferred_flat_ticks_never_become_observed_events_and_no_data_is_not_omission(self):
        out = self.rows['flat_middle']['decoded']
        labels = out['selected_assignment_equivalence']['inferred_emitted_labels']
        self.assertEqual(sum(480 <= t < 672 for t, _ in labels), 8)
        self.assertIs(out['selected_assignment_equivalence']['labels_are_detected_events'], False)
        for name in ('flat', 'all_unavailable'):
            d = self.rows[name]['decoded']
            self.assertAlmostEqual(d['clock_family_log_ratio'], 0., places=8)
            self.assertEqual(d['selected_assignment_equivalence']['inferred_emitted_labels'], [])
            self.assertGreater(d['clock_family_probabilities'][0], .99)
        unknown = self.rows['all_unavailable']['decoded']
        for clock in unknown['clocks']:
            self.assertTrue(all(p is None for p in clock['label_probabilities']))
            self.assertAlmostEqual(clock['mean_pulse_retention'], .5, places=9)
            self.assertAlmostEqual(clock['mean_accent_retention'], .5, places=9)
        noise = self.rows['fixed_seed_noise']['decoded']
        self.assertLess(noise['clock_family_log_ratio'], 0.)
        self.assertEqual(noise['selected_assignment_equivalence']['inferred_emitted_labels'], [])


if __name__ == '__main__':
    unittest.main()
