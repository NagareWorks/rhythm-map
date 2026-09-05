"""Independent local medians, pairwise midranks and shared clock reconstruction."""
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np
from test_shared_clock import inputs, independent_inference

ROOT = Path(__file__).resolve().parents[2]
BASES = ("constant_intact", "half_speed_intact", "double_speed_weak_alternating")


def reconstruct(name):
    base = name.removesuffix("_tiny_contrast").removesuffix("_middle_offset")
    beat, bar, available = inputs(base)
    if name.endswith("_tiny_contrast"):
        beat, bar = -8 + (beat + 8)/4096, -8 + (bar + 8)/4096
    if name.endswith("_middle_offset"):
        beat[384:768] -= 16
        bar[384:768] -= 16
    return beat, bar, available


def independent_features(beat, bar, available, background):
    ids = np.flatnonzero(available)
    scores = np.full((len(beat), 2), np.nan)
    source = np.column_stack((beat, bar))
    runs = np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1)
    for run in runs:
        smooth = np.empty((len(run), 2))
        for offset, t in enumerate(run):
            near = run[max(0, offset-1):offset+2]
            weights = np.where(near == t, 2., 1.)
            smooth[offset] = np.average(source[near], axis=0, weights=weights)
        for offset, t in enumerate(run):
            scores[t] = smooth[offset]
            if background:
                scores[t] -= np.median(smooth[max(0, offset-4):offset+5], axis=0)
    for h in range(2):
        values = scores[ids, h]
        # Pairwise comparisons, independent of the Rust sorted tie-group scan.
        lower = np.sum(values[:, None] > values[None, :], axis=1)
        equal = np.sum(values[:, None] == values[None, :], axis=1)
        mid = lower + equal/2
        scores[ids, h] = np.log(mid) - np.log(len(ids)-mid)
    if len(ids):
        scores[ids] -= scores[ids].max(axis=0)
    return scores


def paired_table(features, available):
    n = sum(available)
    coeff = np.full((min(n, 64)+1, min(n, 32)+1), -np.inf)
    coeff[0, 0] = 0
    for b, d in features[available]:
        old = coeff.copy()
        coeff[1:] = np.logaddexp(coeff[1:], old[:-1]+b)
        coeff[:, 1:] = np.logaddexp(coeff[:, 1:], old[:, :-1]+b+d)
    for a in range(coeff.shape[0]):
        for d in range(min(coeff.shape[1]-1, n-a)+1):
            coeff[a, d] -= (math.lgamma(n+1)-math.lgamma(a+1)
                            - math.lgamma(d+1)-math.lgamma(n-a-d+1))
    return coeff


class RankClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT/'evaluation/parity/rank-clock-v1.json').read_bytes())
        cls.rows = {r['case']: r for r in cls.report['cases']}

    def test_frozen_sources_family_inputs_and_limited_scope(self):
        for field, path in (
            ('audit_source_sha256', 'crates/rhythm-map-eval/examples/rank_clock.rs'),
            ('fixture_source_sha256', 'crates/rhythm-map-eval/examples/support/rank_fixtures.rs'),
            ('rank_source_sha256', 'crates/rhythm-map-eval/examples/support/rank_frames.rs'),
            ('feature_source_sha256', 'crates/rhythm-map-eval/examples/support/shared_frames.rs'),
            ('meter_source_sha256', 'crates/rhythm-map-eval/examples/support/frame_meter.rs'),
            ('shared_report_sha256', 'evaluation/parity/shared-clock-v1.json'),
        ):
            self.assertEqual(self.report[field], hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
        for key in ('production_output_changed', 'training_run', 'holdout_opened', 'real_music_evaluated',
                    'unrestricted_clock_search', 'dropout_states_marginalized', 'calibrated_confidence'):
            self.assertIs(self.report[key], False)
        for key in ('supplied_clock_templates', 'truth_assisted_clock_family', 'meter_paths_searched'):
            self.assertIs(self.report[key], True)
        self.assertEqual(self.report['background_window_frames'], 9)
        self.assertEqual(self.report['minimum_period_frames'], 10)
        old = json.loads((ROOT/'evaluation/parity/shared-clock-v1.json').read_bytes())
        self.assertEqual(len(self.rows), 20)
        for a, b in zip(self.report['clock_templates'], old['cases'][0]['raw']['clocks']):
            self.assertEqual(a, {k: b[k] for k in a})
        for r in old['cases']:
            for key in ('beat_f64_le_sha256', 'bar_f64_le_sha256', 'authored_clock'):
                self.assertEqual(self.rows[r['case']][key], r[key])

    def test_every_input_count_normalizer_meter_and_clock_output(self):
        templates = self.report['clock_templates']
        prior = np.array([t['duration_prior_log_weight'] for t in templates])
        prior -= np.logaddexp.reduce(prior)
        for row in self.rows.values():
            beat, bar, available = reconstruct(row['case'])
            for name, array in (('beat', beat), ('bar', bar)):
                self.assertEqual(row[name+'_f64_le_sha256'], hashlib.sha256(array.astype('<f8').tobytes()).hexdigest())
            self.assertEqual(row['availability_u8_sha256'], hashlib.sha256(available.astype('u1').tobytes()).hexdigest())
            for mode in ('raw_rank', 'ranked'):
                out = row[mode]
                scores = independent_features(beat, bar, available, mode == 'ranked')
                coeff = paired_table(scores, available)
                self.assertEqual(out['available_frames'], sum(available))
                weights = []
                for i, (result, template) in enumerate(zip(out['clocks'], templates)):
                    path = template['given_ticks']
                    ids = [t for t, _ in path if available[t]]
                    n = len(ids)
                    self.assertEqual(result['clock'], template['clock'])
                    self.assertEqual(result['visible_beats'], n)
                    self.assertEqual(result['unobserved_ticks'], len(path)-n)
                    norms = [coeff[n-d, d] for d in range(min(n, (len(path)+1)//2)+1)]
                    np.testing.assert_allclose(result['paired_log_normalizers'], norms, atol=1e-8, rtol=0)
                    marks = [float(scores[t, 1]) if available[t] else None for t, _ in path]
                    self.assertEqual([m is None for m in result['bar_mark_scores']], [m is None for m in marks])
                    np.testing.assert_allclose([m for m in result['bar_mark_scores'] if m is not None],
                                               [m for m in marks if m is not None], atol=1e-12, rtol=0)
                    z, rate, _, downbeats, counts = independent_inference(marks, norms)
                    beat_sum = scores[ids, 0].sum()
                    self.assertAlmostEqual(result['beat_score_sum'], beat_sum, places=9)
                    self.assertAlmostEqual(result['meter_log_ratio'], z, places=8)
                    self.assertAlmostEqual(result['mean_meter_change_probability'], rate, places=9)
                    np.testing.assert_allclose(result['meter_count_probabilities'], counts, atol=1e-9, rtol=0)
                    np.testing.assert_allclose(result['downbeat_probabilities'], downbeats, atol=1e-9, rtol=0)
                    self.assertAlmostEqual(result['joint_log_ratio'], z+beat_sum, places=8)
                    weights.append(z+beat_sum+prior[i])
                    self.assertAlmostEqual(result['family_log_weight'], weights[-1], places=8)
                total = np.logaddexp.reduce(weights)
                self.assertAlmostEqual(out['clock_family_log_ratio'], total, places=8)
                np.testing.assert_allclose(out['clock_family_probabilities'], np.exp(np.array(weights)-total), atol=1e-9, rtol=0)

    def test_joint_change_gate_and_retained_raw_rank_offset_failures(self):
        expected = dict(constant_intact=0, constant_weak_alternating=0, half_speed_intact=1,
                        double_speed_intact=2, double_speed_weak_alternating=2, non_octave_intact=3,
                        constant_all_weak=0, constant_erased_beats=0, flat_middle=0, unavailable_gap=0)
        for name, target in expected.items():
            self.assertEqual(np.argmax(self.rows[name]['ranked']['clock_family_probabilities']), target)
        for base in BASES:
            target = expected[base]
            self.assertEqual(np.argmax(self.rows[base+'_middle_offset']['ranked']['clock_family_probabilities']), target)
        for base in ('constant_intact', 'double_speed_weak_alternating'):
            self.assertEqual(np.argmax(self.rows[base+'_middle_offset']['raw_rank']['clock_family_probabilities']), 1)

    def test_rank_certainty_cannot_identify_missing_pulses_or_absolute_signal_strength(self):
        half, erased = self.rows['half_speed_intact'], self.rows['constant_erased_beats_and_bars']
        for key in ('beat_f64_le_sha256', 'bar_f64_le_sha256', 'availability_u8_sha256', 'raw_rank', 'ranked'):
            self.assertEqual(half[key], erased[key])
        self.assertNotEqual(half['authored_clock'], erased['authored_clock'])
        self.assertGreater(half['ranked']['clock_family_probabilities'][1], 0.99)
        for base in BASES:
            for mode in ('raw_rank', 'ranked'):
                self.assertEqual(self.rows[base][mode], self.rows[base+'_tiny_contrast'][mode])
        for name in ('flat', 'all_unavailable'):
            for mode in ('raw_rank', 'ranked'):
                self.assertAlmostEqual(self.rows[name][mode]['clock_family_log_ratio'], 0., places=8)
                self.assertGreater(self.rows[name][mode]['clock_family_probabilities'][0], 0.99)
        for mode in ('raw_rank', 'ranked'):
            self.assertLess(self.rows['fixed_seed_noise'][mode]['clock_family_log_ratio'], 0.)
        witness = self.report['ambiguous_observation_witness']
        self.assertIs(witness['diagnostic_only'], True)
        self.assertIs(witness['latent_clock_identifiable_from_these_heads'], False)


if __name__ == '__main__':
    unittest.main()
