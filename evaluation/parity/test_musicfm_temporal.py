"""Model-free geometry/algebra tests; separate from real authored PCM inference."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import musicfm_temporal as temporal
import musicfm_temporal_probe as probe


class MusicFMTemporalControls(unittest.TestCase):
    def test_protocol_identity_and_no_product_or_music_access(self):
        plan, _, _ = probe.protocol()
        self.assertEqual(plan['authored']['cases'], list(temporal.CASES))
        self.assertEqual(plan['authored']['paired_denominator'], 6)
        for key in ('music_access', 'holdout_access', 'training', 'production_change',
                    'new_user_parameters', 'commercial_distribution_approved'):
            self.assertFalse(plan[key])

    def test_changed_protocol_stops_before_loading(self):
        with patch.object(Path, 'read_bytes', return_value=b'{}'), patch.object(probe.fidelity, 'protocol') as base:
            with self.assertRaisesRegex(ValueError, 'temporal protocol changed'):
                probe.protocol()
            base.assert_not_called()

    def test_changed_helper_stops(self):
        raw = (probe.HERE / 'musicfm-temporal-lock-v1.json').read_bytes()
        with patch.object(Path, 'read_bytes', side_effect=[raw, b'changed']):
            with self.assertRaisesRegex(ValueError, 'temporal helper changed'):
                probe.protocol()

    def test_geometry_formula_and_boundaries(self):
        for samples in (1025, 1199, 1200, 1919, 1920, 48000, 192000, 192001, 288240, temporal.MAX_SAMPLES):
            self.assertEqual(temporal.tokens(samples), (samples // 240 + 3) // 4)
        for samples in (0, 1024, True, 192000.0, temporal.MAX_SAMPLES + 1):
            with self.assertRaises(ValueError):
                temporal.chunks(samples)

    def test_fixed_complete_ownership_and_unpadded_tail(self):
        for samples in (1025, 1200, 48000, 191999, 192000, 192001, 288000, 288001, 288240, 480001):
            plan = temporal.chunks(samples)
            self.assertEqual(plan[-1]['sample_stop'], samples)
            self.assertEqual(plan[0]['sample_start'], 0)
            self.assertEqual(plan[0]['own_start'], 0)
            self.assertEqual(plan[-1]['own_stop'], temporal.tokens(samples))
            for a, b in zip(plan, plan[1:]):
                self.assertEqual(b['sample_start'] - a['sample_start'], 96000)
                self.assertEqual(a['own_stop'], b['own_start'])
                self.assertEqual(b['own_start'] * 960, b['sample_start'] + 48000)
                self.assertGreater(b['sample_stop'] - b['sample_start'], 96000)
            self.assertTrue(all(r['sample_start'] % 960 == 0 for r in plan))

    def test_authored_owns_both_seams_exactly_once(self):
        plan = temporal.chunks(288240)
        self.assertEqual([r['token_count'] for r in plan], [200, 200, 101])
        values = [np.full((r['token_count'], 1024), r['id'], np.float32) for r in plan]
        h, owner = temporal.stitch(values, plan, 288240)
        self.assertEqual(owner.tolist(), [0] * 150 + [1] * 100 + [2] * 51)
        np.testing.assert_array_equal(h[:, 0], owner)
        self.assertEqual(owner[150], 1)
        self.assertEqual(owner[250], 2)

    def test_stitch_rejects_missing_altered_or_nonfinite_context(self):
        plan = temporal.chunks(288240)
        arrays = [np.zeros((r['token_count'], 1024), np.float32) for r in plan]
        for invalid in (arrays[:-1], [a.astype(np.float64) for a in arrays],
                        [np.full(a.shape, np.nan, np.float32) for a in arrays]):
            with self.assertRaises(ValueError):
                temporal.stitch(invalid, plan, 288240)
        changed = [dict(r) for r in plan]
        changed[1]['own_start'] += 1
        with self.assertRaises(ValueError):
            temporal.stitch(arrays, changed, 288240)

    def test_native_mapping_retains_odd_start_offset_and_boundary(self):
        for start in (100, 101):
            result = temporal.window_from_50hz(start, Fraction(start - 25, 1), Fraction(25),
                                               Fraction(25, 2), start + 100)
            self.assertEqual(result['token_stop'] - result['token_start'], 100)
            self.assertEqual(result['first_target_offset_seconds'], '0' if start % 2 == 0 else '1/50')
            self.assertEqual(result['boundary'] + result['token_start'], Fraction(start + 100, 2))
            self.assertEqual(result['pre'], Fraction(25, 2))

    def test_phase_and_inverse_are_exact_across_boundary(self):
        for t in (Fraction(-1, 2), Fraction(0), Fraction(50), Fraction(201, 4)):
            p = temporal.phase_at(t, Fraction(-3, 2), Fraction(25, 2), Fraction(20), Fraction(50))
            self.assertEqual(temporal.time_at(p, Fraction(-3, 2), Fraction(25, 2), Fraction(20), Fraction(50)), t)

    def test_absolute_phase_cancels_and_half_names_are_duplicate_conditions(self):
        a = temporal.queries(0, Fraction(25, 2), 20, 50)
        b = temporal.queries(Fraction(17, 3), Fraction(25, 2), 20, 50)
        self.assertEqual(a, b)
        for shape in ('constant', 'step'):
            self.assertEqual(a[1][shape + '/half_zero'], a[1][shape + '/half_one'])

    def test_common_support_never_extrapolates(self):
        targets, rows, _ = temporal.queries(0, Fraction(25, 2), 20, 50)
        self.assertGreater(len(targets), 0)
        self.assertLess(len(targets), 100)
        for t in targets:
            self.assertTrue(all(0 <= q <= 99 for values in rows.values() for q in values[t]))

    def test_exact_float64_interpolation_including_cancellation(self):
        h = np.zeros((100, 1024), np.float32)
        h[0, 0], h[1, 0] = 1, -3
        v = temporal.sample(h, Fraction(1, 4))
        self.assertEqual(v.dtype, np.float64)
        self.assertEqual(v[0], 0)
        with self.assertRaisesRegex(ValueError, 'zero-norm'):
            temporal.unit(v)

    def test_authored_cue_retains_shape_and_density_despite_pulse_omissions(self):
        owner = np.zeros(100, np.int32)
        for post in (Fraction(25, 4), Fraction(20)):
            change = temporal.evaluate(temporal.authored_hidden(post), owner, 0, Fraction(25, 2), post, 50)
            for kind in ('clean', 'omitted', 'weak'):
                constant = temporal.evaluate(temporal.authored_hidden(Fraction(25, 2), kind), owner,
                                             0, Fraction(25, 2), post, 50)
                result = temporal.verdict(constant, change)
                self.assertTrue(result['shape_supported'], (post, kind, result))
                self.assertTrue(result['density_resolved'], (post, kind, result))

    def test_flat_features_tie_and_zero_features_reject_all(self):
        for kind, expected in (('flat', 'eligible'), ('zero', 'zero_norm_support')):
            result = temporal.evaluate(temporal.authored_hidden(20, kind), np.zeros(100, np.int32),
                                       0, Fraction(25, 2), 20, 50)
            self.assertEqual(result['status'], expected)
            if kind == 'flat':
                self.assertTrue(all(v['score'] == 0 for v in result['clocks'].values()))
            else:
                self.assertIsNone(result['clocks'])

    def test_density_counterexample_is_not_a_native_success(self):
        for multiplier in (0.5, 2):
            result = temporal.evaluate(temporal.authored_hidden(Fraction(25, 2), multiplier=multiplier),
                                       np.zeros(100, np.int32), 0, Fraction(25, 2), 20, 50)
            native = result['clocks']['constant/native']['score']
            rival = result['clocks']['constant/half_zero' if multiplier == 0.5 else 'constant/double']['score']
            self.assertEqual(temporal.compare(rival, native), 'left')

    def test_owner_crossings_retained_without_changing_scores(self):
        h = temporal.authored_hidden(20)
        owner = np.zeros(100, np.int32)
        a = temporal.evaluate(h, owner, 0, Fraction(25, 2), 20, 50)
        owner[50:] = 1
        b = temporal.evaluate(h, owner, 0, Fraction(25, 2), 20, 50)
        for name in temporal.NAMES:
            self.assertEqual(a['clocks'][name]['score'], b['clocks'][name]['score'])
            self.assertGreater(b['clocks'][name]['owner_crossing_targets'], 0)

    def test_no_orphan_win_and_no_support_is_unavailable(self):
        h = temporal.authored_hidden(20)
        row = temporal.evaluate(h, np.zeros(100, np.int32), 0, 1000, 1000, 50)
        self.assertEqual(row['status'], 'no_common_support')
        verdict = temporal.verdict(row, dict(status='eligible', clocks={}))
        self.assertFalse(verdict['shape_supported'])
        self.assertFalse(verdict['density_resolved'])

    def test_invalid_window_owner_and_nonfinite_input_refused(self):
        h = temporal.authored_hidden(20)
        for owner in (np.zeros(100), np.full(100, -1, np.int32), np.arange(100, dtype=np.int32)[::-1]):
            with self.assertRaises(ValueError):
                temporal.evaluate(h, owner, 0, 12, 20, 50)
        h[99, 0] = np.inf
        with self.assertRaises(ValueError):
            temporal.evaluate(h, np.zeros(100, np.int32), 0, 12, 20, 50)

    def test_inputs_immutable(self):
        h, owner = temporal.authored_hidden(20), np.zeros(100, np.int32)
        old_h, old_owner = h.copy(), owner.copy()
        temporal.evaluate(h, owner, 0, Fraction(25, 2), 20, 50)
        np.testing.assert_array_equal(h, old_h)
        np.testing.assert_array_equal(owner, old_owner)

    def test_pcm_population_complete_deterministic_and_bounded(self):
        for name in temporal.CASES:
            a, b = temporal.authored_pcm(name), temporal.authored_pcm(name)
            self.assertEqual(a.dtype, np.float32)
            self.assertEqual(a.shape, (288240,))
            self.assertTrue(np.isfinite(a).all())
            self.assertLessEqual(np.max(np.abs(a)), 0.38)
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.any(temporal.authored_pcm('silence')))
        with self.assertRaises(ValueError):
            temporal.authored_pcm('unregistered')

    def test_complete_scoring_requires_every_case(self):
        with self.assertRaisesRegex(ValueError, 'incomplete authored population'):
            probe.score_cases({})

    def test_complete_authored_ledger_and_no_dropped_weak_condition(self):
        cases = {}
        for name in temporal.CASES:
            hidden = np.zeros((301, 1024), np.float32)
            post = Fraction(25, 4) if name == 'step-fast' else Fraction(20) if name == 'step-slow' else Fraction(25, 2)
            kind = 'omitted' if name.endswith('omitted') else 'weak' if name.endswith('weak') else 'flat' if name == 'silence' else 'clean'
            hidden[100:200] = temporal.authored_hidden(post, kind)
            owner = np.zeros(301, np.int32)
            owner[150:] = 1
            cases[name] = hidden, owner
        result = probe.score_cases(cases)
        self.assertEqual(len(result['pairs']), 6)
        self.assertEqual(result['shape_supported'], 6)
        self.assertEqual(result['density_resolved'], 6)
        self.assertTrue(result['silence_abstains'])
        self.assertTrue(result['rule_survives_authored'])
        cases['constant-weak'][0][100:200] = 0
        rejected = probe.score_cases(cases)
        self.assertEqual(len(rejected['pairs']), 6)
        self.assertEqual(rejected['paired_denominator'], 6)
        self.assertEqual(sum(p['status'] == 'pair_rejected' for p in rejected['pairs']), 2)
        self.assertFalse(rejected['rule_survives_authored'])

    def test_same_physical_queries_as_previous_50hz_formula(self):
        import prehead_recurrence as previous
        old = previous.query_plan(0, 25, 40, 100)
        _, native, _ = temporal.queries(0, Fraction(25, 2), 20, 50)
        self.assertEqual(tuple(old['queries']), temporal.NAMES)
        for name in temporal.NAMES:
            for t in range(100):
                self.assertEqual(native[name][t], tuple(q / 2 for q in old['queries'][name][2 * t]))

    def test_numerical_ties_are_not_confidence(self):
        self.assertEqual(temporal.compare(0, 0.5e-6), 'unresolved')
        self.assertEqual(temporal.compare(0, 2e-6), 'right')
        with self.assertRaises(ValueError):
            temporal.compare(float('nan'), 0)

    def test_retained_report_matches_frozen_code_and_execution_identity(self):
        raw = (probe.HERE / 'musicfm-temporal-v1.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), 'b0badf814d197c30e3f4e83df221b43570795e396586351904080a7bdd5683a4')
        report = json.loads(raw)
        self.assertTrue(report['complete'])
        self.assertTrue(report['extraction_fidelity_pass'])
        self.assertEqual(report['protocol_sha256'], probe.LOCK_SHA)
        self.assertEqual(report['temporal_source_sha256'], probe.TEMPORAL_SHA)
        self.assertEqual(report['probe_source_sha256'], hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest())
        self.assertEqual(report['initial_model_state_sha256'], report['final_model_state_sha256'])
        self.assertEqual([c['id'] for c in report['cases']], list(temporal.CASES))
        self.assertEqual(sum(len(c['contexts']) for c in report['cases']), 18)
        for case in report['cases']:
            self.assertEqual(case['samples'], 288240)
            self.assertEqual(case['tokens'], 301)
            self.assertTrue(case['state_and_input_unchanged'])
            self.assertTrue(case['private_archive_roundtrip_exact'])
            self.assertTrue(all(c['reference_hook_repeat_exact'] and c['all_blocks_and_projection'] for c in case['contexts']))

    def test_retained_six_pair_successes_do_not_hide_silence_failure(self):
        report = json.loads((probe.HERE / 'musicfm-temporal-v1.json').read_bytes())
        result = report['discrimination']
        self.assertEqual((result['shape_supported'], result['density_resolved'], result['paired_denominator']), (6, 6, 6))
        self.assertEqual(len(result['pairs']), 6)
        for pair in result['pairs']:
            speed = pair['change'].split('-')[1]
            verdict = temporal.verdict(result['scores'][pair['constant']][speed], result['scores'][pair['change']][speed])
            self.assertEqual(verdict, {k: pair[k] for k in verdict})
        for silence in result['scores']['silence'].values():
            values = [c['score'] for c in silence['clocks'].values()]
            self.assertGreater(max(values) - min(values), temporal.TIE)
        self.assertFalse(result['silence_abstains'])
        self.assertFalse(result['rule_survives_authored'])
        self.assertEqual(report['decision'], 'close_fixed_rule_before_music')

    def test_retained_context_difference_and_no_music_claim(self):
        report = json.loads((probe.HERE / 'musicfm-temporal-v1.json').read_bytes())
        for key in ('music_access', 'holdout_access', 'training', 'production_change'):
            self.assertFalse(report[key])
        for case in report['cases']:
            self.assertEqual(len(case['overlaps']), 2)
            self.assertTrue(all(not row['exact'] and row['max_abs_delta'] > 0 for row in case['overlaps']))
        for rows in report['discrimination']['scores'].values():
            for row in rows.values():
                self.assertEqual(set(row['clocks']), set(temporal.NAMES))
                self.assertEqual((row['target_frames'], row['pre_boundary_frames'], row['post_boundary_frames']), (75, 25, 50))
                self.assertTrue(any(c['owner_crossing_targets'] > 0 for c in row['clocks'].values()))


if __name__ == '__main__':
    unittest.main()
