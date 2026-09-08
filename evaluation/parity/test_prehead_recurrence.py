import copy
from fractions import Fraction
import json
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

import prehead_recurrence as audit
import prehead_recurrence_audit as replay
from test_annotation_exposure import data


OWNER = np.zeros(200, np.int32)


def score(hidden, post=40, anchor=0, boundary=100, owner=OWNER):
    return audit.evaluate(hidden, owner, Fraction(anchor), Fraction(25),
                          Fraction(post), Fraction(boundary))


class PreheadRecurrenceControls(unittest.TestCase):
    def test_registered_population_still_accounts_for_all_forty_ids(self):
        lock, _, _, _, cases, plan = replay.verified_plan()
        self.assertEqual(len(cases), 40)
        self.assertEqual(len(plan), 7)
        self.assertEqual(sum(c['data']['untyped'] for c in cases), 26)
        self.assertEqual(sum(c['data']['untyped'] and c['cohort'] == 'rubato' for c in cases), 25)
        self.assertEqual([c['identity']['id'] for c in cases if c['data']['untyped'] and
                          c['cohort'] == 'artbeat'], ['artbeat-18-piano-rubato'])
        self.assertEqual(len({c['identity']['id'] for c in cases}), 40)
        self.assertFalse(lock['real_features_opened_for_discrimination'])
        self.assertTrue(all(p['window_frames'] == 200 and p['cohort'] == 'artbeat' for p in plan))

    def test_changed_helper_stops_before_annotation_or_private_access(self):
        with patch.object(Path, 'read_bytes', return_value=b'changed'), \
             patch.object(replay.oracle.replay, 'verified_plan') as reader:
            with self.assertRaisesRegex(ValueError, 'helper changed'):
                replay.verified_plan()
            reader.assert_not_called()

    def test_geometry_dependency_change_is_rejected_and_not_only_the_top_level(self):
        real_sha = replay.sha
        def modified(path):
            return '0' * 64 if Path(path).name == 'prefix_clock_drift_audit.py' else real_sha(path)
        with patch.object(replay, 'sha', side_effect=modified), \
             patch.object(replay.oracle.replay, 'verified_plan') as reader:
            with self.assertRaisesRegex(ValueError, 'oracle dependency changed'):
                replay.verified_plan()
            reader.assert_not_called()

    def test_pair_adapter_reuses_geometry_without_turning_flat_features_into_truth(self):
        hidden = np.zeros((1501, 512), np.float32)
        hidden[:, 0] = 1
        pair = dict(constant_start_frame=401, change_start_frame=650, window_frames=200,
                    pre_segment_index=0, change_index=0)
        row = replay.evaluate_pair(hidden, np.zeros(1501, np.int32), data(), pair)
        self.assertTrue(row['geometry_compatible'])
        self.assertFalse(row['verdict']['shape_supported'])
        self.assertTrue(all(len(v) == 4 for v in row['windows'].values()))
        hidden[650:850] = 0
        rejected = replay.evaluate_pair(hidden, np.zeros(1501, np.int32), data(), pair)
        self.assertEqual(rejected['verdict']['status'], 'pair_rejected')
        self.assertTrue(all(r['clocks'] is None for rows in rejected['windows'].values() for r in rows))

    def test_capture_validation_requires_exact_pcm_heads_and_complete_recording(self):
        pcm = np.zeros(441, np.float32)
        trace = dict(suite_id='artbeat-v1', suite_sha256=replay.oracle.dense.SUITES['artbeat'][1],
            case_id='authored', audio_sha256='audio', observation_contract='contract',
            prefix_seconds=60, sample_rate=22050, decoded_sample_count=441, mono_samples=pcm.tolist(),
            mel_shape=[1, 2, 128], observations={'authored': True, 'source':
                dict(backend='beat-this-rten', frame_rate_hz=50., model='beat-this-full-rten',
                     version=replay.oracle.dense.MODEL)},
            beat_logits=[0., 1.], downbeat_logits=[0., -1.])
        case = dict(id='authored', audio_sha256='audio', sample_rate=22050, sample_count=441,
                    pcm_sha256=replay.capture.array_hash(pcm))
        old = dict(observation_contract='contract', observations={'authored': True, 'source':
                   dict(backend='beat-this-rten', frame_rate_hz=50., model='beat_this.onnx', version=None)},
                   beat_logits=[0., 1.], downbeat_logits=[0., -1.])
        with patch('compare_reference.validate_trace'):
            replay.validate_trace(trace, case, {'frame_count': 2}, old, {})
            for key, value in (('decoded_sample_count', 440), ('beat_logits', [0., 2.]),
                               ('mono_samples', [1.] * 441), ('mel_shape', [1, 1, 128]),
                               ('observations', {'authored': True, 'source': {'model': 'other'}})):
                with self.assertRaises(ValueError):
                    replay.validate_trace(dict(trace, **{key: value}), case, {'frame_count': 2}, old, {})

    def test_complete_clock_population_and_shared_interpolation_support(self):
        plan = audit.query_plan(0, 25, 40, 100)
        self.assertEqual(set(plan['queries']), set(audit.NAMES))
        expected = [t for t in range(200) if all(0 <= q <= 199
                    for rows in plan['queries'].values() for q in rows[t])]
        self.assertEqual(plan['targets'], expected)
        self.assertTrue(expected)
        self.assertTrue(all(t >= 50 for t in expected))

    def test_exact_lags_cancel_anchor_and_half_phase_without_selecting_one(self):
        plan = audit.query_plan(Fraction(-7, 3), 25, Fraction(25, 2), Fraction(301, 3))
        for anchor in (0, Fraction(1, 7), 99, -81):
            self.assertEqual(plan, audit.query_plan(anchor, 25, Fraction(25, 2), Fraction(301, 3)))
        for shape in ('constant', 'step'):
            self.assertEqual(plan['queries'][shape + '/half_zero'], plan['queries'][shape + '/half_one'])

    def test_cross_boundary_queries_keep_accumulated_phase(self):
        p = audit.query_plan(7, 25, 50, 103)
        self.assertEqual(p['queries']['step/native'][113], (Fraction(83), Fraction(191, 2)))
        self.assertNotEqual(p['queries']['step/native'][113][0], 113 - 50)

    def test_linear_interpolation_precedes_normalization(self):
        values = np.zeros((200, 512), np.float32)
        values[3, 0], values[4, 1] = 2, 4
        sampled = audit.sample(values, Fraction(13, 4))
        self.assertEqual(sampled[0], 1.5)
        self.assertEqual(sampled[1], 1.)
        np.testing.assert_allclose(audit.unit(sampled)[:2], np.array([1.5, 1]) / np.sqrt(3.25))

    def test_cancelling_endpoints_do_not_create_direction_from_float32_rounding(self):
        values = np.zeros((200, 512), np.float32)
        values[3, 0], values[4, 0] = 1., -2.
        sampled = audit.sample(values, Fraction(10, 3))
        self.assertLess(float(np.linalg.norm(sampled)), audit.NORM_EPS)
        with self.assertRaisesRegex(ValueError, 'zero-norm'):
            audit.unit(sampled)

    def test_clean_omitted_and_weak_pulse_controls_with_retained_clock_cue(self):
        for post in (Fraction(25, 2), Fraction(40)):
            for kind in ('clean', 'omitted', 'weak'):
                rows = {role: [score(audit.authored_hidden(25, 25 if role == 'constant' else post,
                                                         kind=kind), post, anchor)
                               for anchor in (0, Fraction(1, 4), Fraction(-1, 2), 3)]
                        for role in ('constant', 'change')}
                verdict = audit.verdict(rows)
                self.assertTrue(verdict['shape_supported'], (post, kind, verdict))
                self.assertTrue(verdict['density_resolved'], (post, kind, verdict))

    def test_flat_features_do_not_become_constant_clock_evidence(self):
        row = score(audit.authored_hidden(25, 40, kind='flat'))
        self.assertEqual(row['status'], 'eligible')
        self.assertTrue(all(c['score'] == 0 for c in row['clocks'].values()))
        verdict = audit.verdict({r: [row] * 4 for r in ('constant', 'change')})
        self.assertFalse(verdict['shape_supported'])
        self.assertFalse(verdict['density_resolved'])

    def test_zero_norm_invalidates_all_clocks_and_both_pair_members(self):
        row = score(audit.authored_hidden(25, 40, kind='zero'))
        self.assertEqual(row['status'], 'zero_norm_support')
        self.assertIsNone(row['clocks'])
        good = score(audit.authored_hidden(25, 25))
        result = audit.verdict({'constant': [good] * 4, 'change': [row] * 4})
        self.assertEqual(result['status'], 'pair_rejected')

    def test_owner_seams_are_counted_without_changing_or_reinferring_features(self):
        hidden = audit.authored_hidden(25, 40)
        owner = OWNER.copy()
        owner[100:] = 1
        original = score(hidden)
        seam = score(hidden, owner=owner)
        self.assertTrue(any(c['owner_crossing_targets'] > 0 for c in seam['clocks'].values()))
        for name in audit.NAMES:
            self.assertEqual(original['clocks'][name]['score'], seam['clocks'][name]['score'])

    def test_half_double_semantic_labels_cannot_change_identical_feature_evidence(self):
        # Assigning two incompatible human beat units to the same observation
        # cannot change a label-free score. This is not a musical success case.
        for multiplier in (0.5, 2):
            hidden = audit.authored_hidden(25, 25, multiplier=multiplier)
            np.testing.assert_array_equal(hidden, hidden.copy())
            self.assertEqual(score(hidden), score(hidden.copy()))
            rows = score(hidden)['clocks']
            rival = 'constant/half_zero' if multiplier == 0.5 else 'constant/double'
            self.assertEqual(audit.compare(rows[rival]['score'], rows['constant/native']['score']), 'left')

    def test_file_start_no_extrapolation_and_insufficient_support_is_explicit(self):
        hidden = audit.authored_hidden(25, 25)
        result = audit.evaluate(hidden, OWNER, 0, 500, 500, 100)
        self.assertEqual(result['status'], 'no_common_support')
        self.assertIsNone(result['clocks'])
        self.assertEqual(result['excluded_edge_frames'], 200)

    def test_change_near_tail_retains_small_post_boundary_denominator(self):
        row = score(audit.authored_hidden(25, 40, boundary=199), boundary=199)
        self.assertEqual(row['post_boundary_frames'], 1)
        self.assertEqual(row['pre_boundary_frames'] + row['post_boundary_frames'], row['target_frames'])

    def test_inputs_are_not_mutated_and_malformed_data_fail_closed(self):
        hidden = audit.authored_hidden(25, 40)
        before = hidden.tobytes()
        score(hidden)
        self.assertEqual(hidden.tobytes(), before)
        for value in (hidden.astype(np.float64), hidden[:, :2], hidden[:199]):
            with self.assertRaises(ValueError):
                score(value)
        bad = hidden.copy()
        bad[4, 1] = np.nan
        with self.assertRaises(ValueError):
            score(bad)
        bad_owner = OWNER.copy()
        bad_owner[3] = 1
        with self.assertRaises(ValueError):
            score(hidden, owner=bad_owner)

    def test_ties_are_numerical_not_calibrated_probabilities(self):
        self.assertEqual(audit.compare(0., 0.0000005), 'unresolved')
        self.assertEqual(audit.compare(0.01, 0.), 'left')
        self.assertEqual(audit.compare(0., 0.01), 'right')
        with self.assertRaises(ValueError):
            audit.compare(float('nan'), 0.)

    def test_no_favorable_phase_or_orphan_member_can_hide_failure(self):
        windows = {r: [score(audit.authored_hidden(25, 25 if r == 'constant' else 40))
                       for _ in range(4)] for r in ('constant', 'change')}
        self.assertTrue(audit.verdict(windows)['shape_supported'])
        modified = copy.deepcopy(windows)
        scores = modified['change'][2]['clocks']
        scores['step/native']['score'] = scores['constant/native']['score']
        self.assertEqual(audit.verdict(modified)['shape_phases'], 3)
        self.assertFalse(audit.verdict(modified)['shape_supported'])
        with self.assertRaises(ValueError):
            audit.verdict({'constant': windows['constant']})


class FrozenPreheadRecurrence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = replay.HERE / 'prehead-recurrence-v1.json'
        cls.report = json.loads(cls.path.read_bytes())

    def test_report_identity_source_and_complete_population(self):
        r = self.report
        self.assertEqual(replay.sha(self.path), '81b3d8a98c66d9362bf073187583c44d9f21b9f695bf178261ac24f6c5ee1c09')
        self.assertEqual(r['lock_sha256'], replay.sha(replay.LOCK_PATH))
        self.assertEqual(r['contract'], json.loads(replay.LOCK_PATH.read_bytes()))
        for name, value in r['source_sha256'].items():
            self.assertEqual(value, replay.sha(replay.HERE / name))
        for name, value in r['helper_sha256'].items():
            self.assertEqual(value, replay.sha(replay.HERE / name))
        self.assertTrue(replay.oracle.PINS.items() <= r['helper_sha256'].items())
        self.assertEqual(len(r['input_cases']), 40)
        self.assertEqual(sum(c['feature_capture'] for c in r['input_cases']), 7)
        self.assertEqual(sum(c['disposition'] == 'expressive_untyped' for c in r['input_cases']), 26)
        self.assertEqual(r['summary'], dict(selected_pairs=7, eligible_pairs=7,
            geometry_compatible_pairs=7, robust_shape_pairs=3, robust_density_pairs=2))
        self.assertEqual([c['case_id'] for c in r['captures']], [c['id'] for c in r['pairs']])
        for c in r['captures']:
            self.assertEqual(c['hidden_shape'], [c['frames'], 512])
            self.assertTrue(c['private_archive']['roundtrip_bit_exact'])
            self.assertTrue(all(v['passed'] for v in c['checks'].values()))
            for chunk in c['chunk_checks']:
                self.assertTrue(all(v['passed'] and v['bit_exact'] for v in chunk['checks'].values()))

    def test_all_failed_pairs_phases_and_density_alternatives_remain_visible(self):
        r = self.report
        shape, density, ledgers = [], [], 0
        for pair in r['pairs']:
            self.assertTrue(pair['geometry_compatible'])
            self.assertEqual(audit.verdict(pair['windows']), pair['verdict'])
            for rows in pair['windows'].values():
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(row == rows[0] for row in rows))
                for row in rows:
                    self.assertEqual(set(row['clocks']), set(audit.NAMES))
                    self.assertEqual(row['target_frames'] + row['excluded_edge_frames'], 200)
                    self.assertEqual(row['pre_boundary_frames'] + row['post_boundary_frames'], row['target_frames'])
                    for values in row['clocks'].values():
                        self.assertTrue(-2 <= values['score'] <= 2)
                        self.assertAlmostEqual(values['score'], values['same_cycle_cosine'] - values['half_cycle_cosine'])
                    ledgers += len(row['clocks'])
            if pair['verdict']['shape_supported']:
                shape.append(pair['id'])
            if pair['verdict']['density_resolved']:
                density.append(pair['id'])
        self.assertEqual(ledgers, 448)  # Correlated conditions, not independent observations.
        self.assertEqual(shape, ['artbeat-09-90-to-80', 'artbeat-10-90-to-120', 'artbeat-14-240-to-96'])
        self.assertEqual(density, shape[:2])
        self.assertEqual(r['decision'], 'do_not_promote_fixed_recurrence_rule')

    def test_public_report_has_no_reconstructable_features_or_private_paths(self):
        raw = self.path.read_text()
        self.assertNotRegex(raw, r'[A-Z]:[\\/]|/Users/|/home/|/dev/shm/')
        forbidden = {'mono_samples', 'mel_values', 'beat_logits', 'downbeat_logits',
                     'upstream_beats', 'upstream_downbeats', 'observations'}
        def visit(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden & value.keys())
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(self.report)
        for key in ('training', 'holdout_access', 'production_changed', 'new_user_parameters'):
            self.assertFalse(self.report[key])


if __name__ == '__main__':
    unittest.main()
