import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import annotation_exposure_audit as audit


def authored():
    return dict(schema_version=1, id='authored', duration_s=30,
                beats=[dict(time_s=i / 2) for i in range(30)] + [dict(time_s=i / 4) for i in range(60, 121)],
                tempo_segments=[dict(start_s=0, end_s=15, kind='constant', start_bpm=120, end_bpm=120),
                                dict(start_s=15, end_s=30, kind='constant', start_bpm=240, end_bpm=240)],
                change_points=[dict(time_s=15, kind='tempo_jump')])


def data():
    return audit.normalize(authored(), 1501)


class AnnotationExposureControls(unittest.TestCase):
    def test_time_quantization_and_collisions_are_explicit(self):
        self.assertEqual(audit.microseconds(0.0000005), 1)
        self.assertEqual(audit.microseconds(0.00000049), 0)
        for invalid in (True, '1', -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                audit.microseconds(invalid)
        truth = authored()
        truth['beats'][1]['time_s'] = 0.0000001
        with self.assertRaisesRegex(ValueError, 'colliding'):
            audit.normalize(truth, 1501)
        with self.assertRaisesRegex(ValueError, 'duration mismatch'):
            audit.normalize(authored(), 1400)

    def test_normalization_rejects_wrong_segment_and_change_semantics(self):
        for field, value in (('kind', 'unknown'), ('end_bpm', 121), ('start_bpm', 0), ('end_s', 31)):
            truth = authored()
            truth['tempo_segments'][0][field] = value
            with self.assertRaises(ValueError):
                audit.normalize(truth, 1501)
        truth = authored()
        truth['tempo_segments'][1]['start_s'] = 14
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            audit.normalize(truth, 1501)
        truth = authored()
        truth['change_points'] *= 2
        with self.assertRaisesRegex(ValueError, 'ordering'):
            audit.normalize(truth, 1501)

    def test_fractional_full_query_matches_independent_ledger_expression(self):
        for length in audit.LOCK['durations_frames']:
            for center in (Fraction(2), Fraction(2000001, 1000000), Fraction(3),
                           Fraction(length - 3), Fraction(length - 3) - Fraction(1, 1000000),
                           Fraction(length - 2)):
                expected = math.ceil(center - 3) >= 0 and math.floor(center + 3) < length
                self.assertEqual(expected, 2 * audit.FRAME_US < center * audit.FRAME_US < (length - 3) * audit.FRAME_US)

    def test_prefix_is_strictly_prior_and_inside_constant_segment(self):
        d = data()
        self.assertEqual(audit.classify(d, 75, 100)[0], 'insufficient_prefix')
        self.assertEqual(audit.classify(d, 76, 200), ('constant', 0))
        self.assertEqual(audit.classify(d, 755, 100)[0], 'no_stable_prefix_segment')
        d['segments'][0]['kind'] = 'ramp'
        self.assertEqual(audit.classify(d, 100, 100)[0], 'no_stable_prefix_segment')
        d = data()
        d['changes'].insert(0, dict(at=1000000, kind='authored_conflict'))
        self.assertEqual(audit.classify(d, 100, 100)[0], 'prefix_contains_change')

    def test_constant_end_may_touch_step_without_containing_it(self):
        self.assertEqual(audit.classify(data(), 550, 200), ('constant', 0))
        self.assertEqual(audit.classify(data(), 551, 200)[0], 'step_too_few_interior_beats')
        self.assertEqual(audit.classify(data(), 697, 100), ('change', (0, 0)))

    def test_full_query_minima_do_not_count_boundary_censored_beats(self):
        d = data()
        # A two-second window aligned to the 120 BPM grid has only three full-query beats.
        self.assertEqual(audit.classify(d, 100, 100)[0], 'constant_too_few_interior_beats')
        self.assertEqual(audit.classify(d, 104, 100), ('constant', 0))
        # The change window at 14s has one interior pre-step beat; shifting one frame earlier exposes two.
        self.assertEqual(audit.classify(d, 700, 100)[0], 'step_too_few_interior_beats')
        self.assertEqual(audit.classify(d, 697, 100), ('change', (0, 0)))

    def test_step_requires_adjacent_different_constant_tempos_and_single_change(self):
        for mutation in ('same_bpm', 'ramp', 'gap', 'extra_change', 'not_jump', 'post_end'):
            d = data()
            if mutation == 'same_bpm':
                d['segments'][1]['bpm'] = 120
            elif mutation == 'ramp':
                d['segments'][1]['kind'] = 'ramp'
            elif mutation == 'gap':
                d['segments'][1]['start'] += 1
            elif mutation == 'extra_change':
                d['changes'].append(dict(at=15500000, kind='tempo_jump'))
            elif mutation == 'not_jump':
                d['changes'][0]['kind'] = 'ramp_start'
            else:
                d['segments'][1]['end'] = 15500000
            self.assertNotEqual(audit.classify(d, 697, 100)[0], 'change', mutation)

    def test_tail_short_tracks_and_untyped_rubato_keep_exclusions(self):
        self.assertEqual(audit.classify(data(), 1450, 100)[0], 'annotation_tail_uncovered')
        d = audit.normalize(authored(), 1501, untyped=True)
        p = audit.profile(d, 100)
        self.assertEqual(p['roles'], {'untyped_rubato': 1402})
        self.assertIsNone(p['pair'])
        d['frame_count'] = 90
        self.assertEqual(audit.profile(d, 100)['candidate_starts'], 0)
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            audit.profile(d, 101)

    def test_clock_budget_counts_half_open_ticks_and_both_half_phases(self):
        beats = list(range(0, 10000001, 50000))
        before = 40
        self.assertTrue(audit.clock_budget(beats, before, 2000000, 5200000))  # double: exactly 128
        self.assertFalse(audit.clock_budget(beats, before, 2000000, 5200001))
        d = data()
        with patch.object(audit, 'clock_budget', return_value=False):
            self.assertEqual(audit.classify(d, 101, 100)[0], 'clock_budget_exceeded')

    def test_pair_selection_and_combination_counts_match_brute_force(self):
        controls = {0: [0, 10, 20, 25], 1: [0, 5, 15]}
        changes = [dict(at=1500000), dict(at=1700000)]
        windows = [(90, 0, 1), (50, 0, 0), (51, 0, 0), (49, 0, 0), (50, 1, 0), (10, 2, 0)]
        length = 20
        brute = [(i, abs(2 * changes[i]['at'] - (2 * s + length) * audit.FRAME_US), s, -c, pre)
                 for s, pre, i in windows for c in controls.get(pre, []) if c + length <= s]
        expected = min(brute)
        pair, combinations, starts = audit.choose_pair(controls, windows, changes, length)
        self.assertEqual(combinations, len(brute))
        self.assertEqual(starts, len({(p[2], p[4], p[0]) for p in brute}))
        self.assertEqual((pair['change_index'], pair['change_start_frame'], -pair['constant_start_frame']),
                         (expected[0], expected[2], expected[3]))
        self.assertLessEqual(pair['constant_start_frame'] + length, pair['change_start_frame'])
        self.assertEqual(audit.choose_pair({0: [31]}, [(50, 0, 0)], changes, 20), (None, 0, 0))
        self.assertEqual(audit.choose_pair({1: [0]}, [(50, 0, 0)], changes, 20), (None, 0, 0))

    def test_positive_profile_has_one_pair_and_every_start_has_one_role(self):
        for length in audit.LOCK['durations_frames']:
            p = audit.profile(data(), length)
            self.assertEqual(sum(p['roles'].values()), p['candidate_starts'])
            if p['pair'] is not None:
                pair = p['pair']
                self.assertEqual(audit.classify(data(), pair['constant_start_frame'], length), ('constant', pair['pre_segment_index']))
                self.assertEqual(audit.classify(data(), pair['change_start_frame'], length), ('change', (pair['pre_segment_index'], pair['change_index'])))
                self.assertLessEqual(pair['constant_start_frame'] + length, pair['change_start_frame'])
        self.assertIsNotNone(audit.profile(data(), 100)['pair'])

    def test_global_duration_uses_track_coverage_then_longer_tie_and_null(self):
        empty, yes = {'pair': None}, {'pair': {'fixture': True}}
        self.assertEqual(audit.choose_duration({100: [yes, yes], 200: [yes, empty], 400: [yes, empty]}), 100)
        self.assertEqual(audit.choose_duration({100: [yes], 200: [yes], 400: [yes]}), 400)
        self.assertIsNone(audit.choose_duration({100: [empty], 200: [empty], 400: [empty]}))

    def test_metadata_projection_ignores_poisoned_response_outcomes(self):
        source = {'cohorts': [dict(cohort='artbeat', suite_sha256='suite', cases=[
            dict(id='x', frame_count=1501, truth_sha256='truth', capture_sha256='capture', raw_matches=99)])]}
        before = audit.metadata_projection(source)
        source['cohorts'][0]['cases'][0]['raw_matches'] = {'arbitrary': [float('nan')]}
        source['cohorts'][0]['cases'][0]['scores'] = object()
        source['cohorts'][0]['inventories'] = object()
        self.assertEqual(audit.metadata_projection(source), before)
        self.assertEqual(set(before[0]['cases'][0]), {'id', 'frame_count', 'truth_sha256', 'capture_sha256'})

    def test_report_positive_and_no_pair_paths_publish_hashes_not_coordinates(self):
        identity = dict(id='authored', frame_count=1501, truth_sha256='truth', capture_sha256='capture')
        for untyped in (False, True):
            with patch.object(audit, 'load_annotations', return_value=([], [dict(cohort='artbeat', identity=identity,
                             data=audit.normalize(authored(), 1501, untyped))])):
                report = audit.build_report()
            self.assertEqual(report['selected_pair_count'], 0 if untyped else 1)
            if untyped:
                self.assertIsNone(report['selected_window_frames'])
                self.assertEqual(report['plan_sha256'], audit.canonical_hash([]))
            self.assertNotIn('constant_start_frame', json.dumps(report))
            self.assertNotIn('change_start_frame', json.dumps(report))
            self.assertEqual(report['response_budget_verification'], 'not_run')
            self.assertFalse(report['response_fields_used'])

    def test_public_input_hash_is_checked_before_json_parsing(self):
        with patch.object(Path, 'read_bytes', return_value=b'not json'):
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                audit.read_json('authored', 'wrong')


class FrozenAnnotationExposure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = audit.HERE / 'annotation-exposure-v1.json'
        cls.report = json.loads(cls.path.read_bytes())
        cls.reads = []
        original = Path.read_bytes

        def recording_read(path):
            cls.reads.append(path.resolve())
            return original(path)

        with patch.object(Path, 'read_bytes', autospec=True, side_effect=recording_read):
            cls.reproduced = audit.build_report()

    def test_complete_report_reproduces_from_only_public_inputs(self):
        self.assertEqual(self.report, self.reproduced)
        self.assertEqual(audit.sha(self.path.read_bytes()), 'e9bb2086e906bda366e0b8d2d4dd87b9ff9d7c44441696d223a75a96471b1346')
        allowed = {audit.HERE / audit.SOURCE, audit.LOCK_PATH, Path(audit.__file__),
                   *(audit.ROOT / 'evaluation/suites' / (name + '.json') for name in audit.SUITES.values())}
        self.assertEqual(len(self.reads), 45)  # source, two suites, 40 truths, code and lock
        for path in self.reads:
            self.assertTrue(path in allowed or path.is_relative_to(audit.ROOT / 'evaluation/suites/truth'), path)

    def test_every_profile_retains_all_tracks_starts_and_pair_exclusions(self):
        source = audit.metadata_projection(json.loads((audit.HERE / audit.SOURCE).read_bytes()))
        self.assertEqual(self.report['metadata_projection_sha256'], audit.canonical_hash(source))
        expected_ids = [c['id'] for cohort in source for c in cohort['cases']]
        for length, p in self.report['profiles'].items():
            self.assertEqual([c['id'] for c in p['cases']], expected_ids)
            for c in p['cases']:
                self.assertEqual(c['candidate_starts'], max(0, c['frame_count'] - int(length) + 1))
                self.assertEqual(sum(c['roles'].values()), c['candidate_starts'])
                self.assertEqual(c['pair_available'], c['pairable_change_starts'] > 0)
                self.assertGreaterEqual(c['pair_combinations'], c['pairable_change_starts'])
                self.assertEqual(c['pair_sha256'] is not None, c['pair_available'])
            for cohort, aggregate in p['cohorts'].items():
                rows = [c for c in p['cases'] if c['cohort'] == cohort]
                self.assertEqual(aggregate['tracks'], len(rows))
                self.assertEqual(aggregate['pairable_tracks'], sum(c['pair_available'] for c in rows))
                for key in ('candidate_starts', 'pair_combinations', 'pairable_change_starts'):
                    self.assertEqual(aggregate[key], sum(c[key] for c in rows))
                self.assertEqual(sum(aggregate['roles'].values()), aggregate['candidate_starts'])
                if cohort == 'rubato':
                    self.assertEqual(aggregate['pairable_tracks'], 0)
                    self.assertEqual(aggregate['roles'], {'untyped_rubato': aggregate['candidate_starts']})

    def test_frozen_global_selection_and_claim_boundaries(self):
        self.assertEqual([self.report['profiles'][str(n)]['cohorts']['artbeat']['pairable_tracks']
                          for n in (100, 200, 400)], [4, 7, 0])
        self.assertEqual(self.report['selected_window_frames'], 200)
        self.assertEqual(self.report['selected_pair_count'], 7)
        selected = [c['id'] for c in self.report['profiles']['200']['cases'] if c['pair_available']]
        self.assertEqual(selected, self.report['selected_case_ids'])
        self.assertEqual(self.report['plan_sha256'], '5c15e677d0277dcac111aaebfb2be0b788996cbd44019a315c171fcf39bfe4f2')
        for flag in ('private_captures_read', 'response_fields_used', 'neural_inference', 'decoder_replayed',
                     'fitted_mapping', 'holdout_opened', 'training_run', 'production_output_changed',
                     'new_user_parameters', 'accuracy_improvement_claimed'):
            self.assertIs(self.report[flag], False)
        self.assertIsNone(self.report['selected_tempo'])
        self.assertEqual(self.report['response_budget_verification'], 'not_run')

    def test_loader_rejects_suite_role_identity_and_annotation_path_escape(self):
        original = audit.read_json
        for mutation, message in (('purpose', 'suite role'), ('order', 'ordered case'),
                                  ('path', 'outside annotation'), ('truth_id', 'truth id')):
            def altered(path, expected):
                result = copy.deepcopy(original(path, expected))
                if Path(path).name == 'artbeat-v1.json':
                    if mutation == 'purpose':
                        result['purpose'] = 'holdout'
                    elif mutation == 'order':
                        result['cases'].reverse()
                    elif mutation == 'path':
                        result['cases'][0]['input']['truth'] = '../not-an-annotation.json'
                elif mutation == 'truth_id' and Path(path).is_relative_to(audit.ROOT / 'evaluation/suites/truth'):
                    result['id'] = 'wrong-case'
                return result

            with patch.object(audit, 'read_json', side_effect=altered):
                with self.assertRaisesRegex(ValueError, message):
                    audit.load_annotations()


if __name__ == '__main__':
    unittest.main()
