"""Model-free conditional-gate controls; no neural or private data access."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import musicfm_paired as paired
import musicfm_paired_probe as probe
import check_musicfm_paired as replay
from test_annotation_exposure import data


class MusicFMPairedControls(unittest.TestCase):
    def setUp(self):
        self.hidden = np.zeros((751, 1024), np.float32)
        self.hidden[:, 0] = 1
        self.owner = np.zeros(751, np.int32)
        self.pair = dict(constant_start_frame=401, change_start_frame=650, window_frames=200,
                         pre_segment_index=0, change_index=0)

    def test_protocol_keeps_seven_pairs_and_forty_id_denominator(self):
        plan, _, _, cases, selected = probe.protocol()
        self.assertEqual(len(cases), 40)
        self.assertEqual(sum(c['data']['untyped'] for c in cases), 26)
        self.assertEqual([p['id'] for p in selected], plan['selected_ids'])
        self.assertEqual(len(selected), 7)
        self.assertTrue(plan['conditional_only'])
        for key in ('training', 'holdout_access', 'production_change'):
            self.assertFalse(plan[key])

    def test_changed_helper_stops_before_population_access(self):
        raw = probe.LOCK.read_bytes()
        with patch.object(Path, 'read_bytes', side_effect=[raw, b'changed']), \
             patch.object(paired.prior, 'verified_plan') as reader:
            with self.assertRaisesRegex(ValueError, 'paired helper changed'):
                probe.protocol()
            reader.assert_not_called()

    def test_flat_features_do_not_get_truth_from_supplied_clocks(self):
        row = paired.pair_result(self.hidden, self.owner, data(), self.pair)
        self.assertTrue(row['geometry_compatible'])
        self.assertEqual(row['verdict']['status'], 'eligible')
        self.assertFalse(row['verdict']['shape_supported'])
        self.assertFalse(row['verdict']['density_resolved'])
        self.assertEqual(len(row['windows']['constant']), 4)

    def test_independently_authored_continuous_phase_cue_survives_missing_pulses(self):
        # 120 -> 240 at 15 seconds, independent of the adapter's supplied clocks.
        t = np.arange(751) / 25
        phase = np.where(t < 15, 2 * t, 30 + 4 * (t - 15))
        self.hidden[:, 0], self.hidden[:, 1] = np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)
        pulse = 0.25 * (np.minimum(phase % 1, 1 - phase % 1) < 1 / 16)
        pulse[np.floor(phase).astype(int) % 2 == 1] = 0
        self.hidden[:, 2] = pulse
        row = paired.pair_result(self.hidden, self.owner, data(), self.pair)
        self.assertTrue(row['verdict']['shape_supported'])
        self.assertTrue(row['verdict']['density_resolved'])
        # This cue is authored into vectors, not evidence of a model's ability.

    def test_odd_start_keeps_same_physical_boundary_without_upsampling(self):
        row = paired.pair_result(self.hidden, self.owner, data(), self.pair)
        for role in paired.oracle.replay.ROLES:
            original = paired.oracle.prepare(data(), self.pair, role, {s: [] for s in paired.oracle.replay.SOURCES})
            start = self.pair[role + '_start_frame']
            for coord in row['coordinates'][role]:
                self.assertEqual(coord['token_stop'] - coord['token_start'], 100)
                self.assertEqual(Fraction(coord['boundary']) + coord['token_start'], (start + original[3]) / 2)
                self.assertEqual(coord['first_target_offset_seconds'], '1/50' if start % 2 else '0')

    def test_zero_support_rejects_both_members_and_retains_pair(self):
        self.hidden[325:425] = 0
        row = paired.pair_result(self.hidden, self.owner, data(), self.pair)
        self.assertEqual(row['verdict']['status'], 'pair_rejected')
        self.assertFalse(row['verdict']['shape_supported'])
        self.assertTrue(all(r['clocks'] is None for rows in row['windows'].values() for r in rows))

    def test_geometry_failure_cannot_become_a_passing_pair(self):
        real = paired.oracle.prepare
        def incompatible(*args):
            row = copy.deepcopy(real(*args))
            row[2][0]['geometry']['compatible'] = False
            return row
        with patch.object(paired.oracle, 'prepare', side_effect=incompatible):
            row = paired.pair_result(self.hidden, self.owner, data(), self.pair)
        self.assertEqual(row['verdict']['status'], 'geometry_incompatible')
        self.assertFalse(row['verdict']['density_resolved'])
        self.assertTrue(all(r['clocks'] is None for rows in row['windows'].values() for r in rows))

    def test_complete_context_owners_preserved_and_inputs_immutable(self):
        self.owner[360:] = 1
        old_hidden, old_owner = self.hidden.copy(), self.owner.copy()
        row = paired.pair_result(self.hidden, self.owner, data(), self.pair)
        self.assertTrue(any(v['owner_crossing_targets'] > 0 for r in row['windows']['change'] for v in r['clocks'].values()))
        np.testing.assert_array_equal(old_hidden, self.hidden)
        np.testing.assert_array_equal(old_owner, self.owner)

    def test_invalid_feature_and_owner_contracts_fail(self):
        for hidden, owner in ((self.hidden.astype(np.float64), self.owner),
                              (self.hidden[:350], self.owner[:350]),
                              (self.hidden, self.owner.astype(np.int64)),
                              (self.hidden, np.full(751, -1, np.int32)),
                              (np.full_like(self.hidden, np.nan), self.owner)):
            with self.assertRaises(ValueError):
                paired.pair_result(hidden, owner, data(), self.pair)

    def test_no_orphan_or_replacement_or_duplicate_in_summary(self):
        ids = [str(i) for i in range(7)]
        rows = [dict(id=k, verdict=dict(status='eligible', shape_supported=True, density_resolved=True)) for k in ids]
        result = paired.summarize(rows, ids)
        self.assertEqual(result['both_native_shapes'], 7)
        for key in ('product_admission', 'automatic_accuracy_measured', 'independent_acceptance', 'no_rhythm_failure_resolved'):
            self.assertFalse(result[key])
        for changed in (rows[:-1], rows[::-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                paired.summarize(changed, ids)

    def test_cached_trace_hash_checked_before_pcm_interpretation(self):
        selected = [dict(id='authored')]
        old = dict(trace_sha256={'authored': '0' * 64})
        with patch.object(Path, 'read_bytes', return_value=b'not JSON'):
            with self.assertRaisesRegex(ValueError, 'trace changed'):
                probe.read_pcm(Path('unused'), selected, old)

    def test_cached_trace_complete_length_and_rate_checked(self):
        trace = dict(case_id='authored', sample_rate=22050, suite_id='artbeat-v1', prefix_seconds=60,
                     decoded_sample_count=1100, mono_samples=[0.] * 1100, audio_sha256='audio')
        for change in ({}, {'sample_rate': 24000}, {'decoded_sample_count': 1099}, {'case_id': 'replacement'}):
            raw = json.dumps(dict(trace, **change)).encode()
            old = dict(trace_sha256={'authored': probe.loader.digest(raw)})
            with patch.object(Path, 'read_bytes', return_value=raw):
                if change:
                    with self.assertRaises(ValueError):
                        probe.read_pcm(Path('unused'), [dict(id='authored')], old)
                else:
                    inputs, sources = probe.read_pcm(Path('unused'), [dict(id='authored')], old)
                    self.assertEqual(inputs['authored'].shape, (1100,))
                    self.assertEqual(sources[0]['input_sample_rate'], 22050)

    def test_old_negative_failure_is_still_pinned_and_not_reclassified(self):
        plan, *_ = probe.protocol()
        raw = (probe.HERE / 'musicfm-negative-v1.json').read_bytes()
        self.assertEqual(probe.loader.digest(raw), plan['helpers']['musicfm-negative-v1.json'])
        old = json.loads(raw)
        self.assertEqual(old['decision'], 'close_composed_rule_before_music')
        self.assertEqual(old['discrimination']['negative_families_abstaining'], 0)
        self.assertFalse(old['discrimination']['rule_survives_authored'])

    def test_incomplete_replay_receipt_never_opens_feature_archives(self):
        with patch.object(replay.np, 'load') as reader:
            with self.assertRaisesRegex(ValueError, 'completed conditional'):
                replay.verify(dict(schema='rhythm-map.musicfm-paired.v1', complete=False), Path('unused'))
            reader.assert_not_called()

    def test_changed_replay_protocol_never_opens_feature_archives(self):
        report = dict(schema='rhythm-map.musicfm-paired.v1', complete=True,
            extraction_fidelity_pass=True, conditional_only=True,
            decision='conditional_diagnostic_only_no_product_admission',
            training=False, holdout_access=False, production_change=False, helper_sha256={})
        with patch.object(replay.np, 'load') as reader:
            with self.assertRaisesRegex(ValueError, 'report protocol differs'):
                replay.verify(report, Path('unused'))
            reader.assert_not_called()

    def test_measured_report_retains_exact_scope_and_model_capture_counts(self):
        raw = (probe.HERE / 'musicfm-paired-v1.json').read_bytes()
        self.assertEqual(probe.loader.digest(raw), 'e71ebd237e5c37c10496d2a58f51ef607788caf3ed2ed5630ccc6ec2f643b203')
        report = json.loads(raw)
        self.assertTrue(report['complete'])
        self.assertTrue(report['extraction_fidelity_pass'])
        self.assertEqual(sum(len(c['contexts']) for c in report['captures']), 27)
        self.assertEqual(report['initial_model_state_sha256'], report['final_model_state_sha256'])
        self.assertEqual(len(report['input_cases']), 40)
        self.assertEqual(report['helper_sha256'], probe.protocol()[0]['helpers'])
        self.assertLess(report['peak_rss_bytes'], 4 * 1024 ** 3)

    def test_measured_shape_and_density_failures_remain_distinct(self):
        report = json.loads((probe.HERE / 'musicfm-paired-v1.json').read_bytes())
        ids = [p['id'] for p in report['pairs']]
        self.assertEqual(paired.summarize(report['pairs'], ids), report['summary'])
        self.assertEqual(report['summary']['both_native_shapes'], 3)
        self.assertEqual(report['summary']['all_density_rivals'], 0)
        supported = [p['id'] for p in report['pairs'] if p['verdict']['shape_supported']]
        self.assertEqual(supported, ['artbeat-09-90-to-80', 'artbeat-10-90-to-120', 'artbeat-15-85-to-127-5'])
        for pair in report['pairs']:
            phases = [probe.temporal.verdict(pair['windows']['constant'][i], pair['windows']['change'][i]) for i in range(4)]
            self.assertEqual(all(p['shape_supported'] for p in phases), pair['verdict']['shape_supported'])
            self.assertEqual(all(p['density_resolved'] for p in phases), pair['verdict']['density_resolved'])

    def test_measured_context_confounds_and_negative_failure_not_hidden(self):
        report = json.loads((probe.HERE / 'musicfm-paired-v1.json').read_bytes())
        crossing_windows = sum(any(c['owner_crossing_targets'] > 0 for c in p['windows'][role][0]['clocks'].values())
                               for p in report['pairs'] for role in ('constant', 'change'))
        self.assertEqual(crossing_windows, 11)
        self.assertFalse(report['summary']['product_admission'])
        self.assertFalse(report['summary']['no_rhythm_failure_resolved'])
        self.assertEqual(report['known_no_rhythm_failure_report_sha256'],
                         probe.protocol()[0]['helpers']['musicfm-negative-v1.json'])


if __name__ == '__main__':
    unittest.main()
