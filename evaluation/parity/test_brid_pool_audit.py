"""Failure-preserving aggregate controls; no corpus download or model access."""
import copy
import json
import unittest

import brid_pool_audit as audit


def authored_rows():
    return [dict(recording_id=recording_id, style='SA', audio_size_bytes=100,
                 beat_count=3, downbeat_count=2, annotated_span_seconds=1,
                 native_audio=dict(pcm_sha256=recording_id, duration_seconds=2,
                                   structural_gate_passed=True, zero_channels=[],
                                   full_scale_sample_fraction_by_channel=[0]))
            for recording_id in audit.IDS]


class BridPoolAuditTests(unittest.TestCase):
    def test_retained_pool_has_all_source_bytes_and_no_training_admission(self):
        inventory, assets = audit.load_pool()
        report = json.loads((audit.ROOT / 'evaluation/datasets/brid-audio-pool-v1.json').read_text())
        self.assertEqual(len(inventory['records']), 93)
        self.assertEqual(len(assets), 279)
        self.assertEqual(report['fetch_lock_sha256'], audit.POOL_LOCK_SHA)
        self.assertEqual([row['recording_id'] for row in report['records']], list(audit.IDS))
        self.assertEqual(report['audio_bytes'], 469990540)
        self.assertAlmostEqual(report['duration_seconds'], 2663.980634921)
        self.assertEqual(report['annotated_span_seconds'], 2527.192)
        self.assertEqual(report['beat_count'], 4342)
        self.assertEqual(report['downbeat_count'], 2205)
        self.assertEqual(report['style_counts'], dict(MA=3, PA=28, SA=41, SE=21))
        self.assertEqual(report, audit.summarize(report['records']))
        self.assertEqual(report['admitted_training_recordings'], 0)
        self.assertEqual({row['leakage_group'] for row in report['records']}, {'brid-corpus-v1'})
        smoke = json.loads((audit.ROOT / 'evaluation/datasets/brid-audio-smoke-v1.json').read_text())
        rows = {row['recording_id']: row for row in report['records']}
        for row in smoke['records']:
            self.assertEqual(rows[row['recording_id']], row)

    def test_summary_requires_all_ids_and_keeps_training_closed(self):
        result = audit.summarize(authored_rows())
        self.assertEqual(result['audio_recordings_decoded'], 93)
        self.assertEqual(result['beat_count'], 279)
        self.assertEqual(result['downbeat_count'], 186)
        self.assertEqual(result['duration_seconds'], 186)
        self.assertEqual(result['admitted_training_recordings'], 0)
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertIsNone(result['independent_work_count'])
        for key in ('framewise_tempo_targets_created', 'tempo_change_labels_created',
                    'feature_access', 'model_inference', 'project_holdout_access', 'public_default_changed'):
            self.assertFalse(result[key])

    def test_missing_reordered_or_duplicate_ids_cannot_pass_as_full_pool(self):
        rows = authored_rows()
        for changed in (rows[:-1], rows[::-1], [rows[0]] + rows[:-1]):
            with self.assertRaisesRegex(ValueError, 'exact 93'):
                audit.summarize(changed)

    def test_failures_are_retained_without_filtering_or_mutating(self):
        rows = authored_rows()
        rows[9]['native_audio']['structural_gate_passed'] = False
        rows[9]['native_audio']['zero_channels'] = [0]
        rows[12]['native_audio']['full_scale_sample_fraction_by_channel'] = [.01]
        original = copy.deepcopy(rows)
        result = audit.summarize(rows)
        self.assertFalse(result['structural_gate_passed'])
        self.assertEqual(result['structural_failure_ids'], ['0010'])
        self.assertEqual(result['zero_channel_ids'], ['0010'])
        self.assertEqual(result['full_scale_sample_ids'], ['0013'])
        self.assertEqual(len(result['records']), 93)
        self.assertEqual(rows, original)

    def test_exact_pcm_duplicates_do_not_become_independent_works(self):
        rows = authored_rows()
        rows[1]['native_audio']['pcm_sha256'] = rows[0]['native_audio']['pcm_sha256']
        result = audit.summarize(rows)
        self.assertEqual(result['exact_duplicate_native_pcm_groups'], [['0001', '0002']])
        self.assertIsNone(result['independent_work_count'])
        self.assertEqual(result['conservative_leakage_group_count'], 1)


if __name__ == '__main__':
    unittest.main()
