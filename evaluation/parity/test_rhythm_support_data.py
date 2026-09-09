"""Keep the initial publication-only worksheet distinct from verified data."""
import json
from pathlib import Path
import unittest


class RhythmSupportWorksheetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parent.parent / 'datasets/rhythm-support-worksheet-v1.json'
        cls.worksheet = json.loads(path.read_text(encoding='utf-8'))
        cls.sources = {row['id']: row for row in cls.worksheet['sources']}

    def test_source_routes_are_distinct_and_draft_not_a_fetch_manifest(self):
        self.assertEqual(len(self.sources), len(self.worksheet['sources']))
        self.assertEqual(set(self.sources), {'fsd50k-dev-4060432', 'existing-project-regressions',
                                           'newly-authored-recordings'})
        self.assertEqual(self.worksheet['status'], 'draft_no_clip_inventory')
        self.assertEqual(self.worksheet['purpose'],
                         'metadata_and_label_planning_not_an_acquisition_or_training_manifest')

    def test_published_counts_are_not_verified_candidates(self):
        row = self.sources['fsd50k-dev-4060432']
        counts = row['published_license_counts']
        self.assertEqual(counts, {'CC0': 14959, 'CC-BY': 20017, 'CC-BY-NC': 4616, 'Sampling+': 1374})
        self.assertEqual(sum(counts.values()), row['published_total_clips'])
        self.assertEqual(row['initial_license_screen'], ['CC0', 'CC-BY'])
        self.assertEqual(sum(counts[key] for key in row['initial_license_screen']),
                         row['nominal_candidate_clips_from_publication'])
        for key in ('verified_clip_count', 'verified_candidate_clip_count',
                    'distinct_uploaders', 'independent_source_group_count'):
            self.assertIsNone(row[key])
        self.assertEqual(self.worksheet['candidate_records'], [])
        self.assertEqual(self.worksheet['admitted_independent_groups'], [])

    def test_failed_metadata_acquisition_has_no_byte_identity_claim(self):
        row = self.sources['fsd50k-dev-4060432']
        self.assertEqual(row['evidence_level'], 'publication_only')
        self.assertEqual(row['metadata_status'], 'not_acquired')
        self.assertEqual(row['published_metadata_md5'], 'b9ea0c829a411c1d42adb9da539ed237')
        self.assertIsNone(row['metadata_sha256'])
        self.assertIsNone(row['dev_member_sha256'])
        self.assertEqual(row['scope'], 'development_metadata_only')
        self.assertTrue(row['metadata_url'].endswith('/FSD50K.metadata.zip'))
        self.assertFalse(row['audio_acquired'])
        self.assertFalse(row['eval_metadata_inspected'])
        self.assertTrue(row['acquisition_obstacles'])

    def test_license_family_is_not_a_completed_rights_or_overlap_review(self):
        row = self.sources['fsd50k-dev-4060432']
        self.assertEqual(row['dataset_conditions_review'], 'unreviewed_commercial_contact_note')
        self.assertEqual(row['encoder_overlap_review'], 'unreviewed')
        rights = self.worksheet['candidate_record_template']['rights']
        for key in ('training_permission', 'evaluation_permission', 'redistribution_permission',
                    'dataset_conditions'):
            self.assertEqual(rights[key], 'unreviewed')
        for key in ('audio_license_url', 'annotation_license_url', 'attribution'):
            self.assertIsNone(rights[key])
        self.assertEqual(rights['evidence_urls'], [])

    def test_provenance_template_does_not_fabricate_independent_groups(self):
        row = self.worksheet['candidate_record_template']
        for key in ('source_id', 'clip_id', 'source_url', 'source_version', 'metadata_sha256', 'pcm_sha256'):
            self.assertIsNone(row[key])
        provenance = row['provenance']
        for key in ('work_id', 'recording_id', 'synthesis_generator_id'):
            self.assertIsNone(provenance[key])
        for key in ('creator_ids', 'derived_asset_ids', 'evidence_urls'):
            self.assertEqual(provenance[key], [])
        self.assertEqual(provenance['encoder_overlap'], 'unreviewed')
        self.assertEqual(row['split'], 'unassigned')
        self.assertEqual(row['windows'], [])

    def test_all_required_slices_remain_without_collected_labels(self):
        rows = self.worksheet['label_coverage']
        self.assertEqual(len({row['slice'] for row in rows}), 8)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row['labeled_independent_groups'] == 0 for row in rows))
        row = self.worksheet['window_record_template']
        for key in ('query_start_seconds', 'query_end_seconds', 'context_start_seconds',
                    'context_end_seconds', 'context_policy_version', 'resolved_label'):
            self.assertIsNone(row[key])
        self.assertEqual(row['label_status'], 'not_collected')  # Not listener-voted unknown.
        self.assertEqual(row['listener_votes'], [])
        self.assertEqual(row['adjudication_history'], [])

    def test_previous_regressions_and_holdouts_do_not_become_new_data(self):
        row = self.sources['existing-project-regressions']
        self.assertEqual(row['role'], 'regression_only')
        self.assertEqual(row['exposed_calibration_id_count'], 40)
        self.assertEqual(row['exposed_authored_gate_input_count'], 11)
        self.assertFalse(row['new_independent_acceptance_eligible'])
        self.assertEqual(row['holdout_policy'], 'unchanged_sealed')

    def test_authoring_and_training_are_not_implicitly_authorized(self):
        row = self.sources['newly-authored-recordings']
        self.assertEqual(row['recordings_created'], 0)
        self.assertEqual(row['grants_obtained'], 0)
        self.assertIsNone(row['independent_source_group_count'])
        self.assertFalse(self.worksheet['training_authorized'])
        self.assertFalse(self.worksheet['audio_selection_made'])


if __name__ == '__main__':
    unittest.main()
