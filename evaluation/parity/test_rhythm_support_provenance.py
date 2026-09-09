"""Authored sampling controls and retained evidence boundaries; no external data."""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from rhythm_support_inventory import LICENSES, SCREEN
import rhythm_support_provenance as provenance


def rows():
    result = {}
    for index in range(1, 25):
        license_url = ('http://creativecommons.org/licenses/by/3.0/' if index <= 12
                       else 'http://creativecommons.org/publicdomain/zero/1.0/')
        result[str(index)] = dict(license=license_url, uploader=f'authored-{index}',
                                  title='Authored fixture', description='', tags=[])
    return result


def exposure():
    return dict(id='freesound-loop-tempo-v1',
                assets=[dict(path=f'audio/{index}.wav', role='audio') for index in range(100, 115)])


class ProvenanceSamplingTests(unittest.TestCase):
    def test_fixed_quota_and_no_duplicate_ids(self):
        selected = provenance.select_review_sample(rows(), set())
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({item['clip_id'] for item in selected}), 8)
        self.assertEqual(Counter(item['license_id'] for item in selected),
                         {license_id: 4 for license_id in SCREEN})

    def test_input_order_and_semantic_descriptions_do_not_select_outcomes(self):
        source = rows()
        expected = provenance.select_review_sample(source, set())
        changed = dict(reversed(list(copy.deepcopy(source).items())))
        for row in changed.values():
            row.update(title='Different title', description='pulse tempo silence', tags=['music'])
        self.assertEqual(provenance.select_review_sample(changed, set()), expected)
        self.assertEqual(source, rows())

    def test_known_project_exposure_is_excluded(self):
        source = rows()
        exposed = {item['clip_id'] for item in provenance.select_review_sample(source, set())}
        selected = provenance.select_review_sample(source, exposed)
        self.assertFalse(exposed.intersection(item['clip_id'] for item in selected))
        self.assertEqual(len(selected), 8)

    def test_noncommercial_and_sampling_licenses_cannot_fill_a_quota(self):
        source = rows()
        expected = provenance.select_review_sample(source, set())
        for index, url in enumerate(LICENSES, 1000):
            if LICENSES[url] not in SCREEN:
                source[str(index)] = dict(license=url, uploader=f'excluded-{index}')
        self.assertEqual(provenance.select_review_sample(source, set()), expected)

    def test_casefold_hint_is_excluded_across_both_license_quotas(self):
        source = rows()
        preliminary = provenance.select_review_sample(source, set())
        left, right = preliminary[0]['clip_id'], preliminary[4]['clip_id']
        source[left]['uploader'] = 'SameHint'
        source[right]['uploader'] = 'SAMEHINT'
        selected = provenance.select_review_sample(source, set())
        selected_ids = {item['clip_id'] for item in selected}
        self.assertIn(left, selected_ids)
        self.assertNotIn(right, selected_ids)
        self.assertEqual(len({source[key]['uploader'].casefold() for key in selected_ids}), 8)

    def test_insufficient_hints_fail_without_relaxing_quota(self):
        source = rows()
        for row in source.values():
            row['uploader'] = 'one hint'
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            provenance.select_review_sample(source, set())

    def test_tied_ranks_use_numeric_not_insertion_or_lexical_order(self):
        with patch.object(provenance, 'rank', return_value='same'):
            selected = provenance.select_review_sample(dict(reversed(list(rows().items()))), set())
        self.assertEqual([item['clip_id'] for item in selected], ['1', '2', '3', '4', '13', '14', '15', '16'])

    def test_exposure_manifest_identity_and_duplicate_ids_are_rejected(self):
        for change in ('wrong_id', 'duplicate', 'too_few', 'path', 'unicode', 'zero'):
            manifest = exposure()
            if change == 'wrong_id':
                manifest['id'] = 'not-the-exposed-suite'
            elif change == 'duplicate':
                manifest['assets'][1] = manifest['assets'][0]
            elif change == 'too_few':
                manifest['assets'].pop()
            elif change == 'path':
                manifest['assets'][0]['path'] = '../100.wav'
            elif change == 'unicode':
                manifest['assets'][0]['path'] = 'audio/１００.wav'
            else:
                manifest['assets'][0]['path'] = 'audio/0.wav'
            with self.subTest(change=change), self.assertRaises(ValueError):
                provenance.exposure_ids(json.dumps(manifest))

    def test_exposure_matches_are_not_encoder_membership_results(self):
        source = rows()
        manifest = exposure()
        manifest['assets'][0]['path'] = 'audio/1.wav'
        report = provenance.build_report(source, {'archive_locks': []}, json.dumps(manifest).encode())
        self.assertEqual(report['exposure_matches']['screened_ids'], ['1'])
        self.assertEqual(report['encoder_recording_overlap'], 'unknown')
        self.assertEqual(report['independent_groups_admitted'], 0)
        self.assertIsNone(report['population_rhythm_coverage'])
        self.assertEqual(report['listener_labels_collected'], 0)
        for key in ('audio_selection_made', 'audio_acquired', 'feature_access', 'training_authorized',
                    'project_holdout_access', 'production_change'):
            self.assertFalse(report[key])


class RetainedAdmissionReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent.parent / 'datasets'
        cls.sample = json.loads((root / 'rhythm-support-provenance-v1.json').read_text(encoding='utf-8'))
        cls.review = json.loads((root / 'rhythm-support-admission-review-v1.json').read_text(encoding='utf-8'))

    def test_exposure_hash_and_two_known_regressions_are_retained(self):
        payload = provenance.EXPOSURE_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), self.sample['exposure_manifest']['sha256'])
        self.assertEqual(len(provenance.exposure_ids(payload)), 15)
        self.assertEqual(self.sample['exposure_matches']['screened_ids'], ['210835', '418991'])
        self.assertFalse({'210835', '418991'}.intersection(row['clip_id'] for row in self.sample['review_sample']))

    def test_every_frozen_sample_has_one_finding_and_no_inferred_label(self):
        samples = self.sample['review_sample']
        findings = self.review['sample_findings']
        self.assertEqual([row['clip_id'] for row in findings], [row['clip_id'] for row in samples])
        self.assertEqual(len(findings), 8)
        for sample, finding in zip(samples, findings):
            self.assertEqual(sample['rank_sha256'], provenance.rank(sample['clip_id']))
            self.assertIsNone(finding['rhythm_label'])
            self.assertFalse(finding['independent_group_verified'])
            self.assertFalse(finding['current_source_verified'])
            self.assertTrue(finding['open_questions'])

    def test_stale_snapshot_and_unavailable_pages_are_not_live_verification(self):
        findings = self.review['sample_findings']
        unavailable = [row for row in findings if row['source_page_retrieval'] == 'tool_unavailable_nonretryable']
        self.assertEqual(len(unavailable), 7)
        cached = next(row for row in findings if row['clip_id'] == '340968')
        self.assertEqual(cached['source_page_retrieval'], 'cached_primary_page_crawler_reports_7_months_old')
        self.assertTrue(cached['cached_page_agrees_with_metadata'])
        self.assertFalse(cached['current_source_verified'])

    def test_corpus_absence_does_not_claim_recording_level_independence(self):
        encoder = self.review['encoder_provenance']
        self.assertEqual(len(encoder['published_corpora']), 16)
        self.assertEqual(len(set(encoder['published_corpora'])), 16)
        self.assertNotIn('fsd50k', encoder['published_corpora'])
        self.assertNotIn('fsld', encoder['published_corpora'])
        self.assertEqual(encoder['final0_documented_excluded_corpus'], 'gtzan')
        self.assertEqual(encoder['recording_level_overlap'], 'unknown')
        self.assertFalse(encoder['membership_comparison_completed'])

    def test_standard_cc_terms_are_not_individual_clearance_or_training_authority(self):
        rights = self.review['license_review']
        self.assertFalse(rights['separate_ai_specific_grant_required_by_standard_cc_license'])
        self.assertFalse(rights['other_rights_cleared'])
        conclusion = self.review['conclusion']
        self.assertEqual(conclusion['new_independent_groups_admitted'], 0)
        self.assertEqual(conclusion['new_region_labels_collected'], 0)
        for key in ('training_necessity_established', 'audio_selection_made', 'audio_acquired',
                    'feature_access', 'training_authorized', 'project_holdout_access', 'production_change'):
            self.assertFalse(conclusion[key])
        self.assertEqual(conclusion['next_target'],
                         'return_to_missed_beat_vs_real_tempo_change_evidence_not_presence_only')


if __name__ == '__main__':
    unittest.main()
