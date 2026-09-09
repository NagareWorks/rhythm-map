"""Contract fixtures are not pretrained-model outcomes or music acceptance."""
import copy
import json
import math
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

import rhythm_presence_audit as audit


class RhythmPresenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.source, self.reports = audit.protocol()
        self.cases = audit.gate_cases(self.source, self.reports)
        self.predictions = [dict(pcm_sha256=c['pcm_sha256'], status=c['expected_status']) for c in self.cases]

    def test_complete_population_and_separate_denominators(self):
        self.assertEqual(len(self.cases), 11)
        self.assertEqual(sum(c['expected_status'] == 'supported' for c in self.cases), 7)
        self.assertEqual(sum(c['expected_status'] == 'not_supported' for c in self.cases), 4)
        self.assertEqual([c['id'] for c in self.cases], [
            'constant-clean', 'constant-omitted', 'constant-weak', 'step-fast', 'step-slow', 'silence',
            'quiet-gain', 'short-rest', 'steady-tone', 'dc', 'noise'])

    def test_previous_decisions_unchanged(self):
        self.assertEqual(self.reports['musicfm-negative-v1.json']['decision'], 'close_composed_rule_before_music')
        self.assertEqual(self.reports['musicfm-temporal-v1.json']['decision'], 'close_fixed_rule_before_music')
        self.assertFalse(self.reports['beat-this-semantics-v1.json']['adapter_accepted'])

    def test_changed_source_stops_before_reading_reports(self):
        with patch.object(Path, 'read_bytes', return_value=b'{}') as read:
            with self.assertRaisesRegex(ValueError, 'changed bytes: rhythm-presence-source'):
                audit.protocol()
        self.assertEqual(read.call_count, 1)

    def test_changed_prior_report_stops(self):
        raw = (audit.HERE / 'rhythm-presence-source-v1.json').read_bytes()
        with patch.object(Path, 'read_bytes', side_effect=[raw, b'{}']):
            with self.assertRaisesRegex(ValueError, 'changed bytes: musicfm-temporal'):
                audit.protocol()

    def test_missing_extra_or_duplicate_retained_case_rejected(self):
        for operation in ('missing', 'extra', 'duplicate'):
            reports = copy.deepcopy(self.reports)
            rows = reports['musicfm-temporal-v1.json']['cases']
            if operation == 'missing':
                rows.pop()
            elif operation == 'extra':
                rows.append(dict(id='unregistered', pcm_sha256='a' * 64))
            else:
                rows.append(copy.deepcopy(rows[0]))
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                audit.gate_cases(self.source, reports)

    def test_invalid_or_duplicate_pcm_identity_rejected(self):
        for digest in ('not-a-sha256', self.cases[0]['pcm_sha256']):
            reports = copy.deepcopy(self.reports)
            reports['musicfm-temporal-v1.json']['cases'][1]['pcm_sha256'] = digest
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                audit.gate_cases(self.source, reports)

    def test_duplicate_expectation_rejected(self):
        source = copy.deepcopy(self.source)
        source['gate']['negative_ids'].append('constant-clean')
        with self.assertRaisesRegex(ValueError, 'duplicate declared identity'):
            audit.gate_cases(source, self.reports)

    def test_unrun_producer_has_no_pass_or_unknown_count(self):
        self.assertEqual(audit.assess_admissibility(self.cases, None),
                         dict(status='not_run', passes=None, case_count=11, unknown_count=None))

    def test_all_expected_fixtures_pass_only_authored_contract(self):
        result = audit.assess_admissibility(self.cases, self.predictions)
        self.assertTrue(result['passes'])
        self.assertEqual(result['status'], 'authored_gate_pass_not_music_acceptance')
        self.assertEqual((result['positives_supported'], result['negatives_not_supported']), (7, 4))

    def test_prediction_order_does_not_change_accounting(self):
        self.assertEqual(audit.assess_admissibility(self.cases, self.predictions),
                         audit.assess_admissibility(self.cases, list(reversed(self.predictions))))

    def test_each_single_failure_stays_visible(self):
        for index in range(len(self.cases)):
            rows = copy.deepcopy(self.predictions)
            rows[index]['status'] = 'not_supported' if rows[index]['status'] == 'supported' else 'supported'
            result = audit.assess_admissibility(self.cases, rows)
            with self.subTest(index=index):
                self.assertFalse(result['passes'])
                self.assertEqual(sum(not r['matches'] for r in result['cases']), 1)
                self.assertEqual(result['case_count'], 11)

    def test_unknown_never_becomes_rejection_or_a_dropped_row(self):
        for index in range(len(self.cases)):
            rows = copy.deepcopy(self.predictions)
            rows[index]['status'] = 'unknown'
            result = audit.assess_admissibility(self.cases, rows)
            with self.subTest(index=index):
                self.assertFalse(result['passes'])
                self.assertEqual((result['case_count'], result['unknown_count']), (11, 1))

    def test_all_support_all_reject_all_unknown_cannot_pass(self):
        for status in ('supported', 'not_supported', 'unknown'):
            rows = [dict(pcm_sha256=c['pcm_sha256'], status=status) for c in self.cases]
            with self.subTest(status=status):
                self.assertFalse(audit.assess_admissibility(self.cases, rows)['passes'])

    def test_missing_duplicate_or_extra_predictions_rejected(self):
        populations = (self.predictions[:-1], self.predictions + [self.predictions[0]],
                       self.predictions + [dict(pcm_sha256='0' * 64, status='supported')])
        for rows in populations:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                audit.assess_admissibility(self.cases, rows)

    def test_oracle_fields_cannot_be_smuggled_into_prediction_schema(self):
        for field in ('oracle_bpm', 'case_id', 'expected_status', 'supplied_clock_family'):
            rows = copy.deepcopy(self.predictions)
            rows[0][field] = 120
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'invalid prediction fields'):
                audit.assess_admissibility(self.cases, rows)

    def test_invalid_status_or_digest_rejected(self):
        for field, value in (('status', True), ('status', 'confident'), ('status', None),
                             ('status', 0.99), ('pcm_sha256', []), ('pcm_sha256', '0' * 64)):
            rows = copy.deepcopy(self.predictions)
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                audit.assess_admissibility(self.cases, rows)

    def test_empty_or_duplicate_case_contract_rejected(self):
        for cases in ([], self.cases + [self.cases[0]]):
            with self.assertRaisesRegex(ValueError, 'invalid case population'):
                audit.assess_admissibility(cases, None)

    def test_fixed_gate_does_not_supply_oracle_clocks(self):
        gate = self.source['gate']
        self.assertEqual(gate['producer_inputs'], ['complete_pcm', 'sample_rate', 'physical_window_seconds'])
        self.assertEqual(gate['window_seconds'], [4.0, 8.0])
        self.assertEqual((gate['sample_rate'], gate['complete_input_samples']), (24000, 288240))
        self.assertIsNone(gate['producer'])
        self.assertIn('oracle_bpm', gate['forbidden_producer_inputs'])

    def test_perfect_frame_labels_can_be_mostly_nonbeat_in_rhythmic_input(self):
        witness = audit.semantic_witnesses()['frame_nonbeat']
        self.assertEqual(witness['regular_offtick_fraction'], 0.96)
        self.assertTrue(witness['same_summary'])
        self.assertNotEqual(witness['regular_ticks'], witness['irregular_ticks'])
        self.assertIsNone(witness['region_presence_probability'])

    def test_weighted_ce_inverse_is_only_a_frame_identity(self):
        p = [0.03, 0.01, 0.96]
        q = audit.weighted_posterior(p, [60, 200, 1])
        direct = [p[0] * 60, p[1] * 200, p[2]]
        for a, b in zip(q, [v / sum(direct) for v in direct]):
            self.assertAlmostEqual(a, b, places=14)
        for a, b in zip(audit.weighted_posterior(q, [60, 200, 1], inverse=True), p):
            self.assertAlmostEqual(a, b, places=14)

    def test_zero_probability_and_large_weights_are_finite(self):
        result = audit.weighted_posterior([0, 0.5, 0.5], [1, 1e308, 1e308])
        self.assertEqual(result, [0, 0.5, 0.5])

    def test_invalid_probability_vectors_and_weights_rejected(self):
        for p in ([], [True, 0], [0.5, 0.6], [-1, 2], [math.nan, 1], [math.inf, 0]):
            with self.subTest(p=p), self.assertRaises(ValueError):
                audit.weighted_posterior(p, [1] * len(p))
        for w in ([1], [0, 1], [-1, 1], [math.inf, 1], [True, 1]):
            with self.subTest(w=w), self.assertRaises(ValueError):
                audit.weighted_posterior([0.5, 0.5], w)

    def test_shannon_concentration_does_not_measure_acoustic_cause(self):
        self.assertEqual(audit.histogram_information_bits([1] * 40), 0)
        self.assertEqual(audit.histogram_information_bits([100] + [0] * 39), math.log2(40))
        self.assertEqual(audit.histogram_information_bits([500] + [0] * 39), math.log2(40))

    def test_invalid_histograms_rejected(self):
        for counts in ([], [1], [0, 0], [-1, 2], [True, 1], [0.5, 0.5]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                audit.histogram_information_bits(counts)

    def test_prior_only_preference_remains_unqualified(self):
        witness = audit.semantic_witnesses()['prior_only']
        masses = [p * math.exp(ll) for p, ll in
                  zip(witness['family_prior'], witness['signal_log_likelihood_ratios'])]
        self.assertEqual(witness['family_posterior'], [x / sum(masses) for x in masses])
        self.assertIsNone(witness['acoustic_presence_probability'])

    def test_report_runs_offline_without_new_inference(self):
        with patch.object(socket, 'socket', side_effect=AssertionError('no network')):
            report = audit.build_report()
        self.assertEqual((report['positive_count'], report['negative_count']), (7, 4))
        self.assertEqual(report['producer_gate']['status'], 'not_run')
        for key in ('new_inference', 'new_model_acquisition', 'music_access', 'holdout_access', 'training',
                    'training_necessity_proven', 'production_change', 'new_user_parameters'):
            self.assertFalse(report[key])

    def test_new_sources_are_not_misreported_as_measured_failures(self):
        candidates = {c['id']: c for c in self.source['candidates']}
        self.assertEqual(len(candidates), 5)
        self.assertEqual(candidates['beatnet-plus']['revision'], 'bb90eb0a9065b101a4b4c4cb2b2061950266cb4b')
        self.assertEqual(len(candidates['beatnet-plus']['weights_observed_not_downloaded']), 3)
        self.assertEqual(candidates['beatfcos']['disposition'],
                         'no_acquirable_task_checkpoint_identified_in_reviewed_surfaces')

    def test_retained_report_reconstructs_without_audio_or_network(self):
        expected = json.loads((audit.HERE / 'rhythm-presence-audit-v1.json').read_text(encoding='utf-8'))
        self.assertEqual(audit.build_report(), expected)


if __name__ == '__main__':
    unittest.main()
