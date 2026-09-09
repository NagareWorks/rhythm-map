"""Model-free controls for frozen neural negatives; fixtures are not model evidence."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import musicfm_negative as negative
import musicfm_temporal as temporal


def ledger(correct=None):
    return dict(status='eligible', clocks={name: dict(score=float(name == correct)) for name in temporal.NAMES})


def passing_scores(plan):
    return {case['id']: {speed: ledger('constant/native' if case['kind'] == 'constant' else None)
                        for speed, _ in negative.FAMILIES} for case in plan['cases']}


class MusicFMNegativeControls(unittest.TestCase):
    def test_frozen_population_and_denominators(self):
        plan, _, old, base, _ = negative.protocol()
        self.assertEqual([c['id'] for c in plan['cases']], ['quiet-gain', 'short-rest', 'steady-tone', 'dc', 'noise'])
        self.assertEqual((plan['new_case_count'], plan['new_context_count'], plan['new_forward_count']), (5, 15, 45))
        self.assertEqual((plan['pair_denominator'], plan['negative_family_denominator']), (4, 6))
        self.assertEqual(old['decision'], 'close_fixed_rule_before_music')
        self.assertEqual(base['resources']['process_memory_limit_bytes'], 4 * 1024 ** 3)
        for key in ('music_access', 'holdout_access', 'training', 'production_change', 'new_user_parameters'):
            self.assertFalse(plan[key])

    def test_protocol_refuses_changed_bytes_before_prior_protocols(self):
        with patch.object(Path, 'read_bytes', return_value=b'{}'), patch.object(negative.availability, 'protocol') as prior:
            with self.assertRaisesRegex(ValueError, 'negative protocol changed'):
                negative.protocol()
            prior.assert_not_called()

    def test_previous_helper_and_report_drift_refused(self):
        raw = (negative.HERE / 'musicfm-negative-lock-v1.json').read_bytes()
        helper = Path(negative.availability.__file__).read_bytes()
        for reads, message in (([raw, b'changed'], 'availability helper changed'),
                                ([raw, helper, b'changed'], 'availability report changed')):
            with patch.object(Path, 'read_bytes', side_effect=reads):
                with self.assertRaisesRegex(ValueError, message):
                    negative.protocol()

    def test_pcm_identities_reuse_all_five_activity_only_cases(self):
        plan, *_ = negative.protocol()
        cases = negative.inputs(plan)
        self.assertEqual(list(cases), [c['id'] for c in plan['cases']])
        self.assertTrue(all(x.dtype == np.float32 and x.shape == (288240,) for x in cases.values()))

    def test_changed_pcm_is_refused_before_model_use(self):
        plan, *_ = negative.protocol()
        controls = negative.availability.controls()
        controls['quiet-gain'][123] += np.float32(0.01)
        with patch.object(negative.availability, 'controls', return_value=controls):
            with self.assertRaisesRegex(ValueError, 'frozen control PCM changed'):
                negative.inputs(plan)

    def test_tied_ledger_abstains_and_does_not_create_a_null_model(self):
        result = negative.negative_verdict(ledger())
        self.assertEqual(result, dict(abstains=True, reason='numerical_tie', clock_spread=0.0))

    def test_fixed_numerical_budget_boundary_no_retuning(self):
        for delta, expected in ((temporal.TIE, True), (temporal.TIE * 1.01, False)):
            row = ledger()
            row['clocks']['constant/native']['score'] = delta
            self.assertEqual(negative.negative_verdict(row)['abstains'], expected)

    def test_any_clock_preference_fails_not_only_a_wrong_shape(self):
        for name in temporal.NAMES:
            result = negative.negative_verdict(ledger(name))
            self.assertFalse(result['abstains'])
            self.assertEqual(result['reason'], 'spurious_clock_preference')

    def test_unavailable_numerical_support_is_explicit(self):
        for status in ('zero_norm_support', 'no_common_support'):
            result = negative.negative_verdict(dict(status=status, clocks=None))
            self.assertTrue(result['abstains'])
            self.assertIsNone(result['clock_spread'])
        with self.assertRaises(ValueError):
            negative.negative_verdict(dict(status='unknown_activity', clocks=None))

    def test_malformed_ledger_cannot_be_counted_as_abstention(self):
        for bad in (float('nan'), float('inf'), True):
            row = ledger()
            row['clocks']['constant/native']['score'] = bad
            with self.assertRaises(ValueError):
                negative.negative_verdict(row)
        row = ledger()
        row['clocks'].pop('constant/native')
        with self.assertRaises(ValueError):
            negative.negative_verdict(row)

    def test_all_pairs_and_both_negative_families_are_required(self):
        plan, _, old, *_ = negative.protocol()
        result = negative.judge(passing_scores(plan), old, plan)
        self.assertTrue(result['rule_survives_authored'])
        self.assertEqual((result['shape_supported'], result['density_resolved']), (4, 4))
        self.assertEqual(result['negative_families_abstaining'], 6)

    def test_one_negative_family_failure_is_not_hidden(self):
        plan, _, old, *_ = negative.protocol()
        scores = passing_scores(plan)
        scores['noise']['slow'] = ledger('step/double')
        result = negative.judge(scores, old, plan)
        self.assertFalse(result['rule_survives_authored'])
        self.assertEqual((len(result['negatives']), result['negative_families_abstaining']), (6, 5))
        self.assertEqual(result['paired_denominator'], 4)

    def test_robustness_failure_keeps_all_pairs(self):
        plan, _, old, *_ = negative.protocol()
        scores = passing_scores(plan)
        scores['short-rest']['fast'] = dict(status='zero_norm_support', clocks=None)
        result = negative.judge(scores, old, plan)
        self.assertFalse(result['rule_survives_authored'])
        self.assertEqual((len(result['pairs']), result['density_resolved']), (4, 3))

    def test_missing_case_or_family_refused(self):
        plan, _, old, *_ = negative.protocol()
        for kind in ('case', 'family'):
            scores = passing_scores(plan)
            if kind == 'case':
                scores.pop('noise')
            else:
                scores['noise'].pop('slow')
            with self.assertRaises(ValueError):
                negative.judge(scores, old, plan)

    def test_verdict_does_not_mutate_old_evidence(self):
        plan, _, old, *_ = negative.protocol()
        snapshot = copy.deepcopy(old)
        negative.judge(passing_scores(plan), old, plan)
        self.assertEqual(old, snapshot)

    def test_all_activity_preflights_use_the_existing_native_boundary(self):
        plan, activity, *_ = negative.protocol()
        cases = negative.inputs(plan)
        with patch.object(negative.availability, 'run_native', return_value=(
                dict(source_sha256={}), [True] * 301, 'authored')) as native:
            rows = negative.prepare_activity(Path('/native'), Path('/private'), cases, plan, activity)
        self.assertEqual(native.call_count, 5)
        self.assertEqual([r['id'] for r in rows], list(cases))
        self.assertTrue(all(g['status'] == 'available_for_periodic_test' for r in rows for g in r['gates'].values()))

    def test_changed_activity_stops_before_neural_inputs(self):
        plan, activity, *_ = negative.protocol()
        cases = negative.inputs(plan)
        with patch.object(negative.availability, 'run_native', return_value=(
                dict(source_sha256={}), [False] * 301, 'authored')):
            with self.assertRaisesRegex(ValueError, 'prior activity availability changed'):
                negative.prepare_activity(Path('/native'), Path('/private'), cases, plan, activity)

    def test_scoring_reuses_fixed_complete_window_without_extrapolation(self):
        plan, *_ = negative.protocol()
        hidden = np.ones((301, 1024), np.float32)
        owner = np.array([0] * 150 + [1] * 100 + [2] * 51, np.int32)
        extracted = {c['id']: (hidden.copy(), owner.copy()) for c in plan['cases']}
        gates = [dict(id=name, gates={speed: dict(status='available_for_periodic_test')
                                     for speed, _ in negative.FAMILIES}) for name in extracted]
        scores = negative.score_extracted(extracted, gates, plan)
        for name in extracted:
            for speed, period in negative.FAMILIES:
                self.assertEqual(scores[name][speed], temporal.evaluate(hidden[100:200], owner[100:200],
                                  0, temporal.Fraction(25, 2), period, 50))
        np.testing.assert_array_equal(extracted['dc'][0], hidden)

    def test_failed_activity_is_not_scored_on_a_favorable_subset(self):
        plan, *_ = negative.protocol()
        extracted = {c['id']: (None, None) for c in plan['cases']}
        gates = [dict(id=name, gates={speed: dict(status='low_activity') for speed, _ in negative.FAMILIES})
                 for name in extracted]
        with patch.object(temporal, 'evaluate') as evaluate:
            with self.assertRaisesRegex(ValueError, 'activity changed after preflight'):
                negative.score_extracted(extracted, gates, plan)
            evaluate.assert_not_called()

    def test_retained_report_identity_and_complete_extraction(self):
        raw = (negative.HERE / 'musicfm-negative-v1.json').read_bytes()
        self.assertEqual(negative.loader.digest(raw), 'b2f67387286130b0b1c98e568134caa3def2b9b2cb4b0287a4f9056e5167d79c')
        report = json.loads(raw)
        self.assertTrue(report['complete'])
        self.assertTrue(report['extraction_fidelity_pass'])
        self.assertEqual(report['source_sha256'], negative.loader.digest(Path(negative.__file__).read_bytes()))
        self.assertEqual(report['protocol_sha256'], negative.LOCK_SHA)
        self.assertEqual(len(report['cases']), 5)
        contexts = [row for case in report['cases'] for row in case['contexts']]
        self.assertEqual(len(contexts), 15)
        self.assertTrue(all(row['reference_hook_repeat_exact'] and row['all_blocks_and_projection'] for row in contexts))
        self.assertEqual(report['initial_model_state_sha256'], report['final_model_state_sha256'])

    def test_retained_result_recomputes_all_denominators_and_failed_negatives(self):
        report = json.loads((negative.HERE / 'musicfm-negative-v1.json').read_bytes())
        plan, _, old, *_ = negative.protocol()
        self.assertEqual(negative.judge(report['discrimination']['scores'], old, plan), report['discrimination'])
        result = report['discrimination']
        self.assertEqual((result['paired_denominator'], result['shape_supported'], result['density_resolved']), (4, 4, 4))
        self.assertEqual((result['negative_family_denominator'], result['negative_families_abstaining']), (6, 0))
        self.assertEqual(report['decision'], 'close_composed_rule_before_music')
        self.assertFalse(result['rule_survives_authored'])
        self.assertTrue(all(row['reason'] == 'spurious_clock_preference' for row in result['negatives']))

    def test_retained_failure_does_not_change_availability_or_claim_music_accuracy(self):
        report = json.loads((negative.HERE / 'musicfm-negative-v1.json').read_bytes())
        for row in report['activity']:
            self.assertTrue(all(g['status'] == 'available_for_periodic_test' for g in row['gates'].values()))
        for key in ('music_access', 'holdout_access', 'training', 'production_change', 'new_user_parameters',
                    'independent_music_acceptance', 'commercial_distribution_approved'):
            self.assertFalse(report[key])


if __name__ == '__main__':
    unittest.main()
