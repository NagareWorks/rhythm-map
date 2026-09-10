"""Authored geometry controls; no music, listener claims, or private inputs."""
import copy
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest

import clock_proposal_diagnosis as d
import clock_proposal_diagnosis_audit as audit


def linear(period=1, offset=0, support=(0, 12)):
    return {'knots': [[t, (t - offset) / period] for t in support]}


def generated(clocks=None, query=None):
    return dict(query_s=[4, 8] if query is None else query,
                clocks=[linear()] if clocks is None else clocks)


REFERENCE = list(range(13))


class DiagnosticGeometryTests(unittest.TestCase):
    def test_original_hit_and_all_candidates_are_unchanged(self):
        g = generated([linear(), linear(offset=.2), linear(period=2)])
        old = copy.deepcopy(g)
        result = d.diagnose(g, REFERENCE, REFERENCE, REFERENCE)
        self.assertEqual(result['partition'], 'original_covered')
        self.assertEqual(len(result['candidates']), 3)
        self.assertEqual(result['original_reference'], d.clocks.reference_check(old, REFERENCE))
        self.assertEqual(g, old)

    def test_crop_crossing_is_witness_not_promoted_coverage(self):
        clock = linear(offset=-.02)
        result = d.diagnose(generated([clock]), REFERENCE, REFERENCE, [])
        self.assertEqual(result['partition'], 'query_boundary_only_witness')
        self.assertEqual(result['original_reference']['status'], 'native_grid_missed')
        self.assertEqual(result['candidates'][0]['boundary'], dict(
            unmatched_reference=0, unmatched_prediction=0, boundary_crossing_pairs=2))

    def test_count_mismatch_can_be_only_a_boundary_crossing(self):
        # The first reference tick moves inside; the last stays outside.
        reference = [0, 4.02, 5, 6, 7, 8, 12]
        clock = {'knots': [[0, -4], [3.99, 0], [5, 1], [12, 8]]}
        row = d.candidate_diagnosis(clock, reference, [4, 8])
        self.assertEqual(row['count_delta'], -1)
        self.assertEqual(row['kind'], 'query_boundary_only_witness')

    def test_boundary_halo_cannot_repair_an_interior_missing_event(self):
        clock = {'knots': [[0, -4], [3.98, 0], [5.98, 1], [6.98, 2], [7.98, 3], [12, 7]]}
        row = d.candidate_diagnosis(clock, REFERENCE, [4, 8])
        self.assertEqual(row['kind'], 'count_mismatch')
        self.assertEqual(row['boundary']['unmatched_reference'], 1)

    def test_clock_support_is_not_extrapolated_for_halo(self):
        row = d.candidate_diagnosis(linear(offset=-.02, support=(4, 8)), REFERENCE, [4, 8])
        self.assertEqual(row['boundary']['unmatched_reference'], 1)
        self.assertNotEqual(row['kind'], 'query_boundary_only_witness')

    def test_constant_offset_is_descriptive_not_a_fitted_clock(self):
        clock = linear(offset=.2)
        before = copy.deepcopy(clock)
        row = d.candidate_diagnosis(clock, REFERENCE, [4, 8])
        self.assertEqual(row['kind'], 'equal_count_offset_compatible')
        self.assertFalse(row['strict_match'])
        self.assertAlmostEqual(row['ordered_residuals']['median_s'], .2)
        self.assertAlmostEqual(row['ordered_residuals']['span_s'], 0)
        self.assertEqual(clock, before)

    def test_nonuniform_residual_is_not_called_a_translatable_offset(self):
        clock = {'knots': [[0, -4], [4.1, 0], [5.4, 1], [6.1, 2], [7.4, 3], [12, 8]]}
        row = d.candidate_diagnosis(clock, REFERENCE, [4, 8])
        self.assertEqual(row['kind'], 'equal_count_nonuniform_error')
        self.assertAlmostEqual(row['ordered_residuals']['span_s'], .3)
        result = d.diagnose(generated([clock]), REFERENCE, REFERENCE, [])
        self.assertEqual(result['partition'], 'equal_count_nonuniform_error')

    def test_no_equal_count_candidate_keeps_native_reference_scope(self):
        result = d.diagnose(generated([linear(period=2)]), REFERENCE, [0, 2, 4, 6, 8, 10, 12], [])
        self.assertEqual(result['partition'], 'no_count_matched_candidate')
        # This does not say the half-level is perceptually wrong.
        self.assertEqual(result['anchors']['anchored_interval_advances'], {'2': 2})

    def test_empty_candidates_are_misses_not_dropped(self):
        result = d.diagnose(generated([]), REFERENCE, [], [])
        self.assertEqual(result['partition'], 'no_candidates')
        self.assertEqual(result['original_reference']['status'], 'native_grid_missed')
        self.assertEqual(result['anchors']['unmatched_reference'], 4)

    def test_missing_reference_is_unavailable_not_negative(self):
        for reference in ([], [5, 6, 7]):
            result = d.diagnose(generated(), reference, REFERENCE, [])
            self.assertEqual(result['partition'], 'reference_unavailable')
            self.assertIsNone(result['anchors'])
            self.assertEqual(result['candidates'], [])

    def test_missing_anchor_peak_inventory_never_inserts_events(self):
        raw, peaks = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12], [5.02]
        before = copy.deepcopy((raw, peaks, REFERENCE))
        row = d.anchor_diagnosis(raw, peaks, REFERENCE, [4, 8])
        self.assertEqual(row['unmatched_reference'], 2)
        self.assertEqual(row['missing_reference_with_candidate_peak'], 1)
        self.assertEqual(row['missing_reference_without_candidate_peak'], 1)
        self.assertEqual((raw, peaks, REFERENCE), before)

    def test_peaks_match_one_to_one_not_multiple_missing_anchors(self):
        row = d.anchor_diagnosis([0, 2], [.5], [0, .48, .52, 1, 2], [0, 2])
        self.assertEqual(row['missing_reference_with_candidate_peak'], 1)
        self.assertEqual(row['missing_reference_without_candidate_peak'], 2)

    def test_mixed_reference_advances_do_not_assume_each_event_is_one_beat(self):
        row = d.anchor_diagnosis([0, 1, 3, 4], [], [0, 1, 2, 3, 4], [0, 4])
        self.assertEqual(row['anchored_interval_advances'], {'1': 2, '2': 1})
        self.assertTrue(row['mixed_reference_advances'])
        self.assertEqual(row['interior_reference_ticks_between_matched_raw_anchors'], 1)
        self.assertEqual(row['linear_interpolation_outside_tolerance'], 0)

    def test_nonlinear_between_anchor_timing_is_not_cured_by_uniform_subdivision(self):
        row = d.anchor_diagnosis([0, 2, 3], [], [0, .5, 2, 3], [0, 3])
        self.assertEqual(row['linear_interpolation_outside_tolerance'], 1)
        self.assertEqual(row['interior_reference_ticks_between_matched_raw_anchors'], 1)

    def test_unmatched_endpoint_never_gets_a_reference_advance(self):
        row = d.anchor_diagnosis([0, 1.3, 2], [], [0, 1, 2], [0, 2])
        self.assertEqual(row['intervals_without_two_reference_anchors'], 2)
        self.assertEqual(row['anchored_interval_advances'], {})

    def test_original_tick_budget_failure_stays_explicit(self):
        result = d.diagnose(generated([linear(period=.01)]), REFERENCE, REFERENCE, [])
        self.assertEqual(result['partition'], 'diagnostic_budget_unavailable')
        self.assertEqual(result['candidates'][0]['kind'], 'original_tick_budget_exceeded')
        self.assertEqual(result['original_reference']['tick_budget_exceeded_candidates'], 1)

    def test_halo_budget_cannot_demote_a_strict_original_hit(self):
        clock = linear(period=1 / 32)
        reference = [i / 32 for i in range(385)]
        result = d.diagnose(generated([clock]), reference, reference, [])
        self.assertEqual(result['partition'], 'original_covered')
        self.assertEqual(result['candidates'][0]['kind'], 'diagnostic_tick_budget_exceeded')
        self.assertTrue(result['candidates'][0]['strict_match'])

    def test_invalid_timeline_query_and_clock_rejected(self):
        for reference in ([1, 1], [2, 1], [False, 1], [0, float('nan')], [-1, 0]):
            with self.assertRaises(ValueError):
                d.candidate_diagnosis(linear(), reference, [4, 8])
        for query in ([4, 4], [-1, 8], [4, True], [4, float('inf')]):
            with self.assertRaises(ValueError):
                d.candidate_diagnosis(linear(), REFERENCE, query)
        with self.assertRaises(ValueError):
            d.candidate_diagnosis({'knots': [[0, 1], [12, 1]]}, REFERENCE, [4, 8])

    def test_half_open_query_ownership(self):
        self.assertEqual(d.within([3.99, 4, 7.99, 8], [4, 8]), [4, 7.99])

    def test_chronological_matching_is_not_a_least_error_alignment(self):
        self.assertEqual(d.matches([1, 1.04], [1.03]), [(0, 0)])

    def test_partition_precedence_preserves_all_candidate_diagnostics(self):
        result = d.diagnose(generated([linear(period=2), linear(offset=.2), linear(offset=-.02)]),
                            REFERENCE, REFERENCE, [])
        self.assertEqual(result['partition'], 'query_boundary_only_witness')
        self.assertEqual(len(result['candidates']), 3)

    def test_summary_flags_overlap_but_query_partitions_conserve_denominator(self):
        rows = [dict(diagnosis=d.diagnose(generated(clocks), reference, [], []))
                for clocks, reference in (([linear()], REFERENCE), ([], REFERENCE), ([], []))]
        summary = d.summarize(rows)
        self.assertEqual(summary['queries'], 3)
        self.assertEqual(sum(summary['partition'].values()), 3)
        self.assertEqual(sum(summary['original_reference_status'].values()), 3)
        self.assertEqual(summary['failed_query_flags']['missing_raw_reference_anchor'], 1)
        self.assertEqual(summary['failed_query_flags']['some_missing_raw_anchor_has_no_candidate_peak'], 1)


class DiagnosticReportTests(unittest.TestCase):
    def report(self):
        return audit.read_pinned(audit.HERE / 'clock-proposal-diagnosis-v1.json',
                                 '0806f2725d842bd9ee421dc1bb4c8cc9ffaf1565de44b2ef20ef73363355d5fa')

    def test_source_and_prior_contracts_are_byte_pinned(self):
        report = self.report()
        lock = json.loads((audit.HERE / 'clock-proposal-diagnosis-lock-v1.json').read_bytes())
        self.assertEqual(lock, report['contract'])
        self.assertEqual(audit.sha(audit.HERE / 'clock-proposal-diagnosis-lock-v1.json'), report['lock_sha256'])
        for name, value in lock['source_sha256'].items():
            self.assertEqual(audit.sha(audit.HERE / name), value)
        prior = audit.read_pinned(audit.HERE / 'clock-proposal-v1.json', lock['prior_report_sha256'])
        self.assertEqual(report['prior_generation_sha256'], prior['generation_sha256'])
        self.assertEqual(lock['tolerance_seconds'], d.TOLERANCE)

    def test_all_original_input_identities_and_query_counts_are_retained(self):
        report = self.report()
        prior = json.loads((audit.HERE / 'clock-proposal-v1.json').read_bytes())
        keys = ('id', 'cohort', 'capture_sha256', 'truth_sha256', 'frame_count', 'queries')
        self.assertEqual([{k: r[k] for k in keys} for r in report['input_cases']],
                         [{k: r[k] for k in keys} for r in prior['input_cases']])
        self.assertEqual(len(report['input_cases']), 40)

    def test_denominators_flags_and_anchor_occurrences_conserve_all_slices_and_inputs(self):
        report = self.report()
        for groups in (report['input_cases'], list(report['slices'].values())):
            self.assertEqual(sum(g['queries'] for g in groups), 3348)
            for field in ('partition', 'original_reference_status', 'failed_query_flags', 'failed_query_anchor_totals'):
                totals = Counter()
                for group in groups:
                    totals.update(group[field])
                    if field in ('partition', 'original_reference_status'):
                        self.assertEqual(sum(group[field].values()), group['queries'])
                    if field == 'failed_query_flags':
                        self.assertTrue(all(0 <= v <= group['original_reference_status'].get('native_grid_missed', 0)
                                            for v in group[field].values()))
                self.assertEqual(dict(totals), report['summary'][field])
            for group in groups:
                anchors = group['failed_query_anchor_totals']
                self.assertEqual(anchors['unmatched_reference'], anchors['missing_reference_with_candidate_peak'] +
                                 anchors['missing_reference_without_candidate_peak'])
                self.assertLessEqual(anchors['linear_interpolation_outside_tolerance'],
                                     anchors['interior_reference_ticks_between_matched_raw_anchors'])

    def test_old_scores_are_not_promoted_by_posthoc_witnesses(self):
        summary = self.report()['summary']
        self.assertEqual(summary['original_reference_status'], dict(
            native_grid_covered=484, native_grid_missed=2690, reference_unavailable=174))
        self.assertEqual(summary['partition'], dict(original_covered=484, reference_unavailable=174,
            no_candidates=46, query_boundary_only_witness=38, equal_count_offset_compatible=427,
            equal_count_nonuniform_error=1179, no_count_matched_candidate=1000))

    def test_step_missing_path_and_peak_inventory_are_not_erased(self):
        step = self.report()['slices']['step']
        self.assertEqual(step['partition'], dict(original_covered=5, no_count_matched_candidate=17,
                                                equal_count_offset_compatible=1, equal_count_nonuniform_error=1))
        self.assertEqual(step['failed_query_flags']['missing_raw_reference_anchor'], 18)
        self.assertEqual(step['failed_query_flags']['all_missing_raw_anchors_have_candidate_peaks'], 14)
        self.assertEqual(step['failed_query_anchor_totals']['missing_reference_with_candidate_peak'], 50)
        self.assertEqual(step['failed_query_anchor_totals']['missing_reference_without_candidate_peak'], 4)

    def test_no_product_accuracy_or_training_claim(self):
        report = self.report()
        for key in ('new_candidates_generated', 'phase_shift_applied', 'neural_inference', 'training',
                    'holdout_access', 'production_change', 'accuracy_improvement_claimed'):
            self.assertFalse(report[key])
        self.assertTrue(report['all_original_candidates_and_scores_unchanged'])
        self.assertTrue(report['posthoc_reference_assisted_diagnosis'])
        self.assertIsNone(report['selected_clock'])
        self.assertEqual(report['new_annotations_collected'], 0)

    def test_altered_pinned_bytes_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'authored.json'
            path.write_text('{"value": 1}', encoding='utf-8')
            expected = audit.sha(path)
            self.assertEqual(audit.read_pinned(path, expected), {'value': 1})
            path.write_text('{"value": 2}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'pinned input bytes'):
                audit.read_pinned(path, expected)


if __name__ == '__main__':
    unittest.main()
