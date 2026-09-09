"""Authored planning/label semantics, not collected annotations or training."""
import copy
import hashlib
import itertools
import json
import unittest

import clock_readout_proposal as plan


def votes(left=None, right=None):
    return [dict(listener_id='listener-a', view_sha256='a' * 64,
                 labels=left or {'c0': 'supported', 'c1': 'contradicted'}),
            dict(listener_id='listener-b', view_sha256='a' * 64,
                 labels=right or {'c0': 'supported', 'c1': 'contradicted'})]


class ClockReadoutProposalTests(unittest.TestCase):
    def test_design_is_not_training_or_data_admission(self):
        spec = plan.proposal()
        self.assertEqual(spec['status'], 'design_only_not_training_authorized')
        for key in ('training_authorized', 'generator_implemented', 'generator_coverage_measured',
                    'training_readiness_established', 'production_change', 'holdout_access'):
            self.assertFalse(spec[key])
        self.assertEqual(spec['new_annotations_collected'], 0)
        self.assertEqual(spec['new_independent_groups_admitted'], 0)

    def test_prior_measured_failure_stays_byte_identified(self):
        raw = (plan.HERE / 'musicfm-paired-v1.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), plan.proposal()['baseline_report_sha256'])
        report = json.loads(raw)
        self.assertEqual(report['summary']['both_native_shapes'], 3)
        self.assertEqual(report['summary']['all_density_rivals'], 0)
        self.assertEqual(len(report['input_cases']), 40)

    def test_parameters_count_biases_and_shared_vs_candidate_costs(self):
        b = plan.budget()
        self.assertEqual(b['projection_parameters'], 512 * 32 + 32)
        self.assertEqual(b['temporal_parameters'], 6 * (5 * 32 + 32 + 32 * 32 + 32))
        self.assertEqual(b['fusion_parameters'], 40 * 32 + 32)
        self.assertEqual(b['output_parameters'], 33)
        self.assertEqual(b['parameters'], 25249)
        self.assertEqual(b['weight_bytes'], 100996)

    def test_receptive_field_is_not_encoder_latency(self):
        b = plan.budget()
        self.assertEqual(b['radius_frames'], 126)
        self.assertEqual(b['receptive_field_frames'], 253)
        self.assertEqual(b['center_span_seconds'], 5.04)
        self.assertFalse(b['runtime_measured'])

    def test_overlapping_query_costs_count_fusion_twice_where_reused(self):
        b = plan.budget()
        self.assertEqual(b['example_queries'], 149)
        self.assertEqual(b['example_query_frames'], 29800)
        self.assertEqual(b['example_multiply_accumulates'],
                         15000 * (512 * 32 + 6 * (5 * 32 + 32 * 32)) + 149 * 8 * (200 * 40 * 32 + 32))
        self.assertEqual(b['example_multiply_accumulates'], 657510144)

    def test_input_buffer_count_includes_both_readout_halos(self):
        b = plan.budget()
        self.assertEqual(b['feature_chunk_bytes'], 3072000)
        self.assertEqual(b['feature_chunk_with_halo_bytes'], 3588096)
        self.assertEqual(b['feature_chunk_with_halo_bytes'] - b['feature_chunk_bytes'], 2 * 126 * 512 * 4)

    def test_query_tail_is_not_lost_or_needlessly_duplicated(self):
        self.assertEqual(plan.query_lengths(15000), [200] * 149)
        self.assertEqual(plan.query_lengths(15001), [200] * 149 + [101])
        self.assertEqual(plan.query_lengths(50), [50])
        self.assertEqual(plan.query_lengths(200), [200])
        self.assertEqual(plan.query_lengths(201), [200, 101])

    def test_invalid_query_geometry_fails(self):
        for args in ((0,), (True,), (2.5,), (100, 0, 10), (100, 10, 11), (100, 10, -1)):
            with self.assertRaises(ValueError):
                plan.query_lengths(*args)

    def test_small_query_grids_cover_every_frame_and_own_no_outside_frame(self):
        for frames in range(1, 40):
            for window in range(1, 8):
                for hop in range(1, window + 1):
                    lengths = plan.query_lengths(frames, window, hop)
                    covered = {i * hop + j for i, count in enumerate(lengths) for j in range(count)}
                    self.assertEqual(covered, set(range(frames)))

    def test_only_equal_known_votes_produce_targets(self):
        for left, right in itertools.product(plan.proposal()['labels'], repeat=2):
            row = plan.consensus(['c0'], votes({'c0': left}, {'c0': right}))
            expected = 1 if left == right == 'supported' else 0 if left == right == 'contradicted' else None
            self.assertEqual(row['targets'], {'c0': expected})
            self.assertEqual(row['known'] + row['unknown'], 1)

    def test_multiple_supported_clocks_are_not_forced_into_one_class(self):
        both = {'c0': 'supported', 'c1': 'supported'}
        row = plan.consensus(['c0', 'c1'], votes(both, both))
        self.assertEqual(row['targets'], {'c0': 1, 'c1': 1})
        self.assertEqual(row['known'], 2)

    def test_all_negative_proposals_do_not_infer_absent_rhythm(self):
        neither = {'c0': 'contradicted', 'c1': 'contradicted'}
        row = plan.consensus(['c0', 'c1'], votes(neither, neither))
        self.assertEqual(row['targets'], {'c0': 0, 'c1': 0})
        self.assertNotIn('rhythm_present', row)
        self.assertFalse(plan.proposal()['all_negative_proposals_imply_no_rhythm'])

    def test_disagreement_and_shared_uncertainty_remain_separate(self):
        row = plan.consensus(['c0', 'c1'], votes(
            {'c0': 'supported', 'c1': 'uncertain'}, {'c0': 'contradicted', 'c1': 'uncertain'}))
        self.assertEqual(row['targets'], {'c0': None, 'c1': None})
        self.assertEqual(row['reasons'], {'c0': 'no_consensus', 'c1': 'both_uncertain'})
        self.assertEqual(row['unknown'], 2)

    def test_mismatched_view_or_duplicate_listener_fails(self):
        for key, value in (('view_sha256', 'b' * 64), ('listener_id', 'listener-a')):
            records = votes()
            records[1][key] = value
            with self.assertRaisesRegex(ValueError, 'duplicate listener or mismatched'):
                plan.consensus(['c0', 'c1'], records)

    def test_incomplete_or_invalid_votes_fail(self):
        for labels in ({'c0': 'supported'}, {'c0': 'supported', 'c1': False},
                       {'c0': 'supported', 'c1': 'not sure'}, {'c0': 'supported', 'c1': 'uncertain', 'extra': 'supported'}):
            records = votes()
            records[0]['labels'] = labels
            with self.assertRaises(ValueError):
                plan.consensus(['c0', 'c1'], records)
        with self.assertRaises(ValueError):
            plan.consensus(['c0', 'c1'], votes()[:1])

    def test_invalid_view_and_record_fields_fail(self):
        for key, value in (('view_sha256', 'not-a-hash'), ('listener_id', ''), ('unexpected', True)):
            records = votes()
            records[0][key] = value
            with self.assertRaises(ValueError):
                plan.consensus(['c0', 'c1'], records)

    def test_empty_overflow_or_duplicate_candidates_fail_without_removal(self):
        for ids in ([], ['c0', 'c0'], list(map(str, range(9))), [1, 'c1'], ['', 'c1']):
            with self.assertRaises(ValueError):
                plan.consensus(ids, votes())

    def test_original_votes_are_immutable_and_mapping_order_is_irrelevant(self):
        records = votes()
        old = copy.deepcopy(records)
        a = plan.consensus(['c0', 'c1'], records)
        records[1]['labels'] = dict(reversed(list(records[1]['labels'].items())))
        b = plan.consensus(['c0', 'c1'], records)
        self.assertEqual(a, b)
        self.assertEqual(records, old)

    def test_pilot_counts_are_group_based_and_not_an_accuracy_certificate(self):
        p = plan.proposal()['proposed_label_pilot']
        self.assertEqual(p['max_provenance_groups'], p['primary_strata'] * p['groups_per_stratum'])
        self.assertEqual(p['max_provenance_groups'] * p['queries_per_group'], 48)
        self.assertEqual(p['listeners'], 2)
        self.assertIn('not_fitting_or_acceptance', p['role'])

    def test_fit_envelope_has_a_primary_seed_and_counts_clock_only_control(self):
        cap = plan.proposal()['proposed_fit_caps_not_allocations']
        self.assertEqual(cap['primary_seed'], cap['main_seeds'][0])
        self.assertEqual(cap['max_fits'], len(cap['main_seeds']) + 1)
        self.assertEqual(cap['max_total_fit_minutes'], cap['max_fits'] * cap['max_minutes_per_fit'])
        self.assertEqual(cap['max_fixed_form_calibration_fits'], 1)


if __name__ == '__main__':
    unittest.main()
