"""Authored contract tests, not musical data or listener annotations."""
import copy
import hashlib
import json
import unittest

import clock_proposals as c
import clock_annotation_packet as p
import clock_proposal_audit as audit


def observations(times=None):
    return dict(duration_s=12, beat_times_s=list(range(13)) if times is None else times, valid_spans_s=[[0, 12]])


def provenance():
    return dict(work_id='authored-work', recording_id='authored-recording', performance_id='authored-performance',
                creator_session_id='authored-session', shared_asset_ids=[], transform_family_ids=['authored-control'],
                rights_evidence_sha256='a' * 64, training_review='unreviewed', redistribution_review='unreviewed',
                encoder_overlap='unknown', input_relation=dict(kind='complete', source_interval_s=[0, 12]))


def bundle(obs=None):
    return p.build(dict(pcm_sha256='b' * 64, sample_rate=50, sample_count=600), provenance(),
                   observations() if obs is None else obs, [4, 8], 'c' * 64)


def first_records(pack, followable=True):
    return [dict(listener_id=name, view_sha256=p.digest(pack['first_view']), context_sufficient=True,
                 followability='followable' if followable else 'uncertain',
                 accepted_clocks=[{'knots': [[0, 0], [12, 12]]}] if followable else [], transition_intervals_s=[])
            for name in ('listener-a', 'listener-b')]


def vote_records(pack, first, label='supported'):
    return [dict(listener_id=r['listener_id'], view_sha256=p.digest(pack['candidate_view']),
                 first_pass_sha256=p.digest(r), labels={c['id']: label for c in pack['candidate_view']['candidates']}) for r in first]


class ClockProposalTests(unittest.TestCase):
    def test_regular_events_deduplicate_constructions_not_half_phases(self):
        result = c.generate(observations(), [4, 8])
        self.assertEqual((result['status'], result['proposed_count'], result['unique_count']), ('ready', 8, 4))
        ticks = [c.ticks(clock, [4, 8]) for clock in result['clocks']]
        self.assertEqual(ticks, [[4, 5, 6, 7], [4, 6], [5, 7], [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5]])

    def test_clock_equivalence_all_knots_and_integer_phase_only(self):
        a = {'knots': [[0, 0], [4, 4]]}
        self.assertTrue(c.equivalent(a, {'knots': [[0, 3], [2, 5], [4, 7]]}))
        self.assertFalse(c.equivalent(a, {'knots': [[0, .5], [4, 4.5]]}))
        self.assertFalse(c.equivalent(a, {'knots': [[0, 0], [2, 3], [4, 4]]}))

    def test_overflow_is_explicit_and_returns_no_favorable_prefix(self):
        result = c.bounded_set([{'knots': [[0, 0], [12, rate]]} for rate in range(1, 10)])
        self.assertEqual((result['status'], result['unique_count'], result['clocks']), ('overflow', 9, []))

    def test_invalid_clock_geometry_refused(self):
        for clock in ({'knots': [[0, 0], [1, 0]]}, {'knots': [[0, 1], [1, 0]]},
                      {'knots': [[0, 0], [0, 1]]}, {'knots': [[0, 0], [1, float('nan')]]},
                      {'knots': [[0, 0], [1, 1]], 'score': 1}):
            with self.assertRaises(ValueError):
                c.validate_clock(clock)

    def test_piecewise_clock_is_phase_continuous_at_step(self):
        obs = observations([0, .5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        result = c.generate(obs, [1, 5])
        self.assertTrue(any(c.ticks(clock, [1, 5]) == [1, 1.5, 2, 3, 4] for clock in result['clocks']))
        for clock in result['clocks']:
            c.validate_clock(clock)
            for t, value in clock['knots'][1:-1]:
                self.assertAlmostEqual(c.phase(clock, t), value)

    def test_empty_edges_and_gap_remain_unavailable(self):
        self.assertEqual(c.generate(observations([]), [4, 8])['status'], 'empty')
        self.assertEqual(c.generate(observations([1, 2, 3]), [0, 4])['status'], 'unbracketed_query')
        obs = dict(duration_s=12, beat_times_s=[0, 1, 2, 8, 9, 10, 11, 12], valid_spans_s=[[0, 2], [8, 12]])
        self.assertEqual(c.generate(obs, [1, 5])['status'], 'invalid_query_support')
        self.assertEqual(c.generate(obs, [8, 12])['status'], 'ready')

    def test_valid_silence_is_not_missing_frame_support(self):
        result = c.generate(observations([]), [4, 8])
        self.assertEqual(result['status'], 'empty')
        self.assertNotEqual(result['status'], 'invalid_query_support')
        self.assertTrue(result['full_readout_context'])

    def test_dense_event_budget_fails_without_decimation(self):
        result = c.generate(observations([i / 100 for i in range(1201)]), [4, 8])
        self.assertEqual((result['status'], result['clocks']), ('observation_budget_exceeded', []))

    def test_extra_annotation_score_or_identity_fields_are_refused(self):
        for key in ('truth', 'bpm', 'case_id', 'confidence', 'role'):
            obs = observations() | {key: 1}
            with self.assertRaisesRegex(ValueError, 'unexpected observation'):
                c.generate(obs, [4, 8])

    def test_invalid_observation_values_and_query_fail(self):
        for obs in (observations([0, 0]), observations([0, float('nan')]), observations([0, 13]),
                    observations() | {'valid_spans_s': [[0, 6], [5, 12]]}):
            with self.assertRaises(ValueError):
                c.generate(obs, [4, 8])
        for query in ([4, 4], [-1, 4], [0, 13], [0, True]):
            with self.assertRaises(ValueError):
                c.generate(observations(), query)

    def test_query_grid_keeps_short_input_and_partial_tail(self):
        self.assertEqual(list(c.query_grid(1)), [[0, 1]])
        self.assertEqual(list(c.query_grid(5)), [[0, 4], [2, 5]])
        self.assertEqual(list(c.query_grid(8)), [[0, 4], [2, 6], [4, 8]])

    def test_context_flags_do_not_invent_physical_edges(self):
        self.assertFalse(c.generate(observations(), [0, 4])['full_readout_context'])
        self.assertTrue(c.generate(observations(), [4, 8])['full_readout_context'])
        result = c.generate(observations(list(range(3, 13))), [4, 8])
        self.assertTrue(result['full_readout_context'])
        self.assertFalse(result['full_clock_context'])

    def test_native_grid_is_posthoc_proxy_and_empty_stays_in_denominator(self):
        result = c.generate(observations(), [4, 8])
        old = copy.deepcopy(result)
        self.assertEqual(c.reference_check(result, list(range(13)))['status'], 'native_grid_covered')
        self.assertEqual(c.reference_check(result, [i * .7 for i in range(18)])['status'], 'native_grid_missed')
        self.assertEqual(c.reference_check(c.generate(observations([]), [4, 8]), list(range(13)))['status'], 'native_grid_missed')
        self.assertEqual(c.reference_check(result, [5, 6, 7])['status'], 'reference_unavailable')
        self.assertEqual(result, old)

    def test_tick_budget_is_counted_not_silently_scored(self):
        result = dict(query_s=[4, 8], clocks=[{'knots': [[0, 0], [12, 1200]]}])
        scored = c.reference_check(result, list(range(13)))
        self.assertEqual(scored['tick_budget_exceeded_candidates'], 1)
        self.assertEqual(scored['status'], 'native_grid_missed')

    def test_projection_ignores_case_identity_labels_and_scores(self):
        base = dict(frame_count=601, frame_rate_hz=50, start_time_s=0,
                    observations=dict(duration_s=12, beats=[dict(time_s=t, confidence=.1) for t in range(13)]))
        result = audit.generate_capture(base)
        other = copy.deepcopy(base)
        other.update(case_id='different', truth=[123], bpm=999)
        for b in other['observations']['beats']:
            b['confidence'] = .99
        self.assertEqual(audit.generate_capture(other), result)


class ClockPacketTests(unittest.TestCase):
    def test_two_views_separate_provenance_candidates_and_scores(self):
        pack = bundle()
        p.validate_bundle(pack)
        p.verify_replay(pack, observations())
        self.assertNotIn('candidates', pack['first_view'])
        self.assertNotIn('provenance', pack['candidate_view'])
        self.assertTrue(all(set(c) == {'id', 'knots'} for c in pack['candidate_view']['candidates']))
        self.assertEqual(pack['private_ledger']['provenance']['training_review'], 'unreviewed')

    def test_randomized_order_reproducible_and_bound_to_identity(self):
        pack = bundle()
        self.assertEqual(pack, bundle())
        changed = copy.deepcopy(pack)
        changed['candidate_view']['candidates'].reverse()
        with self.assertRaises(ValueError):
            p.validate_bundle(changed)

    def test_changed_context_query_or_pcm_cannot_reuse_view_hash(self):
        for change in ('query', 'pcm', 'rate'):
            pack = bundle()
            if change == 'query':
                pack['first_view']['query_s'][0] += .1
            elif change == 'pcm':
                pack['first_view']['input']['pcm_sha256'] = 'd' * 64
            else:
                pack['candidate_view']['input']['sample_rate'] += 1
            with self.assertRaises(ValueError):
                p.validate_bundle(pack)

    def test_pcm_duration_and_incomplete_provenance_fail(self):
        for prov in (provenance() | {'extra': 'label'}, provenance() | {'work_id': ''},
                     provenance() | {'training_review': 'probably okay'}):
            with self.assertRaises(ValueError):
                p.provenance(prov)
        with self.assertRaises(ValueError):
            p.build(dict(pcm_sha256='b' * 64, sample_rate=50, sample_count=599), provenance(), observations(), [4, 8], 'c' * 64)

    def test_crop_relation_does_not_masquerade_as_natural_input_edge(self):
        binding = dict(pcm_sha256='b' * 64, sample_rate=50, sample_count=600)
        for relation in (dict(kind='complete', source_interval_s=[4, 16]), dict(kind='crop', source_interval_s=[4, 12])):
            with self.assertRaisesRegex(ValueError, 'crop relation'):
                p.build(binding, provenance() | {'input_relation': relation}, observations(), [4, 8], 'c' * 64)
        pack = p.build(binding, provenance() | {'input_relation': dict(kind='crop', source_interval_s=[4, 16])}, observations(), [4, 8], 'c' * 64)
        p.verify_replay(pack, observations())
        self.assertNotIn('input_relation', pack['candidate_view'])

    def test_replay_rejects_other_observations_even_if_geometry_matches(self):
        with self.assertRaisesRegex(ValueError, 'replay changed'):
            p.verify_replay(bundle(), observations() | {'beat_times_s': [t for t in range(13) if t != 1]})

    def test_multiple_supported_candidates_preserved(self):
        pack = bundle()
        first = first_records(pack)
        result = p.reconcile(pack, first, vote_records(pack, first))
        self.assertEqual(result['known'], 4)
        self.assertEqual(set(result['targets'].values()), {1})
        self.assertEqual(result['coverage_status'], 'supported_candidate')
        self.assertFalse(result['training_authorized'])

    def test_unknown_or_insufficient_context_is_not_negative(self):
        pack = bundle()
        first = first_records(pack)
        first[0]['context_sufficient'] = None
        result = p.reconcile(pack, first, vote_records(pack, first, 'contradicted'))
        self.assertEqual(set(result['targets'].values()), {None})
        self.assertEqual((result['known'], result['unknown'], result['coverage_status']), (0, 4, 'unassessed'))

    def test_disagreement_does_not_manufacture_generator_miss(self):
        pack = bundle()
        first = first_records(pack)
        votes = vote_records(pack, first, 'contradicted')
        votes[1]['labels'] = dict.fromkeys(votes[1]['labels'], 'supported')
        result = p.reconcile(pack, first, votes)
        self.assertEqual(set(result['targets'].values()), {None})
        self.assertEqual(result['coverage_status'], 'unassessed')

    def test_all_negative_with_followable_first_pass_is_generator_miss(self):
        pack = bundle()
        first = first_records(pack)
        result = p.reconcile(pack, first, vote_records(pack, first, 'contradicted'))
        self.assertEqual(result['coverage_status'], 'generator_miss')
        self.assertNotIn('rhythm_present', result)

    def test_empty_candidates_keep_followable_and_unassessed_cases(self):
        pack = bundle(observations([]))
        for followable, expected in ((True, 'generator_miss'), (False, 'unassessed')):
            first = first_records(pack, followable)
            result = p.reconcile(pack, first, vote_records(pack, first))
            self.assertEqual(result['coverage_status'], expected)
            self.assertEqual(result['targets'], {})

    def test_first_pass_cannot_be_replaced_after_candidate_votes(self):
        pack = bundle()
        first = first_records(pack)
        votes = vote_records(pack, first)
        first[0]['transition_intervals_s'] = [[5, 6]]
        with self.assertRaisesRegex(ValueError, 'original first pass'):
            p.reconcile(pack, first, votes)

    def test_duplicate_listener_and_cross_input_votes_fail(self):
        pack = bundle()
        first = first_records(pack)
        votes = vote_records(pack, first)
        votes[0]['view_sha256'] = 'd' * 64
        with self.assertRaises(ValueError):
            p.reconcile(pack, first, votes)
        first[1]['listener_id'] = first[0]['listener_id']
        with self.assertRaisesRegex(ValueError, 'duplicate listener'):
            p.reconcile(pack, first, vote_records(pack, first))

    def test_first_pass_followable_needs_own_clock_not_candidate_id(self):
        pack = bundle()
        record = first_records(pack)[0]
        record['accepted_clocks'] = []
        with self.assertRaisesRegex(ValueError, 'own clock'):
            p.first_pass(pack, record)

    def test_no_mutation_of_inputs_records_or_bundle(self):
        pack = bundle()
        first = first_records(pack)
        votes = vote_records(pack, first)
        before = copy.deepcopy((pack, first, votes))
        p.reconcile(pack, first, votes)
        self.assertEqual((pack, first, votes), before)


class ClockProposalReportTests(unittest.TestCase):
    def report(self):
        raw = (audit.HERE / 'clock-proposal-v1.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), 'd17f36dd8f3ea3d4c9ccc46eb78c41c3ab9ee11d97f22e2869e5d714582411cf')
        return json.loads(raw)

    def test_complete_frozen_identity_and_no_training_or_listening_claim(self):
        report = self.report()
        metadata = audit.exposure.metadata_projection(audit.exposure.read_json(
            audit.HERE / audit.exposure.SOURCE, audit.exposure.SOURCE_HASH))
        self.assertEqual([r['id'] for r in report['input_cases']], [r['id'] for g in metadata for r in g['cases']])
        self.assertEqual(report['capture_count'], 40)
        for key in ('neural_inference', 'training', 'holdout_access', 'production_change'):
            self.assertFalse(report[key])
        self.assertIsNone(report['independently_labeled_coverage'])
        self.assertEqual(report['new_annotations_collected'], 0)
        self.assertTrue(report['generation_completed_before_annotation_load'])

    def test_source_and_contract_bytes_remain_locked(self):
        report = self.report()
        lock = json.loads((audit.HERE / 'clock-proposal-lock-v1.json').read_bytes())
        self.assertEqual(report['contract'], lock)
        self.assertEqual(audit.sha(audit.HERE / 'clock-proposal-lock-v1.json'), report['lock_sha256'])
        for name, expected in lock['source_sha256'].items():
            self.assertEqual(audit.sha(audit.HERE / name), expected)
        self.assertEqual((lock['max_candidates'], lock['max_intervals']), (c.CAP, c.MAX_INTERVALS))
        self.assertEqual(lock['context_halo_seconds'], c.HALO_S)

    def test_all_query_denominators_conserved_across_tracks_and_slices(self):
        report = self.report()
        summary = report['summary']
        self.assertEqual(summary['queries'], 3348)
        for groups in (report['input_cases'], list(report['slices'].values())):
            self.assertEqual(sum(g['queries'] for g in groups), 3348)
            for field in ('reference_status', 'generator_status', 'candidate_counts'):
                totals = audit.Counter()
                for group in groups:
                    self.assertEqual(sum(group[field].values()), group['queries'])
                    totals.update(group[field])
                self.assertEqual(dict(totals), summary[field])

    def test_measured_failures_and_missing_edges_cannot_disappear(self):
        report = self.report()
        self.assertEqual(report['summary']['reference_status'], dict(native_grid_covered=484, native_grid_missed=2690, reference_unavailable=174))
        self.assertEqual(report['summary']['generator_status'], dict(ready=3167, unbracketed_query=181))
        self.assertEqual(report['slices']['step']['reference_status'], dict(native_grid_covered=5, native_grid_missed=19))
        self.assertEqual(report['summary']['full_readout_context_queries'], 3177)
        self.assertEqual(report['summary']['full_clock_context_queries'], 3070)


if __name__ == '__main__':
    unittest.main()
