import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import paired_response_replay_audit as audit


def source(frames, name='raw', marks=None):
    return [dict(source=name, source_index=i, coordinate=audit.packets.rational.encode(Fraction(t)),
                 score_frame=math.floor(t + Fraction(1, 2)), beat_logit=4. if marks is None else marks[i],
                 downbeat_logit=-8., published={'authored_identity': i}, plateau_index=i)
            for i, t in enumerate(frames)]


def pair():
    return dict(constant_start_frame=200, change_start_frame=600, window_frames=200, pre_segment_index=0, change_index=0)


def constant_times():
    return list(map(Fraction, range(0, 1001, 25)))


def step_times():
    return list(map(Fraction, list(range(0, 700, 25)) + list(range(700, 1001, 50))))


def packets(times):
    return {'raw': source(times), 'candidate': source(times, 'candidate')}


class PairedReplayControls(unittest.TestCase):
    def test_exact_step_exposes_unexplained_continuation_ticks_not_a_winner_score(self):
        result = audit.analyze_pair(packets(step_times()), step_times(), pair())
        self.assertEqual(result['status'], 'eligible')
        for s in audit.SOURCES:
            constant = result['windows']['constant']['versus_continuation'][s]['annotated']
            step = result['windows']['change']['versus_continuation'][s]['annotated']
            self.assertEqual(constant, dict(matched_delta=0, unmatched_ticks_delta=0, relation='equal'))
            self.assertEqual(step, dict(matched_delta=0, unmatched_ticks_delta=-2, relation='left_dominates'))
        self.assertEqual(audit.summarize_pairs([result])['paired_relations']['raw']['annotated'], {'equal/left_dominates': 1})
        self.assertNotIn('selected_tempo', result)

    def test_same_observations_allow_omission_and_true_change_interpretations(self):
        observations = packets(step_times())
        omitted = audit.analyze_pair(observations, constant_times(), pair())
        changed = audit.analyze_pair(observations, step_times(), pair())
        self.assertEqual(omitted['windows']['change']['owned_responses'], changed['windows']['change']['owned_responses'])
        self.assertEqual(omitted['windows']['change']['versus_continuation']['raw']['annotated']['relation'], 'equal')
        self.assertEqual(changed['windows']['change']['versus_continuation']['raw']['annotated']['relation'], 'left_dominates')
        self.assertEqual(omitted['windows']['change']['annotated_tick_causes']['raw_never_no_candidate_in_radius'], 2)

    def test_weak_candidate_support_does_not_erase_off_annotation_background(self):
        observations = packets(step_times())
        times = constant_times()
        candidates = sorted(times + [Fraction(612), Fraction(662), Fraction(712), Fraction(762)])
        observations['candidate'] = source(candidates, 'candidate', [-4.] * len(candidates))
        result = audit.analyze_pair(observations, times, pair())['windows']['change']
        self.assertEqual(result['annotated_tick_causes']['raw_never_candidate_always'], 2)
        self.assertEqual(result['annotated_miss_candidate_marks']['always/nonpositive_only'], 2)
        inventory = result['inventories']['candidate']['annotated']
        self.assertEqual(inventory['response_strata']['never/interior/off_annotation/nonpositive'], 4)

    def test_any_member_budget_rejection_suppresses_both_ledgers_and_keeps_ownership(self):
        observations = packets(constant_times())
        observations['candidate'] = source([Fraction(600 + i) for i in range(129)], 'candidate')
        with patch.object(audit.packets, 'packet_ledger', side_effect=AssertionError('must not run an orphan side')):
            result = audit.analyze_pair(observations, constant_times(), pair())
        self.assertEqual(result['status'], 'pair_rejected')
        self.assertEqual(result['windows']['constant']['status'], 'eligible')
        self.assertEqual(result['windows']['change']['status'], 'response_budget_exceeded')
        self.assertTrue(all('inventories' not in w for w in result['windows'].values()))
        summary = audit.summarize_pairs([result])
        self.assertEqual((summary['selected_pairs'], summary['eligible_pairs'], summary['compared_windows']), (1, 0, 0))
        self.assertEqual(summary['all_selected_owned_responses']['candidate'], 129)
        self.assertEqual(summary['by_context']['constant']['windows'], 0)

    def test_fractional_right_edge_support_rejects_entire_pair_without_moving_packet(self):
        observations = packets(constant_times())
        observations['raw'] = source([Fraction(799, 2)])
        before = copy.deepcopy(observations)
        result = audit.analyze_pair(observations, constant_times(), pair())
        self.assertEqual(result['windows']['constant']['status'], 'packet_support_crosses_window')
        self.assertEqual(result['status'], 'pair_rejected')
        self.assertEqual(observations, before)
        self.assertEqual(result['windows']['constant']['owned_responses']['raw'], 1)

    def test_half_open_ownership_and_original_mark_identity_are_preserved(self):
        observations = packets([Fraction(200), Fraction(399), Fraction(400), Fraction(600), Fraction(799), Fraction(800)])
        prepared = audit.prepare_window(observations, constant_times(), 200, 400)
        self.assertEqual([p['source_index'] for p in prepared[1]['raw']], [0, 1])
        self.assertEqual([p['coordinate'] for p in prepared[1]['raw']], [[0, 1], [199, 1]])
        self.assertEqual(prepared[1]['raw'][1]['published'], observations['raw'][1]['published'])
        self.assertEqual(prepared[1]['raw'][1]['beat_logit'], observations['raw'][1]['beat_logit'])

    def test_no_phase_refit_to_future_annotations_or_responses(self):
        times = constant_times()
        changed = [t + Fraction(1, 2) if 600 <= t < 800 else t for t in times]
        left = audit.prepare_window(packets(times), times, 600, 800)[2]
        right = audit.prepare_window(packets(changed), changed, 600, 800)[2]
        self.assertNotEqual(left['annotated'], right['annotated'])
        for name in audit.CLOCKS[1:]:
            self.assertEqual(left[name], right[name])
        self.assertFalse(set(left['half_phase_zero']) & set(left['half_phase_one']))
        self.assertEqual(set(left['continuation']), set(left['half_phase_zero']) | set(left['half_phase_one']))

    def test_prefix_tail_and_clock_rejections_are_not_truncated(self):
        observations = packets(constant_times())
        self.assertEqual(audit.prepare_window(observations, constant_times(), 0, 200)[0]['status'], 'insufficient_prefix')
        self.assertEqual(audit.prepare_window(observations, constant_times(), 900, 1100)[0]['status'], 'annotation_tail_uncovered')
        self.assertEqual(audit.prepare_window(observations, list(map(Fraction, range(1001))), 200, 400)[0]['status'], 'clock_budget_exceeded')
        for start, end in ((200, 400.0), (200, 600), (-1, 199)):
            with self.assertRaises(ValueError):
                audit.prepare_window(observations, constant_times(), start, end)

    def test_overlap_and_duplicate_source_identity_rejected(self):
        invalid = pair() | {'change_start_frame': 399}
        with self.assertRaisesRegex(ValueError, 'overlap'):
            audit.analyze_pair(packets(constant_times()), constant_times(), invalid)
        observations = packets(constant_times())
        observations['raw'][24]['source_index'] = 8  # same identity in opposite members
        with self.assertRaisesRegex(ValueError, 'duplicate source identity'):
            audit.analyze_pair(observations, constant_times(), pair())

    def test_contrast_reports_both_axes_and_all_relations(self):
        reference = dict(matched=3, unmatched_ticks=3)
        for values, expected in (((3, 3), 'equal'), ((4, 2), 'left_dominates'), ((3, 2), 'left_dominates'),
                                 ((2, 4), 'right_dominates'), ((4, 4), 'tradeoff'), ((2, 2), 'tradeoff')):
            result = audit.contrast(dict(matched=values[0], unmatched_ticks=values[1]), reference)
            self.assertEqual(result['relation'], expected)
            self.assertEqual(result['matched_delta'], values[0] - 3)
            self.assertEqual(result['unmatched_ticks_delta'], values[1] - 3)

    def test_all_optimal_assignments_and_censored_queries_remain_visible(self):
        # Symmetric integer responses compete equally for the same annotated tick.
        times = constant_times()
        observations = packets([Fraction(249), Fraction(251)])
        prepared = audit.prepare_window(observations, times, 200, 400)
        result = audit.evaluate_window(*prepared)
        inventory = result['inventories']['raw']['annotated']
        self.assertEqual(inventory['assignment_count'], 2)
        self.assertEqual(inventory['tick_strata']['never/boundary'], 1)
        self.assertEqual(inventory['response_strata']['sometimes/interior/near_annotation/positive'], 2)
        self.assertEqual(result['annotated_tick_causes']['raw_never_boundary_censored'], 1)

    def test_reused_evaluator_is_exactly_the_frozen_eight_second_inventory(self):
        times = list(map(Fraction, range(0, 1601, 25)))
        observations = packets([t for t in times if t != 600])
        prior = audit.residual.analyze_window(observations, times, 400, 800)
        cropped = {s: audit.residual.crop_packets(observations[s], 400, 800) for s in audit.SOURCES}
        grids, status = audit.residual.supplied_clocks(times, 400, 800)
        row = dict(frames=400, status=status, owned_responses={s: len(cropped[s]) for s in audit.SOURCES})
        result = audit.evaluate_window(row, cropped, grids)
        result.pop('versus_continuation')
        self.assertEqual(result, prior)

    def test_bad_public_plan_fails_before_any_private_input_access(self):
        with patch.object(audit, 'verified_helpers', return_value=({}, {})), \
             patch.object(audit, 'verified_plan', side_effect=ValueError('changed plan')), \
             patch.object(audit.dense, 'read_json') as read:
            with self.assertRaisesRegex(ValueError, 'changed plan'):
                audit.build_report('private-evidence', 'private-captures')
        read.assert_not_called()

    def test_frozen_public_plan_and_helper_identities_reproduce(self):
        helpers, _ = audit.verified_helpers()
        metadata, cases, plan = audit.verified_plan()
        self.assertEqual(len(cases), 40)
        self.assertEqual([len(c['cases']) for c in metadata], [15, 25])
        self.assertEqual(len(plan), 7)
        self.assertEqual(audit.exposure.canonical_hash(plan), audit.LOCK['selected_plan_sha256'])
        self.assertIn('rational_response_ledger.py', helpers)
        for p in plan:
            self.assertLessEqual(p['constant_start_frame'] + 200, p['change_start_frame'])


    def test_whole_report_reads_only_selected_capture_and_retains_unselected_inputs(self):
        cases = [dict(cohort=cohort, identity=dict(id=f'{cohort}-{i}', frame_count=1001,
                     truth_sha256=f'truth-{i}', capture_sha256=f'capture-{i}'),
                     data=dict(beats=[int(t * 20000) for t in constant_times()]))
                 for cohort, count in (('artbeat', 15), ('rubato', 25)) for i in range(count)]
        metadata = [dict(cohort=cohort, suite_sha256='authored', cases=[c['identity'] for c in cases if c['cohort'] == cohort])
                    for cohort in ('artbeat', 'rubato')]
        evidence = {'cases': [dict(id=c['identity']['id'], truth_sha256=c['identity']['truth_sha256'],
                                  truth_times_s=[float(t / 50) for t in constant_times()]) for c in cases[:15]]}
        records = [dict(capture_sha256=c['identity']['capture_sha256']) for c in cases[:15]]
        old = dict(cohort='artbeat', capture_summary_sha256='summary', cases=[dict(id=c['identity']['id'],
                   source_inventory={'authored': True}) for c in cases[:15]])
        selected = dict(cohort='artbeat', id='artbeat-0', **pair())
        inputs = {'evidence.json': evidence, 'summary.json': {}, 'artbeat-0.json':
                  dict(downbeat_logits=[], observations={'duration_s': 20})}
        assembled = dict(raw=source(constant_times()), candidates=source(constant_times(), 'candidate'))
        with patch.object(audit, 'verified_helpers', return_value=({}, {'cohorts': [old]})), \
             patch.object(audit, 'verified_plan', return_value=(metadata, cases, [selected])), \
             patch.object(audit.dense, 'read_json', side_effect=lambda p, h: (inputs[Path(p).name], h)) as reads, \
             patch.object(audit.dense, 'validate_summary', return_value=records), \
             patch.object(audit.dense, 'validate_capture', return_value=[0.] * 1001), \
             patch.object(audit.packets, 'assemble', return_value=assembled), \
             patch.object(audit.packets, 'summarize', return_value={'authored': True}):
            report = audit.build_report('evidence.json', 'captures')
        self.assertEqual([Path(c.args[0]).name for c in reads.call_args_list], ['evidence.json', 'summary.json', 'artbeat-0.json'])
        self.assertEqual(len(report['input_cases']), 40)
        self.assertEqual(sum(c['private_capture_read'] for c in report['input_cases']), 1)
        self.assertEqual(report['private_capture_count'], 1)
        self.assertEqual(report['summary']['eligible_pairs'], 1)


class FrozenPairedReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = audit.HERE / 'paired-response-replay-v1.json'
        cls.report = json.loads(cls.path.read_bytes())

    def test_report_hashes_plan_and_all_input_identities_are_frozen(self):
        report = self.report
        self.assertEqual(audit.dense.sha(self.path.read_bytes()), '57d3005924fd6cf64b8b65632e77e1b2baff9e162dc6e4f9471fee14d60312f3')
        self.assertEqual(report['script_sha256'], audit.dense.sha(Path(audit.__file__).read_bytes()))
        self.assertEqual(report['lock_sha256'], audit.dense.sha(audit.LOCK_PATH.read_bytes()))
        self.assertEqual(report['helper_sha256'], audit.verified_helpers()[0])
        metadata, cases, plan = audit.verified_plan()
        self.assertEqual(report['metadata_projection_sha256'], audit.exposure.canonical_hash(metadata))
        self.assertEqual([c['id'] for c in report['input_cases']], [c['identity']['id'] for c in cases])
        self.assertEqual([p['id'] for p in report['pairs']], [p['id'] for p in plan])
        for actual, selected in zip(report['pairs'], plan):
            self.assertEqual(actual['pair_sha256'], audit.exposure.canonical_hash({k: v for k, v in selected.items() if k not in ('id', 'cohort')}))
        self.assertEqual(sum(c['private_capture_read'] for c in report['input_cases']), 7)
        self.assertEqual(report['private_capture_count'], 7)

    def test_pair_aggregation_and_each_ledger_conserve_responses_and_ticks(self):
        report = self.report
        self.assertEqual(report['summary'], audit.summarize_pairs(report['pairs']))
        self.assertEqual((report['summary']['selected_pairs'], report['summary']['eligible_pairs'], report['summary']['compared_windows']), (7, 7, 14))
        self.assertEqual(report['summary']['all_selected_owned_responses'], {'raw': 91, 'candidate': 340})
        for pair_row in report['pairs']:
            for s in audit.SOURCES:
                self.assertEqual(sum(pair_row['windows'][r]['owned_responses'][s] for r in audit.ROLES) + pair_row['outside_selected_windows'][s],
                                 pair_row['whole_capture_responses'][s])
            for window in pair_row['windows'].values():
                for s in audit.SOURCES:
                    for name, ledger in window['inventories'][s].items():
                        self.assertEqual(ledger['matched'] + ledger['unmatched_ticks'], ledger['ticks'])
                        self.assertEqual(ledger['matched'] + ledger['unmatched_responses'], ledger['responses'])
                        self.assertEqual(sum(ledger['tick_strata'].values()), ledger['ticks'])
                        self.assertEqual(sum(ledger['response_strata'].values()), ledger['responses'])
                        self.assertEqual(ledger['responses'], window['owned_responses'][s])
                        self.assertLessEqual(ledger['states'], 16641)
                        if name != 'continuation':
                            self.assertEqual(window['versus_continuation'][s][name], audit.contrast(ledger, window['inventories'][s]['continuation']))
                self.assertEqual(sum(window['annotated_tick_causes'].values()), window['inventories']['raw']['annotated']['ticks'])

    def test_constant_misses_competing_background_and_bad_step_comparisons_remain(self):
        summary = self.report['summary']
        for role, supported, absent, background in (('constant', 10, 0, 117), ('change', 10, 2, 102)):
            context = summary['by_context'][role]
            self.assertEqual(context['annotated_tick_causes']['raw_never_candidate_always'], supported)
            self.assertEqual(context['annotated_tick_causes'].get('raw_never_no_candidate_in_radius', 0), absent)
            self.assertEqual(context['annotated_miss_candidate_marks']['always/nonpositive_only'], supported)
            self.assertEqual(context['ledgers']['candidate']['annotated']['response_strata']['never/interior/off_annotation/nonpositive'], background)
        raw = summary['paired_relations']['raw']['annotated']
        self.assertEqual(sum(v for k, v in raw.items() if k.startswith('left_dominates/')), 3)
        self.assertEqual(sum(v for k, v in raw.items() if k.endswith('/right_dominates')), 2)
        candidate = summary['paired_relations']['candidate']['double']
        self.assertEqual(sum(v for k, v in candidate.items() if k.startswith('left_dominates/')), 4)

    def test_public_report_is_inventory_not_events_scores_or_a_decoder(self):
        report = self.report
        for key in ('source_pooling', 'phase_search', 'neural_inference', 'decoder_replayed', 'fitted_mapping',
                    'holdout_opened', 'training_run', 'production_output_changed', 'new_user_parameters', 'accuracy_improvement_claimed'):
            self.assertIs(report[key], False)
        self.assertIsNone(report['selected_tempo'])
        self.assertTrue(report['truth_assisted'])
        self.assertTrue(report['observation_ledger_replayed'])
        serialized = json.dumps(report)
        for forbidden in ('"beat_logits"', '"downbeat_logits"', '"coordinate"', '"published"',
                          '"constant_start_frame"', '"change_start_frame"', 'D:/', 'C:/', '/home/'):
            self.assertNotIn(forbidden, serialized)


if __name__ == '__main__':
    unittest.main()
