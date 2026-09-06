import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import window_response_residual_audit as audit


def source(frames, name='raw', marks=None):
    return [dict(source=name, source_index=i, coordinate=audit.packets.rational.encode(Fraction(frame)),
                 score_frame=math.floor(frame + Fraction(1, 2)), beat_logit=4. if marks is None else marks[i],
                 downbeat_logit=-8., published={'unchanged': True}, plateau_index=i)
            for i, frame in enumerate(frames)]


def constant_times():
    return list(map(Fraction, range(0, 1001, 25)))


class WindowResidualControls(unittest.TestCase):
    def test_rubato_cache_without_optional_tags_reaches_full_track_inventory(self):
        case = {'id': 'authored-rubato', 'truth_times_s': [i / 50 for i in range(0, 901, 25)], 'truth_sha256': 'truth-hash'}
        record = {'capture_sha256': 'capture-hash'}
        captured = {'raw': source([450]), 'candidates': source([450], 'candidate')}
        prior = {'script_sha256': audit.dense.sha(Path(audit.packets.__file__).read_bytes()),
                 'lock_sha256': audit.dense.sha(audit.packets.LOCK_PATH.read_bytes()), 'helper_sha256': {},
                 'cohorts': [{'cohort': 'rubato', 'capture_summary_sha256': 'summary-hash',
                              'cases': [{'id': case['id'], 'source_inventory': {'authored': True}}]}]}
        inputs = {'evidence.json': {'cases': [case]}, 'backend-response-packets-v1.json': prior,
                  'summary.json': {}, 'rubato-calibration-v1.json': {'cases': [{'id': case['id'], 'input': {'truth': 'truth.json'}}]},
                  'authored-rubato.json': {'downbeat_logits': [-8.] * 900, 'observations': {'duration_s': 18.}}, 'truth.json': {}}
        with patch.object(audit.dense, 'INPUTS', {'rubato': (1, 'evidence-hash')}), \
             patch.object(audit.dense, 'read_json', side_effect=lambda path, expected=None: (inputs[Path(path).name], expected)), \
             patch.object(audit.dense, 'validate_summary', return_value=[record]), \
             patch.object(audit.dense, 'validate_capture', return_value=[-4.] * 900), \
             patch.object(audit.packets, 'assemble', return_value=captured), \
             patch.object(audit.packets, 'summarize', return_value={'authored': True}), \
             patch.object(audit.annotation, 'annotation_labels'):
            result = audit.audit('rubato', 'evidence.json', 'captures')
        self.assertTrue(result['complete'])
        self.assertEqual(result['all']['eligible_windows'], 1)
        self.assertEqual(result['by_regime']['rubato']['eligible_windows'], 1)
        self.assertEqual(result['all']['owned_responses'], {'raw': 1, 'candidate': 1})

    def test_prefix_only_clock_construction_keeps_both_half_phases(self):
        times = constant_times()
        grids, reason = audit.supplied_clocks(times, 400, 800)
        self.assertEqual(reason, 'eligible')
        self.assertEqual(grids['annotated'], grids['continuation'])
        self.assertEqual(grids['half_phase_zero'], list(map(Fraction, range(25, 400, 50))))
        self.assertEqual(grids['half_phase_one'], list(map(Fraction, range(0, 400, 50))))
        self.assertFalse(set(grids['half_phase_zero']) & set(grids['half_phase_one']))
        self.assertEqual(set(grids['continuation']), set(grids['half_phase_zero']) | set(grids['half_phase_one']))
        self.assertTrue(set(grids['continuation']) < set(grids['double']))
        altered = [t if t < 400 or t >= 800 else t + 1 for t in times]
        changed, _ = audit.supplied_clocks(altered, 400, 800)
        self.assertNotEqual(grids['annotated'], changed['annotated'])
        for name in audit.CLOCKS[1:]:
            self.assertEqual(grids[name], changed[name])  # no future response/annotation phase fitting

    def test_constant_density_and_doubled_tick_cost_remain_separate(self):
        frames = range(400, 800, 25)
        row = audit.analyze_window({'raw': source(frames), 'candidate': source(frames, 'candidate')}, constant_times(), 400, 800)
        self.assertEqual(row['status'], 'eligible')
        ledgers = row['inventories']['raw']
        self.assertEqual((ledgers['continuation']['matched'], ledgers['continuation']['unmatched_ticks']), (16, 0))
        for name in ('half_phase_zero', 'half_phase_one'):
            self.assertEqual((ledgers[name]['matched'], ledgers[name]['unmatched_responses']), (8, 8))
        self.assertEqual((ledgers['double']['matched'], ledgers['double']['unmatched_ticks']), (16, 16))
        self.assertEqual(row['comparisons']['raw']['double'], 'equal/fewer')

    def test_slowdown_and_omission_interpretations_are_not_scored_as_a_winner(self):
        times = list(map(Fraction, list(range(0, 400, 25)) + list(range(400, 1001, 50))))
        observed = {'raw': source(range(400, 800, 50)), 'candidate': source(range(400, 800, 50), 'candidate')}
        row = audit.analyze_window(observed, times, 400, 800)
        ledgers = row['inventories']['raw']
        self.assertEqual(ledgers['annotated']['matched'], 8)
        self.assertEqual(ledgers['continuation']['matched'], 8)
        self.assertEqual(ledgers['continuation']['unmatched_ticks'], 8)
        self.assertEqual(ledgers['half_phase_one']['unmatched_ticks'], 0)
        self.assertNotIn('selected_tempo', row)
        self.assertIsNone(audit.LOCK['selected_tempo'])

    def test_weak_subdivision_candidates_can_favor_double_coverage_without_being_beats(self):
        frames = [Fraction(400) + Fraction(25, 2) * i for i in range(32)]
        inputs = {'raw': source(range(400, 800, 25)),
                  'candidate': source(frames, 'candidate', [4. if i % 2 == 0 else -4. for i in range(32)])}
        row = audit.analyze_window(inputs, constant_times(), 400, 800)
        ledgers = row['inventories']['candidate']
        self.assertEqual(ledgers['double']['matched'], 32)
        self.assertEqual(ledgers['continuation']['matched'], 16)
        strata = ledgers['continuation']['response_strata']
        self.assertEqual(strata['never/interior/off_annotation/nonpositive'], 16)
        stronger = copy.deepcopy(inputs)
        for p in stronger['candidate']:
            p['beat_logit'] = 4.
        changed = audit.analyze_window(stronger, constant_times(), 400, 800)
        for name in audit.CLOCKS:
            for key in ('matched', 'unmatched_ticks', 'unmatched_responses', 'assignment_count'):
                self.assertEqual(ledgers[name][key], changed['inventories']['candidate'][name][key])

    def test_candidate_absence_competition_and_ambiguity_are_distinct(self):
        times = list(map(Fraction, [300, 325, 350, 375, 410, 414, 800]))
        for location, expected in ((411, {'raw_never_candidate_always': 1, 'raw_never_candidate_competition': 1}),
                                   (412, {'raw_never_candidate_sometimes': 2})):
            row = audit.analyze_window({'raw': [], 'candidate': source([location], 'candidate')}, times, 400, 800)
            self.assertEqual(row['annotated_tick_causes'], expected)
        absent = audit.analyze_window({'raw': [], 'candidate': []}, times, 400, 800)
        self.assertEqual(absent['annotated_tick_causes'], {'raw_never_no_candidate_in_radius': 2})
        edge_times = list(map(Fraction, [300, 325, 350, 375, 400, 800]))
        boundary = audit.analyze_window({'raw': [], 'candidate': source([400], 'candidate')}, edge_times, 400, 800)
        self.assertEqual(boundary['annotated_tick_causes'], {'raw_never_boundary_censored': 1})

    def test_miss_marks_keep_all_optimal_candidate_pairs_not_just_one_witness(self):
        times = list(map(Fraction, [300, 325, 350, 375, 410, 800]))
        for marks, expected in (([-4., -4.], 'nonpositive_only'), ([4., 4.], 'positive_only'), ([-4., 4.], 'mixed')):
            inputs = {'raw': [], 'candidate': source([409, 411], 'candidate', marks)}
            row = audit.analyze_window(inputs, times, 400, 800)
            self.assertEqual(row['annotated_tick_causes'], {'raw_never_candidate_always': 1})
            self.assertEqual(row['annotated_miss_candidate_marks'], {'always/' + expected: 1})
            self.assertEqual(row['inventories']['candidate']['annotated']['assignment_count'], 2)

    def test_half_open_ownership_preserves_fractional_identity_and_does_not_reextract(self):
        original = source([399, Fraction(799, 2), 400, Fraction(801, 2), 799, 800])
        untouched = copy.deepcopy(original)
        crops = [audit.crop_packets(original, a, b) for a, b in ((0, 400), (400, 800), (800, 900))]
        self.assertEqual([[p['source_index'] for p in c] for c in crops], [[0, 1], [2, 3, 4], [5]])
        self.assertEqual(crops[1][1]['coordinate'], [1, 2])
        self.assertEqual(crops[1][1]['published'], {'unchanged': True})
        self.assertEqual(original, untouched)
        crossing = source([Fraction(1599, 2)])
        row = audit.analyze_window({'raw': crossing, 'candidate': []}, constant_times(), 400, 800)
        self.assertEqual(row['status'], 'packet_support_crosses_window')
        self.assertEqual(row['owned_responses']['raw'], 1)
        self.assertNotIn('inventories', row)

    def test_shared_eligibility_keeps_partial_prefix_tail_and_budget_denominators(self):
        empty = {'raw': [], 'candidate': []}
        cases = [(empty, constant_times(), 0, 400, 'insufficient_prefix'),
                 (empty, constant_times(), 800, 900, 'partial_window'),
                 (empty, constant_times(), 800, 1200, 'annotation_tail_uncovered'),
                 ({'raw': [], 'candidate': source(range(400, 529), 'candidate')}, constant_times(), 400, 800, 'response_budget_exceeded'),
                 (empty, list(map(Fraction, range(1001))), 400, 800, 'clock_budget_exceeded')]
        for inputs, times, start, end, expected in cases:
            row = audit.analyze_window(inputs, times, start, end)
            self.assertEqual(row['status'], expected)
            self.assertNotIn('inventories', row)
        summary = audit.summarize([audit.analyze_window(empty, constant_times(), a, b) for a, b in ((0, 400), (400, 800), (800, 900))])
        self.assertEqual((summary['windows'], summary['frames'], summary['eligible_windows']), (3, 900, 1))
        self.assertEqual(sum(summary['status'].values()), 3)

    def test_annotation_quantization_and_complete_fractional_grids(self):
        found = audit.annotation_coordinates([0., 10.0000005, 10.1])
        self.assertEqual(found[1], Fraction(10000001, 20000))
        times = audit.annotation_coordinates([i * .333333 for i in range(61)])
        grids, status = audit.supplied_clocks(times, 400, 800)
        self.assertEqual(status, 'eligible')
        for ticks in grids.values():
            self.assertTrue(all(t.denominator <= 1000000 and 0 <= t < 400 for t in ticks))
        self.assertEqual(audit.periodic_ticks(Fraction(0), Fraction(3, 2), 4, 10), [Fraction(1, 2), Fraction(2), Fraction(7, 2), Fraction(5)])
        for bad in ([0, 0.0000001], [1, 0], [0, math.nan], [0, -1]):
            with self.assertRaises(ValueError):
                audit.annotation_coordinates(bad)

    def test_regime_requires_complete_prefix_and_window_annotation(self):
        truth = {'tempo_segments': [{'start_s': 0., 'end_s': 20., 'kind': 'constant'}], 'change_points': []}
        self.assertEqual(audit.context_regime(truth, constant_times(), 400, 800, 'artbeat', []), 'constant_context')
        truth['change_points'] = [{'time_s': 10.}]
        self.assertEqual(audit.context_regime(truth, constant_times(), 400, 800, 'artbeat', []), 'change_context')
        truth['change_points'] = []
        truth['tempo_segments'][0]['kind'] = 'ramp'
        self.assertEqual(audit.context_regime(truth, constant_times(), 400, 800, 'artbeat', []), 'ramp_context')
        self.assertEqual(audit.context_regime({}, constant_times(), 400, 800, 'rubato', []), 'rubato')
        truth['tempo_segments'][0]['end_s'] = 12.
        with self.assertRaises(ValueError):
            audit.context_regime(truth, constant_times(), 400, 800, 'artbeat', [])


class FrozenWindowResidualReport(unittest.TestCase):
    def test_report_identity_scope_and_helper_hashes(self):
        here = Path(__file__).parent
        report = json.loads((here / 'window-response-residuals-v1.json').read_text())
        self.assertEqual(audit.dense.sha((here / 'window-response-residuals-v1.json').read_bytes()),
                         '35eb07cfc8c96471710ecf0a40d3fae6da836638845090cc21583ba337769a8a')
        self.assertEqual(report['contract'], audit.LOCK)
        self.assertEqual(report['script_sha256'], audit.dense.sha(Path(audit.__file__).read_bytes()))
        self.assertEqual(report['lock_sha256'], audit.dense.sha(audit.LOCK_PATH.read_bytes()))
        for name, value in (report['helper_sha256'] | report['predecessor_sha256']).items():
            self.assertEqual(value, audit.dense.sha((here / name).read_bytes()))
        self.assertTrue(report['truth_assisted'])
        self.assertIsNone(report['selected_tempo'])
        for key in ('phase_search', 'neural_inference', 'fitted_mapping', 'holdout_opened', 'training_run',
                    'production_output_changed', 'new_user_parameters', 'accuracy_improvement_claimed'):
            self.assertIs(report[key], False)
        for private in ('"time_s"', '"coordinate"', '"published"', '"beat_logits"', '"downbeat_logits"', 'C:/', 'D:/'):
            self.assertNotIn(private, json.dumps(report))

    def test_all_track_ownership_and_shared_denominators(self):
        here = Path(__file__).parent
        report = json.loads((here / 'window-response-residuals-v1.json').read_text())
        prior = json.loads((here / 'backend-response-packets-v1.json').read_text())
        self.assertEqual(len(report['cohorts']), 2)
        for cohort, old in zip(report['cohorts'], prior['cohorts']):
            self.assertTrue(cohort['complete'])
            self.assertEqual([(c['id'], c['capture_sha256'], c['frame_count']) for c in cohort['cases']],
                             [(c['id'], c['capture_sha256'], c['frame_count']) for c in old['cases']])
            for field in ('cohort', 'frozen_evidence_sha256', 'capture_summary_sha256', 'source_hashes'):
                self.assertEqual(cohort[field], old[field])
            overall = cohort['all']
            expected = {'artbeat': (39, 9, 9, 99, 468, 134, 93, 129),
                        'rubato': (824, 766, 25, 8998, 32819, 6435, 3792, 4948)}[cohort['cohort']]
            self.assertEqual((overall['windows'], overall['eligible_windows'],
                              sum(c['inventory']['eligible_windows'] > 0 for c in cohort['cases']),
                              overall['eligible_responses']['raw'], overall['eligible_responses']['candidate'],
                              overall['ledgers']['raw']['annotated']['ticks'], overall['ledgers']['raw']['annotated']['matched'],
                              overall['ledgers']['candidate']['annotated']['matched']), expected)
            self.assertEqual(cohort['by_regime']['constant_context']['eligible_windows'], 0)
            self.assertEqual(cohort['by_regime']['ramp_context']['eligible_windows'], 0)
            expected_causes = {'artbeat': {'raw_always': 93, 'raw_never_boundary_censored': 1,
                                           'raw_never_candidate_always': 36, 'raw_never_no_candidate_in_radius': 4},
                               'rubato': {'raw_always': 3792, 'raw_never_boundary_censored': 42,
                                          'raw_never_candidate_always': 1143, 'raw_never_candidate_competition': 3,
                                          'raw_never_no_candidate_in_radius': 1455}}
            self.assertEqual(overall['annotated_tick_causes'], expected_causes[cohort['cohort']])
            self.assertEqual(overall['annotated_miss_candidate_marks'],
                             {'always/nonpositive_only': 36} if cohort['cohort'] == 'artbeat' else
                             {'always/nonpositive_only': 1064, 'always/positive_only': 79})
            self.assertEqual(overall['frames'], old['total_frames_per_head'])
            self.assertEqual(overall['owned_responses'], {'raw': old['totals']['raw_events'], 'candidate': old['totals']['candidates']})
            for field in ('windows', 'frames', 'eligible_windows'):
                self.assertEqual(overall[field], sum(c['inventory'][field] for c in cohort['cases']))
            self.assertEqual(overall['eligible_windows'], sum(g['eligible_windows'] for g in cohort['by_regime'].values()))
            for summary in [overall] + list(cohort['by_regime'].values()) + [c['inventory'] for c in cohort['cases']]:
                self.assertEqual(summary['windows'], sum(summary['status'].values()))
                self.assertEqual(sum(summary['annotated_tick_causes'].values()), summary['ledgers']['raw']['annotated']['ticks'])
                self.assertEqual(sum(summary['annotated_miss_candidate_marks'].values()),
                                 sum(summary['annotated_tick_causes'].get('raw_never_candidate_' + s, 0) for s in ('always', 'sometimes')))
                for s in audit.SOURCES:
                    for name in audit.CLOCKS:
                        ledger = summary['ledgers'][s][name]
                        self.assertEqual(ledger['responses'], summary['eligible_responses'][s])
                        self.assertEqual(ledger['ticks'], ledger['matched'] + ledger['unmatched_ticks'])
                        self.assertEqual(ledger['responses'], ledger['matched'] + ledger['unmatched_responses'])
                        self.assertLessEqual(ledger['max_states'] or 0, 16641)
                        if 'tick_strata' in ledger:
                            self.assertEqual(sum(ledger['tick_strata'].values()), ledger['ticks'])
                            self.assertEqual(sum(ledger['response_strata'].values()), ledger['responses'])
                    if 'continuation_comparisons' in summary:
                        for counts in summary['continuation_comparisons'][s].values():
                            self.assertEqual(sum(counts.values()), summary['eligible_windows'])


if __name__ == '__main__':
    unittest.main()
