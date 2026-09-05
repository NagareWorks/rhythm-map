import json
import math
from pathlib import Path
import unittest

import metrical_window_audit as audit


def fixture(times=None):
    times = [.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5] if times is None else times
    truth = dict(id='authored', beats=[dict(time_s=t, downbeat=i % 3 == 0) for i, t in enumerate(times)],
                 tempo_segments=[dict(start_s=0., end_s=5., kind='constant')], change_points=[])
    case = dict(id='authored', truth_times_s=times, tags=[], beat_tolerance_s=.07,
                raw_truth_pairs=[[0, 0]], observations=dict(beats=[dict(time_s=times[0])], beat_candidates=[]))
    return truth, case


def rows(cohort='rubato'):
    truth, case = fixture()
    b, d = [-4.] * 251, [-4.] * 251
    for i, t in enumerate(case['truth_times_s']):
        b[round(t * 50) + 1] = 4.
        if i % 3 == 0:
            d[round(t * 50) + 2] = 3.
    return audit.rows_for_case(b, d, truth, case, cohort)


class MetricalWindowControls(unittest.TestCase):
    def test_joint_readout_does_not_combine_separate_peaks(self):
        b, d = [-4.] * 51, [-4.] * 51
        b[24], d[27] = 4., 6.
        f, reason = audit.window(b, d, 25)
        self.assertEqual(reason, 'available')
        self.assertEqual((f['beat_peak'], f['downbeat_at_beat_peak'], f['downbeat_peak']), (4., -4., 6.))
        self.assertEqual((f['peak_separation_frames'], f['beat_peak_gain']), (3, 8.))
        b[26] = 4.
        d[24] = 2.
        self.assertEqual(audit.window(b, d, 25)[0]['downbeat_at_beat_peak'], 2.)  # equal distance -> earlier
        b[25], d[25] = 4., 1.
        self.assertEqual(audit.window(b, d, 25)[0]['downbeat_at_beat_peak'], 1.)  # nearest center

    def test_missing_windows_are_not_clipped_padded_or_absence(self):
        self.assertEqual(audit.window([0.] * 7, [0.] * 7, 2), (None, 'out_of_capture'))
        mask = [True] * 7
        mask[3] = False
        self.assertEqual(audit.window([0.] * 7, [0.] * 7, 3, mask), (None, 'unavailable_frame'))
        flat, _ = audit.window([0.] * 7, [0.] * 7, 3)
        self.assertEqual(flat['beat_peak'], 0.)
        self.assertEqual(flat['peak_separation_frames'], 0)

    def test_fixed_quantization_and_overlap_exclusions(self):
        truth, case = fixture([.21, .41, .81])  # rounded centers 10,20,40; ties to even
        b, d = [-4.] * 60, [-4.] * 60
        b[10] = 3.
        found = audit.rows_for_case(b, d, truth, case, 'artbeat')
        self.assertEqual(found[0]['features']['beat_peak_gain'], 0.)
        self.assertEqual(found[0]['pair_status'], 'control_overlaps_annotation')
        self.assertEqual(found[1]['pair_status'], 'eligible')
        self.assertEqual(found[2]['pair_status'], 'final_beat')
        self.assertTrue(audit.overlaps(10, [10, 10, 10, 10], excluded=0))
        self.assertTrue(audit.overlaps(16, [10]))
        self.assertFalse(audit.overlaps(17, [10]))
        truth, case = fixture([.20, .30, .9])
        found = audit.rows_for_case(b, d, truth, case, 'artbeat')
        self.assertEqual(found[0]['pair_status'], 'canonical_overlaps_neighbor')

    def test_annotation_absence_is_not_negative_downbeat(self):
        truth, case = fixture()
        truth['beats'] = [dict(b, downbeat=False) for b in truth['beats']]
        self.assertEqual(set(audit.annotation_labels(truth, case, 'artbeat')[0]), {'downbeat_unknown'})
        self.assertEqual(set(audit.annotation_labels(truth, case, 'rubato')[0]), {'non_downbeat'})
        ranking = audit.downbeat_ranking(rows('artbeat'))
        self.assertEqual((ranking['positive_windows'], ranking['negative_windows'], ranking['unknown_label_queries']), (0, 0, 9))
        self.assertEqual(set(ranking['auc'].values()), {None})

    def test_change_geometry_is_not_a_model_or_tempo_fit(self):
        truth, case = fixture()
        truth['change_points'] = [dict(time_s=2.5, kind='tempo_jump')]
        regimes = audit.annotation_labels(truth, case, 'artbeat')[1]
        self.assertEqual(regimes[2:7], ['annotated_change_neighborhood'] * 5)
        self.assertEqual([regimes[i] for i in (0, 1, 7, 8)], ['annotated_constant_interior'] * 4)
        self.assertEqual(set(audit.annotation_labels(truth, case, 'rubato')[1]), {'rubato'})
        case['tags'] = ['rubato']
        self.assertEqual(set(audit.annotation_labels(truth, case, 'artbeat')[1]), {'rubato'})
        case['tags'], truth['change_points'] = [], []
        truth['tempo_segments'][0]['kind'] = 'ramp'
        self.assertEqual(set(audit.annotation_labels(truth, case, 'artbeat')[1]), {'annotated_ramp'})

    def test_same_observations_cannot_resolve_omissions_versus_slowdown(self):
        # Authored controls only: identical alternating pulse evidence can be
        # labeled a constant clock with omissions or a slow clock without them.
        b = [-4.] * 251
        for t in (25, 50, 75, 125, 175, 225):
            b[t] = 4.
        constant_truth, constant = fixture()
        slow_truth, slow = fixture([.5, 1., 1.5, 2.5, 3.5, 4.5])
        slow_truth['change_points'] = [dict(time_s=1.5, kind='tempo_jump')]
        const_rows = audit.rows_for_case(b, b, constant_truth, constant, 'artbeat')
        slow_rows = audit.rows_for_case(b, b, slow_truth, slow, 'artbeat')
        shared = [const_rows[i] for i in (0, 1, 2, 4, 6, 8)]
        self.assertEqual([r['features'] for r in shared], [r['features'] for r in slow_rows])
        self.assertNotEqual([r['regime'] for r in shared], [r['regime'] for r in slow_rows])
        self.assertEqual(audit.summarize(const_rows)['acoustic_presence_unknown'], 9)

    def test_larger_window_is_not_automatic_recovery(self):
        result = rows()
        summary = audit.summarize(result)
        self.assertEqual(summary['raw_missed'], 8)
        self.assertEqual(summary['center_nonpositive_peak_positive'], 9)
        self.assertEqual(summary['comparison']['eligible'], 8)
        self.assertEqual(summary['comparison']['wins'], 8)
        self.assertEqual(summary['acoustic_presence_unknown'], 9)
        self.assertEqual(summary['label_counts'], {'downbeat': 3, 'non_downbeat': 6})
        ranking = audit.downbeat_ranking(result)
        self.assertEqual(ranking['auc']['downbeat_peak'], 1.)
        self.assertEqual(ranking['auc']['downbeat_at_beat_peak'], .5)

    def test_incomplete_queries_keep_denominators_and_nulls(self):
        truth, case = fixture([0., .5, 1.])
        result = audit.rows_for_case([-4.] * 51, [-4.] * 51, truth, case, 'artbeat')
        summary = audit.summarize(result)
        self.assertEqual((summary['queries'], summary['full_windows']), (3, 1))
        self.assertEqual(summary['window_status']['out_of_capture'], 2)
        self.assertEqual(summary['comparison']['eligible'], 1)
        self.assertEqual(summary['comparison']['ties'], 1)
        self.assertIsNone(audit.summarize([])['comparison']['win_fraction'])
        self.assertIsNone(audit.summarize([])['feature_quantiles']['beat_peak'])
        self.assertEqual(audit.quantiles([0., 10.]), [1., 5., 9.])

    def test_invalid_inputs_and_changed_truth_matching_fail_closed(self):
        for b, d, center in (([0.] * 7, [0.] * 6, 3), ([math.inf] * 7, [0.] * 7, 3),
                              ([0.] * 7, [0.] * 7, 3.)):
            with self.assertRaises(ValueError):
                audit.window(b, d, center)
        truth, case = fixture()
        case['raw_truth_pairs'] = [[0, 1]]
        with self.assertRaisesRegex(ValueError, 'matching changed'):
            audit.rows_for_case([0.] * 251, [0.] * 251, truth, case, 'artbeat')
        truth['beats'][0]['time_s'] = .6
        with self.assertRaisesRegex(ValueError, 'truth identity'):
            audit.annotation_labels(truth, case, 'artbeat')


class MetricalWindowFrozenReport(unittest.TestCase):
    def test_frozen_contract_provenance_and_scope(self):
        here = Path(__file__).parent
        report = json.loads((here / 'metrical-window-v1.json').read_text())
        self.assertEqual(report['contract'], audit.LOCK)
        self.assertEqual(audit.LOCK['radius_frames'], 3)
        self.assertEqual(audit.LOCK['frame_rate_hz'], 50)
        self.assertEqual(audit.LOCK['feature_quantiles'], [.1, .5, .9])
        self.assertEqual(audit.LOCK['change_neighborhood_beats_each_side'], 2)
        self.assertEqual(report['script_sha256'], audit.dense.sha(Path(audit.__file__).read_bytes()))
        self.assertEqual(report['lock_sha256'], audit.dense.sha(audit.LOCK_PATH.read_bytes()))
        self.assertEqual(report['semantics_source_sha256'], audit.dense.sha((here / 'beat-this-semantics-source-v1.json').read_bytes()))
        self.assertEqual(report['dense_report_sha256'], audit.dense.sha((here / 'dense-clock-evidence-v1.json').read_bytes()))
        for name, identity in report['helper_sha256'].items():
            self.assertEqual(identity, audit.dense.sha((here / name).read_bytes()))
        self.assertTrue(report['truth_assisted'])
        for key in ('holdout_opened', 'training_run', 'neural_inference', 'fitted_mapping', 'decoder_replayed',
                    'production_observations_changed', 'accuracy_improvement_claimed'):
            self.assertIs(report[key], False)

    def test_all_tracks_truth_hashes_and_every_denominator(self):
        here = Path(__file__).parent
        report = json.loads((here / 'metrical-window-v1.json').read_text())
        old = json.loads((here / 'dense-clock-evidence-v1.json').read_text())
        self.assertEqual(len(report['cohorts']), 2)
        for cohort, prior in zip(report['cohorts'], old['cohorts']):
            name = cohort['cohort']
            suite_path = audit.ROOT / 'evaluation/suites' / f'{audit.dense.SUITES[name][0]}.json'
            suite = json.loads(suite_path.read_text())
            for key in ('cohort', 'frozen_evidence_sha256', 'capture_summary_sha256', 'source_hashes', 'total_frames_per_head'):
                self.assertEqual(cohort[key], prior[key])
            self.assertEqual(cohort['suite_sha256'], audit.dense.sha(suite_path.read_bytes()))
            self.assertEqual([(t['id'], t['capture_sha256']) for t in cohort['cases']],
                             [(t['id'], t['capture_sha256']) for t in prior['cases']])
            for track, item in zip(cohort['cases'], suite['cases']):
                self.assertEqual(track['truth_sha256'], audit.dense.sha((suite_path.parent / item['input']['truth']).read_bytes()))
                self.assertEqual(sum(g['queries'] for g in track['by_regime'].values()), track['all']['queries'])
            for group in ('all', 'raw_missed'):
                actual = cohort['groups'][group]
                for key in ('queries', 'full_windows', 'raw_missed', 'candidate_absent_misses', 'acoustic_presence_unknown',
                            'beat_center_above_zero', 'beat_peak_above_zero', 'center_nonpositive_peak_positive', 'head_peaks_separated'):
                    self.assertEqual(actual[key], sum(t[group][key] for t in cohort['cases']))
            for stats in list(cohort['groups'].values()) + [t[g] for t in cohort['cases'] for g in ('all', 'raw_missed')]:
                self.assertEqual(stats['queries'], sum(stats['window_status'].values()))
                self.assertEqual(stats['queries'], sum(stats['label_counts'].values()))
                self.assertEqual(stats['full_windows'], stats['window_status'].get('available', 0))
                self.assertEqual(stats['beat_peak_above_zero'], stats['beat_center_above_zero'] + stats['center_nonpositive_peak_positive'])
                pair = stats['comparison']
                self.assertEqual(pair['queries'], stats['queries'])
                self.assertEqual(pair['queries'], sum(pair['pair_status'].values()))
                self.assertEqual(pair['eligible'], pair['wins'] + pair['ties'] + pair['losses'])
                self.assertEqual(pair['eligible'], pair['pair_status'].get('eligible', 0))
                self.assertEqual(pair['win_fraction'], pair['wins'] / pair['eligible'] if pair['eligible'] else None)
            if name == 'artbeat':
                self.assertEqual(cohort['groups']['all']['label_counts'], {'downbeat_unknown': 460})
                self.assertIsNone(cohort['ranks']['downbeat_ranking']['pooled']['auc']['downbeat_peak'])
            else:
                self.assertEqual(cohort['groups']['rubato']['queries'], 6726)
                self.assertEqual(cohort['groups']['annotated_constant_interior']['queries'], 0)
        public = json.dumps(report)
        for private in ('beat_logits', 'downbeat_logits', 'truth_times_s', '"time_s"', 'D:/', 'C:/'):
            self.assertNotIn(private, public)


if __name__ == '__main__':
    unittest.main()
