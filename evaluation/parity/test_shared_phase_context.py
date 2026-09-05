import json
import math
from pathlib import Path
import statistics
import unittest

import shared_phase_context_audit as audit
from test_metrical_window import fixture


class SharedPhaseControls(unittest.TestCase):
    def test_coherent_offset_and_downbeat_readout_not_separate_maxima(self):
        centers = [20, 40, 60, 80, 100]
        b, d = [-4.] * 130, [-7.] * 130
        for t in centers:
            b[t + 2], d[t + 2], d[t - 1] = 4., 3., 9.
        result = audit.compare(b, d, centers, [25, 45, 65, 85, 105])
        self.assertEqual(result['left'], dict(shared=4., independent=4., phase_penalty=0.,
                                             offset=2, downbeat_at_shared=3.))
        self.assertEqual(result['right']['offset'], -3)  # cross-template windows may overlap
        self.assertEqual(result['shared_margin'], 0.)

    def test_shared_phase_requires_coherence_and_uses_fixed_ties(self):
        centers = [20, 40, 60, 80, 100]
        b = [-4.] * 130
        for t, d in zip(centers, [-2, -1, 0, 1, 2]):
            b[t + d] = 6.
        result = audit.template_score(b, b, centers)
        self.assertEqual((result['independent'], result['shared'], result['phase_penalty']), (6., -2., 8.))
        self.assertEqual(result['offset'], 0)
        brute = [sum(b[t + d] for t in centers) / len(centers) for d in range(-3, 4)]
        self.assertEqual(result['shared'], max(brute))
        for t in centers:
            b[t - 1] = b[t + 1] = 6.
        self.assertEqual(audit.template_score(b, b, centers)['offset'], -1)
        for t in centers:
            b[t] = 6.
        self.assertEqual(audit.template_score(b, b, centers)['offset'], 0)

    def test_complete_common_domain_is_required_even_for_unused_frames(self):
        left, right = [20, 40, 60, 80, 100], [20, 50, 80, 110, 140]
        b, mask = [0.] * 150, [True] * 150
        mask[30] = False  # neither template reads this point
        self.assertEqual(audit.compare(b, b, left, right, mask)['status'], 'unavailable_common_frame')
        b[30] = math.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite common frame'):
            audit.compare(b, b, left, right)
        self.assertEqual(audit.compare([0.] * 143, [0.] * 143, left, right)['status'], 'out_of_capture')
        self.assertEqual(audit.compare([0.] * 150, [0.] * 150, left, [2, 30, 60, 90, 120])['status'], 'out_of_capture')

    def test_equal_grids_and_overlaps_are_not_fake_accuracy(self):
        b, centers = [0.] * 120, [20, 40, 60, 80, 100]
        same = audit.compare(b, b, centers, centers)
        self.assertEqual(same['status'], 'identical_grids')
        self.assertEqual(same['shared_margin'], 0.)
        for other in ([20, 26, 60, 80, 100], [20, 20, 60, 80, 100]):
            self.assertEqual(audit.compare(b, b, centers, other)['status'], 'within_template_overlap')
        self.assertEqual(audit.compare(b, b, centers, [20, 27, 60, 80, 100])['status'], 'informative')
        row = dict(comparisons={'pair': same}, label='downbeat_unknown')
        summary = audit.summarize([row], 'pair')
        self.assertEqual(summary['informative'], 0)
        self.assertIsNone(summary['shared']['positive_fraction'])

    def test_constant_pulses_still_alias_half_tempo(self):
        b = [-4.] * 210
        for t in range(20, 201, 20):
            b[t] = 4.
        result = audit.compare(b, b, [20, 40, 60, 80, 100], [20, 60, 100, 140, 180])
        self.assertEqual(result['status'], 'informative')
        self.assertEqual((result['shared_margin'], result['independent_margin']), (0., 0.))
        self.assertEqual(result['left']['shared'], 4.)

    def test_omission_and_slowdown_can_have_identical_observations(self):
        # These two interpretations use identical arrays, not a sound-presence label.
        b = [-4.] * 450
        for t in (25, 50, 75, 100, 150, 200, 250, 300):
            b[t] = 4.
        constant_truth, constant = fixture([.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5])
        slow_truth, slow = fixture([.5, 1., 1.5, 2., 3., 4., 5., 6., 7.])
        slow_truth['tempo_segments'][0]['end_s'] = 8.
        slow_truth['change_points'] = [dict(time_s=2., kind='tempo_jump')]
        a = audit.case_rows(b, b, constant_truth, constant, 'artbeat')[3]
        z = audit.case_rows(b, b, slow_truth, slow, 'artbeat')[3]
        self.assertNotEqual(a['regime'], z['regime'])
        for pair in audit.PAIR_NAMES[1:]:
            self.assertEqual(a['comparisons'][pair], z['comparisons'][pair])
        # The truth-supplied slow template wins, but this is not truth-free recovery.
        self.assertGreater(z['comparisons']['annotated_vs_continuation']['shared_margin'], 0)
        self.assertEqual((a['acoustic_presence'], z['acoustic_presence']), ('unknown', 'unknown'))

    def test_equal_count_offset_scale_invariance_and_unused_frame_limitation(self):
        left, right = [20, 40, 60, 80, 100], [20, 50, 80, 110, 140]
        b = [float((i * 7) % 17) - 8. for i in range(150)]
        original = audit.compare(b, b, left, right)
        transformed = audit.compare([v * 2 + 100 for v in b], b, left, right)
        for method in ('shared', 'independent'):
            self.assertAlmostEqual(transformed[method + '_margin'], original[method + '_margin'] * 2)
        self.assertEqual(transformed['left']['offset'], original['left']['offset'])
        b[30] = 1000000.  # known but unused: no density or absence evidence
        self.assertEqual(audit.compare(b, b, left, right), original)

    def test_whole_context_labels_and_missing_annotations(self):
        truth, case = fixture()
        self.assertEqual(audit.context_regime(truth, case, 'artbeat', 3), 'constant_context')
        truth['change_points'] = [dict(time_s=.5)]  # prefix, not just queried suffix
        self.assertEqual(audit.context_regime(truth, case, 'artbeat', 3), 'change_context')
        truth['change_points'] = []
        truth['tempo_segments'][0]['kind'] = 'ramp'
        self.assertEqual(audit.context_regime(truth, case, 'artbeat', 3), 'ramp_context')
        self.assertEqual(audit.context_regime(truth, case, 'rubato', 3), 'rubato')
        case['tags'] = ['rubato']
        self.assertEqual(audit.context_regime(truth, case, 'artbeat', 3), 'rubato')
        case['tags'] = []
        truth['tempo_segments'] = [dict(start_s=0., end_s=1.2, kind='constant'),
                                  dict(start_s=1.3, end_s=5., kind='constant')]
        with self.assertRaisesRegex(ValueError, 'context annotation'):
            audit.context_regime(truth, case, 'artbeat', 3)  # gap even between annotated beats

    def test_prefix_suffix_queries_raw_identity_and_null_summaries(self):
        truth, case = fixture()
        rows = audit.case_rows([0.] * 500, [0.] * 500, truth, case, 'artbeat')
        self.assertEqual(sum(r['raw_missed'] for r in rows), 8)
        self.assertEqual(set(r['label'] for r in rows), {'downbeat_unknown'})
        stats = audit.summarize(rows, audit.PAIR_NAMES[0])
        self.assertEqual(stats['status'], dict(insufficient_prefix=3, identical_grids=2, insufficient_suffix=4))
        self.assertEqual(stats['acoustic_presence_unknown'], 9)
        self.assertIsNone(audit.summarize([], audit.PAIR_NAMES[0])['shared']['mean_margin'])
        case['raw_truth_pairs'] = [[0, 1]]
        with self.assertRaisesRegex(ValueError, 'matching changed'):
            audit.case_rows([0.] * 500, [0.] * 500, truth, case, 'artbeat')

    def test_malformed_inputs_fail_closed(self):
        b, centers = [0.] * 120, [20, 40, 60, 80, 100]
        for bad in ([20., 40, 60, 80, 100], [20, 40], [20, 60, 40, 80, 100]):
            with self.assertRaises(ValueError):
                audit.compare(b, b, bad, centers)
        with self.assertRaises(ValueError):
            audit.compare(b, b[:-1], centers, centers)
        with self.assertRaises(ValueError):
            audit.compare(b, b, centers, centers, [True])


class SharedPhaseFrozenReport(unittest.TestCase):
    def test_hashes_contract_and_no_production_claims(self):
        here = Path(__file__).parent
        report = json.loads((here / 'shared-phase-context-v1.json').read_text())
        self.assertEqual(report['contract'], audit.LOCK)
        self.assertEqual(audit.LOCK['pairs'], list(audit.PAIR_NAMES))
        self.assertEqual(audit.LOCK['points_per_template'], 5)
        self.assertEqual(audit.LOCK['offsets_frames'], list(range(-3, 4)))
        self.assertEqual(report['script_sha256'], audit.dense.sha(Path(audit.__file__).read_bytes()))
        self.assertEqual(report['lock_sha256'], audit.dense.sha(audit.LOCK_PATH.read_bytes()))
        self.assertEqual(report['window_report_sha256'], audit.dense.sha((here / 'metrical-window-v1.json').read_bytes()))
        for name, identity in report['helper_sha256'].items():
            self.assertEqual(identity, audit.dense.sha((here / name).read_bytes()))
        self.assertTrue(report['truth_assisted'])
        for key in ('fitted_mapping', 'neural_inference', 'decoder_replayed', 'holdout_opened', 'training_run',
                    'production_observations_changed', 'accuracy_improvement_claimed'):
            self.assertIs(report[key], False)
        for key in ('equal_count_templates_are_full_clock_hypotheses', 'unused_frames_supply_density_evidence',
                    'independent_frame_likelihood_claimed', 'fit_mapping', 'holdout_access'):
            self.assertIs(audit.LOCK[key], False)

    def test_all_tracks_provenance_and_denominators(self):
        here = Path(__file__).parent
        report = json.loads((here / 'shared-phase-context-v1.json').read_text())
        old = json.loads((here / 'metrical-window-v1.json').read_text())
        self.assertEqual(len(report['cohorts']), 2)
        for cohort, prior in zip(report['cohorts'], old['cohorts']):
            expected = {
                'artbeat': {'all': [(296, 145, 232), (297, 199, 196), (333, 277, 268)],
                            'raw_missed': [(91, 61, 74), (97, 73, 74), (100, 74, 67)],
                            'candidate_absent_misses': [(3, 3, 3), (5, 2, 2), (7, 4, 4)]},
                'rubato': {'all': [(6500, 4372, 4465), (6519, 4218, 4514), (6541, 4307, 4357)],
                           'raw_missed': [(2368, 1855, 1848), (2373, 1210, 1309), (2389, 1237, 1324)],
                           'candidate_absent_misses': [(1151, 959, 933), (1152, 577, 622), (1157, 600, 630)]},
            }[cohort['cohort']]
            for group, values in expected.items():
                for pair, counts in zip(audit.PAIR_NAMES, values):
                    stats = cohort['groups'][group][pair]
                    self.assertEqual((stats['informative'], stats['independent']['positive'], stats['shared']['positive']), counts)
            for key in ('cohort', 'frozen_evidence_sha256', 'capture_summary_sha256', 'suite_sha256',
                        'source_hashes', 'total_frames_per_head'):
                self.assertEqual(cohort[key], prior[key])
            self.assertTrue(cohort['complete'])
            self.assertEqual([(t['id'], t['capture_sha256'], t['truth_sha256']) for t in cohort['cases']],
                             [(t['id'], t['capture_sha256'], t['truth_sha256']) for t in prior['cases']])
            for group in ('all', 'raw_missed'):
                for p in audit.PAIR_NAMES:
                    stats = cohort['groups'][group][p]
                    self.assertEqual(stats['queries'], prior['groups'][group]['queries'])
                    self.assertEqual(stats['queries'], sum(t[group][p]['queries'] for t in cohort['cases']))
                    self.assertEqual(stats['informative'], sum(t[group][p]['informative'] for t in cohort['cases']))
                    for method in ('shared', 'independent'):
                        values = [t[group][p][method]['positive_fraction'] for t in cohort['cases']
                                  if t[group][p][method]['positive_fraction'] is not None]
                        macro = cohort['macro'][group][p][method]
                        self.assertEqual(macro['contributing_tracks'], len(values))
                        self.assertEqual(macro['mean_track_positive_fraction'], statistics.mean(values) if values else None)
            for p in audit.PAIR_NAMES:
                for group, suffix in (('all', ''), ('raw_missed', '_missed')):
                    self.assertEqual(cohort['groups'][group][p]['queries'],
                                     sum(cohort['groups'][g + suffix][p]['queries'] for g in audit.REGIMES))
            blocks = list(cohort['groups'].values()) + [t[g] for t in cohort['cases'] for g in ('all', 'raw_missed')]
            for block in blocks:
                for stats in block.values():
                    self.assertEqual(stats['queries'], sum(stats['status'].values()))
                    self.assertEqual(stats['queries'], stats['acoustic_presence_unknown'])
                    self.assertEqual(stats['queries'], sum(stats['label_counts'].values()))
                    n = stats['informative']
                    self.assertEqual(n, stats['status'].get('informative', 0))
                    self.assertEqual(n, sum(stats['joint_signs'].values()))
                    for method in ('shared', 'independent'):
                        s = stats[method]
                        self.assertEqual(n, sum(s[k] for k in ('positive', 'zero', 'negative')))
                        self.assertEqual(s['positive_fraction'], s['positive'] / n if n else None)
                        for polarity in ('positive', 'zero', 'negative'):
                            self.assertEqual(s[polarity], sum(count for key, count in stats['joint_signs'].items()
                                if key.split('_to_')[method == 'shared'] == polarity))
                    for penalty in stats['mean_phase_penalty'].values():
                        if penalty is not None:
                            self.assertGreaterEqual(penalty, -1e-12)
            if cohort['cohort'] == 'artbeat':
                self.assertEqual(cohort['groups']['all'][audit.PAIR_NAMES[0]]['label_counts'], {'downbeat_unknown': 460})
            else:
                self.assertEqual(cohort['groups']['constant_context'][audit.PAIR_NAMES[0]]['queries'], 0)
        public = json.dumps(report)
        for private in ('beat_logits', 'downbeat_logits', 'truth_times_s', '"time_s"', 'D:/', 'C:/'):
            self.assertNotIn(private, public)


if __name__ == '__main__':
    unittest.main()
