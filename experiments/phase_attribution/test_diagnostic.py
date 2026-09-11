import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.phase_attribution.diagnostic import (
    BINS, analyze, components, feedback_diagnosis, macro, rate_diagnosis,
)
from experiments.phase_attribution.run import HERE, PINS, REPORT, audit_metric, source_hashes, verify


class DecompositionTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.arange(501, dtype=np.float64) / 25
        self.valid = np.ones(501, dtype=bool)

    def test_exact_native_clock(self):
        r = analyze(self.reference, self.reference, self.valid)
        self.assertAlmostEqual(r['detail']['native_fraction'], 1)
        self.assertEqual(r['detail']['native_longest_s'], 10)
        self.assertEqual(r['detail']['total_loss'], 0)
        self.assertEqual(r['components'][0]['terminal_drift_cycles'], 0)

    def test_half_and_double_are_not_repaired_scores(self):
        for factor, name, expected in ((.5, 'half', 50), (2, 'double', 100)):
            r = analyze(factor * self.reference, self.reference, self.valid)
            self.assertAlmostEqual(r['original']['tempo_median_error_percent'], expected)
            self.assertEqual(r['detail'][name + '_fraction'], 1)
            self.assertAlmostEqual(r['detail']['oracle_recording_tempo_error_percent'], 0)
            self.assertAlmostEqual(r['detail']['count_loss'], 1)
            self.assertGreater(r['original']['phase_mean_absolute_cycles'], .24)

    def test_constant_integer_origin_is_not_count_drift(self):
        r = analyze(self.reference + 9, self.reference, self.valid)
        self.assertAlmostEqual(r['original']['phase_mean_absolute_cycles'], 0)
        self.assertAlmostEqual(r['detail']['relative_drift_mae_cycles'], 0)
        self.assertEqual(r['detail']['relative_integer_offset_mae'], 0)

    def test_integer_slip_can_hide_in_wrapped_phase_but_not_count(self):
        q = self.reference.copy()
        q[250:] += 1
        r = analyze(q, self.reference, self.valid)
        self.assertLess(r['original']['phase_mean_absolute_cycles'], 1e-12)
        self.assertAlmostEqual(r['components'][0]['terminal_drift_cycles'], 1)
        self.assertEqual(r['components'][0]['integer_boundary_transitions'], 1)
        self.assertGreater(r['original']['count_4s_mean_absolute_cycles'], .5)

    def test_off_grid_local_error_survives_oracle_octave(self):
        r = analyze(self.reference * 1.3, self.reference, self.valid)
        self.assertEqual(r['detail']['off_grid_fraction'], 1)
        self.assertAlmostEqual(r['detail']['oracle_recording_tempo_error_percent'], 30)
        self.assertEqual(sum(r['detail'][name + '_fraction'] for name in BINS), 1)

    def test_cell_folding_does_not_imply_one_track_level(self):
        rates = np.r_[np.ones(250), np.full(250, 4)]
        q = np.r_[0., np.cumsum(rates / 50)]
        r = analyze(q, self.reference, self.valid)['detail']
        self.assertAlmostEqual(r['oracle_cell_folded_tempo_error_percent'], 0)
        self.assertGreater(r['oracle_recording_tempo_error_percent'], 1)
        self.assertEqual(r['half_fraction'], .5)
        self.assertEqual(r['double_fraction'], .5)

    def test_missing_support_splits_runs_and_drift(self):
        mask = self.valid.copy()
        mask[200:300] = False
        reference = self.reference.copy()
        reference[~mask] = np.nan
        q = self.reference.copy()
        q[300:] += 4
        r = analyze(q, reference, mask)
        self.assertEqual(components(mask), [(0, 200), (300, 501)])
        self.assertEqual(len(r['components']), 2)
        self.assertAlmostEqual(r['detail']['relative_drift_mae_cycles'], 0)
        self.assertEqual(r['detail']['native_longest_s'], 4)
        self.assertEqual(r['original']['reference_4s_intervals'], 1)

    def test_empty_support_is_not_zero_error(self):
        r = analyze(self.reference, np.full(501, np.nan), np.zeros(501, bool))
        self.assertIsNone(r['detail']['native_fraction'])
        self.assertIsNone(r['detail']['total_loss'])
        self.assertIsNone(r['detail']['relative_drift_mae_cycles'])
        self.assertEqual(r['components'], [])

    def test_supported_bad_rate_fails_not_filters(self):
        for value in (np.nan, np.inf, self.reference[99], self.reference[99] - 1):
            q = self.reference.copy()
            q[100] = value
            with self.assertRaises(ValueError):
                analyze(q, self.reference, self.valid)

    def test_bin_boundaries_and_no_gap_join(self):
        rates = np.array([2 ** .5, 1.05, 1.09, 1.12, 8., .125])
        r = rate_diagnosis(rates, np.ones(6), np.ones(6, bool))
        self.assertEqual(r['native_fraction'], 2 / 6)
        self.assertEqual(r['off_grid_fraction'], 2 / 6)
        self.assertEqual(r['other_octave_fraction'], 2 / 6)


class FeedbackAndAuditTests(unittest.TestCase):
    def test_zero_observation_is_unavailable_not_perfect_phase(self):
        q = np.arange(251, dtype=np.float64) / 25
        fields = np.zeros((251, 4))
        fields[:, 0] = -1
        r = feedback_diagnosis(q, fields, q, np.ones(251, bool), .3)
        self.assertEqual(r['observation_available_fraction'], 0)
        self.assertIsNone(r['observation_phase_mae_cycles'])
        self.assertLess(r['correction_abs_log2_mean'], 1e-12)

    def test_saved_feedback_and_reachability_bound(self):
        fields = np.zeros((251, 4))
        fields[:, 0] = -1
        fields[:, 1] = 2
        fields[:, 2] = .5
        gain = .3
        q = [0.]
        for _ in range(250):
            angle = 2 * math.pi * ((q[-1] + .04) % 1)
            error = (.5 * math.cos(angle) - 2 * math.sin(angle)) / math.hypot(1, math.hypot(2, .5))
            q.append(q[-1] + .04 * math.exp(gain * error))
        q = np.array(q)
        reference = np.arange(251) * .12
        result = feedback_diagnosis(q, fields, reference, np.ones(251, bool), gain)
        self.assertEqual(result['reference_outside_feedback_bound_fraction'], 1)
        self.assertLess(result['feedback_equation_max_abs_log2_residual'], 1e-12)
        broken = q.copy()
        broken[30] += .001
        with self.assertRaisesRegex(ValueError, 'equation differs'):
            feedback_diagnosis(broken, fields, reference, np.ones(251, bool), gain)

    def test_origin_and_field_geometry_are_audited(self):
        q = np.arange(251, dtype=np.float64) / 25
        fields = np.zeros((251, 4))
        fields[:, 0] = -1
        with self.assertRaisesRegex(ValueError, 'origin differs'):
            feedback_diagnosis(q + 1, fields, q, np.ones(251, bool), 0)
        with self.assertRaises(ValueError):
            feedback_diagnosis(q, fields[:-1], q, np.ones(251, bool), 0)

    def test_work_macro_not_recording_or_frame_pool(self):
        rows = [dict(work=w, common=dict(audio=dict(detail=dict(x=v))))
                for w, v in (('a', 0), ('a', 2), ('b', 10))]
        self.assertEqual(macro(rows, 'common', 'audio', 'detail'), {'x': 5.5})
        rows[0]['common']['audio']['detail']['x'] = None
        self.assertIsNone(macro(rows, 'common', 'audio', 'detail')['x'])

    def test_metric_drift_or_missing_denominator_fails(self):
        actual = dict(error=1., reference_cells=500)
        audit_metric(actual, copy.deepcopy(actual))
        with self.assertRaises(ValueError):
            audit_metric(actual, dict(error=1., reference_cells=499))
        with self.assertRaises(ValueError):
            audit_metric(actual, dict(error=None, reference_cells=500))

    def test_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'artifact'
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                verify(path, '0' * 64)

    def test_analysis_has_no_torch_model_or_autograd_work(self):
        import torch
        q = np.arange(251) / 25
        with (mock.patch.object(torch.nn.Module, '__call__', side_effect=AssertionError('forward')),
              mock.patch.object(torch.autograd, 'grad', side_effect=AssertionError('gradient'))):
            analyze(q, q, np.ones(251, bool))


class RetainedOutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((HERE / 'results-v1.json').read_bytes())
        cls.parent = json.loads(REPORT.read_bytes())

    def test_source_and_all_prior_artifacts_unchanged(self):
        for path, digest in PINS.items():
            verify(path, digest)
        self.assertEqual(self.report['source_sha256'], source_hashes())
        self.assertEqual(self.report['prior_decision'], self.parent['decision'])

    def test_every_record_and_original_common_metric_kept(self):
        rows = self.report['cases']
        self.assertEqual(len(rows), 40)
        self.assertEqual(self.report['original_metric_blocks_audited'], 320)
        for row, old in zip(rows, self.parent['cases']):
            for key in ('id', 'role', 'work', 'frames', 'packet_sha256', 'prediction_sha256'):
                self.assertEqual(row[key], old[key])
            for kind, data in row['common'].items():
                audit_metric(data['original'], old[kind + '_common'])
                self.assertAlmostEqual(sum(data['detail'][k + '_fraction'] for k in BINS), 1)

    def test_every_public_common_macro_recomputes(self):
        for role, supports in self.report['summary'].items():
            rows = [r for r in self.report['cases'] if r['role'] == role]
            for kind, sections in supports['common'].items():
                for section, expected in sections.items():
                    self.assertEqual(macro(rows, 'common', kind, section), expected)

    def test_no_training_forward_or_new_acceptance(self):
        for key in ('training', 'model_forward', 'weights_loaded', 'hidden_features_loaded',
                    'holdout_access', 'independent_acceptance', 'production_change'):
            self.assertIs(self.report[key], False)
        self.assertEqual(self.report['decision'], 'fixed_prediction_diagnosis_only_no_automatic_fit')
        self.assertEqual(len(self.report['private_report_sha256']), 64)
        self.assertIn('full native/common', self.report['public_case_support'])


if __name__ == '__main__':
    unittest.main()
