from fractions import Fraction
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import prefix_clock_drift_audit as audit
from test_annotation_exposure import data


def window(d=None, start=401, role='constant'):
    return audit.diagnose_window(data() if d is None else d, start, 200, 0, role, 0)


class PrefixClockDriftControls(unittest.TestCase):
    def test_perfect_constant_grid_has_zero_terms_and_identical_ownership(self):
        w = window()
        self.assertTrue(w['identical_grids'])
        self.assertEqual(w['all_ticks']['ticks'], 8)
        for term in ('error', 'relative_prefix_period_drift', 'relative_annotation_departure'):
            self.assertEqual(w['all_ticks'][term]['max_abs_us'], '0')
        self.assertEqual(w['first_selected_beat_error_us'], '0')
        self.assertEqual(audit.gate([w])['constant_windows_beyond_normalization'], 0)

    def test_period_bias_accumulates_without_annotation_departure(self):
        d = data()
        d['beats'][13] -= 300000  # endpoint span: 1.8s / 3, not 1.5s / 3
        w = window(d)
        self.assertEqual(w['prefix_period_us'], '600000')
        self.assertEqual(w['first_selected_beat_error_us'], '-100000')
        self.assertEqual(w['all_ticks']['relative_prefix_period_drift']['min_us'], '-700000')
        self.assertEqual(w['all_ticks']['relative_annotation_departure']['max_abs_us'], '0')
        self.assertEqual(w['all_ticks']['error']['max_abs_us'], '800000')
        self.assertGreater(w['all_ticks']['both_full_query_beyond_radius'], 0)

    def test_constant_offset_is_not_period_bias_or_internal_deformation(self):
        d = data()
        d['beats'][17:30] = [t + 20000 for t in d['beats'][17:30]]
        w = window(d)
        self.assertEqual(w['first_selected_beat_error_us'], '20000')
        self.assertEqual(w['all_ticks']['error']['min_us'], '20000')
        self.assertEqual(w['all_ticks']['relative_prefix_period_drift']['max_abs_us'], '0')
        self.assertEqual(w['all_ticks']['relative_annotation_departure']['max_abs_us'], '0')
        self.assertEqual(w['all_ticks']['beyond_radius'], 0)

    def test_internal_interval_variation_survives_zero_prefix_bias(self):
        d = data()
        for i in range(18, 30, 2):
            d['beats'][i] += 10000
        w = window(d)
        self.assertEqual(w['first_selected_beat_error_us'], '0')
        self.assertEqual(w['all_ticks']['relative_prefix_period_drift']['max_abs_us'], '0')
        self.assertEqual(w['all_ticks']['relative_annotation_departure']['max_abs_us'], '10000')

    def test_prefix_instability_can_cancel_at_endpoints(self):
        d = data()
        d['beats'][14] += 20000
        d['beats'][15] -= 10000
        w = window(d)
        self.assertTrue(w['identical_grids'])
        self.assertEqual(w['prefix_interval_minus_nominal']['min_us'], '-30000')
        self.assertEqual(w['prefix_interval_minus_nominal']['max_us'], '20000')

    def test_microsecond_bound_does_not_become_annotation_precision(self):
        for n in range(1, 30):
            bound = audit.normalization_bound(n)
            for a in (Fraction(-1, 2), Fraction(1, 2)):
                for b in (Fraction(-1, 2), Fraction(1, 2)):
                    for c in (Fraction(-1, 2), Fraction(1, 2)):
                        self.assertLessEqual(abs(a - (1 + Fraction(n, 3)) * b + Fraction(n, 3) * c), bound)
        d = data()
        d['beats'][13] -= 1
        self.assertEqual(window(d)['all_ticks']['beyond_normalization_bound'], 0)
        d['beats'][13] -= 29
        self.assertEqual(window(d)['all_ticks']['beyond_normalization_bound'], 8)
        for n in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                audit.normalization_bound(n)

    def test_half_open_ownership_and_fractional_query_support(self):
        w = window(start=400)
        self.assertEqual(w['ownership']['annotation_ticks'], 8)
        self.assertEqual(w['ownership']['annotation_full_query'], 7)
        self.assertEqual(audit.periodic_ordinals(0, Fraction(500000), 500000, 1500000), {1, 2})
        for center in (Fraction(0), Fraction(2), Fraction(2000001, 1000000), Fraction(3),
                       Fraction(197) - Fraction(1, 1000000), Fraction(197), Fraction(200)):
            expected = 0 <= center < 200 and math.ceil(center - 3) >= 0 and math.floor(center + 3) < 200
            self.assertEqual(audit.full_query(center * 20000, 0, 4000000), expected)

    def test_count_difference_is_kept_even_when_error_is_only_one_microsecond(self):
        # A -1us shift in the four-beat anchor moves the tick at 8s before the start.
        d = data()
        d['beats'][15] -= 1
        w = window(d, start=400)
        self.assertGreater(w['ownership']['annotation_only'] + w['ownership']['continuation_only'], 0)
        self.assertEqual(w['all_ticks']['beyond_normalization_bound'], 0)

    def test_cycle_error_is_not_hidden_by_nearest_neighbor_or_phase_wrapping(self):
        d = data()
        d['beats'][17:] = [t + 500000 for t in d['beats'][17:]]
        w = window(d)
        self.assertEqual(w['all_ticks']['error']['min_us'], '500000')
        self.assertEqual(w['all_ticks']['relative_annotation_departure']['max_abs_us'], '0')
        self.assertFalse(w['identical_grids'])
        self.assertEqual(w['ownership']['continuation_only'], 1)

    def test_uncovered_virtual_ordinals_are_not_fabricated(self):
        d = data()
        d['beats'] = d['beats'][:26]  # 12.5s covers the selected window, not arbitrary forecasts
        d['beats'][13:17] = [7780000, 7850000, 7920000, 7990000]
        w = window(d)
        self.assertGreater(w['ownership']['continuation_only_annotation_location']['annotation_uncovered'], 0)
        self.assertEqual(sum(w['ownership']['continuation_only_annotation_location'].values()), w['ownership']['continuation_only'])

    def test_change_contrast_splits_at_step_and_cannot_drive_constant_gate(self):
        const = window()
        change = window(start=650, role='change')
        self.assertEqual(change['strata']['pre_step']['error']['max_abs_us'], '0')
        self.assertGreater(Fraction(change['strata']['post_step']['error']['max_abs_us']), 0)
        self.assertEqual(sum(x['ticks'] for x in change['strata'].values()), change['all_ticks']['ticks'])
        self.assertEqual(audit.gate([const, change])['decision'], 'no_beyond_normalization_drift_found_not_accuracy_validated')
        with self.assertRaisesRegex(ValueError, 'no constant'):
            audit.gate([change])

    def test_decomposition_matches_independent_fractional_calculation(self):
        d = data()
        d['beats'][13] -= 40000
        d['beats'][17:30] = [t + (i % 3) * 7000 for i, t in enumerate(d['beats'][17:30])]
        w = window(d)
        errors = [Fraction(d['beats'][i]) - 8000000 - (i - 16) * Fraction(1540000, 3) for i in range(17, 25)]
        self.assertEqual(w['all_ticks']['error'], audit.stats(errors))

    def test_invalid_window_clock_budget_and_empty_stats(self):
        self.assertEqual(audit.stats([]), dict(count=0, min_us=None, max_us=None, max_abs_us=None))
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            audit.diagnose_window(data(), 400, 100, 0, 'constant', 0)
        with self.assertRaisesRegex(ValueError, 'role changed'):
            window(start=650)
        for args in ((0, 0, 1, 2), (2, 1, 1, 2), (0, 1, 1, 131)):
            with self.assertRaises(ValueError):
                audit.periodic_ordinals(*args)

    def test_pinned_inputs_fail_before_real_annotation_read(self):
        with patch.object(Path, 'read_bytes', return_value=b'changed'), patch.object(audit.exposure, 'load_annotations') as loader:
            with self.assertRaisesRegex(ValueError, 'dependency changed'):
                audit.verified_plan()
            loader.assert_not_called()

    def test_report_assembly_keeps_unselected_denominators_and_excludes_coordinates(self):
        identity = dict(id='authored', frame_count=1501, truth_sha256='truth', capture_sha256='capture')
        cases = [dict(cohort='artbeat', identity=identity, data=data()),
                 dict(cohort='rubato', identity=dict(identity, id='unselected'), data=None)]
        plan = [dict(cohort='artbeat', id='authored', constant_start_frame=401, change_start_frame=650,
                     window_frames=200, pre_segment_index=0, change_index=0)]
        with patch.object(audit, 'verified_plan', return_value=([], cases, plan)):
            report = audit.build_report()
        self.assertEqual(len(report['cases']), 2)
        self.assertNotIn('windows', report['cases'][1])
        for forbidden in ('start_frame', 'time_s', 'logits', 'private.json'):
            self.assertNotIn(forbidden, json.dumps(report))
        self.assertEqual(report['window_count'], 2)
        for flag in ('private_captures_read', 'response_fields_used', 'fitted_mapping', 'production_output_changed'):
            self.assertIs(report[flag], False)


class FrozenPrefixClockDrift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = audit.HERE / 'prefix-clock-drift-v1.json'
        cls.report = json.loads(cls.path.read_bytes())
        cls.reads = []
        original = Path.read_bytes

        def recording_read(path):
            cls.reads.append(path.resolve())
            return original(path)

        with patch.object(Path, 'read_bytes', autospec=True, side_effect=recording_read):
            cls.reproduced = audit.build_report()

    def test_frozen_result_reproduces_byte_identity_and_uses_public_inputs_only(self):
        self.assertEqual(self.report, self.reproduced)
        self.assertEqual(audit.exposure.sha(self.path.read_bytes()), '5c9445e4e00801a11fc92301b77fa38a29f0c604a96268a5c5c0e1963599761a')
        allowed = {audit.HERE / name for name in audit.PINS}
        allowed.update((audit.LOCK_PATH, Path(audit.__file__), audit.HERE / audit.exposure.SOURCE))
        allowed.update(audit.exposure.ROOT / 'evaluation/suites' / (name + '.json') for name in audit.exposure.SUITES.values())
        self.assertEqual(len(self.reads), 94)
        for path in self.reads:
            self.assertTrue(path in allowed or path.is_relative_to(audit.exposure.ROOT / 'evaluation/suites/truth'), path)

    def test_same_forty_identities_seven_pairs_and_no_new_selection(self):
        report = self.report
        frozen = json.loads((audit.HERE / 'annotation-exposure-v1.json').read_bytes())
        self.assertEqual(report['plan_sha256'], frozen['plan_sha256'])
        self.assertEqual(report['metadata_projection_sha256'], frozen['metadata_projection_sha256'])
        self.assertEqual([c['id'] for c in report['cases']], [c['id'] for c in frozen['profiles']['200']['cases']])
        self.assertEqual([c['id'] for c in report['cases'] if c['selected']], frozen['selected_case_ids'])
        self.assertEqual((len(report['cases']), report['selected_pair_count'], report['window_count']), (40, 7, 14))
        for c in report['cases']:
            self.assertEqual('windows' in c, c['selected'])
            if c['selected']:
                self.assertEqual([w['role'] for w in c['windows']], ['constant', 'change'])

    def test_all_ordinal_and_stratum_denominators_conserve_without_nearest_matching(self):
        for case in self.report['cases']:
            for w in case.get('windows', []):
                o = w['ownership']
                self.assertEqual(o['annotation_ticks'], o['shared_ordinals'] + o['annotation_only'])
                self.assertEqual(o['continuation_ticks'], o['shared_ordinals'] + o['continuation_only'])
                self.assertEqual(o['continuation_only'], sum(o['continuation_only_annotation_location'].values()))
                self.assertEqual(o['annotation_only'], o['annotation_only_prediction_before_window'] + o['annotation_only_prediction_after_window'])
                for key in ('ticks', 'beyond_normalization_bound', 'beyond_radius', 'full_query_annotation_ticks',
                            'both_full_query_ticks', 'full_query_annotation_beyond_radius', 'both_full_query_beyond_radius'):
                    self.assertEqual(w['all_ticks'][key], sum(s[key] for s in w['strata'].values()))

    def test_gate_preserves_constant_confound_without_training_or_accuracy_verdict(self):
        g = self.report['gate']
        self.assertEqual(g['constant_windows'], 7)
        self.assertEqual(g['constant_windows_beyond_normalization'], 6)
        self.assertEqual(g['constant_windows_with_ownership_difference'], 1)
        self.assertEqual(g['constant_windows_with_both_full_query_error_beyond_radius'], 3)
        self.assertEqual(g['decision'], 'require_clock_nuisance_control_before_model_discrimination')
        for flag in ('private_captures_read', 'response_fields_used', 'neural_inference', 'fitted_mapping', 'decoder_replayed',
                     'holdout_opened', 'training_run', 'production_output_changed', 'new_user_parameters', 'accuracy_improvement_claimed'):
            self.assertIs(self.report[flag], False)
        self.assertIsNone(self.report['selected_tempo'])
        self.assertIn('unknown', self.report['upstream_annotation_precision'])


if __name__ == '__main__':
    unittest.main()
