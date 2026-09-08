import copy
from fractions import Fraction
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import oracle_period_paired_audit as audit
from test_annotation_exposure import data
from test_paired_response_replay import source


def pair():
    return dict(constant_start_frame=401, change_start_frame=650, window_frames=200,
                pre_segment_index=0, change_index=0)


def observations(d):
    times = list(map(audit.coordinate, d['beats']))
    return {s: source(times, s) for s in audit.replay.SOURCES}


class OraclePeriodControls(unittest.TestCase):
    def test_piecewise_phase_is_continuous_and_invertible_without_phase_reset(self):
        anchor, pre, post, boundary = map(Fraction, (5, 25, 50, 103))
        for t in map(Fraction, (-20, 0, 102, 103, 104, 199, 500)):
            phase = audit.phase_at(t, anchor, pre, post, boundary)
            self.assertEqual(audit.time_at(phase, anchor, pre, post, boundary), t)
        self.assertEqual(audit.phase_at(boundary, anchor, pre, post, boundary), (boundary - anchor) / pre)
        self.assertNotEqual((boundary - anchor) % pre, 0)  # transition not forced onto a beat

    def test_clock_half_open_scale_partition_and_start_crossing_anchor(self):
        args = tuple(map(Fraction, (25, 25, 25, 100, 0, 200)))
        native = audit.clock(*args, *audit.SCALES['native'])
        half0 = audit.clock(*args, *audit.SCALES['half_zero'])
        half1 = audit.clock(*args, *audit.SCALES['half_one'])
        double = audit.clock(*args, *audit.SCALES['double'])
        self.assertEqual(native, list(map(Fraction, range(0, 200, 25))))
        self.assertEqual(set(native), set(half0) | set(half1))
        self.assertFalse(set(half0) & set(half1))
        self.assertTrue(set(native) < set(double))

    def test_oracle_period_eliminates_endpoint_bias_without_response_fit(self):
        d = data()
        obs = observations(d)
        d['beats'][13] -= 30000
        prepared = audit.prepare(d, pair(), 'constant', obs)
        self.assertEqual(len(prepared[2]), 4)
        grids = [p['grids']['constant/native'] for p in prepared[2]]
        self.assertNotEqual(grids[0], grids[-1])
        for grid in grids:
            self.assertTrue(all(b - a == 25 for a, b in zip(grid, grid[1:])))
        changed = copy.deepcopy(obs)
        for items in changed.values():
            for p in items:
                p['beat_logit'], p['downbeat_logit'] = -99., 80.
        self.assertEqual(prepared[2], audit.prepare(d, pair(), 'constant', changed)[2])

    def test_counterfactual_boundary_same_elapsed_time_and_same_pre_step_grid(self):
        d = data()
        obs = observations(d)
        prepared = [audit.prepare(d, pair(), r, obs) for r in audit.replay.ROLES]
        self.assertEqual(prepared[0][3], prepared[1][3])
        self.assertEqual(prepared[0][3], 100)
        for p in prepared:
            for condition in p[2]:
                for scale in audit.SCALES:
                    a, b = [condition['grids'][shape + '/' + scale] for shape in audit.SHAPES]
                    self.assertEqual([t for t in a if t < p[3]], [t for t in b if t < p[3]])

    def test_ideal_acceleration_and_slowdown_can_pass_registered_shape_gate(self):
        for bpm in (240, 60):
            d = data()
            d['segments'][1]['bpm'] = bpm
            period = 60000000 // bpm
            d['beats'] = list(range(0, 15000000, 500000)) + list(range(15000000, 30000001, period))
            result = audit.analyze_pair(observations(d), d, pair())
            self.assertEqual(result['status'], 'eligible')
            self.assertTrue(result['verdict']['geometry_compatible'])
            for s in audit.replay.SOURCES:
                self.assertTrue(result['verdict']['sources'][s]['all_phases_shape_supported'])
                self.assertTrue(result['verdict']['sources'][s]['all_phases_density_resolved'])

    def test_omissions_can_share_observations_with_change_without_becoming_proof(self):
        d = data()
        obs = observations(d)
        for s in audit.replay.SOURCES:
            obs[s] = [p for p in obs[s] if not (p['coordinate'][0] / p['coordinate'][1] >= 750 and p['source_index'] % 2)]
        result = audit.analyze_pair(obs, d, pair())
        self.assertFalse(result['verdict']['sources']['raw']['all_phases_shape_supported'])
        self.assertEqual(audit.summarize([result])['by_source']['raw']['decision'], 'do_not_promote_count_sign_selector')

    def test_downbeat_sign_is_joint_metadata_not_a_negative_beat_label(self):
        packets = source([10, 40], 'candidate', [-1., 2.])
        packets[0]['downbeat_logit'] = 3.
        r = audit.packets.packet_ledger([True] * 200, packets, [Fraction(10), Fraction(40)])['result']
        self.assertEqual(audit.inventory(r, packets), [2, 0, 0, 1, 1, 1, 1])
        context = audit.context_states(packets, r, r, Fraction(40))
        self.assertEqual(context['pre/interior/always/always/-+'], 1)
        self.assertEqual(context['post/interior/always/always/+-'], 1)

    def test_all_optimal_assignments_yield_conservative_positive_bounds(self):
        packets = source([9, 11], 'candidate', [1., -1.])
        r = audit.packets.packet_ledger([True] * 200, packets, [Fraction(10)])['result']
        self.assertEqual(r['optimal_assignment_count'], 2)
        self.assertEqual(audit.inventory(r, packets), [1, 0, 1, 0, 1, 0, 0])
        self.assertEqual(audit.relation(audit.inventory(r, packets), audit.inventory(r, packets)), 'unresolved')
        self.assertEqual(sum(audit.context_states(packets, r, r, 20).values()), 2)

    def test_dominance_requires_guaranteed_head_axes_not_an_arbitrary_weight(self):
        base = [3, 1, 3, 2, 2, 1, 1]
        self.assertEqual(audit.relation(base, base), 'equal')
        better = [4, 0, 2, 2, 2, 1, 1]
        self.assertEqual(audit.relation(better, base), 'left_dominates')
        self.assertEqual(audit.relation(base, better), 'right_dominates')
        better[-2:] = [0, 0]
        self.assertEqual(audit.relation(better, base), 'unresolved')

    def test_geometry_checks_both_directions_and_keeps_cycle_slip(self):
        times = list(map(Fraction, range(0, 401, 25)))
        exact = audit.geometry_check(times, 3, Fraction(75), Fraction(25), Fraction(25), 200, 100, 300)
        self.assertTrue(exact['compatible'])
        slipped = audit.geometry_check(times, 3, Fraction(100), Fraction(25), Fraction(25), 200, 100, 300)
        self.assertFalse(slipped['compatible'])
        self.assertEqual(slipped['max_annotation_error_frames'], '25')
        truncated = audit.geometry_check(times[:8], 3, Fraction(75), Fraction(25), Fraction(25), 200, 100, 300)
        self.assertGreater(truncated['uncovered_clock_ordinals'], 0)

    def test_any_phase_geometry_failure_suppresses_pair_verdict_not_its_inventory(self):
        d = data()
        d['beats'][13] -= 150000
        result = audit.analyze_pair(observations(d), d, pair())
        self.assertEqual(result['status'], 'eligible')
        self.assertFalse(result['verdict']['geometry_compatible'])
        summary = audit.summarize([result])
        self.assertEqual(summary['geometry_incompatible_pairs'], 1)
        self.assertEqual(summary['by_source']['raw']['decision'], 'inconclusive_clock_geometry')

    def test_budget_and_support_failure_reject_both_members_without_ledger(self):
        d = data()
        for mutation, status in (('budget', 'response_budget_exceeded'), ('support', 'packet_support_crosses_window')):
            obs = observations(d)
            times = [Fraction(650 + i) for i in range(129)] if mutation == 'budget' else [Fraction(1699, 2)]
            obs['candidate'] = source(times, 'candidate')
            with patch.object(audit.packets, 'packet_ledger', side_effect=AssertionError('no orphan')):
                result = audit.analyze_pair(obs, d, pair())
            self.assertEqual(result['status'], 'pair_rejected')
            self.assertEqual(result['windows']['change']['status'], status)
            self.assertEqual(audit.summarize([result])['eligible_pairs'], 0)

    def test_overlap_duplicate_ids_and_clock_budget_fail_closed(self):
        d = data()
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            audit.analyze_pair(observations(d), d, pair() | {'change_start_frame': 600})
        obs = observations(d)
        obs['raw'][26]['source_index'] = 17
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit.analyze_pair(obs, d, pair())
        self.assertIsNone(audit.clock(Fraction(0), Fraction(1), Fraction(1), 100, 0, 200, Fraction(1), 0))

    def test_no_phase_selection_geometry_or_density_failure_hidden_by_one_success(self):
        d = data()
        result = audit.analyze_pair(observations(d), d, pair())
        windows = result['windows']
        damaged = windows['change']['phases'][1]['sources']['raw']['counts']
        damaged['step/native'] = damaged['constant/native'][:]
        verdict = audit.verdict(windows)
        self.assertEqual(verdict['sources']['raw']['shape_supported_phases'], 3)
        self.assertFalse(verdict['sources']['raw']['all_phases_shape_supported'])
        self.assertTrue(verdict['sources']['candidate']['all_phases_shape_supported'])

    def test_dependency_identity_checked_before_any_private_read(self):
        with patch.object(Path, 'read_bytes', return_value=b'wrong'), patch.object(audit.dense, 'read_json') as reader:
            with self.assertRaisesRegex(ValueError, 'dependency changed'):
                audit.build_report('not-opened', 'not-opened')
            reader.assert_not_called()


class FrozenOraclePeriodPair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = audit.HERE / 'oracle-period-paired-v1.json'
        cls.report = json.loads(cls.path.read_bytes())

    def test_frozen_artifact_script_contract_and_helpers_have_exact_identities(self):
        r = self.report
        self.assertEqual(audit.dense.sha(self.path.read_bytes()), 'f2666efba8c9c048143d13a4dd47a4cecbdcf64fc55b13e12dbbd0e058469e49')
        self.assertEqual(r['script_sha256'], audit.dense.sha(Path(audit.__file__).read_bytes()))
        self.assertEqual(r['lock_sha256'], audit.dense.sha(audit.LOCK_PATH.read_bytes()))
        self.assertEqual(r['contract'], audit.LOCK)
        for name, expected in r['helper_sha256'].items():
            self.assertEqual(audit.dense.sha((audit.HERE / name).read_bytes()), expected)

    def test_public_annotation_plan_reproduces_all_selected_and_unselected_identities(self):
        _, cases, plan = audit.replay.verified_plan()
        self.assertEqual(self.report['selected_plan_sha256'], audit.replay.LOCK['selected_plan_sha256'])
        self.assertEqual([c['id'] for c in self.report['input_cases']], [c['identity']['id'] for c in cases])
        self.assertEqual(len(cases), 40)
        self.assertEqual([c['id'] for c in self.report['pairs']], [p['id'] for p in plan])
        self.assertEqual(sum(c['private_capture_read'] for c in self.report['input_cases']), 7)
        for result, selected in zip(self.report['pairs'], plan):
            self.assertEqual(result['pair_sha256'], audit.exposure.canonical_hash(selected))

    def test_every_phase_clock_source_and_context_retains_owned_denominators(self):
        self.assertEqual(self.report['count_vector_fields'], list(audit.VECTOR))
        ledgers = 0
        for pair_result in self.report['pairs']:
            self.assertEqual(pair_result['status'], 'eligible')
            for source in audit.replay.SOURCES:
                owned = sum(w['owned_responses'][source] for w in pair_result['windows'].values())
                self.assertEqual(owned + pair_result['outside_selected_windows'][source], pair_result['whole_capture_responses'][source])
            for w in pair_result['windows'].values():
                self.assertEqual(len(w['phases']), 4)
                for ph in w['phases']:
                    self.assertTrue(ph['geometry']['compatible'])
                    for source, inventory in ph['sources'].items():
                        self.assertEqual(set(inventory['counts']), set(audit.NAMES))
                        self.assertEqual(sum(inventory['context'].values()), w['owned_responses'][source])
                        self.assertEqual(inventory['ambiguous_ledgers'], 0)
                        self.assertEqual(inventory['max_assignment_count'], 1)
                        self.assertLessEqual(inventory['max_states'], 16641)
                        for values in inventory['counts'].values():
                            self.assertEqual(len(values), 7)
                            self.assertEqual(values[0] + values[2], w['owned_responses'][source])
                            self.assertLessEqual(values[0] + values[1], 128)
                            self.assertTrue(0 <= values[3] <= values[4] <= w['owned_responses'][source])
                            self.assertTrue(0 <= values[5] <= values[6] <= w['owned_responses'][source])
                            ledgers += 1
        self.assertEqual(ledgers, 896)  # repeated conditions, not independent examples

    def test_verdict_and_summary_recompute_without_best_phase_selection(self):
        for p in self.report['pairs']:
            self.assertEqual(p['verdict'], audit.verdict(p['windows']))
        summary = audit.summarize(self.report['pairs'])
        self.assertEqual(summary, self.report['summary'])
        self.assertEqual((summary['selected_pairs'], summary['eligible_pairs'], summary['geometry_compatible_pairs']), (7, 7, 7))
        self.assertEqual(summary['all_selected_owned_responses'], {'raw': 91, 'candidate': 340})
        self.assertEqual([summary['by_source'][s]['robust_shape_pairs'] for s in audit.replay.SOURCES], [2, 1])
        for s in audit.replay.SOURCES:
            self.assertEqual(summary['by_source'][s]['robust_density_pairs'], 0)
            self.assertEqual(summary['by_source'][s]['decision'], 'do_not_promote_count_sign_selector')

    def test_scope_does_not_claim_accuracy_training_or_publish_packet_arrays(self):
        r = self.report
        self.assertIs(r['truth_assisted'], True)
        self.assertIsNone(r['selected_tempo'])
        for flag in ('neural_inference', 'decoder_replayed', 'fitted_mapping', 'phase_search', 'source_pooling',
                     'holdout_evidence_opened', 'training_run', 'production_output_changed', 'new_user_parameters', 'accuracy_improvement_claimed'):
            self.assertIs(r[flag], False)
        serialized = json.dumps(r)
        for key in ('"time_s"', '"coordinate"', '"beat_logits"', '"downbeat_logits"', '"start_frame"'):
            self.assertNotIn(key, serialized)


if __name__ == '__main__':
    unittest.main()
