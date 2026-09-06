import copy
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import random
import struct
import unittest

import backend_response_packet_audit as adapter
import clock_response_ledger_audit as integer
import rational_response_ledger as rational


def authored(beat, downbeat=None, duration=None):
    downbeat = [-8.] * len(beat) if downbeat is None else downbeat
    duration = len(beat) / 50 if duration is None else duration
    observations = adapter.reconstruct(beat, downbeat, duration)[0]
    return adapter.assemble(beat, downbeat, duration, observations)


def brute(ticks, responses):
    possibilities = []
    for count in range(min(len(ticks), len(responses)) + 1):
        for left in itertools.combinations(range(len(ticks)), count):
            for right in itertools.combinations(range(len(responses)), count):
                pairs = list(zip(left, right))
                if all(abs(ticks[i] - responses[j]) <= 3 for i, j in pairs):
                    possibilities.append(((-count, sum((abs(ticks[i] - responses[j]) for i, j in pairs), Fraction(0))), pairs))
    best = min(key for key, _ in possibilities)
    return best, [pairs for key, pairs in possibilities if key == best]


class BackendPacketControls(unittest.TestCase):
    def test_default_centroid_published_time_and_score_frame_are_distinct(self):
        b, d = [-4.] * 15, [-8.] * 15
        b[5:7], d[6] = [4., 4.], 3.
        result = authored(b, d)
        raw = result['raw'][0]
        self.assertEqual(raw['coordinate'], [11, 2])
        self.assertEqual(raw['published']['time_s'], struct.unpack('<f', struct.pack('<f', .11))[0])
        self.assertEqual(raw['score_frame'], 5)  # f32 publication falls below the half-frame tie
        self.assertEqual(math.floor(Fraction(*raw['coordinate']) + Fraction(1, 2)), 6)
        self.assertEqual(raw['downbeat_logit'], -8.)
        self.assertEqual(raw['published']['downbeat_confidence'], .5)  # snapped and floored, not sigmoid(-8)
        candidate = next(p for p in result['candidates'] if p['coordinate'] == [5, 1])
        self.assertEqual(candidate['published']['downbeat_confidence'], adapter.sigmoid(-8.))

    def test_long_plateau_lineage_is_not_nearest_time_or_independent_packets(self):
        b = [-4.] * 22
        b[5:16] = [4.] * 11
        result = authored(b)
        self.assertEqual(len(result['raw']), 6)
        self.assertEqual(len(result['candidates']), 3)  # positive plateau plus two negative background plateaus
        candidate = next(p for p in result['candidates'] if p['coordinate'] == [10, 1])
        parent = candidate['plateau_index']
        self.assertEqual(result['plateaus'][parent]['raw_indices'], list(range(6)))
        self.assertEqual({p['plateau_index'] for p in result['raw']}, {parent})
        self.assertEqual(adapter.summarize(result)['many_raw_to_one_plateau'], 1)
        self.assertGreater(adapter.summarize(result)['raw_center_more_than_three_frames_from_its_candidate'], 0)
        self.assertEqual([float(Fraction(*p['coordinate'])) for p in result['raw']], [5.5, 7.5, 9.5, 11.5, 13.5, 15.])

    def test_candidate_threshold_radius_and_flat_background_semantics(self):
        b = [-4.] * 20
        b[5], b[7], b[14] = 4., 3., -1.
        result = authored(b)
        self.assertEqual([p['coordinate'] for p in result['raw']], [[5, 1]])
        self.assertTrue(any(p['coordinate'] == [7, 1] for p in result['candidates']))
        counts = adapter.summarize(result)['candidate_reasons']
        self.assertEqual(counts['has_raw_lineage'], 1)
        self.assertEqual(counts['radius_three_competition'], 1)
        self.assertGreater(counts['nonpositive_default_threshold'], 0)
        flat = authored([-4.] * 10)
        self.assertEqual(flat['raw'], [])
        self.assertEqual(flat['candidates'][0]['coordinate'], [4, 1])
        self.assertLess(flat['candidates'][0]['published']['confidence'], .5)
        zero = authored([0.] * 10)
        self.assertEqual(zero['raw'], [])  # strict > zero, not >= zero
        self.assertEqual(zero['candidates'][0]['published']['confidence'], .5)

    def test_duration_filter_and_source_edge_coverage_remain_explicit(self):
        b = [-4.] * 8
        b[-1] = 4.
        result = authored(b, duration=.14)
        self.assertGreater(result['raw'][0]['published']['time_s'], .14)
        self.assertEqual([p['coordinate'] for p in result['candidates']], [[2, 1]])  # negative background survives
        self.assertEqual(adapter.summarize(result)['excluded_candidate_plateaus'], 1)
        self.assertEqual(adapter.summarize(result)['raw_without_included_candidate_lineage'], 1)
        self.assertFalse(result['plateaus'][result['raw'][0]['plateau_index']]['included'])

    def test_published_records_are_preserved_and_changed_output_is_rejected(self):
        b, d = [-4., 4., -4.], [-8., 2., -8.]
        obs = adapter.reconstruct(b, d, .06)[0]
        original = copy.deepcopy(obs)
        packets = adapter.assemble(b, d, .06, obs)
        self.assertEqual(obs, original)
        self.assertEqual(packets['raw'][0]['published'], obs['beats'][0])
        for source, field in (('beats', 'time_s'), ('beats', 'confidence'), ('beat_candidates', 'downbeat_confidence')):
            changed = copy.deepcopy(obs)
            changed[source][0][field] += .001
            with self.assertRaises(ValueError):
                adapter.assemble(b, d, .06, changed)
        changed = copy.deepcopy(obs)
        changed['beat_candidates'] = []
        with self.assertRaisesRegex(ValueError, 'count changed'):
            adapter.assemble(b, d, .06, changed)

    def test_numeric_formula_allowance_is_pinned_not_fitted_or_absolute(self):
        value = .25
        for _ in range(8):
            value = math.nextafter(value, 1.)
        self.assertEqual(adapter.probability_ulps(value, .25), 8)
        with self.assertRaises(ValueError):
            adapter.probability_ulps(math.nextafter(value, 1.), .25)
        self.assertEqual(adapter.sigmoid(-1000.), 0.)
        for b, d, duration in (([], [], 0.), ([1.], [], .02), ([math.nan], [0.], .02),
                                ([0.], [math.inf], .02), ([0.], [0.], -.1), ([0.], [0.], 1.)):
            with self.assertRaises(ValueError):
                adapter.reconstruct(b, d, duration)

    def test_rational_bridge_preserves_fractional_matches_without_pooling_sources(self):
        b = [-4.] * 15
        b[5:7] = [4., 4.]
        packets = authored(b)
        result = adapter.packet_ledger([True] * 15, packets['raw'], [Fraction(11, 2)])['result']
        self.assertEqual(result['optimal_matched_count'], 1)
        self.assertEqual(result['minimum_total_absolute_offset'], [0, 1])
        self.assertEqual(result['tick_queries'][0]['expected_query_frames'], 6)
        # Exact distance 3.5 is outside the fixed radius; rounding to 5 would falsely admit it.
        miss = adapter.packet_ledger([True] * 15, packets['raw'], [2])['result']
        self.assertEqual(miss['optimal_matched_count'], 0)
        self.assertIsNone(result['selected_tempo'])
        with self.assertRaisesRegex(ValueError, 'must not be pooled'):
            adapter.packet_ledger([True] * 15, packets['raw'] + packets['candidates'], [5])
        with self.assertRaisesRegex(ValueError, 'duplicate packet'):
            adapter.packet_ledger([True] * 15, packets['raw'] * 2, [5])
        mask = [True] * 15
        mask[6] = False
        with self.assertRaisesRegex(ValueError, 'unavailable response support'):
            adapter.packet_ledger(mask, packets['raw'], [5])


class RationalLedgerControls(unittest.TestCase):
    def test_all_seventy_integer_ledgers_and_ambiguities_remain_identical(self):
        for case in integer.authored_cases():
            observed = [i not in case['unavailable_frames'] for i in range(case['frame_count'])]
            for spec in case['clocks']:
                ticks = integer.clock_ticks(case['frame_count'], spec['period'], spec['phase'], spec['changes'])
                old = integer.ledger(observed, case['responses'], ticks)
                new = rational.ledger(observed, ticks, [p['frame'] for p in case['responses']])
                for key in ('optimal_matched_count', 'unmatched_tick_count', 'unmatched_response_count',
                            'optimal_assignment_count', 'complete_observed_response_cover', 'optimal_pair_counts', 'witness_pairs'):
                    self.assertEqual(new[key], old[key])
                self.assertEqual(new['minimum_total_absolute_offset'], [old['minimum_total_absolute_offset_frames'], 1])
                for side in ('tick', 'response'):
                    self.assertEqual(new[side + '_matched_assignments'],
                                     [p['matched_assignments'] for p in old['tick_queries' if side == 'tick' else 'response_coverage']])
                self.assertEqual([q['available_query_frames'] for q in new['tick_queries']],
                                 [q['available_query_frames'] for q in old['tick_queries']])

    def test_fractional_assignments_match_independent_enumeration(self):
        rng = random.Random(20260906)
        for _ in range(80):
            ticks = sorted(Fraction(v, 2) for v in rng.sample(range(22), rng.randrange(6)))
            responses = sorted(Fraction(v, 2) for v in rng.sample(range(22), rng.randrange(6)))
            rank, paths = brute(ticks, responses)
            result = rational.ledger([True] * 12, ticks, responses)
            self.assertEqual((-result['optimal_matched_count'], Fraction(*result['minimum_total_absolute_offset'])), rank)
            self.assertEqual(result['optimal_assignment_count'], len(paths))
            self.assertIn([tuple(p) for p in result['witness_pairs']], paths)
            for i, count in enumerate(result['tick_matched_assignments']):
                self.assertEqual(count, sum(any(a == i for a, _ in p) for p in paths))
            for j, count in enumerate(result['response_matched_assignments']):
                self.assertEqual(count, sum(any(b == j for _, b in p) for p in paths))
            expected = {(i, j): sum((i, j) in p for p in paths) for i in range(len(ticks)) for j in range(len(responses))}
            self.assertEqual({(p['tick_index'], p['response_index']): p['assignments'] for p in result['optimal_pair_counts']},
                             {key: count for key, count in expected.items() if count})

    def test_unavailable_padding_large_counts_and_limits(self):
        original = rational.ledger([True] * 10, [Fraction(1, 2), Fraction(15, 2)], [1, Fraction(15, 2)])
        padded = rational.ledger([False] * 4 + [True] * 10 + [False] * 3,
                                 [Fraction(9, 2), Fraction(23, 2)], [5, Fraction(23, 2)])
        self.assertEqual(original, padded)
        ticks = [8 * i + j for i in range(64) for j in (1, 3)]
        found = rational.ledger([True] * 512, ticks, [8 * i + 2 for i in range(64)])
        self.assertEqual(found['optimal_assignment_count'], 2 ** 64)
        for observed, ticks, responses, budget in (([], [], [], 10), ([1] * 5, [], [], 10),
            ([True] * 5, [1.5], [], 10), ([True] * 5, [1, 1], [], 10), ([True] * 5, [], [2, 2], 10),
            ([True] * 5, [], [Fraction(9, 2)], 10), ([True] * 5, [1], [1], 1),
            ([True] * 5, [], [], 16642), ([True] * 5, [], [Fraction(1, 1000001)], 10)):
            with self.assertRaises(ValueError):
                rational.ledger(observed, ticks, responses, max_states=budget)


class BackendPacketsFrozenReport(unittest.TestCase):
    def test_contract_hashes_and_scope(self):
        here = Path(__file__).parent
        report = json.loads((here / 'backend-response-packets-v1.json').read_text())
        self.assertEqual(adapter.dense.sha((here / 'backend-response-packets-v1.json').read_bytes()),
                         '6bf7aa6b88aae92b7b91767b1e63a7fe78b4b4275080dba42edd4c39ceef23e0')
        self.assertEqual(report['contract'], adapter.LOCK)
        self.assertEqual(report['script_sha256'], adapter.dense.sha(Path(adapter.__file__).read_bytes()))
        self.assertEqual(report['lock_sha256'], adapter.dense.sha(adapter.LOCK_PATH.read_bytes()))
        for name, digest in (report['helper_sha256'] | report['predecessor_sha256']).items():
            self.assertEqual(digest, adapter.dense.sha((here / name).read_bytes()))
        for key in ('neural_inference', 'production_output_changed', 'new_user_parameters', 'holdout_opened',
                    'training_run', 'fitted_mapping', 'truth_labels_used_for_inventory',
                    'real_clock_ledger_replayed', 'accuracy_improvement_claimed'):
            self.assertIs(report[key], False)

    def test_all_frozen_tracks_and_every_source_denominator(self):
        here = Path(__file__).parent
        report = json.loads((here / 'backend-response-packets-v1.json').read_text())
        prior = json.loads((here / 'dense-clock-evidence-v1.json').read_text())
        self.assertEqual(len(report['cohorts']), 2)
        expected_totals = {
            'artbeat': (375, 1559, 1, 240, 4, 375, 1180, 4),
            'rubato': (9273, 35151, 0, 5141, 83, 9273, 24758, 1120),
        }
        for cohort, old in zip(report['cohorts'], prior['cohorts']):
            self.assertTrue(cohort['complete'])
            s = cohort['totals']
            self.assertEqual(tuple(s[k] for k in ('raw_events', 'candidates', 'excluded_candidate_plateaus',
                                 'raw_downbeat_published_not_direct_frame_sigmoid', 'raw_downbeat_clamped_from_below_half')) +
                             tuple(s['candidate_reasons'][k] for k in ('has_raw_lineage', 'nonpositive_default_threshold',
                                   'radius_three_competition')), expected_totals[cohort['cohort']])
            for key in ('fractional_raw_centers', 'raw_score_frame_differs_from_nominal_round', 'many_raw_to_one_plateau',
                        'raw_without_included_candidate_lineage', 'raw_center_more_than_three_frames_from_its_candidate',
                        'max_confidence_formula_ulps'):
                self.assertEqual(s[key], 0)
            for key in ('cohort', 'frozen_evidence_sha256', 'capture_summary_sha256', 'source_hashes', 'total_frames_per_head'):
                self.assertEqual(cohort[key], old[key])
            self.assertEqual([(t['id'], t['capture_sha256'], t['frame_count']) for t in cohort['cases']],
                             [(t['id'], t['capture_sha256'], t['frame_count']) for t in old['cases']])
            for key, value in cohort['totals'].items():
                if key == 'candidate_reasons':
                    for reason, count in value.items():
                        self.assertEqual(count, sum(t['source_inventory'][key][reason] for t in cohort['cases']))
                elif key == 'max_confidence_formula_ulps':
                    self.assertEqual(value, max(t['source_inventory'][key] for t in cohort['cases']))
                else:
                    self.assertEqual(value, sum(t['source_inventory'][key] for t in cohort['cases']))
            for s in [cohort['totals']] + [t['source_inventory'] for t in cohort['cases']]:
                self.assertEqual(s['candidates'], sum(s['candidate_reasons'].values()))
                self.assertEqual(s['source_plateaus'], s['candidates'] + s['excluded_candidate_plateaus'])
                self.assertLessEqual(s['fractional_raw_centers'], s['raw_events'])
                self.assertLessEqual(s['max_confidence_formula_ulps'], 8)
        public = json.dumps(report)
        for private in ('beat_logits', 'downbeat_logits', 'source_members', '"time_s"', 'truth_times_s', 'D:/', 'C:/'):
            self.assertNotIn(private, public)


if __name__ == '__main__':
    unittest.main()
