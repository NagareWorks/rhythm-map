import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import unittest

import clock_response_ledger_audit as audit


def exhaustive(ticks, frames):
    """Independent combinations of tick/response subsets, not a skip-path DAG."""
    possibilities = []
    for count in range(min(len(ticks), len(frames)) + 1):
        for left in itertools.combinations(range(len(ticks)), count):
            for right in itertools.combinations(range(len(frames)), count):
                pairs = list(zip(left, right))
                if all(abs(ticks[i] - frames[j]) <= 3 for i, j in pairs):
                    possibilities.append(((-count, sum(abs(ticks[i] - frames[j]) for i, j in pairs)), pairs))
    best = min(r for r, _ in possibilities)
    optimal = [pairs for rank, pairs in possibilities if rank == best]
    return best, optimal


def case_map():
    return {c['id']: c for c in audit.authored_cases()}


def ledgers(case):
    return {c['id']: c['ledger'] for c in audit.evaluate(case)['clocks']}


class ClockResponseLedgerControls(unittest.TestCase):
    def test_full_span_exposes_half_time_subset_but_does_not_penalize_omissions(self):
        found = ledgers(case_map()['constant'])
        self.assertEqual((found['constant']['optimal_matched_count'], found['half']['optimal_matched_count']), (12, 6))
        self.assertEqual(found['half']['unmatched_response_count'], 6)
        self.assertEqual(found['constant']['unmatched_response_count'], 0)
        # A denser clock can explain the same responses with omissions; no invented penalty.
        self.assertEqual(found['double']['optimal_matched_count'], 12)
        self.assertEqual(found['double']['unmatched_tick_count'], 12)
        self.assertTrue(found['double']['complete_observed_response_cover'])
        self.assertIsNone(audit.evaluate(case_map()['constant'])['selected_tempo'])

    def test_omissions_and_genuine_changes_keep_identical_response_ambiguity(self):
        cases = case_map()
        for first, second in (('slowdown', 'same_responses_constant_tail_omissions'),
                              ('speedup', 'same_responses_fast_clock_prefix_omissions')):
            self.assertEqual(cases[first]['responses'], cases[second]['responses'])
            self.assertEqual(audit.evaluate(cases[first]), audit.evaluate(cases[second]))
        slow = ledgers(cases['slowdown'])
        self.assertEqual(slow['slowdown']['unmatched_tick_count'], 0)
        self.assertEqual(slow['constant']['unmatched_tick_count'], 3)
        self.assertTrue(slow['constant']['complete_observed_response_cover'])
        self.assertTrue(slow['slowdown']['complete_observed_response_cover'])
        fast = ledgers(cases['speedup'])
        self.assertGreater(fast['constant']['unmatched_response_count'], 0)
        self.assertTrue(fast['speedup']['complete_observed_response_cover'])
        self.assertTrue(fast['double']['complete_observed_response_cover'])
        renamed = copy.deepcopy(cases['constant'])
        renamed['id'] = 'unrelated_truth_label'
        for i, clock in enumerate(renamed['clocks']):
            clock['id'] = f'unknown_{i}'
        self.assertEqual(list(ledgers(renamed).values()), list(ledgers(cases['constant']).values()))

    def test_weak_subdivision_failure_is_not_hidden_by_counting_more_responses(self):
        cases = case_map()
        weak, strong = ledgers(cases['weak_subdivision_responses']), ledgers(cases['strong_subdivision_responses'])
        self.assertEqual(weak, strong)  # marks not a calibrated reliability law
        self.assertEqual(weak['constant']['unmatched_response_count'], 12)
        self.assertEqual(weak['double']['unmatched_response_count'], 0)
        self.assertEqual(ledgers(cases['all_weak_constant_responses']), ledgers(cases['constant']))
        self.assertFalse(audit.evaluate(cases['weak_subdivision_responses'])['detected_beats_emitted'])

    def test_no_response_unavailable_and_weak_are_different_contract_states(self):
        cases = case_map()
        a = ledgers(cases['constant_middle_omissions'])['constant']
        b = ledgers(cases['constant_missing_observations'])['constant']
        self.assertEqual(a['witness_pairs'], b['witness_pairs'])
        self.assertEqual(a['unmatched_tick_count'], 2)
        for x in (64, 76):
            first = next(t for t in a['tick_queries'] if t['frame'] == x)
            second = next(t for t in b['tick_queries'] if t['frame'] == x)
            self.assertEqual(first['status'], 'unmatched_in_all_optima')
            self.assertTrue(first['full_query_observed'])
            self.assertFalse(second['full_query_observed'])
        empty = ledgers(cases['no_responses_observed'])['constant']
        missing = ledgers(cases['no_observations'])['constant']
        self.assertEqual((empty['observed_frames'], missing['observed_frames']), (148, 0))
        self.assertIsNone(empty['complete_observed_response_cover'])
        self.assertIsNone(missing['complete_observed_response_cover'])

    def test_one_response_cannot_support_two_ticks_and_witness_is_not_unique(self):
        result = audit.ledger([True] * 11, audit.response_rows([5]), [2, 8])
        self.assertEqual((result['optimal_matched_count'], result['unmatched_tick_count']), (1, 1))
        self.assertEqual(result['optimal_assignment_count'], 2)
        self.assertEqual([t['status'] for t in result['tick_queries']], ['matched_in_some_optima'] * 2)
        self.assertEqual(result['response_coverage'][0]['status'], 'matched_in_all_optima')
        self.assertEqual(result['witness_pairs'], [[0, 0]])
        self.assertEqual(result['optimal_pair_counts'], [dict(tick_index=i, response_index=0, assignments=1) for i in (0, 1)])
        flipped = audit.ledger([True] * 11, audit.response_rows([2, 8]), [5])
        self.assertEqual(flipped['optimal_assignment_count'], 2)
        self.assertEqual([t['status'] for t in flipped['response_coverage']], ['matched_in_some_optima'] * 2)

    def test_exhaustive_matchings_verify_counts_marginals_and_witness(self):
        rng = random.Random(20260906)
        for _ in range(80):
            ticks = sorted(rng.sample(range(12), rng.randrange(7)))
            frames = sorted(rng.sample(range(12), rng.randrange(7)))
            best, optimal = exhaustive(ticks, frames)
            found = audit.ledger([True] * 12, audit.response_rows(frames), ticks)
            self.assertEqual((-found['optimal_matched_count'], found['minimum_total_absolute_offset_frames']), best)
            self.assertEqual(found['optimal_assignment_count'], len(optimal))
            self.assertIn([tuple(p) for p in found['witness_pairs']], optimal)
            for i, query in enumerate(found['tick_queries']):
                self.assertEqual(query['matched_assignments'], sum(any(a == i for a, _ in pairs) for pairs in optimal))
            for j, query in enumerate(found['response_coverage']):
                self.assertEqual(query['matched_assignments'], sum(any(b == j for _, b in pairs) for pairs in optimal))
            expected = {(i, j): sum((i, j) in pairs for pairs in optimal)
                        for i in range(len(ticks)) for j in range(len(frames))}
            self.assertEqual({(p['tick_index'], p['response_index']): p['assignments'] for p in found['optimal_pair_counts']},
                             {p: n for p, n in expected.items() if n})

    def test_skip_paths_do_not_duplicate_matchings_and_counts_are_exact_large_integers(self):
        found = audit.ledger([True] * 30, audit.response_rows([20, 25]), [1, 5])
        self.assertEqual(found['optimal_assignment_count'], 1)  # unique empty matching, not six skip paths
        ticks = [8 * i + d for i in range(64) for d in (1, 3)]
        responses = audit.response_rows([8 * i + 2 for i in range(64)])
        found = audit.ledger([True] * 512, responses, ticks)
        self.assertEqual(found['optimal_assignment_count'], 2 ** 64)
        self.assertEqual(found['optimal_matched_count'], 64)
        self.assertEqual({t['matched_assignments'] for t in found['tick_queries']}, {2 ** 63})
        self.assertLessEqual(found['states'], audit.LOCK['max_states'])

    def test_common_unavailable_padding_translation_and_mark_invariance(self):
        observed = [True] * 16
        original = audit.ledger(observed, audit.response_rows([0, 7, 14]), [0, 6, 15])
        padded = audit.ledger([False] * 5 + observed + [False] * 4, audit.response_rows([5, 12, 19]), [5, 11, 20])
        for key in ('optimal_matched_count', 'optimal_assignment_count', 'minimum_total_absolute_offset_frames',
                    'witness_pairs', 'optimal_pair_counts'):
            self.assertEqual(padded[key], original[key])
        self.assertEqual([t['available_query_frames'] for t in padded['tick_queries']],
                         [t['available_query_frames'] for t in original['tick_queries']])
        marks = [dict(r, beat_logit=-999., downbeat_logit=1000.) for r in audit.response_rows([0, 7, 14])]
        self.assertEqual(audit.ledger(observed, marks, [0, 6, 15]), original)

    def test_complete_clock_rendering_change_semantics_and_edges(self):
        self.assertEqual(audit.clock_ticks(13, 4, 0), [0, 4, 8, 12])
        self.assertEqual(audit.clock_ticks(13, 4, 0, [(4, 2)]), [0, 4, 6, 8, 10, 12])
        self.assertEqual(audit.clock_ticks(3, 6, 4), [])
        for args in ((13, 0, 0), (13, 4, 4), (13, 4, 0, [(5, 2)]),
                     (13, 4, 0, [(4, 2), (4, 3)]), (4097, 1, 0), (129, 1, 0)):
            with self.assertRaises(ValueError):
                audit.clock_ticks(*args)

    def test_invalid_inputs_and_budget_never_return_partial_evidence(self):
        for observed, responses, ticks in (([], [], []), ([1] * 5, [], []), ([True] * 5, [], [2, 2]),
            ([True] * 5, [], [5]), ([True] * 5, [], [2.]), ([False] * 5, audit.response_rows([2]), []),
            ([True] * 5, audit.response_rows([2, 2]), []), ([True] * 5, audit.response_rows([3, 2]), []),
            ([True] * 5, [dict(frame=2, beat_logit=1.)], []),
            ([True] * 5, [dict(frame=2, beat_logit=float('nan'), downbeat_logit=0.)], []),
            ([True] * 5, [dict(frame=2, beat_logit=True, downbeat_logit=0.)], []),
            ([True] * 4097, [], []), ([True] * 129, audit.response_rows(list(range(129))), [])):
            with self.assertRaises(ValueError):
                audit.ledger(observed, responses, ticks)
        for budget in (0, 1, 16642, 3.):
            with self.assertRaises(ValueError):
                audit.ledger([True] * 10, audit.response_rows([3, 6]), [3, 6], max_states=budget)


class ClockResponseLedgerReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.here = Path(__file__).parent
        cls.report = json.loads((cls.here / 'clock-response-ledger-v1.json').read_text())

    def test_frozen_inputs_provenance_complete_reproduction_and_scope(self):
        self.assertEqual(self.report, audit.make_report())
        self.assertEqual(len(self.report['cases']), 14)
        self.assertEqual(self.report['contract'], audit.LOCK)
        for name, digest in self.report['predecessor_sha256'].items():
            self.assertEqual(digest, hashlib.sha256((self.here / name).read_bytes()).hexdigest())
        for flag in ('real_music_evaluated', 'neural_inference', 'likelihood_adapter', 'fitted_parameters', 'holdout_opened',
                     'training_run', 'production_output_changed', 'user_parameters_added', 'unknown_clock_search',
                     'accuracy_improvement_claimed'):
            self.assertIs(self.report[flag], False)
        self.assertIsNone(audit.LOCK['unmatched_tick_penalty'])
        self.assertIsNone(audit.LOCK['unmatched_response_likelihood'])
        self.assertIs(audit.LOCK['complete_response_cover_selects_tempo'], False)

    def test_every_response_and_tick_is_accounted_for_without_a_tempo_winner(self):
        for case in self.report['cases']:
            self.assertIsNone(case['result']['selected_tempo'])
            self.assertIs(case['result']['detected_beats_emitted'], False)
            for clock in case['result']['clocks']:
                r = clock['ledger']
                self.assertEqual(r['frame_count'], r['observed_frames'] + r['unavailable_frames'])
                n = r['optimal_matched_count']
                self.assertEqual(r['response_count'], n + r['unmatched_response_count'])
                self.assertEqual(r['clock_tick_count'], n + r['unmatched_tick_count'])
                self.assertEqual(len(r['witness_pairs']), n)
                self.assertEqual(len({i for i, _ in r['witness_pairs']}), n)
                self.assertEqual(len({j for _, j in r['witness_pairs']}), n)
                self.assertEqual(sum(q['matched_assignments'] for q in r['tick_queries']), n * r['optimal_assignment_count'])
                self.assertEqual(sum(q['matched_assignments'] for q in r['response_coverage']), n * r['optimal_assignment_count'])


if __name__ == '__main__':
    unittest.main()
