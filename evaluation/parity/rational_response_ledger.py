"""Exact rational-coordinate extension of the frozen integer response ledger.

This is still a supplied-clock accounting reference, not tempo selection.
The old byte-addressed audit remains unchanged as an independent integer oracle.
"""
from fractions import Fraction
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def coordinates(values, size):
    require(len(values) <= 128, 'event limit exceeded')
    require(all(type(v) in (int, Fraction) for v in values), 'exact integer or Fraction coordinates required')
    values = [Fraction(v) for v in values]
    require(all(0 <= v < size and v.denominator <= 1000000 for v in values) and
            all(a < b for a, b in zip(values, values[1:])), 'invalid or unordered coordinates')
    return values


def encode(value):
    return [value.numerator, value.denominator]


def ledger(observed, ticks, responses, max_states=16641):
    """Maximum cardinality, then minimum exact absolute offset within 3 frames.

Canonical skip-tick/match-later-response transitions count each matching once.
The complete bounded table differs from the old lazy table only in allocation;
all optimal assignments, not just a single witness, contribute integer counts.
"""
    require(1 <= len(observed) <= 4096 and all(type(v) is bool for v in observed), 'invalid availability domain')
    require(type(max_states) is int and 1 <= max_states <= 16641, 'invalid state budget')
    ticks, responses = coordinates(ticks, len(observed)), coordinates(responses, len(observed))
    # A fractional response can require both bracketing frames; no invented
    # interpolated observation at a missing endpoint is accepted.
    require(all(observed[math.floor(r)] and math.ceil(r) < len(observed) and observed[math.ceil(r)]
                for r in responses), 'unavailable response support')
    n, m = len(ticks), len(responses)
    states = (n + 1) * (m + 1)
    require(states <= max_states, 'exact state budget exceeded; no partial result')
    compatible = [[j for j, r in enumerate(responses) if abs(t - r) <= 3] for t in ticks]
    table = [[None] * (m + 1) for _ in range(n + 1)]
    table[n] = [(0, Fraction(0), 1)] * (m + 1)

    def choices(i, first):
        yield None, first, table[i + 1][first]
        for j in compatible[i]:
            if j >= first:
                matched, distance, ways = table[i + 1][j + 1]
                yield j, j + 1, (matched + 1, distance + abs(ticks[i] - responses[j]), ways)

    for i in reversed(range(n)):
        for first in range(m + 1):
            options = list(choices(i, first))
            rank = min((-value[0], value[1]) for _, _, value in options)
            ways = sum(value[2] for _, _, value in options if (-value[0], value[1]) == rank)
            table[i][first] = -rank[0], rank[1], ways

    def optimal_edges(i, first):
        if i == n:
            return []
        return [(j, following) for j, following, value in choices(i, first)
                if value[:2] == table[i][first][:2]]

    prefix = [[0] * (m + 1) for _ in range(n + 1)]
    prefix[0][0] = 1
    tick_counts, response_counts, pair_counts = [0] * n, [0] * m, {}
    for i in range(n):
        for first, count in enumerate(prefix[i]):
            if not count:
                continue
            for j, following in optimal_edges(i, first):
                prefix[i + 1][following] += count
                if j is not None:
                    ways = count * table[i + 1][following][2]
                    tick_counts[i] += ways
                    response_counts[j] += ways
                    pair_counts[i, j] = pair_counts.get((i, j), 0) + ways
    witness, first = [], 0
    for i in range(n):
        j, first = min(optimal_edges(i, first), key=lambda item: (
            item[0] is None, abs(ticks[i] - responses[item[0]]) if item[0] is not None else 0,
            item[0] if item[0] is not None else -1))
        if j is not None:
            witness.append([i, j])
    queries = []
    for t in ticks:
        left, right = math.ceil(t - 3), math.floor(t + 3)
        available = sum(0 <= f < len(observed) and observed[f] for f in range(left, right + 1))
        queries.append(dict(expected_query_frames=right - left + 1, available_query_frames=available,
                            full_query_observed=available == right - left + 1))
    matched, distance, ways = table[0][0]
    return dict(optimal_matched_count=matched, unmatched_tick_count=n - matched, unmatched_response_count=m - matched,
                minimum_total_absolute_offset=encode(distance), optimal_assignment_count=ways,
                tick_matched_assignments=tick_counts, response_matched_assignments=response_counts,
                optimal_pair_counts=[dict(tick_index=i, response_index=j, assignments=count)
                                     for (i, j), count in sorted(pair_counts.items())],
                witness_pairs=witness, tick_queries=queries, states=states,
                complete_observed_response_cover=matched == m if m else None, selected_tempo=None)
