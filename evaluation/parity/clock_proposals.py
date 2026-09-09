"""Bounded observation-only clock proposals, research baseline, not a decoder.

Two constructions in one fixed proposal set: local median-interval continuation
and the continuous clock interpolating raw events. Neither is a selected result.
No confidence, source identity, annotation, score, or per-track option is input.
"""
from bisect import bisect_right
import hashlib
import json
import math
import statistics

HALO_S = 126 / 50
CAP = 8
MAX_INTERVALS = 128
EPS = 1e-9  # Numerical equivalence only, not a musical decision tolerance.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_clock(clock):
    require(type(clock) is dict and set(clock) == {'knots'}, 'invalid clock fields')
    knots = clock['knots']
    require(type(knots) is list and 2 <= len(knots) <= MAX_INTERVALS + 3,
            'invalid clock knot population')
    require(all(type(k) is list and len(k) == 2 and all(number(v) for v in k)
                and k[0] >= 0 for k in knots), 'invalid clock coordinates')
    require(all(a[0] < b[0] and a[1] < b[1] for a, b in zip(knots, knots[1:])),
            'clock must have positive finite slope and continuous phase')
    require(all(number((b[1] - a[1]) / (b[0] - a[0])) for a, b in zip(knots, knots[1:])),
            'nonfinite clock slope')


def phase(clock, time):
    """Piecewise-linear unwrapped cycles; slope may step, phase never resets."""
    knots = clock['knots']
    require(number(time) and knots[0][0] <= time <= knots[-1][0], 'clock support exceeded')
    i = min(len(knots) - 2, max(0, bisect_right([k[0] for k in knots], time) - 1))
    a, b = knots[i:i + 2]
    return a[1] + (time - a[0]) * (b[1] - a[1]) / (b[0] - a[0])


def equivalent(a, b):
    if a['knots'][0][0] != b['knots'][0][0] or a['knots'][-1][0] != b['knots'][-1][0]:
        return False
    points = sorted({k[0] for c in (a, b) for k in c['knots']})
    offset = round(phase(a, points[0]) - phase(b, points[0]))
    return all(abs(phase(a, t) - phase(b, t) - offset) <= EPS for t in points)


def bounded_set(clocks):
    unique = []
    for clock in clocks:
        validate_clock(clock)
        if not any(equivalent(clock, old) for old in unique):
            unique.append(clock)
    # Never return a favorable prefix on overflow.
    return dict(status='overflow' if len(unique) > CAP else 'empty' if not unique else 'ready',
                proposed_count=len(clocks), unique_count=len(unique),
                clocks=[] if len(unique) > CAP else unique)


def validate_observations(observations):
    require(type(observations) is dict and set(observations) ==
            {'duration_s', 'beat_times_s', 'valid_spans_s'}, 'unexpected observation fields')
    duration, times, spans = (observations[k] for k in ('duration_s', 'beat_times_s', 'valid_spans_s'))
    require(number(duration) and duration > 0, 'invalid duration')
    require(type(times) is list and all(number(t) and 0 <= t <= duration for t in times)
            and all(a < b for a, b in zip(times, times[1:])), 'invalid raw event timeline')
    require(type(spans) is list and all(type(s) is list and len(s) == 2 and
            all(number(v) for v in s) and 0 <= s[0] < s[1] <= duration for s in spans)
            and all(a[1] < b[0] for a, b in zip(spans, spans[1:])), 'invalid valid spans')
    require(all(any(a <= t <= b for a, b in spans) for t in times), 'event outside valid support')


def query_grid(duration):
    require(number(duration) and duration > 0, 'invalid query duration')
    start = 0
    while True:
        end = min(start + 4, duration)
        yield [start, end]
        if end == duration:
            return
        start += 2


def generate(observations, query):
    validate_observations(observations)
    require(type(query) is list and len(query) == 2 and all(number(v) for v in query)
            and 0 <= query[0] < query[1] <= observations['duration_s'], 'invalid query')
    start, end = query
    result = dict(query_s=list(query), observation_sha256=canonical_hash(observations),
                  full_readout_context=False, proposed_count=0, unique_count=0, clocks=[])
    spans = [s for s in observations['valid_spans_s'] if s[0] <= start < end <= s[1]]
    if not spans:
        return result | dict(status='invalid_query_support')
    left, right = spans[0]
    result['full_readout_context'] = left <= start - HALO_S and right >= end + HALO_S
    times = [t for t in observations['beat_times_s'] if left <= t <= right]
    if len(times) < 2:
        return result | dict(status='empty')
    if times[0] > start or times[-1] < end:
        return result | dict(status='unbracketed_query')
    a, b = max(left, times[0], start - HALO_S), min(right, times[-1], end + HALO_S)
    intervals = [(i, x, y) for i, (x, y) in enumerate(zip(times, times[1:])) if x < b and y > a]
    if len(intervals) > MAX_INTERVALS:
        return result | dict(status='observation_budget_exceeded')
    result['full_clock_context'] = a <= start - HALO_S and b >= end + HALO_S
    result['intervals_used'] = len(intervals)
    # Phase indices retain the event order within this contiguous valid span.
    first, last = intervals[0][0], intervals[-1][0] + 1
    event_clock = {'knots': [[times[i], float(i)] for i in range(first, last + 1)]}
    cropped = {'knots': [[a, phase(event_clock, a)]] +
               [[t, float(i)] for i, t in enumerate(times) if a < t < b] + [[b, phase(event_clock, b)]]}
    period = statistics.median(y - x for _, x, y in intervals)
    anchor = min(range(first, last + 1), key=lambda i: (abs(times[i] - (start + end) / 2), i))
    constant = {'knots': [[t, anchor + (t - times[anchor]) / period] for t in (a, b)]}
    # Two half-level phases are different clocks. Integer phase aliases are not.
    clocks = [{'knots': [[t, value * scale + shift] for t, value in base['knots']]}
              for base in (constant, cropped) for scale, shift in ((1, 0), (.5, 0), (.5, .5), (2, 0))]
    return result | bounded_set(clocks)


def ticks(clock, query):
    validate_clock(clock)
    start, end = query
    lo, hi = phase(clock, start), phase(clock, end)
    first, stop = math.ceil(lo - EPS), math.ceil(hi - EPS)
    require(stop - first <= 128, 'tick population exceeds audit budget')
    out = []
    knots = clock['knots']
    values = [k[1] for k in knots]
    for cycle in range(first, stop):
        i = min(len(knots) - 2, max(0, bisect_right(values, cycle) - 1))
        a, b = knots[i:i + 2]
        time = a[0] + (cycle - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        if abs(time - start) <= EPS:
            time = start
        if start <= time < end:
            out.append(time)
    return out


def reference_check(result, reference_times):
    """Post-generation native annotation-grid proxy, NOT independent listening."""
    require(type(reference_times) is list and all(number(t) and t >= 0 for t in reference_times)
            and all(a < b for a, b in zip(reference_times, reference_times[1:])), 'invalid reference timeline')
    start, end = result['query_s']
    truth = [t for t in reference_times if start <= t < end]
    if not reference_times or reference_times[0] > start or reference_times[-1] < end or len(truth) < 2:
        return dict(status='reference_unavailable', matching_candidates=0)
    matched = budget_exceeded = 0
    for clock in result['clocks']:
        try:
            predicted = ticks(clock, [start, end])
        except ValueError as error:
            if str(error) != 'tick population exceeds audit budget':
                raise
            budget_exceeded += 1
            continue  # Whole query remains; an over-budget candidate cannot pass.
        matched += len(predicted) == len(truth) and all(abs(a - b) <= .06 for a, b in zip(predicted, truth))
    return dict(status='native_grid_covered' if matched else 'native_grid_missed',
                matching_candidates=matched, tick_budget_exceeded_candidates=budget_exceeded)
