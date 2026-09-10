"""Post-hoc geometry of frozen proposal misses; no new candidates or decisions.

Partitions are diagnostic witnesses, not perceptual truth or causal proofs.
Boundary comparisons and offset-compatible residuals never update old coverage.
"""
from bisect import bisect_left
from collections import Counter
import statistics

import clock_proposals as clocks

TOLERANCE = .06  # Unchanged native-grid tolerance; no sweep or fitted alignment.


def ordered(values):
    clocks.require(type(values) is list and all(clocks.number(t) and t >= 0 for t in values)
                   and all(a < b for a, b in zip(values, values[1:])), 'invalid diagnostic timeline')


def matches(predicted, reference):
    """Chronological maximum-cardinality tolerance matching, not least-error fit."""
    ordered(predicted)
    ordered(reference)
    i = j = 0
    pairs = []
    while i < len(predicted) and j < len(reference):
        delta = predicted[i] - reference[j]
        if abs(delta) <= TOLERANCE:
            pairs.append((i, j))
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return pairs


def within(values, query):
    return values[bisect_left(values, query[0]):bisect_left(values, query[1])]


def validate_query(query):
    clocks.require(type(query) is list and len(query) == 2 and all(clocks.number(t) for t in query)
                   and 0 <= query[0] < query[1], 'invalid diagnostic query')


def boundary_inventory(predicted, reference, query):
    """Match a fixed 60 ms halo, but count only events owned by the old query.

    Inputs are physical timestamps, never translated. Exterior events can only
    witness a match across an old crop boundary, not repair an interior miss.
    """
    a, b = query
    pairs = matches(predicted, reference)
    paired_p, paired_r = {i for i, _ in pairs}, {j for _, j in pairs}
    missing = [j for j, t in enumerate(reference) if a <= t < b and j not in paired_r]
    extra = [i for i, t in enumerate(predicted) if a <= t < b and i not in paired_p]
    crossings = sum((a <= predicted[i] < b) != (a <= reference[j] < b) for i, j in pairs)
    return dict(unmatched_reference=len(missing), unmatched_prediction=len(extra),
                boundary_crossing_pairs=crossings), missing


def candidate_diagnosis(clock, reference, query):
    clocks.validate_clock(clock)
    ordered(reference)
    validate_query(query)
    original = clocks.ticks(clock, query)
    target = within(reference, query)
    clocks.require(len(target) >= 2, 'insufficient query reference ticks')
    same_count = len(original) == len(target)
    residuals = [a - b for a, b in zip(original, target)] if same_count else None
    strict = same_count and all(abs(r) <= TOLERANCE for r in residuals)
    halo = [query[0] - TOLERANCE, query[1] + TOLERANCE]
    supported = [max(clock['knots'][0][0], halo[0]), min(clock['knots'][-1][0], halo[1])]
    try:
        extended = clocks.ticks(clock, supported)
    except ValueError as error:
        if str(error) != 'tick population exceeds audit budget':
            raise
        return dict(kind='diagnostic_tick_budget_exceeded', strict_match=strict,
                    predicted_ticks=len(original), reference_ticks=len(target))
    boundary, _ = boundary_inventory(extended, within(reference, halo), query)
    # No offset is chosen or applied. The span only tests whether the ordered
    # residual intervals have a common translation at the existing tolerance.
    residual = None if not residuals else dict(
        minimum_s=min(residuals), maximum_s=max(residuals), median_s=statistics.median(residuals),
        span_s=max(residuals) - min(residuals), endpoint_delta_s=residuals[-1] - residuals[0])
    edge_only = not strict and boundary['boundary_crossing_pairs'] > 0 and not (
        boundary['unmatched_reference'] or boundary['unmatched_prediction'])
    kind = ('strict_match' if strict else 'query_boundary_only_witness' if edge_only else
            'equal_count_offset_compatible' if residual and residual['span_s'] <= 2 * TOLERANCE else
            'equal_count_nonuniform_error' if same_count else 'count_mismatch')
    return dict(kind=kind, strict_match=strict, predicted_ticks=len(original), reference_ticks=len(target),
                count_delta=len(original) - len(target), boundary=boundary, ordered_residuals=residual)


def anchor_diagnosis(raw, model_candidates, reference, query):
    """Reference-relative anchor inventory, not event recovery or musical labels."""
    for times in (raw, model_candidates, reference):
        ordered(times)
    validate_query(query)
    halo = [query[0] - TOLERANCE, query[1] + TOLERANCE]
    local_raw, local_ref = within(raw, halo), within(reference, halo)
    inventory, missing = boundary_inventory(local_raw, local_ref, query)
    missing_times = [local_ref[j] for j in missing]
    # Potential peaks for missing reference anchors, one-to-one among those
    # anchors only. No joint assignment with retained raw events is claimed.
    possible = len(matches(within(model_candidates, halo), missing_times))
    inventory.update(missing_reference_with_candidate_peak=possible,
                     missing_reference_without_candidate_peak=len(missing) - possible)
    full_matches = dict(matches(raw, reference))
    advances = Counter()
    unanchored = hidden = displaced = 0
    for i, (a, b) in enumerate(zip(raw, raw[1:])):
        if not (a < query[1] and b > query[0]):
            continue
        if i not in full_matches or i + 1 not in full_matches:
            unanchored += 1
            continue
        j, k = full_matches[i], full_matches[i + 1]
        advances[k - j] += 1
        for index in range(j + 1, k):
            if query[0] <= reference[index] < query[1]:
                hidden += 1
                interpolated = a + (b - a) * (index - j) / (k - j)
                displaced += abs(interpolated - reference[index]) > TOLERANCE
    inventory.update(anchored_interval_advances={str(k): advances[k] for k in sorted(advances)},
                     mixed_reference_advances=len(advances) > 1,
                     intervals_without_two_reference_anchors=unanchored,
                     interior_reference_ticks_between_matched_raw_anchors=hidden,
                     linear_interpolation_outside_tolerance=displaced)
    return inventory


def diagnose(generated, reference, raw, model_candidates):
    """Evaluate the existing list in place; labels cannot create another clock."""
    before = clocks.canonical_hash(generated)
    old = clocks.reference_check(generated, reference)
    if old['status'] == 'reference_unavailable':
        return dict(partition='reference_unavailable', original_reference=old, candidates=[], anchors=None)
    rows = []
    for clock in generated['clocks']:
        try:
            rows.append(candidate_diagnosis(clock, reference, generated['query_s']))
        except ValueError as error:
            if str(error) != 'tick population exceeds audit budget':
                raise
            rows.append(dict(kind='original_tick_budget_exceeded', strict_match=False))
    kinds = {r['kind'] for r in rows}
    partition = ('original_covered' if old['status'] == 'native_grid_covered' else
                 'no_candidates' if not rows else
                 'query_boundary_only_witness' if 'query_boundary_only_witness' in kinds else
                 'equal_count_offset_compatible' if 'equal_count_offset_compatible' in kinds else
                 'equal_count_nonuniform_error' if 'equal_count_nonuniform_error' in kinds else
                 'diagnostic_budget_unavailable' if any('budget' in k for k in kinds) else 'no_count_matched_candidate')
    clocks.require(sum(r['strict_match'] for r in rows) == old['matching_candidates'], 'original matching count changed')
    clocks.require(clocks.canonical_hash(generated) == before, 'diagnosis mutated frozen proposals')
    return dict(partition=partition, original_reference=old, candidates=rows,
                anchors=anchor_diagnosis(raw, model_candidates, reference, generated['query_s']))


def summarize(rows):
    partitions = Counter(r['diagnosis']['partition'] for r in rows)
    misses = [r['diagnosis'] for r in rows if r['diagnosis']['original_reference']['status'] == 'native_grid_missed']
    anchors = [r['anchors'] for r in misses]
    return dict(queries=len(rows), partition=dict(sorted(partitions.items())),
                original_reference_status=dict(sorted(Counter(r['diagnosis']['original_reference']['status'] for r in rows).items())),
                failed_query_flags=dict(
                    missing_raw_reference_anchor=sum(a['unmatched_reference'] > 0 for a in anchors),
                    extra_raw_event=sum(a['unmatched_prediction'] > 0 for a in anchors),
                    all_missing_raw_anchors_have_candidate_peaks=sum(a['unmatched_reference'] > 0 and
                        a['missing_reference_without_candidate_peak'] == 0 for a in anchors),
                    some_missing_raw_anchor_has_no_candidate_peak=sum(a['missing_reference_without_candidate_peak'] > 0 for a in anchors),
                    mixed_reference_advances=sum(a['mixed_reference_advances'] for a in anchors),
                    misplaced_linear_interpolation=sum(a['linear_interpolation_outside_tolerance'] > 0 for a in anchors),
                    equal_count_nonuniform_candidate=sum(any(c['kind'] == 'equal_count_nonuniform_error' for c in r['candidates']) for r in misses)),
                failed_query_anchor_totals={key: sum(a[key] for a in anchors) for key in
                    ('unmatched_reference', 'unmatched_prediction', 'missing_reference_with_candidate_peak',
                     'missing_reference_without_candidate_peak', 'intervals_without_two_reference_anchors',
                     'interior_reference_ticks_between_matched_raw_anchors', 'linear_interpolation_outside_tolerance')})
