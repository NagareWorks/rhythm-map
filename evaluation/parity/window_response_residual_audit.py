"""Truth-assisted fixed-window residual accounting, not an automatic decoder.

Windows and prefix-anchored clock families are frozen before cohort execution.
No phase search, source pooling, fitted penalty, recovery or musical winner.
"""
import argparse
from bisect import bisect_left
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path

import backend_response_packet_audit as packets
import metrical_window_audit as annotation

dense = packets.dense
HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / 'window-response-residuals-lock-v1.json'
LOCK = json.loads(LOCK_PATH.read_text())
CLOCKS = tuple(LOCK['clocks'])
SOURCES = tuple(LOCK['sources'])
REGIMES = ('constant_context', 'change_context', 'ramp_context', 'rubato')
PRIOR = '6bf7aa6b88aae92b7b91767b1e63a7fe78b4b4275080dba42edd4c39ceef23e0'


def annotation_coordinates(times):
    dense.require(all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in times), 'invalid annotation time')
    result = [Fraction(math.floor(Fraction(str(t)) * 1000000 + Fraction(1, 2)), 20000) for t in times]
    dense.require(len(result) >= 2 and all(a < b for a, b in zip(result, result[1:])),
                  'unordered or microsecond-colliding annotation')
    return result


def periodic_ticks(anchor, period, start, end):
    dense.require(period > 0 and start < end, 'invalid periodic domain')
    first, stop = math.ceil((start - anchor) / period), math.ceil((end - anchor) / period)
    if stop - first > 128:
        return None
    return [anchor + k * period - start for k in range(first, stop)]


def supplied_clocks(times, start, end):
    before = bisect_left(times, start)
    if before < 4:
        return None, 'insufficient_prefix'
    if end > times[-1]:
        return None, 'annotation_tail_uncovered'
    anchor, period = times[before - 1], (times[before - 1] - times[before - 4]) / 3
    grids = {'annotated': [t - start for t in times[before:bisect_left(times, end)]]}
    for name, phase, multiplier in (('continuation', 0, 1), ('half_phase_zero', 0, 2),
                                    ('half_phase_one', 1, 2), ('double', 0, Fraction(1, 2))):
        grids[name] = periodic_ticks(anchor + phase * period, period * multiplier, start, end)
    if any(g is None or len(g) > 128 for g in grids.values()):
        return None, 'clock_budget_exceeded'
    return grids, 'eligible'


def crop_packets(source_packets, start, end):
    # Coordinates are window-local; identity, publication, marks and plateau
    # provenance remain those of whole-capture extraction. Never re-extract.
    result = []
    for p in source_packets:
        center = Fraction(*p['coordinate'])
        if start <= center < end:
            result.append(p | {'coordinate': packets.rational.encode(center - start),
                               'score_frame': p['score_frame'] - start})
    return result


def support_inside(source_packets, size):
    return all(0 <= p['score_frame'] < size and math.ceil(Fraction(*p['coordinate'])) < size for p in source_packets)


def state(count, total):
    return 'never' if count == 0 else 'always' if count == total else 'sometimes'


def edge(center, size):
    return math.ceil(center - 3) < 0 or math.floor(center + 3) >= size


def ledger_inventory(result, source_packets, annotated, size):
    ways = result['optimal_assignment_count']
    response_strata = Counter()
    for p, count in zip(source_packets, result['response_matched_assignments']):
        center = Fraction(*p['coordinate'])
        key = '/'.join((state(count, ways), 'boundary' if edge(center, size) else 'interior',
                        'near_annotation' if any(abs(t - center) <= 3 for t in annotated) else 'off_annotation',
                        'positive' if p['beat_logit'] > 0 else 'nonpositive'))
        response_strata[key] += 1
    tick_strata = Counter('/'.join((state(count, ways), 'interior' if q['full_query_observed'] else 'boundary'))
                          for count, q in zip(result['tick_matched_assignments'], result['tick_queries']))
    return dict(ticks=len(result['tick_matched_assignments']), responses=len(source_packets),
                matched=result['optimal_matched_count'], unmatched_ticks=result['unmatched_tick_count'],
                unmatched_responses=result['unmatched_response_count'],
                ambiguous_assignment_windows=int(ways > 1), tick_strata=dict(tick_strata),
                response_strata=dict(response_strata), assignment_count=ways, states=result['states'])


def annotated_miss_causes(raw, candidate, candidates, annotated):
    causes, marks = Counter(), Counter()
    for i, count in enumerate(raw['tick_matched_assignments']):
        raw_state = state(count, raw['optimal_assignment_count'])
        if raw_state != 'never':
            key = 'raw_' + raw_state
        elif not raw['tick_queries'][i]['full_query_observed']:
            key = 'raw_never_boundary_censored'
        else:
            candidate_state = state(candidate['tick_matched_assignments'][i], candidate['optimal_assignment_count'])
            if candidate_state != 'never':
                key = 'raw_never_candidate_' + candidate_state
                signs = {candidates[p['response_index']]['beat_logit'] > 0 for p in candidate['optimal_pair_counts'] if p['tick_index'] == i}
                mark = 'positive_only' if signs == {True} else 'nonpositive_only' if signs == {False} else 'mixed'
                marks[candidate_state + '/' + mark] += 1
            elif any(abs(Fraction(*p['coordinate']) - annotated[i]) <= 3 for p in candidates):
                key = 'raw_never_candidate_competition'
            else:
                key = 'raw_never_no_candidate_in_radius'
        causes[key] += 1
    return dict(causes), dict(marks)


def compare_counts(left, right):
    sign = lambda x: 'more' if x > 0 else 'fewer' if x < 0 else 'equal'
    # Two separate quantities, deliberately not a weighted winner score.
    return '/'.join((sign(left['optimal_matched_count'] - right['optimal_matched_count']),
                     sign(left['unmatched_tick_count'] - right['unmatched_tick_count'])))


def analyze_window(source_packets, times, start, end):
    size = end - start
    selected = {s: crop_packets(source_packets[s], start, end) for s in SOURCES}
    row = dict(frames=size, owned_responses={s: len(selected[s]) for s in SOURCES})
    if size != 400:
        return row | {'status': 'partial_window'}
    grids, status = supplied_clocks(times, start, end)
    if status != 'eligible':
        return row | {'status': status}
    if any(len(p) > 128 for p in selected.values()):
        return row | {'status': 'response_budget_exceeded'}
    if not all(support_inside(p, size) for p in selected.values()):
        return row | {'status': 'packet_support_crosses_window'}
    results = {s: {name: packets.packet_ledger([True] * size, selected[s], grids[name])['result'] for name in CLOCKS}
               for s in SOURCES}
    inventories = {s: {name: ledger_inventory(results[s][name], selected[s], grids['annotated'], size) for name in CLOCKS}
                   for s in SOURCES}
    comparisons = {s: {name: compare_counts(results[s]['continuation'], results[s][name])
                       for name in CLOCKS if name != 'continuation'} for s in SOURCES}
    causes, marks = annotated_miss_causes(results['raw']['annotated'], results['candidate']['annotated'],
                                          selected['candidate'], grids['annotated'])
    return row | dict(status='eligible', inventories=inventories, comparisons=comparisons, annotated_tick_causes=causes,
                      annotated_miss_candidate_marks=marks,
                      identical_annotated_continuation=grids['annotated'] == grids['continuation'])


def context_regime(truth, times, start, end, cohort, tags):
    if cohort == 'rubato' or 'rubato' in tags:
        return 'rubato'
    lo = times[bisect_left(times, start) - 4]
    coordinate = lambda t: Fraction(math.floor(Fraction(str(t)) * 1000000 + Fraction(1, 2)), 20000)
    segments = sorted((s for s in truth['tempo_segments'] if coordinate(s['start_s']) < end and coordinate(s['end_s']) > lo),
                      key=lambda s: s['start_s'])
    covered = lo
    for i, s in enumerate(segments):
        a, b = coordinate(s['start_s']), coordinate(s['end_s'])
        dense.require(s['kind'] in ('constant', 'ramp') and b > a and (a <= lo if i == 0 else a == covered),
                      'missing or overlapping context annotation')
        covered = b
    dense.require(covered >= end, 'incomplete context annotation')
    if any(lo <= coordinate(cp['time_s']) < end for cp in truth['change_points']):
        return 'change_context'
    return 'ramp_context' if any(s['kind'] == 'ramp' for s in segments) else 'constant_context'


def merge_counts(values):
    counts = Counter()
    for value in values:
        counts.update(value)
    return dict(sorted(counts.items()))


def summarize(rows, detailed=True):
    eligible = [r for r in rows if r['status'] == 'eligible']
    result = dict(windows=len(rows), frames=sum(r['frames'] for r in rows),
                  status=dict(sorted(Counter(r['status'] for r in rows).items())), eligible_windows=len(eligible),
                  owned_responses={s: sum(r['owned_responses'][s] for r in rows) for s in SOURCES},
                  eligible_responses={s: sum(r['owned_responses'][s] for r in eligible) for s in SOURCES},
                  identical_annotated_continuation_windows=sum(r['identical_annotated_continuation'] for r in eligible),
                  annotated_tick_causes=merge_counts(r['annotated_tick_causes'] for r in eligible),
                  annotated_miss_candidate_marks=merge_counts(r['annotated_miss_candidate_marks'] for r in eligible), ledgers={})
    for s in SOURCES:
        result['ledgers'][s] = {}
        for name in CLOCKS:
            ledgers = [r['inventories'][s][name] for r in eligible]
            inventory = {k: sum(v[k] for v in ledgers) for k in
                         ('ticks', 'responses', 'matched', 'unmatched_ticks', 'unmatched_responses', 'ambiguous_assignment_windows')}
            inventory.update(max_assignment_count=max((v['assignment_count'] for v in ledgers), default=None),
                             max_states=max((v['states'] for v in ledgers), default=None))
            if detailed:
                inventory.update({k: merge_counts(v[k] for v in ledgers) for k in ('tick_strata', 'response_strata')})
            result['ledgers'][s][name] = inventory
    if detailed:
        result['continuation_comparisons'] = {s: {name: dict(sorted(Counter(r['comparisons'][s][name] for r in eligible).items()))
                                                  for name in CLOCKS if name != 'continuation'} for s in SOURCES}
    return result


def audit(cohort, evidence_path, capture_dir):
    count, evidence_hash = dense.INPUTS[cohort]
    evidence, _ = dense.read_json(evidence_path, evidence_hash)
    predecessor, _ = dense.read_json(HERE / 'backend-response-packets-v1.json', PRIOR)
    dense.require(predecessor['script_sha256'] == dense.sha(Path(packets.__file__).read_bytes()) and
                  predecessor['lock_sha256'] == dense.sha(packets.LOCK_PATH.read_bytes()) and
                  all(dense.sha((HERE / p).read_bytes()) == h for p, h in predecessor['helper_sha256'].items()),
                  'frozen packet adapter or ledger implementation changed')
    prior = next(c for c in predecessor['cohorts'] if c['cohort'] == cohort)
    summary, summary_hash = dense.read_json(Path(capture_dir) / 'summary.json', prior['capture_summary_sha256'])
    records = dense.validate_summary(summary, evidence['cases'], count)
    sources = {k: dense.sha((dense.ROOT / p).read_bytes()) for k, p in dense.SOURCES.items()}
    suite_path = dense.ROOT / 'evaluation/suites' / f'{dense.SUITES[cohort][0]}.json'
    suite, suite_hash = dense.read_json(suite_path, dense.SUITES[cohort][1])
    dense.require([c['id'] for c in suite['cases']] == [c['id'] for c in evidence['cases']], 'suite order changed')
    tracks, all_rows = [], []
    for record, case, item, old_case in zip(records, evidence['cases'], suite['cases'], prior['cases']):
        payload, _ = dense.read_json(Path(capture_dir) / f"{case['id']}.json", record['capture_sha256'])
        beats = dense.validate_capture(payload, record, case, cohort, sources)
        source = packets.assemble(beats, payload['downbeat_logits'], payload['observations']['duration_s'], payload['observations'])
        dense.require(case['id'] == old_case['id'] and packets.summarize(source) == old_case['source_inventory'], 'packet inventory changed')
        truth, truth_hash = dense.read_json(suite_path.parent / item['input']['truth'], case['truth_sha256'])
        annotation.annotation_labels(truth, case, cohort)
        times = annotation_coordinates(case['truth_times_s'])
        source_packets = {'raw': source['raw'], 'candidate': source['candidates']}
        rows = []
        for start in range(0, len(beats), 400):
            end = min(start + 400, len(beats))
            row = analyze_window(source_packets, times, start, end)
            if row['status'] == 'eligible':
                row['regime'] = context_regime(truth, times, start, end, cohort, case.get('tags', []))
            rows.append(row)
        inventory = summarize(rows, detailed=False)
        dense.require(inventory['frames'] == len(beats) and all(inventory['owned_responses'][s] == len(source_packets[s]) for s in SOURCES),
                      'window ownership lost a response or frame')
        tracks.append(dict(id=case['id'], capture_sha256=record['capture_sha256'], truth_sha256=truth_hash,
                           frame_count=len(beats), inventory=inventory))
        all_rows.extend(rows)
    return dict(cohort=cohort, complete=True, frozen_evidence_sha256=evidence_hash, capture_summary_sha256=summary_hash,
                suite_sha256=suite_hash, source_hashes=sources, cases=tracks, all=summarize(all_rows),
                by_regime={r: summarize([row for row in all_rows if row.get('regime') == r]) for r in REGIMES})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for cohort in dense.INPUTS:
        parser.add_argument(f'--{cohort}-evidence', type=Path, required=True)
        parser.add_argument(f'--{cohort}-captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    helpers = ('backend_response_packet_audit.py', 'rational_response_ledger.py', 'metrical_window_audit.py',
               'dense_clock_evidence.py', 'resampler_event_audit.py', 'clock_phase_evidence.py', 'compare_reference.py',
               'phase_tail_audit.py', 'verify_bounded_resampler.py', 'candidate_evidence_audit.py')
    report = dict(schema_version=1, purpose=LOCK['purpose'], contract=LOCK,
                  script_sha256=dense.sha(Path(__file__).read_bytes()), lock_sha256=dense.sha(LOCK_PATH.read_bytes()),
                  helper_sha256={p: dense.sha((HERE / p).read_bytes()) for p in helpers},
                  predecessor_sha256={'backend-response-packets-v1.json': PRIOR}, truth_assisted=True,
                  selected_tempo=None, phase_search=False, neural_inference=False, fitted_mapping=False,
                  holdout_opened=False, training_run=False, production_output_changed=False,
                  new_user_parameters=False, accuracy_improvement_claimed=False,
                  cohorts=[audit(c, getattr(args, f'{c}_evidence'), getattr(args, f'{c}_captures')) for c in dense.INPUTS])
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
