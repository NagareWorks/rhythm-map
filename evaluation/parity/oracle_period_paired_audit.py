"""A bounded, truth-assisted count/sign gate, not an automatic tempo decoder.

Oracle nominal periods remove prefix period estimation as a nuisance. All four
prefix-derived phases are retained, shared across competing shape/density clocks.
No clock or phase is optimized against response marks. Failure is not a theorem
about all representations or a proof that training is uniquely necessary.
"""
import argparse
from bisect import bisect_left
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path

import paired_response_replay_audit as replay
import prefix_clock_drift_audit as geometry

HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / 'oracle-period-paired-lock-v1.json'
LOCK = json.loads(LOCK_PATH.read_text())
SHAPES = ('constant', 'step')
SCALES = {'native': (Fraction(1), 0), 'half_zero': (Fraction(2), 0),
          'half_one': (Fraction(2), 1), 'double': (Fraction(1, 2), 0)}
NAMES = tuple(f'{s}/{k}' for s in SHAPES for k in SCALES)
PINS = {
    'paired_response_replay_audit.py': 'e750e5df4f2ff1769419875269e086e632e43c8cf1b87118e66c9d4562509617',
    'paired-response-replay-lock-v1.json': 'd40ba206367c0b4cf970e15f0f3790a063a704e71d59b6da59a44df776fc7aeb',
    'prefix_clock_drift_audit.py': '9a1eb612285cf5fd3449d2fa989a4acb4a0833e4b31d45733ebf25893cefd331',
    'prefix-clock-drift-lock-v1.json': '3261542bcd0add7f0b4bd91d506b4f1007fe5cc85b81aba787d969fde1d1f3cc',
    'prefix-clock-drift-v1.json': '5c9445e4e00801a11fc92301b77fa38a29f0c604a96268a5c5c0e1963599761a',
}
VECTOR = ('matched', 'unmatched_ticks', 'unmatched_responses', 'beat_positive_lower',
          'beat_positive_upper', 'downbeat_positive_lower', 'downbeat_positive_upper')
dense, packets, residual, exposure = replay.dense, replay.packets, replay.residual, replay.exposure
require = dense.require


def coordinate(time_us):
    return Fraction(time_us, exposure.FRAME_US)


def phase_at(time, anchor, pre, post, boundary):
    return (time - anchor) / pre if time <= boundary else (boundary - anchor) / pre + (time - boundary) / post


def time_at(phase, anchor, pre, post, boundary):
    crossing = (boundary - anchor) / pre
    return anchor + phase * pre if phase <= crossing else boundary + (phase - crossing) * post


def clock(anchor, pre, post, boundary, start, end, scale, offset):
    require(pre > 0 and post > 0 and start < boundary < end and scale > 0, 'invalid oracle clock')
    first = math.ceil((phase_at(start, anchor, pre, post, boundary) - offset) / scale)
    stop = math.ceil((phase_at(end, anchor, pre, post, boundary) - offset) / scale)
    if stop - first > 128:
        return None
    return [time_at(offset + k * scale, anchor, pre, post, boundary) - start for k in range(first, stop)]


def geometry_check(times, anchor_index, anchor, pre, post, boundary, start, end):
    # Native lattice ordinal zero always represents the last prior annotated beat.
    annotated = range(bisect_left(times, start), bisect_left(times, end))
    ann_full = ann_bad = con_full = con_bad = uncovered = 0
    errors = []
    full = lambda t: geometry.full_query(t * exposure.FRAME_US, start * exposure.FRAME_US, end * exposure.FRAME_US)
    for i in annotated:
        prediction = time_at(Fraction(i - anchor_index), anchor, pre, post, boundary)
        error = times[i] - prediction
        errors.append(abs(error))
        if full(times[i]):
            ann_full += 1
            ann_bad += abs(error) > 3
    first, stop = math.ceil(phase_at(start, anchor, pre, post, boundary)), math.ceil(phase_at(end, anchor, pre, post, boundary))
    require(stop - first <= 128, 'geometry clock budget exceeded')
    for n in range(first, stop):
        t = time_at(Fraction(n), anchor, pre, post, boundary)
        if full(t):
            con_full += 1
            i = anchor_index + n
            if not 0 <= i < len(times):
                uncovered += 1
                con_bad += 1
            else:
                con_bad += abs(times[i] - t) > 3
    return dict(annotation_full=ann_full, annotation_bad=ann_bad, clock_full=con_full, clock_bad=con_bad,
                uncovered_clock_ordinals=uncovered, max_annotation_error_frames=str(max(errors)) if errors else None,
                compatible=ann_full > 0 and con_full > 0 and ann_bad == con_bad == 0)


def prepare(data, pair, role, source_packets):
    start = pair[role + '_start_frame']
    end = start + pair['window_frames']
    require(pair['window_frames'] == 200, 'unregistered duration')
    expected = pair['pre_segment_index'] if role == 'constant' else (pair['pre_segment_index'], pair['change_index'])
    require(exposure.classify(data, start, 200) == (role, expected), 'selected role changed')
    selected = {s: residual.crop_packets(source_packets[s], start, end) for s in replay.SOURCES}
    row = dict(owned_responses={s: len(v) for s, v in selected.items()}, status='eligible')
    times = list(map(coordinate, data['beats']))
    before = bisect_left(times, start)
    segment = pair['pre_segment_index']
    pre = Fraction(3000) / Fraction(str(data['segments'][segment]['bpm']))
    post = Fraction(3000) / Fraction(str(data['segments'][segment + 1]['bpm']))
    boundary = start + coordinate(data['changes'][pair['change_index']]['at']) - pair['change_start_frame']
    conditions = []
    for index in range(4):
        anchor = times[before - 4 + index] + (3 - index) * pre
        # Transport can cross start; keep the original ordinal-zero identity.
        # A negative/zero lattice ordinal is valid, not a reason to refit phase.
        grids = {name: clock(anchor, pre, pre if shape == 'constant' else post, boundary, start, end, *SCALES[scale])
                 for name in NAMES for shape, scale in [name.split('/')]}
        if any(g is None for g in grids.values()):
            row['status'] = 'clock_budget_exceeded'
            check = None
        else:
            check = geometry_check(times, before - 1, anchor, pre, pre if role == 'constant' else post, boundary, start, end)
        conditions.append(dict(grids=grids, geometry=check))
    if any(len(v) > 128 for v in selected.values()):
        row['status'] = 'response_budget_exceeded'
    if not all(residual.support_inside(v, 200) for v in selected.values()):
        row['status'] = 'packet_support_crosses_window'
    return row, selected, conditions, boundary - start


def inventory(result, source):
    ways = result['optimal_assignment_count']
    values = [result['optimal_matched_count'], result['unmatched_tick_count'], result['unmatched_response_count']]
    for head in ('beat_logit', 'downbeat_logit'):
        counts = [n for p, n in zip(source, result['response_matched_assignments']) if p[head] > 0]
        values.extend((sum(n == ways for n in counts), sum(n > 0 for n in counts)))
    return values


def relation(a, b):
    def dominates(x, y):
        delta = (x[0] - y[0], y[1] - x[1], x[3] - y[4], x[5] - y[6])
        return all(v >= 0 for v in delta) and any(v > 0 for v in delta)
    if dominates(a, b):
        return 'left_dominates'
    if dominates(b, a):
        return 'right_dominates'
    if a == b and a[3] == a[4] and a[5] == a[6]:
        return 'equal'
    return 'unresolved'


def context_states(source, constant, step, boundary):
    counts = Counter()
    for i, packet in enumerate(source):
        center = Fraction(*packet['coordinate'])
        state = lambda r: residual.state(r['response_matched_assignments'][i], r['optimal_assignment_count'])
        sign = ''.join('+' if packet[k] > 0 else '-' for k in ('beat_logit', 'downbeat_logit'))
        key = '/'.join(('pre' if center < boundary else 'post', 'boundary' if residual.edge(center, 200) else 'interior',
                        state(constant), state(step), sign))
        counts[key] += 1
    return dict(sorted(counts.items()))


def evaluate(prepared):
    row, sources, conditions, boundary = prepared
    phase_results = []
    for condition in conditions:
        output = dict(geometry=condition['geometry'], sources={})
        for source, data in sources.items():
            ledgers = {name: packets.packet_ledger([True] * 200, data, ticks)['result'] for name, ticks in condition['grids'].items()}
            values = {name: inventory(r, data) for name, r in ledgers.items()}
            output['sources'][source] = dict(
                counts=values, step_versus_constant=relation(values['step/native'], values['constant/native']),
                context=context_states(data, ledgers['constant/native'], ledgers['step/native'], boundary),
                ambiguous_ledgers=sum(r['optimal_assignment_count'] > 1 for r in ledgers.values()),
                max_assignment_count=max(r['optimal_assignment_count'] for r in ledgers.values()),
                max_states=max(r['states'] for r in ledgers.values()))
        phase_results.append(output)
    return row | dict(phases=phase_results)


def verdict(windows):
    compatible = all(p['geometry']['compatible'] for w in windows.values() for p in w['phases'])
    sources = {}
    for source in replay.SOURCES:
        shape_ok, density_ok = [], []
        for phase in range(4):
            shape, density = True, True
            for role, w in windows.items():
                values = w['phases'][phase]['sources'][source]['counts']
                true_shape = 'constant' if role == 'constant' else 'step'
                correct = true_shape + '/native'
                opposite = ('step' if role == 'constant' else 'constant') + '/native'
                shape &= relation(values[correct], values[opposite]) == 'left_dominates'
                density &= all(relation(values[correct], v) == 'left_dominates' for name, v in values.items() if name != correct)
            shape_ok.append(shape)
            density_ok.append(density)
        sources[source] = dict(shape_supported_phases=sum(shape_ok), density_resolved_phases=sum(density_ok),
                               all_phases_shape_supported=all(shape_ok), all_phases_density_resolved=all(density_ok))
    return dict(geometry_compatible=compatible, sources=sources)


def analyze_pair(source_packets, data, pair):
    require(pair['constant_start_frame'] + pair['window_frames'] <= pair['change_start_frame'], 'overlapping pair')
    prepared = {role: prepare(data, pair, role, source_packets) for role in replay.ROLES}
    for source in replay.SOURCES:
        ids = [(p['source'], p['source_index']) for value in prepared.values() for p in value[1][source]]
        require(len(ids) == len(set(ids)), 'duplicate or shared packet identity')
    if any(value[0]['status'] != 'eligible' for value in prepared.values()):
        return dict(status='pair_rejected', windows={role: v[0] for role, v in prepared.items()})
    windows = {role: evaluate(value) for role, value in prepared.items()}
    return dict(status='eligible', windows=windows, verdict=verdict(windows))


def summarize(pairs):
    eligible = [p for p in pairs if p['status'] == 'eligible']
    compatible = [p for p in eligible if p['verdict']['geometry_compatible']]
    by_source = {}
    for source in replay.SOURCES:
        n = sum(p['verdict']['sources'][source]['all_phases_shape_supported'] for p in compatible)
        resolved = sum(p['verdict']['sources'][source]['all_phases_density_resolved'] for p in compatible)
        by_source[source] = dict(geometry_compatible_pairs=len(compatible), robust_shape_pairs=n, robust_density_pairs=resolved,
            decision='inconclusive_clock_geometry' if not compatible else
                     'do_not_promote_count_sign_selector' if n < len(compatible) or resolved < len(compatible) else
                     'independent_examples_required_not_general_accuracy')
    return dict(selected_pairs=len(pairs), eligible_pairs=len(eligible), rejected_pairs=len(pairs) - len(eligible),
                geometry_compatible_pairs=len(compatible), geometry_incompatible_pairs=len(eligible) - len(compatible),
                all_selected_owned_responses={s: sum(p['windows'][r]['owned_responses'][s] for p in pairs for r in replay.ROLES) for s in replay.SOURCES},
                by_source=by_source)


def build_report(evidence_path, capture_dir):
    require(all(dense.sha((HERE / p).read_bytes()) == h for p, h in PINS.items()), 'frozen dependency changed')
    helpers, packet_report = replay.verified_helpers()
    metadata, cases, plan = replay.verified_plan()  # verified before private input access
    by_id = {c['identity']['id']: c for c in cases}
    count, evidence_hash = dense.INPUTS['artbeat']
    evidence, _ = dense.read_json(evidence_path, evidence_hash)
    prior = next(c for c in packet_report['cohorts'] if c['cohort'] == 'artbeat')
    summary, summary_hash = dense.read_json(Path(capture_dir) / 'summary.json', prior['capture_summary_sha256'])
    records = dense.validate_summary(summary, evidence['cases'], count)
    require([c['id'] for c in evidence['cases']] == [c['id'] for c in metadata[0]['cases']], 'ordered cohort changed')
    lookup = {c['id']: (c, r, old) for c, r, old in zip(evidence['cases'], records, prior['cases'])}
    sources = {k: dense.sha((dense.ROOT / p).read_bytes()) for k, p in dense.SOURCES.items()}
    results = []
    for pair in plan:
        identity, data = (by_id[pair['id']][k] for k in ('identity', 'data'))
        case, record, old = lookup[pair['id']]
        require(case['truth_sha256'] == identity['truth_sha256'] and record['capture_sha256'] == identity['capture_sha256'] and old['id'] == pair['id'],
                'selected identity differs')
        require(residual.annotation_coordinates(case['truth_times_s']) == list(map(coordinate, data['beats'])), 'private truth differs')
        payload, _ = dense.read_json(Path(capture_dir) / (pair['id'] + '.json'), identity['capture_sha256'])
        beats = dense.validate_capture(payload, record, case, 'artbeat', sources)
        require(len(beats) == identity['frame_count'], 'selected extent differs')
        assembled = packets.assemble(beats, payload['downbeat_logits'], payload['observations']['duration_s'], payload['observations'])
        require(packets.summarize(assembled) == old['source_inventory'], 'packet inventory changed')
        result = analyze_pair({'raw': assembled['raw'], 'candidate': assembled['candidates']}, data, pair)
        whole = {'raw': len(assembled['raw']), 'candidate': len(assembled['candidates'])}
        owned = {s: sum(result['windows'][r]['owned_responses'][s] for r in replay.ROLES) for s in replay.SOURCES}
        require(all(owned[s] <= whole[s] for s in replay.SOURCES), 'ownership exceeds whole capture')
        results.append(dict(**identity, pair_sha256=exposure.canonical_hash(pair), whole_capture_responses=whole,
                            outside_selected_windows={s: whole[s] - owned[s] for s in replay.SOURCES}, **result))
    selected = {p['id'] for p in plan}
    return dict(schema_version=1, purpose=LOCK['purpose'], contract=LOCK, count_vector_fields=VECTOR,
                script_sha256=dense.sha(Path(__file__).read_bytes()), lock_sha256=dense.sha(LOCK_PATH.read_bytes()),
                helper_sha256=helpers | PINS, source_hashes=sources, selected_plan_sha256=replay.LOCK['selected_plan_sha256'],
                evidence_sha256=evidence_hash, capture_summary_sha256=summary_hash,
                input_cases=[dict(cohort=c['cohort'], **c['identity'], private_capture_read=c['identity']['id'] in selected) for c in cases],
                pairs=results, summary=summarize(results), truth_assisted=True, selected_tempo=None,
                neural_inference=False, decoder_replayed=False, fitted_mapping=False, phase_search=False,
                source_pooling=False, holdout_evidence_opened=False, training_run=False, production_output_changed=False,
                new_user_parameters=False, accuracy_improvement_claimed=False,
                scope='fixed count/sign summaries and four prefix phases, not all model representations or continuous nuisance values')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artbeat-evidence', type=Path, required=True)
    parser.add_argument('--artbeat-captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.artbeat_evidence, args.artbeat_captures)
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
