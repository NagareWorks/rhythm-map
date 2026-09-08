"""Annotation-only geometry of the already frozen seven constant/step pairs.

This is not a tempo estimator. Same-ordinal comparisons deliberately preserve
cycle errors. No response packets, numerical ML dependencies or fitted phase.
"""
import argparse
from bisect import bisect_left
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path

import annotation_exposure_audit as exposure

HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / 'prefix-clock-drift-lock-v1.json'
LOCK = json.loads(LOCK_PATH.read_text())
PINS = {
    'annotation_exposure_audit.py': '824832bb891637a8eafb40bd05c05e4fb5c5000577c7f08a81a10bdc2892c681',
    'annotation-exposure-lock-v1.json': '2f9fcbc4a75ac6c30acad4d927d681d686a6a2e9d93531831f8e8e9f934c1271',
    'annotation-exposure-v1.json': 'e9bb2086e906bda366e0b8d2d4dd87b9ff9d7c44441696d223a75a96471b1346',
}
PLAN_HASH = '5c15e677d0277dcac111aaebfb2be0b788996cbd44019a315c171fcf39bfe4f2'
FRAME_US = exposure.FRAME_US
RADIUS_US = 3 * FRAME_US
require = exposure.require


def rational(value):
    return str(Fraction(value))


def stats(values):
    values = list(values)
    return dict(count=len(values), min_us=rational(min(values)) if values else None,
                max_us=rational(max(values)) if values else None,
                max_abs_us=rational(max(map(abs, values))) if values else None)


def full_query(time, start, end):
    center = Fraction(time - start, FRAME_US)
    size = (end - start) // FRAME_US
    return start <= time < end and math.ceil(center - 3) >= 0 and math.floor(center + 3) < size


def normalization_bound(ordinal):
    require(type(ordinal) is int and ordinal > 0, 'ordinal must follow anchor')
    # e_n = t_n - (1+n/3)*anchor + (n/3)*oldest. Each input rounds by <= 1/2 us.
    return 1 + Fraction(ordinal, 3)


def periodic_ordinals(anchor, period, start, end):
    period = Fraction(period)
    require(period > 0 and start < end and anchor < start, 'invalid prefix clock')
    first, stop = math.ceil((start - anchor) / period), math.ceil((end - anchor) / period)
    require(0 <= stop - first <= 128, 'clock budget exceeded')
    return set(range(first, stop))


def verified_plan():
    for name, expected in PINS.items():
        require(exposure.sha((HERE / name).read_bytes()) == expected, 'frozen exposure dependency changed')
    frozen = json.loads((HERE / 'annotation-exposure-v1.json').read_bytes())
    require(exposure.build_report() == frozen, 'exposure report does not reproduce')
    require(frozen['selected_window_frames'] == 200 and frozen['selected_pair_count'] == 7 and
            frozen['plan_sha256'] == PLAN_HASH, 'selected plan contract changed')
    metadata, cases = exposure.load_annotations()
    plan = []
    for case in cases:
        pair = exposure.profile(case['data'], 200)['pair']
        if pair is not None:
            plan.append(dict(cohort=case['cohort'], id=case['identity']['id'], **pair))
    require(exposure.canonical_hash(plan) == PLAN_HASH and len(plan) == 7, 'selected coordinates changed')
    return metadata, cases, plan


def summarize_rows(rows):
    rows = list(rows)
    return dict(
        ticks=len(rows), full_query_annotation_ticks=sum(r['annotation_full_query'] for r in rows),
        both_full_query_ticks=sum(r['both_full_query'] for r in rows),
        beyond_normalization_bound=sum(abs(r['error']) > normalization_bound(r['n']) for r in rows),
        beyond_radius=sum(abs(r['error']) > RADIUS_US for r in rows),
        full_query_annotation_beyond_radius=sum(r['annotation_full_query'] and abs(r['error']) > RADIUS_US for r in rows),
        both_full_query_beyond_radius=sum(r['both_full_query'] and abs(r['error']) > RADIUS_US for r in rows),
        error=stats(r['error'] for r in rows),
        relative_prefix_period_drift=stats(r['drift'] for r in rows),
        relative_annotation_departure=stats(r['departure'] for r in rows),
        normalization_bound=stats(normalization_bound(r['n']) for r in rows))


def diagnose_window(data, start_frame, length, pre_index, role, change_index):
    require(length == 200 and role in ('constant', 'change'), 'unregistered window')
    expected = pre_index if role == 'constant' else (pre_index, change_index)
    require(exposure.classify(data, start_frame, length) == (role, expected), 'frozen window role changed')
    start, end = start_frame * FRAME_US, (start_frame + length) * FRAME_US
    beats = data['beats']
    before = bisect_left(beats, start)
    anchor_index = before - 1
    prefix = beats[before - 4:before]
    anchor = prefix[-1]
    period = Fraction(prefix[-1] - prefix[0], 3)
    nominal = Fraction(60000000) / Fraction(str(data['segments'][pre_index]['bpm']))
    annotated = {i - anchor_index for i in range(before, bisect_left(beats, end))}
    continued = periodic_ordinals(anchor, period, start, end)
    require(annotated and len(annotated) <= 128, 'annotation budget or empty window')
    first_n = min(annotated)
    first_time = beats[anchor_index + first_n]
    first_error = first_time - anchor - first_n * period
    change = data['changes'][change_index]['at'] if role == 'change' else None
    rows = []
    for n in sorted(annotated):
        actual, predicted = beats[anchor_index + n], anchor + n * period
        error = actual - predicted
        drift = (n - first_n) * (nominal - period)
        departure = actual - first_time - (n - first_n) * nominal
        require(error == first_error + drift + departure, 'decomposition identity failed')
        rows.append(dict(n=n, error=error, drift=drift, departure=departure,
                         annotation_full_query=full_query(actual, start, end),
                         both_full_query=full_query(actual, start, end) and full_query(predicted, start, end),
                         stratum='constant' if change is None else ('pre_step' if actual < change else 'post_step')))
    ann_only = annotated - continued
    con_only = continued - annotated
    outside = Counter()
    for n in con_only:
        index = anchor_index + n
        if not 0 <= index < len(beats):
            outside['annotation_uncovered'] += 1
        else:
            require(not start <= beats[index] < end, 'ordinal ownership contradiction')
            outside['annotation_before_window' if beats[index] < start else 'annotation_after_window'] += 1
    inventory = dict(annotation_ticks=len(annotated), continuation_ticks=len(continued),
                     shared_ordinals=len(annotated & continued), annotation_only=len(ann_only),
                     continuation_only=len(con_only),
                     annotation_only_prediction_before_window=sum(anchor + n * period < start for n in ann_only),
                     annotation_only_prediction_after_window=sum(anchor + n * period >= end for n in ann_only),
                     continuation_only_annotation_location={key: outside[key] for key in
                         ('annotation_before_window', 'annotation_after_window', 'annotation_uncovered')},
                     annotation_full_query=sum(full_query(beats[anchor_index + n], start, end) for n in annotated),
                     continuation_full_query=sum(full_query(anchor + n * period, start, end) for n in continued))
    require(inventory['annotation_only_prediction_before_window'] + inventory['annotation_only_prediction_after_window'] == len(ann_only),
            'unaccounted annotation ordinal')
    strata = ('constant',) if role == 'constant' else ('pre_step', 'post_step')
    return dict(role=role, prefix_period_us=rational(period), nominal_period_us=rational(nominal),
                prefix_minus_nominal_period_us=rational(period - nominal),
                prefix_interval_minus_nominal=stats(b - a - nominal for a, b in zip(prefix, prefix[1:])),
                first_selected_beat_error_us=rational(first_error),
                prefix_nominal_first_error_us=rational(first_time - anchor - first_n * nominal),
                identical_grids=annotated == continued and all(r['error'] == 0 for r in rows),
                ownership=inventory, all_ticks=summarize_rows(rows),
                strata={key: summarize_rows(r for r in rows if r['stratum'] == key) for key in strata})


def gate(windows):
    constant = [w for w in windows if w['role'] == 'constant']
    require(constant, 'no constant controls')
    affected = sum(w['all_ticks']['beyond_normalization_bound'] > 0 for w in constant)
    return dict(constant_windows=len(constant), constant_windows_beyond_normalization=affected,
                constant_windows_with_ownership_difference=sum(w['ownership']['annotation_only'] + w['ownership']['continuation_only'] > 0 for w in constant),
                constant_windows_with_full_query_error_beyond_radius=sum(w['all_ticks']['full_query_annotation_beyond_radius'] > 0 for w in constant),
                constant_windows_with_both_full_query_error_beyond_radius=sum(w['all_ticks']['both_full_query_beyond_radius'] > 0 for w in constant),
                decision='require_clock_nuisance_control_before_model_discrimination' if affected else
                         'no_beyond_normalization_drift_found_not_accuracy_validated')


def build_report():
    metadata, cases, plan = verified_plan()
    selected = {p['id']: p for p in plan}
    results, windows = [], []
    for case in cases:
        identity = case['identity']
        result = dict(cohort=case['cohort'], **identity, selected=identity['id'] in selected)
        if result['selected']:
            pair = selected[identity['id']]
            result['pair_sha256'] = exposure.canonical_hash(pair)
            result['windows'] = [diagnose_window(case['data'], pair[role + '_start_frame'], pair['window_frames'],
                                  pair['pre_segment_index'], role, pair['change_index']) for role in ('constant', 'change')]
            windows.extend(result['windows'])
        results.append(result)
    return dict(schema_version=1, purpose=LOCK['purpose'], contract=LOCK,
                script_sha256=exposure.sha(Path(__file__).read_bytes()), lock_sha256=exposure.sha(LOCK_PATH.read_bytes()),
                exposure_dependencies=PINS, plan_sha256=PLAN_HASH,
                metadata_projection_sha256=exposure.canonical_hash(metadata),
                selected_pair_count=len(plan), window_count=len(windows), cases=results, gate=gate(windows),
                private_captures_read=False, response_fields_used=False, neural_inference=False,
                fitted_mapping=False, decoder_replayed=False, holdout_opened=False, training_run=False,
                production_output_changed=False, new_user_parameters=False, accuracy_improvement_claimed=False,
                upstream_annotation_precision='unknown; normalization bounds are not annotation uncertainty',
                selected_tempo=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
