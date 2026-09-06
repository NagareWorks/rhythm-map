"""Plan matched exposure using annotations only, never detector outcomes.

Only public identity/duration metadata is projected from the previous report.
No private capture, audio, model, decoder or numerical ML dependency is accepted.
Frame-aligned candidate starts overlap: their counts are availability, not N.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK_PATH = HERE / 'annotation-exposure-lock-v1.json'
LOCK = json.loads(LOCK_PATH.read_text())
SOURCE = 'window-response-residuals-v1.json'
SOURCE_HASH = '35eb07cfc8c96471710ecf0a40d3fae6da836638845090cc21583ba337769a8a'
SUITES = {'artbeat': 'artbeat-v1', 'rubato': 'rubato-calibration-v1'}
FRAME_US = 20000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path, expected):
    data = Path(path).read_bytes()
    require(sha(data) == expected, 'input identity changed')
    return json.loads(data)


def canonical_hash(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def metadata_projection(report):
    # Explicit allowlist: model-derived inventories and counts never leave here.
    return [dict(cohort=c['cohort'], suite_sha256=c['suite_sha256'], cases=[
        {key: case[key] for key in ('id', 'frame_count', 'truth_sha256', 'capture_sha256')} for case in c['cases']])
        for c in report['cohorts']]


def microseconds(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'invalid annotation time')
    return math.floor(Fraction(str(value)) * 1000000 + Fraction(1, 2))


def normalize(truth, frame_count, untyped=False):
    require(type(frame_count) is int and frame_count > 0, 'invalid capture extent')
    duration = microseconds(truth['duration_s'])
    require(abs(duration - (frame_count - 1) * FRAME_US) <= FRAME_US, 'annotation/capture duration mismatch')
    beats = [microseconds(b['time_s']) for b in truth['beats']]
    require(len(beats) >= 4 and all(a < b for a, b in zip(beats, beats[1:])) and beats[-1] <= duration,
            'invalid or colliding annotated beats')
    result = dict(frame_count=frame_count, beats=beats, untyped=untyped, segments=[], changes=[])
    if untyped:
        return result  # labels cannot be derived from rubato tempo summaries
    for s in truth['tempo_segments']:
        a, b = microseconds(s['start_s']), microseconds(s['end_s'])
        require(a < b <= duration and s['kind'] in ('constant', 'ramp'), 'invalid tempo segment')
        require(all(type(s[k]) in (int, float) and math.isfinite(s[k]) and s[k] > 0 for k in ('start_bpm', 'end_bpm')),
                'invalid annotated tempo')
        require(s['kind'] != 'constant' or s['start_bpm'] == s['end_bpm'], 'nonconstant constant segment')
        result['segments'].append(dict(start=a, end=b, kind=s['kind'], bpm=s['start_bpm']))
    require(all(a['end'] <= b['start'] for a, b in zip(result['segments'], result['segments'][1:])), 'overlapping or unordered segments')
    result['changes'] = [dict(at=microseconds(c['time_s']), kind=c['kind']) for c in truth['change_points']]
    require(all(0 <= c['at'] <= duration for c in result['changes']) and
            all(a['at'] < b['at'] for a, b in zip(result['changes'], result['changes'][1:])), 'invalid change ordering')
    return result


def clock_budget(beats, before, start, end):
    anchor, period = beats[before - 1], Fraction(beats[before - 1] - beats[before - 4], 3)
    counts = [bisect_left(beats, end) - before]
    for phase, multiplier in ((0, 1), (0, 2), (1, 2), (0, Fraction(1, 2))):
        step = period * multiplier
        origin = anchor + phase * period
        counts.append(math.ceil((end - origin) / step) - math.ceil((start - origin) / step))
    return max(counts) <= 128


def classify(data, start_frame, length):
    beats, segments, changes = (data[k] for k in ('beats', 'segments', 'changes'))
    start, end = start_frame * FRAME_US, (start_frame + length) * FRAME_US
    before = bisect_left(beats, start)
    if before < 4:
        return 'insufficient_prefix', None
    if end > beats[-1]:
        return 'annotation_tail_uncovered', None
    prefixes = [i for i, s in enumerate(segments) if s['kind'] == 'constant' and s['start'] <= beats[before - 4] and start < s['end']]
    if len(prefixes) != 1:
        return 'no_stable_prefix_segment', None
    pre = prefixes[0]
    if any(beats[before - 4] < c['at'] < start for c in changes):
        return 'prefix_contains_change', None
    if not clock_budget(beats, before, start, end):
        return 'clock_budget_exceeded', None
    # Exactly the existing rational ledger's full-query condition, including
    # fractional centers (ceil(center-3)>=0 and floor(center+3)<length).
    interior = beats[bisect_right(beats, start + 2 * FRAME_US):bisect_left(beats, end - 3 * FRAME_US)]
    inside = [(i, c) for i, c in enumerate(changes) if start <= c['at'] < end]
    if not inside and end <= segments[pre]['end']:
        return ('constant', pre) if len(interior) >= 4 else ('constant_too_few_interior_beats', None)
    if len(inside) != 1:
        return 'not_one_annotated_step', None
    index, change = inside[0]
    if change['kind'] != 'tempo_jump' or pre + 1 >= len(segments):
        return 'not_adjacent_constant_step', None
    post = segments[pre + 1]
    if not (segments[pre]['end'] == change['at'] == post['start'] and post['kind'] == 'constant' and
            segments[pre]['bpm'] != post['bpm'] and end <= post['end']):
        return 'not_adjacent_constant_step', None
    left = sum(t < change['at'] for t in interior)
    if min(left, len(interior) - left) < 2:
        return 'step_too_few_interior_beats', None
    return 'change', (pre, index)


def choose_pair(controls, change_windows, changes, length):
    pairs, combinations = [], 0
    for start, pre, index in change_windows:
        candidates = controls.get(pre, [])
        count = bisect_right(candidates, start - length)
        combinations += count
        if count:
            constant = candidates[count - 1]
            imbalance = abs(2 * changes[index]['at'] - (2 * start + length) * FRAME_US)
            pairs.append(((index, imbalance, start, -constant), dict(
                constant_start_frame=constant, change_start_frame=start, window_frames=length,
                pre_segment_index=pre, change_index=index)))
    return (min(pairs, key=lambda p: p[0])[1] if pairs else None), combinations, len(pairs)


def profile(data, length):
    require(length in LOCK['durations_frames'], 'unregistered duration')
    starts = max(0, data['frame_count'] - length + 1)
    if data['untyped']:
        return dict(candidate_starts=starts, roles={'untyped_rubato': starts} if starts else {},
                    pair_combinations=0, pairable_change_starts=0, pair=None)
    roles, controls, changes = Counter(), {}, []
    for start in range(starts):
        role, identity = classify(data, start, length)
        roles[role] += 1
        if role == 'constant':
            controls.setdefault(identity, []).append(start)
        elif role == 'change':
            changes.append((start, *identity))
    pair, count, pairable = choose_pair(controls, changes, data['changes'], length)
    return dict(candidate_starts=starts, roles=dict(sorted(roles.items())), pair_combinations=count,
                pairable_change_starts=pairable, pair=pair)


def choose_duration(profiles):
    counts = {length: sum(p['pair'] is not None for p in rows) for length, rows in profiles.items()}
    if not any(counts.values()):
        return None
    return max(counts, key=lambda length: (counts[length], length))


def load_annotations():
    report = read_json(HERE / SOURCE, SOURCE_HASH)
    metadata = metadata_projection(report)
    require([c['cohort'] for c in metadata] == ['artbeat', 'rubato'] and
            [len(c['cases']) for c in metadata] == [15, 25], 'input cohort changed')
    cases = []
    for cohort in metadata:
        suite_path = ROOT / 'evaluation/suites' / (SUITES[cohort['cohort']] + '.json')
        suite = read_json(suite_path, cohort['suite_sha256'])
        require(suite['purpose'] == 'calibration' and [c['id'] for c in suite['cases']] == [c['id'] for c in cohort['cases']],
                'suite role or ordered case set changed')
        for item, identity in zip(suite['cases'], cohort['cases']):
            path = (suite_path.parent / item['input']['truth']).resolve()
            require(path.is_relative_to((suite_path.parent / 'truth').resolve()), 'truth path outside annotation directory')
            truth = read_json(path, identity['truth_sha256'])
            require(truth['id'] == identity['id'], 'truth id changed')
            untyped = cohort['cohort'] == 'rubato' or 'rubato' in item.get('tags', [])
            cases.append(dict(cohort=cohort['cohort'], identity=identity, data=normalize(truth, identity['frame_count'], untyped)))
    return metadata, cases


def build_report():
    metadata, cases = load_annotations()
    profiles = {length: [profile(c['data'], length) for c in cases] for length in LOCK['durations_frames']}
    duration = choose_duration(profiles)
    summaries = {}
    for length, rows in profiles.items():
        cohorts = {}
        for cohort in SUITES:
            selected = [p for c, p in zip(cases, rows) if c['cohort'] == cohort]
            roles = Counter()
            for p in selected:
                roles.update(p['roles'])
            cohorts[cohort] = dict(tracks=len(selected), pairable_tracks=sum(p['pair'] is not None for p in selected),
                                  candidate_starts=sum(p['candidate_starts'] for p in selected), roles=dict(sorted(roles.items())),
                                  pair_combinations=sum(p['pair_combinations'] for p in selected),
                                  pairable_change_starts=sum(p['pairable_change_starts'] for p in selected))
        summaries[str(length)] = dict(cohorts=cohorts, cases=[dict(cohort=c['cohort'], **c['identity'],
             **{k: v for k, v in p.items() if k != 'pair'}, pair_available=p['pair'] is not None,
             pair_sha256=canonical_hash(p['pair']) if p['pair'] is not None else None) for c, p in zip(cases, rows)])
    plan = [] if duration is None else [dict(cohort=c['cohort'], id=c['identity']['id'], **p['pair'])
                                      for c, p in zip(cases, profiles[duration]) if p['pair']]
    return dict(schema_version=1, purpose=LOCK['purpose'], contract=LOCK,
                script_sha256=sha(Path(__file__).read_bytes()), lock_sha256=sha(LOCK_PATH.read_bytes()),
                source_report_sha256=SOURCE_HASH, metadata_projection_sha256=canonical_hash(metadata),
                private_captures_read=False, response_fields_used=False, neural_inference=False,
                decoder_replayed=False, fitted_mapping=False, holdout_opened=False, training_run=False,
                production_output_changed=False, new_user_parameters=False, accuracy_improvement_claimed=False,
                profiles=summaries, selected_window_frames=duration, selected_pair_count=len(plan),
                selected_case_ids=[p['id'] for p in plan], plan_sha256=canonical_hash(plan),
                response_budget_verification='not_run', selected_tempo=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
