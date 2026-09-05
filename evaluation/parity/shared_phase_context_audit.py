"""Shared versus individual phase alignment, not a clock decoder/likelihood.

Both templates contain five points and inspect the same complete frame domain.
They are not full clocks: unused frames do not contribute density evidence.
The ideal constant/half-time alias and omission ambiguity must remain visible.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

import metrical_window_audit as windows

dense = windows.dense
ROOT = dense.ROOT
LOCK_PATH = Path(__file__).with_name('shared-phase-context-lock-v1.json')
LOCK = json.loads(LOCK_PATH.read_text())
PAIRS = (('annotated', 'continuation'), ('continuation', 'half_tempo'), ('continuation', 'double_tempo'))
PAIR_NAMES = tuple(f'{a}_vs_{b}' for a, b in PAIRS)
REGIMES = ('constant_context', 'change_context', 'ramp_context', 'rubato', 'insufficient_context')


def template_score(beats, downbeats, centers):
    scores = {d: math.fsum(float(beats[t + d]) for t in centers) / 5 for d in range(-3, 4)}
    offset = max(scores, key=lambda d: (scores[d], -abs(d), -d))
    independent = math.fsum(max(float(beats[t + d]) for d in range(-3, 4)) for t in centers) / 5
    shared = scores[offset]
    return dict(shared=shared, independent=independent, phase_penalty=independent - shared,
                offset=offset, downbeat_at_shared=math.fsum(float(downbeats[t + offset]) for t in centers) / 5)


def compare(beats, downbeats, left, right, available=None):
    """No truth, identities, regime or raw-event labels enter scoring."""
    dense.require(len(beats) == len(downbeats) and
                  (available is None or len(available) == len(beats)), 'invalid frame dimensions')
    for centers in (left, right):
        dense.require(len(centers) == 5 and all(type(t) is int for t in centers) and
                      all(a <= b for a, b in zip(centers, centers[1:])), 'expected five ordered integer centers')
    lo, hi = min(left[0], right[0]) - 3, max(left[-1], right[-1]) + 3
    if lo < 0 or hi >= len(beats):
        return dict(status='out_of_capture')
    if available is not None and not all(available[lo:hi + 1]):
        return dict(status='unavailable_common_frame')
    dense.require(all(math.isfinite(float(v)) for head in (beats, downbeats) for v in head[lo:hi + 1]),
                  'nonfinite common frame')
    if any(b - a <= 6 for centers in (left, right) for a, b in zip(centers, centers[1:])):
        return dict(status='within_template_overlap')
    first, second = template_score(beats, downbeats, left), template_score(beats, downbeats, right)
    dense.require(all(math.isfinite(v) for s in (first, second) for v in s.values()), 'nonfinite template score')
    shared_margin, independent_margin = (first[k] - second[k] for k in ('shared', 'independent'))
    dense.require(math.isfinite(shared_margin) and math.isfinite(independent_margin), 'nonfinite comparison margin')
    return dict(status='identical_grids' if left == right else 'informative', common_frames=hi - lo + 1,
                shared_margin=shared_margin, independent_margin=independent_margin, left=first, right=second)


def context_regime(truth, case, cohort, i):
    if cohort == 'rubato' or 'rubato' in case['tags']:
        return 'rubato'
    times = case['truth_times_s']
    lo, hi = times[i - 3], times[i + 4]
    if any(lo <= cp['time_s'] <= hi for cp in truth['change_points']):
        return 'change_context'
    segments = sorted((s for s in truth['tempo_segments'] if s['start_s'] <= hi and s['end_s'] > lo),
                      key=lambda s: s['start_s'])
    covered = lo
    for j, s in enumerate(segments):
        dense.require(s['kind'] in ('constant', 'ramp') and s['end_s'] > s['start_s'] and
                      (s['start_s'] <= lo if j == 0 else s['start_s'] == covered),
                      'missing or overlapping context annotation')
        covered = s['end_s']
    dense.require(covered > hi, 'incomplete context annotation')
    if any(s['kind'] == 'ramp' for s in segments):
        return 'ramp_context'
    return 'constant_context'


def case_rows(beats, downbeats, truth, case, cohort):
    # Reuse strict annotation identity/semantics and historical raw matching.
    labels, _ = windows.annotation_labels(truth, case, cohort)
    times = case['truth_times_s']
    pairs = dense.matches([b['time_s'] for b in case['observations']['beats']], times, case['beat_tolerance_s'])
    dense.require(pairs == case['raw_truth_pairs'], 'raw/truth matching changed')
    matched = {t for _, t in pairs}
    candidates = [b['time_s'] for b in case['observations']['beat_candidates']]
    dense.require(dense.ordered(candidates), 'invalid candidates')
    rows = []
    for i, t in enumerate(times):
        row = dict(raw_missed=i not in matched, label=labels[i], acoustic_presence='unknown',
                   candidate_absent_miss=i not in matched and not dense.near(candidates, t, case['beat_tolerance_s']))
        if i < 3 or i + 4 >= len(times):
            reason = 'insufficient_prefix' if i < 3 else 'insufficient_suffix'
            row.update(regime='insufficient_context', comparisons={name: dict(status=reason) for name in PAIR_NAMES})
        else:
            period = (t - times[i - 3]) / 3
            grids = {'annotated': [round(v * 50) for v in times[i:i + 5]]}
            grids.update({name: [round((t + j * period * multiplier) * 50) for j in range(5)]
                          for name, multiplier in (('continuation', 1), ('half_tempo', 2), ('double_tempo', .5))})
            row.update(regime=context_regime(truth, case, cohort, i), comparisons={
                name: compare(beats, downbeats, grids[a], grids[b]) for name, (a, b) in zip(PAIR_NAMES, PAIRS)})
        rows.append(row)
    return rows


def sign(x):
    return 'positive' if x > 0 else 'negative' if x < 0 else 'zero'


def summarize(rows, pair):
    informative = [r['comparisons'][pair] for r in rows if r['comparisons'][pair]['status'] == 'informative']
    n = len(informative)
    result = dict(queries=len(rows), status=dict(Counter(r['comparisons'][pair]['status'] for r in rows)),
                  informative=n, acoustic_presence_unknown=len(rows), label_counts=dict(Counter(r['label'] for r in rows)),
                  joint_signs=dict(Counter(f"{sign(c['independent_margin'])}_to_{sign(c['shared_margin'])}" for c in informative)))
    for method in ('shared', 'independent'):
        margins = [c[f'{method}_margin'] for c in informative]
        positive = sum(m > 0 for m in margins)
        result[method] = dict(positive=positive, zero=sum(m == 0 for m in margins), negative=sum(m < 0 for m in margins),
                              positive_fraction=positive / n if n else None,
                              mean_margin=statistics.mean(margins) if n else None, quantiles=windows.quantiles(margins))
    result['mean_phase_penalty'] = {side: statistics.mean(c[side]['phase_penalty'] for c in informative) if n else None
                                     for side in ('left', 'right')}
    result['downbeat_readout_quantiles'] = {side: windows.quantiles([c[side]['downbeat_at_shared'] for c in informative])
                                            for side in ('left', 'right')}
    return result


def audit(cohort, evidence_path, capture_dir):
    count, evidence_hash = dense.INPUTS[cohort]
    evidence, _ = dense.read_json(evidence_path, evidence_hash)
    old, _ = dense.read_json(ROOT / 'evaluation/parity/dense-clock-evidence-v1.json', windows.OLD_DENSE)
    prior = next(c for c in old['cohorts'] if c['cohort'] == cohort)
    summary, summary_hash = dense.read_json(Path(capture_dir) / 'summary.json', prior['capture_summary_sha256'])
    records = dense.validate_summary(summary, evidence['cases'], count)
    sources = {k: dense.sha((ROOT / p).read_bytes()) for k, p in dense.SOURCES.items()}
    suite_path = ROOT / 'evaluation/suites' / f'{dense.SUITES[cohort][0]}.json'
    suite, suite_hash = dense.read_json(suite_path, dense.SUITES[cohort][1])
    dense.require([c['id'] for c in suite['cases']] == [c['id'] for c in evidence['cases']], 'suite order changed')
    all_rows, tracks = [], []
    for record, case, item in zip(records, evidence['cases'], suite['cases']):
        payload, _ = dense.read_json(Path(capture_dir) / f"{case['id']}.json", record['capture_sha256'])
        beats = dense.validate_capture(payload, record, case, cohort, sources)
        truth, truth_hash = dense.read_json(suite_path.parent / item['input']['truth'], case['truth_sha256'])
        rows = case_rows(beats, payload['downbeat_logits'], truth, case, cohort)
        tracks.append(dict(id=case['id'], capture_sha256=record['capture_sha256'], truth_sha256=truth_hash,
                           frame_count=len(beats), context_counts=dict(Counter(r['regime'] for r in rows)),
                           all={p: summarize(rows, p) for p in PAIR_NAMES},
                           raw_missed={p: summarize([r for r in rows if r['raw_missed']], p) for p in PAIR_NAMES}))
        all_rows.extend(rows)
    groups = {'all': all_rows, 'raw_missed': [r for r in all_rows if r['raw_missed']],
              'candidate_absent_misses': [r for r in all_rows if r['candidate_absent_miss']]}
    for regime in REGIMES:
        groups[regime] = [r for r in all_rows if r['regime'] == regime]
        groups[regime + '_missed'] = [r for r in groups[regime] if r['raw_missed']]
    macro = {}
    for group in ('all', 'raw_missed'):
        macro[group] = {}
        for p in PAIR_NAMES:
            macro[group][p] = {}
            for method in ('shared', 'independent'):
                values = [t[group][p][method]['positive_fraction'] for t in tracks
                          if t[group][p][method]['positive_fraction'] is not None]
                macro[group][p][method] = dict(contributing_tracks=len(values), total_tracks=count,
                                               mean_track_positive_fraction=statistics.mean(values) if values else None)
    return dict(cohort=cohort, complete=True, frozen_evidence_sha256=evidence_hash, capture_summary_sha256=summary_hash,
                suite_sha256=suite_hash, source_hashes=sources, total_frames_per_head=sum(t['frame_count'] for t in tracks),
                groups={g: {p: summarize(rows, p) for p in PAIR_NAMES} for g, rows in groups.items()},
                macro=macro, cases=tracks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for cohort in dense.INPUTS:
        parser.add_argument(f'--{cohort}-evidence', type=Path, required=True)
        parser.add_argument(f'--{cohort}-captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cohorts = [audit(c, getattr(args, f'{c}_evidence'), getattr(args, f'{c}_captures')) for c in dense.INPUTS]
    report = dict(schema_version=1, purpose=LOCK['purpose'], contract=LOCK,
                  script_sha256=dense.sha(Path(__file__).read_bytes()), lock_sha256=dense.sha(LOCK_PATH.read_bytes()),
                  helper_sha256={p: dense.sha(Path(__file__).with_name(p).read_bytes()) for p in
                                 ('metrical_window_audit.py', 'dense_clock_evidence.py', 'clock_phase_evidence.py',
                                  'resampler_event_audit.py', 'candidate_evidence_audit.py')},
                  window_report_sha256=dense.sha(Path(__file__).with_name('metrical-window-v1.json').read_bytes()),
                  truth_assisted=True, fitted_mapping=False, neural_inference=False, decoder_replayed=False,
                  holdout_opened=False, training_run=False, production_observations_changed=False,
                  accuracy_improvement_claimed=False, cohorts=cohorts)
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        output.write(serialized)


if __name__ == '__main__':
    main()
