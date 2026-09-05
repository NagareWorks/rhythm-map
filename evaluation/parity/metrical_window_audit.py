"""Fixed, truth-assisted joint-head windows; no decoder or likelihood adapter.

Freeze extraction before the first cohort run. Coordinates and frame arrays
stay private. Off-grid is annotation-relative geometry, never acoustic silence.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import json
import math
from pathlib import Path
import statistics

import dense_clock_evidence as dense
from candidate_evidence_audit import auc

ROOT = dense.ROOT
LOCK_PATH = Path(__file__).with_name('metrical-window-lock-v1.json')
LOCK = json.loads(LOCK_PATH.read_text())
FEATURES = tuple(LOCK['head_features'])
REGIMES = ('annotated_constant_interior', 'annotated_change_neighborhood', 'annotated_ramp', 'rubato')
LABELS = ('downbeat', 'non_downbeat', 'downbeat_unknown')
OLD_DENSE = 'e4826ec6996b58e404a9773f49fe21c46126b960ee1ca83aa719cce6fe18fd12'


def window(beats, downbeats, center, available=None):
    """Observation-only feature function. No annotation, label or case argument.

    Heads are paired at the chosen beat peak. The independent downbeat maximum
    and peak separation are retained separately, not added to a score.
    """
    dense.require(type(center) is int and len(beats) == len(downbeats), 'invalid window dimensions')
    dense.require(available is None or len(available) == len(beats), 'invalid availability')
    left, right = center - 3, center + 3
    if left < 0 or right >= len(beats):
        return None, 'out_of_capture'
    if available is not None and not all(available[left:right + 1]):
        return None, 'unavailable_frame'
    dense.require(all(math.isfinite(float(v)) for head in (beats, downbeats)
                      for v in head[left:right + 1]), 'nonfinite available frame')
    def peak(head):
        return max(range(left, right + 1), key=lambda t: (float(head[t]), -abs(t - center), -t))
    b, d = peak(beats), peak(downbeats)
    features = dict(beat_peak=float(beats[b]), downbeat_at_beat_peak=float(downbeats[b]),
                downbeat_peak=float(downbeats[d]), peak_separation_frames=abs(d - b),
                beat_peak_gain=float(beats[b]) - float(beats[center]),
                center_beat=float(beats[center]))
    dense.require(all(math.isfinite(v) for v in features.values()), 'nonfinite window feature')
    return features, 'available'


def annotation_labels(truth, case, cohort):
    dense.require(truth.get('id') == case['id'] and
                  [b['time_s'] for b in truth['beats']] == case['truth_times_s'], 'truth identity changed')
    times = case['truth_times_s']
    dense.require(len(times) >= 2 and dense.ordered(times), 'invalid truth ordering')
    dense.require(all(type(b.get('downbeat')) is bool for b in truth['beats']), 'invalid downbeat labels')
    dense.require(cohort in ('artbeat', 'rubato'), 'unsupported annotation source')
    if cohort == 'rubato' or 'rubato' in case['tags']:
        regimes = ['rubato'] * len(times)
    else:
        anchors = [min(range(len(times)), key=lambda i: (abs(times[i] - cp['time_s']), i))
                   for cp in truth['change_points']]
        regimes = []
        for i, t in enumerate(times):
            if any(abs(i - a) <= 2 for a in anchors):
                regimes.append('annotated_change_neighborhood')
            else:
                segments = [s for s in truth['tempo_segments'] if s['start_s'] <= t < s['end_s']]
                dense.require(len(segments) == 1 and segments[0]['kind'] in ('constant', 'ramp'),
                              'missing or ambiguous tempo annotation')
                regimes.append('annotated_ramp' if segments[0]['kind'] == 'ramp' else 'annotated_constant_interior')
    labels = [('downbeat' if b['downbeat'] else 'non_downbeat') if cohort == 'rubato'
              else 'downbeat_unknown' for b in truth['beats']]
    return labels, regimes


def overlaps(center, centers, excluded=None):
    """Two closed radius-three windows share a frame when centers differ <= 6."""
    first = bisect_left(centers, center - 6)
    return any(i != excluded and abs(centers[i] - center) <= 6
               for i in range(first, bisect_right(centers, center + 6)))


def rows_for_case(beats, downbeats, truth, case, cohort, available=None):
    labels, regimes = annotation_labels(truth, case, cohort)
    times = case['truth_times_s']
    centers = [round(t * 50) for t in times]
    # Same-frame truth collisions remain explicit overlap exclusions, not deduped labels.
    pairs = dense.matches([b['time_s'] for b in case['observations']['beats']], times, case['beat_tolerance_s'])
    dense.require(pairs == case['raw_truth_pairs'], 'raw/truth matching changed')
    matched = {t for _, t in pairs}
    candidates = [b['time_s'] for b in case['observations']['beat_candidates']]
    dense.require(dense.ordered(candidates), 'invalid candidates')
    result = []
    for i, (t, center) in enumerate(zip(times, centers)):
        features, reason = window(beats, downbeats, center, available)
        pair_reason, margin = 'final_beat', None
        if i + 1 < len(times):
            middle = round((t + times[i + 1]) * 25)
            control, _ = window(beats, downbeats, middle, available)
            if features is None or control is None:
                pair_reason = 'window_unavailable'
            elif overlaps(center, centers, excluded=i):
                pair_reason = 'canonical_overlaps_neighbor'
            elif overlaps(middle, centers):
                pair_reason = 'control_overlaps_annotation'
            else:
                pair_reason = 'eligible'
                margin = features['beat_peak'] - control['beat_peak']
        result.append(dict(features=features, window_status=reason, pair_status=pair_reason,
                           margin=margin, raw_missed=i not in matched,
                           candidate_absent_miss=i not in matched and not dense.near(candidates, t, case['beat_tolerance_s']),
                           label=labels[i], regime=regimes[i]))
    return result


def quantiles(values):
    if not values:
        return None
    values = sorted(values)
    output = []
    for q in (.1, .5, .9):
        index = (len(values) - 1) * q
        left, right = math.floor(index), math.ceil(index)
        output.append(values[left] + (values[right] - values[left]) * (index - left))
    return output


def comparison(rows):
    margins = [r['margin'] for r in rows if r['margin'] is not None]
    wins = sum(m > 0 for m in margins)
    return dict(queries=len(rows), pair_status=dict(Counter(r['pair_status'] for r in rows)),
                eligible=len(margins), wins=wins, ties=sum(m == 0 for m in margins),
                losses=sum(m < 0 for m in margins), win_fraction=wins / len(margins) if margins else None,
                mean_margin=statistics.mean(margins) if margins else None)


def summarize(rows):
    full = [r['features'] for r in rows if r['features'] is not None]
    return dict(queries=len(rows), window_status=dict(Counter(r['window_status'] for r in rows)),
                raw_missed=sum(r['raw_missed'] for r in rows),
                candidate_absent_misses=sum(r['candidate_absent_miss'] for r in rows),
                label_counts=dict(Counter(r['label'] for r in rows)), acoustic_presence_unknown=len(rows),
                full_windows=len(full), feature_quantiles={k: quantiles([f[k] for f in full]) for k in FEATURES},
                beat_center_above_zero=sum(f['center_beat'] > 0 for f in full),
                beat_peak_above_zero=sum(f['beat_peak'] > 0 for f in full),
                center_nonpositive_peak_positive=sum(f['center_beat'] <= 0 < f['beat_peak'] for f in full),
                head_peaks_separated=sum(f['peak_separation_frames'] > 0 for f in full),
                comparison=comparison(rows))


def downbeat_ranking(rows):
    positive = [r['features'] for r in rows if r['label'] == 'downbeat' and r['features'] is not None]
    negative = [r['features'] for r in rows if r['label'] == 'non_downbeat' and r['features'] is not None]
    return dict(positive_windows=len(positive), negative_windows=len(negative),
                unknown_label_queries=sum(r['label'] == 'downbeat_unknown' for r in rows),
                auc={k: auc([f[k] for f in positive], [f[k] for f in negative], 1) if positive and negative else None
                     for k in ('downbeat_at_beat_peak', 'downbeat_peak')})


def audit(cohort, evidence_path, capture_dir):
    count, evidence_hash = dense.INPUTS[cohort]
    evidence, _ = dense.read_json(evidence_path, evidence_hash)
    old, _ = dense.read_json(ROOT / 'evaluation/parity/dense-clock-evidence-v1.json', OLD_DENSE)
    prior = next(c for c in old['cohorts'] if c['cohort'] == cohort)
    summary, summary_hash = dense.read_json(Path(capture_dir) / 'summary.json', prior['capture_summary_sha256'])
    records = dense.validate_summary(summary, evidence['cases'], count)
    sources = {k: dense.sha((ROOT / p).read_bytes()) for k, p in dense.SOURCES.items()}
    suite_path = ROOT / 'evaluation/suites' / f'{dense.SUITES[cohort][0]}.json'
    suite, suite_hash = dense.read_json(suite_path, dense.SUITES[cohort][1])
    dense.require([c['id'] for c in suite['cases']] == [c['id'] for c in evidence['cases']], 'suite order changed')
    tracks, all_rows, frames = [], [], 0
    for record, case, item in zip(records, evidence['cases'], suite['cases']):
        payload, _ = dense.read_json(Path(capture_dir) / f"{case['id']}.json", record['capture_sha256'])
        beats = dense.validate_capture(payload, record, case, cohort, sources)
        truth, truth_hash = dense.read_json(suite_path.parent / item['input']['truth'], case['truth_sha256'])
        rows = rows_for_case(beats, payload['downbeat_logits'], truth, case, cohort)
        missed = [r for r in rows if r['raw_missed']]
        tracks.append(dict(id=case['id'], capture_sha256=record['capture_sha256'], truth_sha256=truth_hash,
                           frame_count=len(beats), all=summarize(rows), raw_missed=summarize(missed),
                           by_regime={regime: comparison([r for r in rows if r['regime'] == regime]) for regime in REGIMES},
                           downbeat_ranking=downbeat_ranking(rows), missed_downbeat_ranking=downbeat_ranking(missed)))
        all_rows.extend(rows)
        frames += len(beats)
    groups = {'all': all_rows, 'raw_missed': [r for r in all_rows if r['raw_missed']],
              'candidate_absent_misses': [r for r in all_rows if r['candidate_absent_miss']]}
    for regime in REGIMES:
        groups[regime] = [r for r in all_rows if r['regime'] == regime]
        groups[regime + '_missed'] = [r for r in groups[regime] if r['raw_missed']]
    macro = {}
    for group in ('all', 'raw_missed'):
        fractions = [t[group]['comparison']['win_fraction'] for t in tracks
                     if t[group]['comparison']['win_fraction'] is not None]
        macro[group] = dict(contributing_tracks=len(fractions), total_tracks=count,
                            mean_track_win_fraction=statistics.mean(fractions) if fractions else None)
    ranks = {}
    for group, selected in (('downbeat_ranking', all_rows), ('missed_downbeat_ranking', groups['raw_missed'])):
        ranks[group] = dict(pooled=downbeat_ranking(selected), macro={})
        for feature in ('downbeat_at_beat_peak', 'downbeat_peak'):
            values = [t[group]['auc'][feature] for t in tracks if t[group]['auc'][feature] is not None]
            ranks[group]['macro'][feature] = dict(contributing_tracks=len(values), total_tracks=count,
                                                  mean_track_auc=statistics.mean(values) if values else None)
    return dict(cohort=cohort, complete=True, frozen_evidence_sha256=evidence_hash,
                capture_summary_sha256=summary_hash, suite_sha256=suite_hash, source_hashes=sources,
                total_frames_per_head=frames, groups={k: summarize(v) for k, v in groups.items()},
                macro=macro, ranks=ranks, cases=tracks)


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
                  dense_report_sha256=OLD_DENSE,
                  helper_sha256={p: dense.sha(Path(__file__).with_name(p).read_bytes()) for p in
                                 ('dense_clock_evidence.py', 'clock_phase_evidence.py', 'resampler_event_audit.py',
                                  'candidate_evidence_audit.py')},
                  semantics_source_sha256=dense.sha(Path(__file__).with_name('beat-this-semantics-source-v1.json').read_bytes()),
                  truth_assisted=True, holdout_opened=False, training_run=False, neural_inference=False,
                  fitted_mapping=False, decoder_replayed=False, production_observations_changed=False,
                  accuracy_improvement_claimed=False, cohorts=cohorts)
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        output.write(serialized)


if __name__ == '__main__':
    main()
