"""Lossless backend response provenance; no tempo fitting, recovery or model run.

Published events, score frames, exact nominal centers and original paired marks
are distinct. Raw/default and candidate sources are inventoried, never blended
or promoted as user strategies. Real frame arrays and packet coordinates stay private.
"""
import argparse
from bisect import bisect_right
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import struct

import dense_clock_evidence as dense
import rational_response_ledger as rational

HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / 'backend-response-packets-lock-v1.json'
LOCK = json.loads(LOCK_PATH.read_text())
OLD_DENSE = 'e4826ec6996b58e404a9773f49fe21c46126b960ee1ca83aa719cce6fe18fd12'


def published_time(frame):
    return struct.unpack('<f', struct.pack('<f', float(frame) / 50))[0]


def sigmoid(value):
    try:
        return 1. / (1. + math.exp(-float(value)))
    except OverflowError:
        return 0.


def ulp_distance(actual, expected):
    bits = lambda value: struct.unpack('<Q', struct.pack('<d', value))[0]
    return abs(bits(actual) - bits(expected))


def probability_ulps(actual, expected):
    dense.require(type(actual) in (int, float) and math.isfinite(actual) and 0 <= actual <= 1,
                  'invalid published confidence')
    distance = ulp_distance(actual, expected)
    dense.require(distance <= LOCK['confidence_check_max_ulps'], 'published confidence formula mismatch')
    return distance


def peak_groups(values):
    """Fixed shipped strict-zero/radius-three local maxima with running-mean merge."""
    peaks = [i for i, v in enumerate(values) if v > 0 and
             all(x <= v for x in values[max(0, i - 3):i + 4])]
    groups = []
    for i in peaks:
        if groups and Fraction(i) - Fraction(sum(groups[-1]), len(groups[-1])) <= 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    # With distinct integer maxima and shipped width one, a group has at most
    # two adjacent members. Its rational mean is exactly representable in f64.
    dense.require(all(len(g) <= 2 for g in groups), 'unexpected default merge geometry')
    return groups


def candidate_plateaus(values):
    maxima = [i for i, v in enumerate(values) if all(x <= v for x in values[max(0, i - 1):i + 2])]
    plateaus = []
    for i in maxima:
        if plateaus and i == plateaus[-1][-1] + 1:
            plateaus[-1].append(i)
        else:
            plateaus.append([i])
    return plateaus


def reconstruct(beat_logits, downbeat_logits, duration_s):
    """Authored/reference output construction; production records are never overwritten."""
    beat = dense.vector(beat_logits).tolist()
    downbeat = dense.vector(downbeat_logits).tolist()
    dense.require(len(beat) == len(downbeat), 'head dimensions differ')
    dense.require(math.isfinite(duration_s) and 0 <= duration_s <= len(beat) / 50, 'invalid duration')
    raw_groups, plateaus = peak_groups(beat), candidate_plateaus(beat)
    times = [published_time(Fraction(sum(g), len(g))) for g in raw_groups]
    down_times = [published_time(Fraction(sum(g), len(g))) for g in peak_groups(downbeat)]
    snapped = sorted({min(times, key=lambda t: (abs(t - d), t)) for d in down_times}) if times else []
    raw = []
    for t in times:
        frame = min(math.floor(t * 50 + .5), len(beat) - 1)
        flagged = any(abs(d - t) <= .07 for d in snapped)
        raw.append(dict(time_s=t, confidence=sigmoid(beat[frame]),
                        downbeat_confidence=max(.5, sigmoid(downbeat[frame])) if flagged else 0.))
    candidates = []
    for group in plateaus:
        frame = group[(len(group) - 1) // 2]
        t = published_time(frame)
        if t <= duration_s:
            candidates.append(dict(time_s=t, confidence=sigmoid(beat[frame]), downbeat_confidence=sigmoid(downbeat[frame])))
    return dict(beats=raw, beat_candidates=candidates), raw_groups, plateaus


def assemble(beat_logits, downbeat_logits, duration_s, observations):
    """Identity-checked packets, using original published values and frame marks."""
    expected, groups, plateaus = reconstruct(beat_logits, downbeat_logits, duration_s)
    beat, downbeat = dense.vector(beat_logits).tolist(), dense.vector(downbeat_logits).tolist()
    maximum_ulps = 0
    for source in ('beats', 'beat_candidates'):
        actual = observations[source]
        dense.require(len(actual) == len(expected[source]), 'published event count changed')
        for a, e in zip(actual, expected[source]):
            dense.require(a['time_s'] == e['time_s'], 'published event time changed')
            for key in ('confidence', 'downbeat_confidence'):
                maximum_ulps = max(maximum_ulps, probability_ulps(a[key], e[key]))
    starts = [g[0] for g in plateaus]
    raw, candidates, all_plateaus = [], [], []
    for i, g in enumerate(plateaus):
        frame = g[(len(g) - 1) // 2]
        included = published_time(frame) <= duration_s
        row = dict(plateau_index=i, support_first=g[0], support_last=g[-1], representative_frame=frame,
                   included=included, raw_indices=[])
        if included:
            row['candidate_index'] = len(candidates)
            candidates.append(dict(source='candidate', source_index=len(candidates), plateau_index=i,
                                   coordinate=[frame, 1], published=dict(observations['beat_candidates'][len(candidates)]),
                                   score_frame=frame, beat_logit=beat[frame], downbeat_logit=downbeat[frame]))
        all_plateaus.append(row)
    for i, members in enumerate(groups):
        center = Fraction(sum(members), len(members))
        parent = bisect_right(starts, members[0]) - 1
        dense.require(parent >= 0 and members[-1] <= plateaus[parent][-1], 'raw peak lineage not contained in one plateau')
        published = dict(observations['beats'][i])
        frame = min(math.floor(published['time_s'] * 50 + .5), len(beat) - 1)
        all_plateaus[parent]['raw_indices'].append(i)
        raw.append(dict(source='raw', source_index=i, plateau_index=parent, coordinate=rational.encode(center),
                        source_members=members, published=published, score_frame=frame,
                        beat_logit=beat[frame], downbeat_logit=downbeat[frame]))
    return dict(frame_count=len(beat), raw=raw, candidates=candidates, plateaus=all_plateaus,
                max_confidence_formula_ulps=maximum_ulps)


def packet_ledger(observed, packets, ticks):
    """Bounded bridge; marks stay in packet metadata, not in a musical score."""
    identities = [(p['source'], p['source_index']) for p in packets]
    dense.require(len(set(identities)) == len(identities), 'duplicate packet identity')
    dense.require(len({p['source'] for p in packets}) <= 1, 'raw and candidate sources must not be pooled')
    for p in packets:
        dense.require(p['source'] in ('raw', 'candidate') and type(p['source_index']) is int and p['source_index'] >= 0,
                      'invalid packet source')
        dense.require(type(p['score_frame']) is int and 0 <= p['score_frame'] < len(observed) and observed[p['score_frame']],
                      'unavailable packet score frame')
        dense.require(all(math.isfinite(p[k]) for k in ('beat_logit', 'downbeat_logit')), 'invalid paired packet marks')
        dense.require(len(p['coordinate']) == 2 and all(type(x) is int for x in p['coordinate']) and
                      p['coordinate'][1] > 0, 'invalid rational packet coordinate')
    result = rational.ledger(observed, ticks, [Fraction(*p['coordinate']) for p in packets])
    return dict(result=result, response_identities=[list(i) for i in identities])


def summarize(packets):
    raw, candidates, plateaus = (packets[k] for k in ('raw', 'candidates', 'plateaus'))
    reasons = Counter()
    for p in candidates:
        parent = plateaus[p['plateau_index']]
        reasons['has_raw_lineage' if parent['raw_indices'] else
                'nonpositive_default_threshold' if p['beat_logit'] <= 0 else 'radius_three_competition'] += 1
    return dict(raw_events=len(raw), candidates=len(candidates), source_plateaus=len(plateaus),
                excluded_candidate_plateaus=sum(not p['included'] for p in plateaus),
                fractional_raw_centers=sum(p['coordinate'][1] != 1 for p in raw),
                raw_score_frame_differs_from_nominal_round=sum(p['score_frame'] != math.floor(Fraction(*p['coordinate']) + Fraction(1, 2)) for p in raw),
                raw_downbeat_published_not_direct_frame_sigmoid=sum(
                    ulp_distance(p['published']['downbeat_confidence'], sigmoid(p['downbeat_logit'])) > LOCK['confidence_check_max_ulps'] for p in raw),
                raw_downbeat_clamped_from_below_half=sum(p['published']['downbeat_confidence'] == .5 and p['downbeat_logit'] < 0 for p in raw),
                many_raw_to_one_plateau=sum(len(p['raw_indices']) > 1 for p in plateaus),
                raw_without_included_candidate_lineage=sum(not plateaus[p['plateau_index']]['included'] for p in raw),
                raw_center_more_than_three_frames_from_its_candidate=sum(
                    abs(Fraction(*p['coordinate']) - plateaus[p['plateau_index']]['representative_frame']) > 3 for p in raw
                    if plateaus[p['plateau_index']]['included']),
                candidate_reasons={key: reasons[key] for key in ('has_raw_lineage', 'nonpositive_default_threshold', 'radius_three_competition')},
                max_confidence_formula_ulps=packets['max_confidence_formula_ulps'])


def audit(cohort, evidence_path, capture_dir):
    count, evidence_hash = dense.INPUTS[cohort]
    evidence, _ = dense.read_json(evidence_path, evidence_hash)
    old, _ = dense.read_json(HERE / 'dense-clock-evidence-v1.json', OLD_DENSE)
    prior = next(c for c in old['cohorts'] if c['cohort'] == cohort)
    summary, summary_hash = dense.read_json(Path(capture_dir) / 'summary.json', prior['capture_summary_sha256'])
    records = dense.validate_summary(summary, evidence['cases'], count)
    sources = {k: dense.sha((dense.ROOT / p).read_bytes()) for k, p in dense.SOURCES.items()}
    tracks = []
    for record, case in zip(records, evidence['cases']):
        payload, _ = dense.read_json(Path(capture_dir) / f"{case['id']}.json", record['capture_sha256'])
        beats = dense.validate_capture(payload, record, case, cohort, sources)
        packets = assemble(beats, payload['downbeat_logits'], payload['observations']['duration_s'], payload['observations'])
        tracks.append(dict(id=case['id'], capture_sha256=record['capture_sha256'], frame_count=len(beats),
                           source_inventory=summarize(packets)))
    totals = {k: sum(t['source_inventory'][k] for t in tracks) for k in tracks[0]['source_inventory']
              if k not in ('candidate_reasons', 'max_confidence_formula_ulps')}
    totals['candidate_reasons'] = {k: sum(t['source_inventory']['candidate_reasons'][k] for t in tracks)
                                   for k in tracks[0]['source_inventory']['candidate_reasons']}
    totals['max_confidence_formula_ulps'] = max(t['source_inventory']['max_confidence_formula_ulps'] for t in tracks)
    return dict(cohort=cohort, complete=True, frozen_evidence_sha256=evidence_hash, capture_summary_sha256=summary_hash,
                source_hashes=sources, total_frames_per_head=sum(t['frame_count'] for t in tracks), cases=tracks, totals=totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for cohort in dense.INPUTS:
        parser.add_argument(f'--{cohort}-evidence', type=Path, required=True)
        parser.add_argument(f'--{cohort}-captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(schema_version=1, purpose=LOCK['purpose'], contract=LOCK,
                  script_sha256=dense.sha(Path(__file__).read_bytes()), lock_sha256=dense.sha(LOCK_PATH.read_bytes()),
                  helper_sha256={p: dense.sha((HERE / p).read_bytes()) for p in
                                 ('rational_response_ledger.py', 'dense_clock_evidence.py', 'resampler_event_audit.py',
                                  'clock_phase_evidence.py', 'compare_reference.py', 'phase_tail_audit.py', 'verify_bounded_resampler.py')},
                  predecessor_sha256={p: dense.sha((HERE / p).read_bytes()) for p in
                                      ('clock-response-ledger-v1.json', 'backend-response-packets-lock-v1.json')},
                  neural_inference=False, production_output_changed=False, new_user_parameters=False,
                  holdout_opened=False, training_run=False, fitted_mapping=False, truth_labels_used_for_inventory=False,
                  real_clock_ledger_replayed=False, accuracy_improvement_claimed=False,
                  cohorts=[audit(c, getattr(args, f'{c}_evidence'), getattr(args, f'{c}_captures')) for c in dense.INPUTS])
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        output.write(serialized)


if __name__ == '__main__':
    main()
