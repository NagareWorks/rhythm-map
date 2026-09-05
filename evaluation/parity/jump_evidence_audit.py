"""Frozen jump-factor and count-conditioned evidence audit; evaluation only.

No fitted rates, decoder alternatives, truth-fed inference, or product changes.
Feature gains are stress tests, not a selected calibration or runtime option.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from clock_boundary_audit import authored_witness, decompose, infer

ROOT = Path(__file__).resolve().parents[2]
GAINS = (1, 2, 4)


def prior_accounting(periods, frame_duration):
    """Reconstruct the frozen law for explicit atoms, separating physical units."""
    if (not periods or any(type(p) is not int or p <= 0 for p in periods)
            or periods != sorted(set(periods)) or not math.isfinite(frame_duration)
            or frame_duration <= 0):
        raise ValueError('invalid ordered period atoms or frame duration')
    affinity = [[0. if p == q else math.exp(-math.log(100) * abs(math.log2(p / q)))
                 for q in periods] for p in periods]
    totals = list(map(sum, affinity))
    rate = sum(math.log1p(a) / p for a, p in zip(totals, periods)) / len(periods)
    stay = [math.exp(-rate * p) for p in periods]
    jump = [-math.expm1(-rate * p) for p in periods]
    destination = [[a / total if total else 0. for a in row] for row, total in zip(affinity, totals)]
    matrix = [[stay[i] if i == j else jump[i] * destination[i][j]
               for j in range(len(periods))] for i in range(len(periods))]
    return dict(periods=periods, frame_duration=frame_duration,
                physical_periods=[p * frame_duration for p in periods],
                rate_per_frame=rate, rate_per_time_unit=rate / frame_duration,
                affinity_totals=totals, stay_probabilities=stay,
                jump_occurrence_probabilities=jump, jump_destination_probabilities=destination,
                transition_matrix=matrix)


def assignment_limit(values, ticks):
    """Exact max-plus coefficient and tie count for INTEGER authored features.

    At fixed plain/accent counts, assignments put disjoint labels on available
    frames. The limit is combinatorial, not a fitted extrapolation from gains.
    None for the limiting log ratio means negative infinity (vanishing ratio).
    """
    if (not values or len(values) > 32 or any(v is not None and
            (len(v) != 2 or any(not math.isfinite(x) or x != int(x) for x in v)) for v in values)):
        raise ValueError('limit audit requires bounded integer feature pairs')
    observed = [tuple(map(int, v)) for v in values if v is not None]
    assignments = {(0, 0): (0, 1)}
    for beat, accent in observed:
        following = {}
        for (plain, down), (score, count) in assignments.items():
            for key, addition in [((plain, down), 0), ((plain + 1, down), beat),
                                  ((plain, down + 1), beat + accent)]:
                value = score + addition
                if key not in following or value > following[key][0]:
                    following[key] = value, count
                elif value == following[key][0]:
                    following[key] = value, following[key][1] + count
        assignments = following
    plain = down = path_score = 0
    seen = set()
    for tick in ticks:
        t, label = tick['frame'], tick['inferred_label']
        if type(t) is not int or not 0 <= t < len(values) or t in seen:
            raise ValueError('invalid or duplicate tick frame')
        seen.add(t)
        pair = values[t]
        if (pair is None and label is not None) or (pair is not None and label not in (0, 1, 2)):
            raise ValueError('invalid label availability')
        plain += label == 1
        down += label == 2
        if label in (1, 2):
            path_score += int(pair[0]) + (int(pair[1]) if label == 2 else 0)
    best, ties = assignments[plain, down]
    n = len(observed)
    count = math.comb(n, plain) * math.comb(n - plain, down)
    return dict(available_frames=n, plain_count=plain, accent_count=down,
                assignment_count=count, maximizing_assignments=ties,
                path_feature_score=path_score, maximum_feature_score=best,
                gain_slope=path_score - best,
                limiting_log_feature_ratio=math.log(count) - math.log(ties) if path_score == best else None)


def scaled(values, gain):
    return [None if v is None else [gain * x for x in v] for v in values]


def compact(result):
    # Keep all reported marginals and scores, but not a duplicate decomposition.
    return {key: value for key, value in result.items() if key != 'map_score_decomposition'}


def fixed_path_audit(values, domain, ticks):
    scores = [decompose(scaled(values, gain), domain, ticks, 'stationary') for gain in GAINS]
    limit = assignment_limit(values, ticks)
    # Centering in the reference cancels between numerator and denominator.
    nonfeature = sum(v for k, v in scores[0].items()
                     if k not in ('total', 'feature_numerator', 'paired_normalizer'))
    ceiling = limit['limiting_log_feature_ratio']
    return dict(path=ticks, gain_scores=scores, feature_limit=limit,
                nonfeature_log_weight=nonfeature,
                limiting_joint_log_weight=None if ceiling is None else nonfeature + ceiling)


def context_controls():
    """Additional observed context, not unavailable padding or raw audio.

    Both early and late acceleration are retained, including their failures.
    These are development controls, not unseen acceptance data.
    """
    plans = {
        'long_constant': (list(range(1, 27, 3)), [3] * 9),
        'long_half': ([1, 4, 7, 10, 16, 22], [3, 3, 3, 6, 6, 6]),
        'long_double_early': ([1, 7, 13, 16, 19, 22, 25], [6, 6, 3, 3, 3, 3, 3]),
        'long_double_late': ([1, 7, 13, 19, 22, 25], [6, 6, 6, 3, 3, 3]),
    }
    for name, (frames, periods) in plans.items():
        ticks = [dict(frame=t, period_frames=p, meter=3, beat_in_bar=i % 3 + 1,
                      inferred_label=2 if i % 3 == 0 else 1) for i, (t, p) in enumerate(zip(frames, periods))]
        values = [[0., 0.] for _ in range(27)]
        for tick in ticks:
            values[tick['frame']] = [4., 3. if tick['inferred_label'] == 2 else 0.]
        yield name, values, ticks


def audit():
    frozen = json.loads((ROOT / 'evaluation/parity/clock-boundary-v1.json').read_text())
    domain = frozen['domain']
    cases = []
    for row in frozen['cases']:
        values = row['feature_pairs']
        results = [compact(row['variants']['stationary'])]
        results.extend(infer(scaled(values, gain), domain, 'stationary') for gain in GAINS[1:])
        entry = dict(case=row['case'], feature_pairs=values, gain_inferences=results)
        if row['case'] in ('half', 'double'):
            # Posthoc only: no authored path reaches infer, its factors or domain.
            entry['fixed_paths'] = {name: fixed_path_audit(values, domain, path) for name, path in {
                'authored': authored_witness(row['case']),
                'frozen_stationary_map': results[0]['inferred_ticks'],
            }.items()}
        cases.append(entry)
    longer = []
    for name, values, authored in context_controls():
        result = infer(values, domain, 'stationary')
        longer.append(dict(case=name, feature_pairs=values, decoded=result,
                           posthoc_authored=fixed_path_audit(values, domain, authored),
                           posthoc_map=fixed_path_audit(values, domain, result['inferred_ticks'])))
    sources = ['evaluation/parity/jump_evidence_audit.py', 'evaluation/parity/clock_boundary_audit.py',
               'evaluation/parity/test_search_omission.py', 'evaluation/parity/clock-boundary-v1.json',
               'crates/rhythm-map-eval/examples/support/time_prior.rs']
    return dict(schema_version=1, purpose='jump_factor_and_conditional_evidence_limits',
                production_output_changed=False, user_parameters_added=False, training_run=False,
                real_music_evaluated=False, holdout_opened=False, calibrated_confidence=False,
                fitted_parameters=False, transition_law_changed=False, boundary_law_changed=False,
                gain_is_calibration=False, labels_are_detected_events=False, full_song_search=False,
                domain=domain, gains=list(GAINS), boundary='stationary',
                prior_controls={
                    'base': prior_accounting([3, 4, 5, 6], 1.),
                    'same_atoms_new_units': prior_accounting([6, 8, 10, 12], .5),
                    'refined_domain': prior_accounting(list(range(6, 13)), .5),
                    'extended_domain': prior_accounting(list(range(3, 8)), 1.),
                    'singleton': prior_accounting([3], 1.),
                }, position_columns=frozen['position_columns'], cases=cases, context_controls=longer,
                source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite a frozen report')
    result = audit()
    with args.output.open('x', encoding='utf-8', newline='\n') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    main()
