"""Common-frame presence likelihood reference; evaluation only, not a decoder.

The input is three class-conditional log densities, NOT three class posteriors
or arbitrary neural logits. A fixed unit-Gaussian sensor supplies an analytic
control on the old authored features. It is not an audio calibration model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from clock_boundary_audit import roots
from jump_evidence_audit import context_controls
from test_search_omission import beta, next_states, tempo_matrix

ROOT = Path(__file__).resolve().parents[2]
NEG_INF = -math.inf


def logadd(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def gaussian_sensor(values):
    """Known normalized toy densities: N((0,0),I), N((1,0),I), N((1,1),I).

    No centering, fitted mean/scale, thresholding, or gain selection. The old
    scores are used as coordinates only; no claim that detector scores follow
    this distribution. None marginalizes both coordinates without a trial.
    """
    result = []
    for pair in values:
        if pair is None:
            result.append(None)
        else:
            if len(pair) != 2 or not all(math.isfinite(x) for x in pair):
                raise ValueError('invalid sensor coordinates')
            b, d = pair
            try:
                densities = [-math.log(2 * math.pi) - .5 * ((b - mb) ** 2 + (d - md) ** 2)
                             for mb, md in [(0., 0.), (1., 0.), (1., 1.)]]
            except OverflowError as error:
                raise ValueError('non-finite sensor density') from error
            if not all(math.isfinite(x) for x in densities):
                raise ValueError('non-finite sensor density')
            result.append(densities)
    return result


def emissions(evidence, phase, counts):
    if evidence is None:
        yield None, counts, 0.
        return
    n, b, z, d, u, c = counts
    for label in range(3 if phase == 0 else 2):
        yield label, (n + 1, b + (label > 0), z + (label > 0 and phase == 0),
                      d + (label == 2), u, c), evidence[label] - evidence[0]


def terminal(counts):
    n, b, z, d, u, c = counts
    return math.log(beta(b, n - b)) + math.log(beta(d, z - d)) + math.log(beta(c, u - c))


def infer(evidence, domain):
    """Exact log-space count-augmented DAG, with one common all-frame null.

    Source clock/meter/omission priors and the stationary window law are frozen.
    Every available non-tick frame is absent; its density is in the common
    background factor. Only ticks need relative-density updates in the graph.
    """
    if any(type(domain[k]) is not int for k in ('min_period', 'max_period', 'min_meter', 'max_meter')):
        raise ValueError('period and meter bounds must be integers')
    if not 2 <= len(evidence) <= 32 or not 2 <= domain['min_period'] <= domain['max_period'] <= len(evidence):
        raise ValueError('invalid bounded frame/period domain')
    if not 2 <= domain['min_meter'] <= domain['max_meter'] <= 7:
        raise ValueError('invalid meter domain')
    if type(domain['max_states']) is not int or not 1 <= domain['max_states'] <= 250000:
        raise ValueError('invalid exact state budget')
    if any(v is not None and (len(v) != 3 or not all(math.isfinite(x) for x in v)
                              or not all(math.isfinite(x - v[0]) for x in v)) for v in evidence):
        raise ValueError('expected three finite class-conditional log densities')
    try:
        background = math.fsum(v[0] for v in evidence if v is not None)
    except OverflowError as error:
        raise ValueError('non-finite background density') from error
    transitions = tempo_matrix(domain)
    layers = [{} for _ in evidence]
    count = 0

    def insert(t, key, mass, maximum, parent):
        nonlocal count
        if not math.isfinite(mass) or not math.isfinite(maximum):
            raise ValueError('non-finite path score; no partial inference returned')
        if key in layers[t]:
            node = layers[t][key]
            node[0] = logadd(node[0], mass)
            if maximum > node[1]:
                node[1:] = [maximum, parent]
        else:
            if count >= domain['max_states']:
                raise ValueError('exact search state budget exceeded; no partial inference returned')
            layers[t][key] = [mass, maximum, parent]
            count += 1

    for t, key, mass in roots(domain, 'stationary'):
        insert(t, key, math.log(mass), math.log(mass), None)
    total, best, ending, edges = NEG_INF, NEG_INF, None, 0
    for t, layer in enumerate(layers):
        for key, (mass, maximum, _) in layer.items():
            p, m, phase, *counts = key
            for label, new_counts, emission in emissions(evidence[t], phase, tuple(counts)):
                if t + p >= len(evidence):
                    weight = emission + terminal(new_counts)
                    total = logadd(total, mass + weight)
                    if maximum + weight > best:
                        best, ending = maximum + weight, (t, key, label)
                    edges += 1
                else:
                    for following, transition in next_states(p, m, phase, new_counts, domain, transitions):
                        weight = emission + math.log(transition)
                        insert(t + p, following, mass + weight, maximum + weight, (t, key, label))
                        edges += 1
    if not all(math.isfinite(x) for x in (background, total, best, background + total)):
        raise ValueError('non-finite evidence; no partial inference returned')
    positions = np.zeros((len(evidence), 7))
    suffix = [{} for _ in evidence]
    for t in reversed(range(len(evidence))):
        for key, (mass, _, _) in layers[t].items():
            p, m, phase, *counts = key
            back = NEG_INF
            for label, new_counts, emission in emissions(evidence[t], phase, tuple(counts)):
                if t + p >= len(evidence):
                    rest = terminal(new_counts)
                else:
                    rest = NEG_INF
                    for following, transition in next_states(p, m, phase, new_counts, domain, transitions):
                        value = math.log(transition) + suffix[t + p][following]
                        rest = logadd(rest, value)
                        probability = math.exp(mass + emission + value - total)
                        positions[t + p, 5] += probability * (following[0] != p)
                        positions[t + p, 6] += probability * (following[1] != m)
                back = logadd(back, emission + rest)
                probability = math.exp(mass + emission + rest - total)
                positions[t, 0] += probability
                positions[t, 4 if label is None else label + 1] += probability
            suffix[t][key] = back
    ticks = []
    while ending is not None:
        t, key, label = ending
        ticks.append(dict(frame=t, period_frames=key[0], meter=key[1], beat_in_bar=key[2] + 1,
                          inferred_label=label))
        ending = layers[t][key][2]
    ticks.reverse()
    # Tick/omission labels are conditional on a latent clock. Absence on a
    # non-tick frame is a separate contribution, included here for all frames.
    emission_positions = [[None, None, None] if value is None else
                          [1. - row[2] - row[3], row[2], row[3]]
                          for value, row in zip(evidence, positions)]
    return dict(log_ratio=total, background_log_density=background, log_evidence=background + total,
                joint_map_log_weight=best, joint_map_probability=math.exp(best - total),
                inferred_ticks=ticks, positions=positions.tolist(),
                emission_positions=emission_positions, states=count, transitions=edges)


def score_path(evidence, domain, ticks):
    """Independent fixed-path factor product, after inference; not a selector."""
    if not ticks:
        raise ValueError('empty clock path')
    first = ticks[0]
    root = next((mass for t, key, mass in roots(domain, 'stationary') if
                 t == first['frame'] and key[:3] == (first['period_frames'], first['meter'], first['beat_in_bar'] - 1)), None)
    if root is None:
        raise ValueError('unsupported initial clock')
    transitions = tempo_matrix(domain)
    score = dict(initial=math.log(root), clock_stay=0., jump_occurrence=0., jump_destination=0.,
                 meter_destination=0., emission_log_ratio=0., terminal_extra=0.)
    counts = (0,) * 6
    for i, tick in enumerate(ticks):
        t, p, m, phase, label = (tick[k] for k in ('frame', 'period_frames', 'meter', 'beat_in_bar', 'inferred_label'))
        if (not 0 <= t < len(evidence) or not domain['min_period'] <= p <= domain['max_period']
                or not domain['min_meter'] <= m <= domain['max_meter'] or not 1 <= phase <= m):
            raise ValueError('tick outside fixed domain')
        if i:
            old = ticks[i - 1]
            if t != old['frame'] + old['period_frames']:
                raise ValueError('invalid tick interval')
            wrap = old['beat_in_bar'] == old['meter']
            if phase != (1 if wrap else old['beat_in_bar'] + 1) or (not wrap and m != old['meter']):
                raise ValueError('invalid meter path')
            choices = domain['max_meter'] - domain['min_meter']
            if wrap and choices:
                counts = (*counts[:4], counts[4] + 1, counts[5] + (m != old['meter']))
                if m != old['meter']:
                    score['meter_destination'] -= math.log(choices)
            previous = old['period_frames']
            if p == previous:
                score['clock_stay'] += math.log(transitions[previous, p])
            else:
                chance = 1. - transitions[previous, previous]
                score['jump_occurrence'] += math.log(chance)
                score['jump_destination'] += math.log(transitions[previous, p] / chance)
        options = [(c, w) for l, c, w in emissions(evidence[t], phase - 1, counts) if l == label]
        if not options:
            raise ValueError('invalid label')
        counts, weight = options[0]
        score['emission_log_ratio'] += weight
    if ticks[-1]['frame'] + ticks[-1]['period_frames'] < len(evidence):
        raise ValueError('unterminated path')
    n, b, z, d, u, c = counts
    score.update(pulse_retention=math.log(beta(b, n - b)), accent_retention=math.log(beta(d, z - d)),
                 meter_changes=math.log(beta(c, u - c)))
    score['total'] = sum(score.values())
    return score


def audit():
    original = json.loads((ROOT / 'evaluation/parity/jump-evidence-v1.json').read_text())
    domain = original['domain']
    cases = []
    for row in original['cases']:
        evidence = gaussian_sensor(row['feature_pairs'])
        result = infer(evidence, domain)
        record = dict(case=row['case'], feature_pairs=row['feature_pairs'], log_densities=evidence, decoded=result,
                      map_score=score_path(evidence, domain, result['inferred_ticks']))
        if 'fixed_paths' in row:
            record['posthoc_fixed_paths'] = {name: dict(path=item['path'], score=score_path(evidence, domain, item['path']))
                                              for name, item in row['fixed_paths'].items()}
        cases.append(record)
    longer = []
    for name, values, authored in context_controls():
        evidence = gaussian_sensor(values)
        result = infer(evidence, domain)
        longer.append(dict(case=name, feature_pairs=values, decoded=result,
                           authored_path=authored, authored_score=score_path(evidence, domain, authored),
                           map_score=score_path(evidence, domain, result['inferred_ticks'])))
    sensor_controls = []
    for name, pair in [('neutral', [.5, .5]), ('absence', [-4., -4.]), ('unavailable', None)]:
        values = [pair] * 18
        sensor_controls.append(dict(case=name, feature_pairs=values, decoded=infer(gaussian_sensor(values), domain)))
    # Additional INPUT contrast, not a new likelihood model or selected setting:
    # only paired zeros become the already-defined absence sensor coordinate.
    # In particular a positive pulse with a zero accent coordinate is untouched.
    absence_contrasts = []
    for row in cases + longer:
        values = [[-4., -4.] if pair == [0., 0.] else pair for pair in row['feature_pairs']]
        evidence = gaussian_sensor(values)
        result = infer(evidence, domain)
        absence_contrasts.append(dict(case=row['case'], feature_pairs=values, decoded=result,
                                      map_score=score_path(evidence, domain, result['inferred_ticks'])))
    source_files = ['evaluation/parity/presence_likelihood_audit.py', 'evaluation/parity/clock_boundary_audit.py',
                    'evaluation/parity/jump_evidence_audit.py', 'evaluation/parity/jump-evidence-v1.json',
                    'evaluation/parity/test_search_omission.py']
    return dict(schema_version=1, purpose='normalized_common_frame_presence_likelihood',
                production_output_changed=False, user_parameters_added=False, training_run=False,
                real_music_evaluated=False, holdout_opened=False, calibrated_audio_confidence=False,
                fitted_parameters=False, transition_law_changed=False, boundary_law_changed=False,
                omission_law_changed=False, labels_are_detected_events=False, full_song_search=False,
                domain=domain, boundary='stationary', sensor='fixed_unit_gaussian_control_not_audio_calibration',
                sensor_means=[[0., 0.], [1., 0.], [1., 1.]],
                position_columns=original['position_columns'], emission_columns=['absent', 'plain', 'accent'],
                cases=cases, context_controls=longer, sensor_controls=sensor_controls,
                absence_contrasts=absence_contrasts,
                source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files})


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
