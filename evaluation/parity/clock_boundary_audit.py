"""Matched arbitrary-window boundary audit; evaluation only, no decoder changes.

Uses the frozen independent Python factors from search-omission v1. Only the
initial clock/meter law changes. No fitting, clock templates, or event gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from test_search_omission import beta, emitted, finish, initial, next_states, reference, tempo_matrix

ROOT = Path(__file__).resolve().parents[2]


def stationary_boundary(domain):
    """First future tick, not length-biasing its outgoing period by mistake."""
    periods = list(range(domain['min_period'], domain['max_period'] + 1))
    transitions = tempo_matrix(domain)
    matrix = np.array([[transitions[p, q] for q in periods] for p in periods])
    system = matrix.T - np.eye(len(periods))
    system[-1] = 1.
    rhs = np.zeros(len(periods))
    rhs[-1] = 1.
    pi = np.linalg.solve(system, rhs)
    mean = float(pi @ periods)
    # The interval containing time zero has duration q, whereas the first
    # future tick chooses outgoing period p through T[q,p]. Marginalize q.
    weights = {(r, p): sum(float(pi[i]) * transitions[q, p]
                           for i, q in enumerate(periods) if q > r) / mean
               for r in range(max(periods)) for p in periods}
    return dict(periods=periods, transition_matrix=matrix.tolist(),
                tick_stationary_probabilities=pi.tolist(), mean_period=mean,
                first_tick_weights=[[weights[r, p] for p in periods] for r in range(max(periods))])


def roots(domain, boundary):
    if boundary == 'fresh':
        yield from initial(domain)
        return
    if boundary != 'stationary':
        raise ValueError('unknown boundary intervention')
    law = stationary_boundary(domain)
    meters = range(domain['min_meter'], domain['max_meter'] + 1)
    # Meter transitions are symmetric at bar wraps. All beat-in-bar states
    # have equal stationary tick mass, for every nonzero change rate h.
    phases = sum(meters)
    for t, weights in enumerate(law['first_tick_weights']):
        for p, weight in zip(law['periods'], weights):
            if weight > 0:
                for m in meters:
                    for phase in range(m):
                        yield t, (p, m, phase, 0, 0, 0, 0, 0, 0), weight / phases


def infer(values, domain, boundary):
    """Same exact count-augmented graph in probability space, bounded by states."""
    if not 2 <= len(values) <= 32 or not 2 <= domain['min_period'] <= domain['max_period'] <= len(values):
        raise ValueError('invalid bounded frame/period domain')
    if not 2 <= domain['min_meter'] <= domain['max_meter'] <= 7:
        raise ValueError('invalid meter domain')
    if any(v is not None and (len(v) != 2 or not all(math.isfinite(x) for x in v)) for v in values):
        raise ValueError('invalid feature pair')
    pairs, normalizers = reference(values)
    transitions = tempo_matrix(domain)
    layers = [{} for _ in pairs]
    count = 0

    def insert(t, key, mass, maximum, parent):
        nonlocal count
        if key in layers[t]:
            node = layers[t][key]
            node[0] += mass
            if maximum > node[1]:
                node[1:] = [maximum, parent]
        else:
            if count >= domain['max_states']:
                raise ValueError('exact search state budget exceeded; no partial inference returned')
            layers[t][key] = [mass, maximum, parent]
            count += 1

    for t, key, mass in roots(domain, boundary):
        insert(t, key, mass, mass, None)
    total, best, ending, edges = 0., 0., None, 0
    for t, layer in enumerate(layers):
        for key, (mass, maximum, _) in layer.items():
            p, m, phase, *counts = key
            for label, new_counts, emission in emitted(pairs[t], phase, tuple(counts)):
                if t + p >= len(pairs):
                    weight = emission * finish(new_counts, normalizers)
                    total += mass * weight
                    if maximum * weight > best:
                        best, ending = maximum * weight, (t, key, label)
                    edges += 1
                else:
                    for following, transition in next_states(p, m, phase, new_counts, domain, transitions):
                        insert(t + p, following, mass * emission * transition,
                               maximum * emission * transition, (t, key, label))
                        edges += 1
    positions = np.zeros((len(values), 7))
    suffix = [{} for _ in pairs]
    for t in reversed(range(len(pairs))):
        for key, (mass, _, _) in layers[t].items():
            p, m, phase, *counts = key
            back = 0.
            for label, new_counts, emission in emitted(pairs[t], phase, tuple(counts)):
                if t + p >= len(pairs):
                    rest = finish(new_counts, normalizers)
                else:
                    rest = 0.
                    for following, transition in next_states(p, m, phase, new_counts, domain, transitions):
                        value = transition * suffix[t + p][following]
                        rest += value
                        probability = mass * emission * value / total
                        positions[t + p, 5] += probability * (following[0] != p)
                        positions[t + p, 6] += probability * (following[1] != m)
                back += emission * rest
                probability = mass * emission * rest / total
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
    return dict(log_ratio=math.log(total), joint_map_log_weight=math.log(best),
                joint_map_probability=best / total, inferred_ticks=ticks, positions=positions.tolist(),
                states=count, transitions=edges)


def decompose(values, domain, ticks, boundary):
    """Post-inference audit of a fixed path; no truth enters search."""
    pairs, normalizers = reference(values)
    transitions = tempo_matrix(domain)
    p0, m0, t0 = (ticks[0][k] for k in ('period_frames', 'meter', 'frame'))
    period_count = domain['max_period'] - domain['min_period'] + 1
    meter_count = domain['max_meter'] - domain['min_meter'] + 1
    if boundary == 'fresh':
        if not 0 <= t0 < p0:
            raise ValueError('unsupported fresh initial offset')
        clock_initial, meter_initial = -math.log(period_count * p0), -math.log(meter_count * m0)
    elif boundary == 'stationary':
        law = stationary_boundary(domain)
        clock_initial = math.log(law['first_tick_weights'][t0][p0 - domain['min_period']])
        meter_initial = -math.log(sum(range(domain['min_meter'], domain['max_meter'] + 1)))
    else:
        raise ValueError('unknown boundary intervention')
    score = dict(clock_initial=clock_initial, meter_initial=meter_initial,
                 clock_stay=0., jump_occurrence=0., jump_destination=0.,
                 feature_numerator=0., meter_destination=0., terminal_extra=0.)
    counts = (0,) * 6
    for i, tick in enumerate(ticks):
        t, p, m, phase, label = (tick[k] for k in ('frame', 'period_frames', 'meter', 'beat_in_bar', 'inferred_label'))
        if i:
            old = ticks[i - 1]
            if t != old['frame'] + old['period_frames']:
                raise ValueError('invalid tick interval')
            previous = old['period_frames']
            if previous == p:
                score['clock_stay'] += math.log(transitions[previous, p])
            else:
                chance = 1. - transitions[previous, previous]
                score['jump_occurrence'] += math.log(chance)
                score['jump_destination'] += math.log(transitions[previous, p] / chance)
            wrap = old['beat_in_bar'] == old['meter']
            if phase != (1 if wrap else old['beat_in_bar'] + 1) or (not wrap and m != old['meter']):
                raise ValueError('invalid meter path')
            if wrap and meter_count > 1:
                counts = (*counts[:4], counts[4] + 1, counts[5] + (m != old['meter']))
                if m != old['meter']:
                    score['meter_destination'] -= math.log(meter_count - 1)
        choice = next(((c, w) for l, c, w in emitted(pairs[t], phase - 1, counts) if l == label), None)
        if choice is None:
            raise ValueError('invalid label')
        counts, weight = choice
        score['feature_numerator'] += math.log(weight)
    if ticks[-1]['frame'] + ticks[-1]['period_frames'] < len(values):
        raise ValueError('unterminated path')
    n, b, z, d, u, c = counts
    score['paired_normalizer'] = -math.log(normalizers[b - d, d])
    score['pulse_retention'] = math.log(beta(b, n - b))
    score['accent_retention'] = math.log(beta(d, z - d))
    score['meter_changes'] = math.log(beta(c, u - c))
    score['total'] = sum(score.values())
    return score


def authored_witness(case):
    """Post-inference development labels; never called by infer or its factors."""
    paths = {
        'half': ([1, 4, 7, 13], [3, 3, 6, 6], [1, 2, 3, 1], [2, 1, 1, 2]),
        'double': ([1, 7, 10, 13, 16], [6, 3, 3, 3, 3], [1, 2, 3, 1, 2], [2, 1, 1, 2, 1]),
    }
    return [dict(frame=t, period_frames=p, meter=3, beat_in_bar=j, inferred_label=l)
            for t, p, j, l in zip(*paths[case])]


def audit():
    path = ROOT / 'evaluation/parity/search-omission-v1.json'
    original = json.loads(path.read_text())
    domain = original['domain']
    rows = []
    for row in original['cases']:
        values = row['feature_pairs']
        variants = {name: infer(values, domain, name) for name in ('fresh', 'stationary')}
        for name, result in variants.items():
            result['map_score_decomposition'] = decompose(values, domain, result['inferred_ticks'], name)
        record = dict(case=row['case'], feature_pairs=values, variants=variants)
        if row['case'] in ('half', 'double'):
            paths = dict(authored=authored_witness(row['case']), frozen_map=row['decoded']['inferred_ticks'])
            record['posthoc_fixed_paths'] = dict(paths=paths, scores={
                name: {key: decompose(values, domain, ticks, name) for key, ticks in paths.items()}
                for name in ('fresh', 'stationary')})
        rows.append(record)
    padding = []
    for case in ('constant', 'half', 'double'):
        row = next(r for r in rows if r['case'] == case)
        for left, right in [(3, 0), (0, 3)]:
            values = [None] * left + row['feature_pairs'] + [None] * right
            variants = {name: infer(values, domain, name) for name in ('fresh', 'stationary')}
            padding.append(dict(case=case, left=left, right=right, variants=variants))
    witness = original['exhaustive_control']
    witness_result = infer(witness['feature_pairs'], witness['domain'], 'stationary')
    source_files = ['evaluation/parity/clock_boundary_audit.py', 'evaluation/parity/test_search_omission.py',
                    'evaluation/parity/search-omission-v1.json']
    return dict(schema_version=1, purpose='matched_boundary_only_intervention',
                production_output_changed=False, user_parameters_added=False, training_run=False,
                real_music_evaluated=False, holdout_opened=False, calibrated_confidence=False,
                feature_pipeline_changed=False, transition_law_changed=False, interior_meter_law_changed=False,
                omission_law_changed=False, terminal_law_changed=False, labels_are_detected_events=False,
                full_song_search=False, fitted_parameters=False, domain=domain,
                position_columns=['latent_tick', 'omitted', 'plain', 'accent', 'unavailable_tick', 'tempo_change', 'meter_change'],
                stationary_boundary=stationary_boundary(domain), cases=rows, padding_controls=padding,
                exhaustive_control=dict(domain=witness['domain'], feature_pairs=witness['feature_pairs'], decoded=witness_result),
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
