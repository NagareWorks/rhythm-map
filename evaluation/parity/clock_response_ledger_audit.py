"""Full-span response accounting, not a likelihood adapter or tempo selector.

Clocks are supplied. Responses are supplied, not extracted or thresholded here.
All optimal one-to-one matchings are counted exactly without duplicate skip paths.
Unmatched ticks remain possible omissions/wrong clocks, never acoustic absence.
"""
import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / 'clock-response-ledger-lock-v1.json'
LOCK = json.loads(LOCK_PATH.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clock_ticks(frame_count, period, phase=4, changes=()):
    """Render every tick, including domain edges; changes occur at clock ticks.

The period at a change tick determines its next interval. These are supplied
candidate parameters, not detected tempo or user-facing product settings.
"""
    require(type(frame_count) is int and 1 <= frame_count <= LOCK['max_frames'], 'invalid frame count')
    require(type(period) is int and period > 0 and type(phase) is int and 0 <= phase < period, 'invalid clock origin')
    require(all(len(c) == 2 and all(type(v) is int for v in c) and
                phase <= c[0] < frame_count and c[1] > 0 for c in changes), 'invalid clock change')
    require(all(a[0] < b[0] for a, b in zip(changes, changes[1:])), 'unordered clock changes')
    transitions = dict(changes)
    ticks, used, t = [], set(), phase
    while t < frame_count:
        require(len(ticks) < LOCK['max_ticks'], 'clock tick limit exceeded')
        ticks.append(t)
        if t in transitions:
            period = transitions[t]
            used.add(t)
        t += period
    require(used == set(transitions), 'change is not on a clock tick')
    return ticks


def ledger(observed, responses, ticks, max_states=LOCK['max_states']):
    """Bounded exact matching inventory. No truth/clock-name/label inputs.

The canonical recurrence either skips the current tick or matches a later
response. Unused responses have no separate skip edge, preventing multiple
paths from counting the same matching. Logit marks are retained by the caller;
their finite shape is validated, but no score or probability is inferred.
"""
    size = len(observed)
    require(1 <= size <= LOCK['max_frames'] and all(type(v) is bool for v in observed), 'invalid availability domain')
    require(type(max_states) is int and 1 <= max_states <= LOCK['max_states'], 'invalid state budget')
    require(len(ticks) <= LOCK['max_ticks'] and len(responses) <= LOCK['max_responses'], 'event limit exceeded')
    require(all(type(t) is int and 0 <= t < size for t in ticks) and
            all(a < b for a, b in zip(ticks, ticks[1:])), 'invalid clock ticks')
    for r in responses:
        require(set(r) == {'frame', 'beat_logit', 'downbeat_logit'} and type(r['frame']) is int and
                0 <= r['frame'] < size and observed[r['frame']], 'invalid response identity or availability')
        require(all(type(r[k]) in (int, float) and math.isfinite(r[k]) for k in ('beat_logit', 'downbeat_logit')),
                'expected paired finite response marks')
    frames = [r['frame'] for r in responses]
    require(all(a < b for a, b in zip(frames, frames[1:])), 'duplicate or unordered responses')
    radius = LOCK['radius_frames']
    compatible = [range(bisect_left(frames, t - radius), bisect_right(frames, t + radius)) for t in ticks]
    nodes = {}

    def solve(i, first):
        key = i, first
        if key in nodes:
            return nodes[key]
        require(len(nodes) < max_states, 'exact state budget exceeded; no partial result')
        nodes[key] = None  # reserve before recursion, so the budget includes active states
        if i == len(ticks):
            node = dict(matched=0, distance=0, ways=1, edges=[])
        else:
            options = []
            for j in [None] + [j for j in compatible[i] if j >= first]:
                destination = i + 1, first if j is None else j + 1
                child = solve(*destination)
                rank = (-child['matched'] - (j is not None),
                        child['distance'] + (0 if j is None else abs(ticks[i] - frames[j])))
                options.append((rank, j, destination, child['ways']))
            best = min(o[0] for o in options)
            selected = [o for o in options if o[0] == best]
            node = dict(matched=-best[0], distance=best[1], ways=sum(o[3] for o in selected),
                        edges=[(o[1], o[2]) for o in selected])
        nodes[key] = node
        return node

    root = solve(0, 0)
    prefixes = {(0, 0): 1}
    tick_counts, response_counts, pair_counts = [0] * len(ticks), [0] * len(responses), {}
    for key in sorted(nodes):
        prefix = prefixes.get(key, 0)
        if not prefix:
            continue
        i, _ = key
        for j, destination in nodes[key]['edges']:
            prefixes[destination] = prefixes.get(destination, 0) + prefix
            if j is not None:
                ways = prefix * nodes[destination]['ways']
                tick_counts[i] += ways
                response_counts[j] += ways
                pair_counts[i, j] = pair_counts.get((i, j), 0) + ways
    witness, key = [], (0, 0)
    while nodes[key]['edges']:
        i, _ = key
        j, destination = min(nodes[key]['edges'], key=lambda edge: (
            edge[0] is None, abs(ticks[i] - frames[edge[0]]) if edge[0] is not None else 0,
            edge[0] if edge[0] is not None else -1))
        if j is not None:
            witness.append([i, j])
        key = destination

    def status(count):
        return 'matched_in_all_optima' if count == root['ways'] else 'matched_in_some_optima' if count else 'unmatched_in_all_optima'

    queries = []
    for i, t in enumerate(ticks):
        available = sum(0 <= frame < size and observed[frame] for frame in range(t - radius, t + radius + 1))
        queries.append(dict(frame=t, available_query_frames=available, full_query_observed=available == 2 * radius + 1,
                            matched_assignments=tick_counts[i], status=status(tick_counts[i])))
    return dict(frame_count=size, observed_frames=sum(observed), unavailable_frames=size - sum(observed),
                clock_tick_count=len(ticks), response_count=len(responses), optimal_matched_count=root['matched'],
                unmatched_tick_count=len(ticks) - root['matched'], unmatched_response_count=len(responses) - root['matched'],
                minimum_total_absolute_offset_frames=root['distance'], optimal_assignment_count=root['ways'],
                complete_observed_response_cover=root['matched'] == len(responses) if responses else None,
                states=len(nodes), tick_queries=queries,
                response_coverage=[dict(frame=t, matched_assignments=response_counts[j], status=status(response_counts[j]))
                                   for j, t in enumerate(frames)],
                optimal_pair_counts=[dict(tick_index=i, response_index=j, assignments=n) for (i, j), n in sorted(pair_counts.items())],
                witness_pairs=witness, witness_unmatched_ticks=[i for i in range(len(ticks)) if i not in {p[0] for p in witness}],
                witness_unmatched_responses=[j for j in range(len(responses)) if j not in {p[1] for p in witness}])


def response_rows(frames, weak=()):
    return [dict(frame=t, beat_logit=-4. if t in weak else 4., downbeat_logit=-6.) for t in frames]


def authored_cases():
    """Fixed authored response packets only. No audio or dense-capture adapter."""
    size = 148
    family = [dict(id='constant', period=12, phase=4, changes=[]),
              dict(id='half', period=24, phase=4, changes=[]),
              dict(id='double', period=6, phase=4, changes=[]),
              dict(id='slowdown', period=12, phase=4, changes=[[64, 24]]),
              dict(id='speedup', period=12, phase=4, changes=[[64, 6]])]
    grids = {s['id']: clock_ticks(size, s['period'], s['phase'], s['changes']) for s in family}
    examples = [
        ('constant', grids['constant'], (), []),
        ('constant_middle_omissions', [t for t in grids['constant'] if t not in (64, 76)], (), []),
        ('slowdown', grids['slowdown'], (), []),
        ('same_responses_constant_tail_omissions', grids['slowdown'], (), []),
        ('speedup', grids['speedup'], (), []),
        ('same_responses_fast_clock_prefix_omissions', grids['speedup'], (), []),
        ('constant_shifted', [t + 2 for t in grids['constant']], (), []),
        ('constant_jitter', [t + (-2 if i % 2 else 2) for i, t in enumerate(grids['constant'])], (), []),
        ('weak_subdivision_responses', grids['double'], set(grids['double']) - set(grids['constant']), []),
        ('strong_subdivision_responses', grids['double'], (), []),
        ('all_weak_constant_responses', grids['constant'], grids['constant'], []),
        ('constant_missing_observations', [t for t in grids['constant'] if t not in (64, 76)], (), list(range(61, 80))),
        ('no_responses_observed', [], (), []),
        ('no_observations', [], (), list(range(size))),
    ]
    return [dict(id=name, frame_count=size, unavailable_frames=unavailable, responses=response_rows(frames, weak), clocks=family)
            for name, frames, weak, unavailable in examples]


def evaluate(case):
    unavailable = set(case['unavailable_frames'])
    observed = [i not in unavailable for i in range(case['frame_count'])]
    clocks = []
    for spec in case['clocks']:
        ticks = clock_ticks(case['frame_count'], spec['period'], spec['phase'], spec['changes'])
        clocks.append(dict(id=spec['id'], ledger=ledger(observed, case['responses'], ticks)))
    return dict(clocks=clocks, complete_response_cover_clocks=[c['id'] for c in clocks if c['ledger']['complete_observed_response_cover']],
                selected_tempo=None, detected_beats_emitted=False)


def make_report():
    source_names = ('shared-phase-context-v1.json', 'presence-likelihood-v1.json', 'beat-this-semantics-source-v1.json')
    return dict(schema_version=1, purpose=LOCK['purpose'], contract=LOCK,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                lock_sha256=hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
                predecessor_sha256={p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in source_names},
                authored_only=True, real_music_evaluated=False, neural_inference=False, likelihood_adapter=False,
                fitted_parameters=False, holdout_opened=False, training_run=False, production_output_changed=False,
                user_parameters_added=False, unknown_clock_search=False, accuracy_improvement_claimed=False,
                cases=[dict(input=c, result=evaluate(c)) for c in authored_cases()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    serialized = json.dumps(make_report(), indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8', newline='\n') as output:
        output.write(serialized)


if __name__ == '__main__':
    main()
