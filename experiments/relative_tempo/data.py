"""Pinned population, sparse query ownership and failure-preserving metrics."""
import json
from pathlib import Path

import numpy as np

from experiments.tempo_pairs.protocol import ROOT, IDS, PROFILES, SOURCE_EDGES, sha, selected_sources
from experiments.tempo_pairs.timeline import Timeline

ALIGNMENT = 'experiments/tempo_alignment/results-v1.json'
ALIGNMENT_SHA = '326e2b853b88f6bad9907c7ba63623c22c25ebb56e541518c16d7c53b8ce4b50'
WEIGHTS_SHA = '1b25f43474ba780d845dccd032a8e4c150ebabc77c50f31cc975e9419c4e214a'
CHECKPOINT_SHA = '8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331'
FIT_PROFILES = ('identity_seams', 'slow', 'fast')
FPS, CLEARANCE = 50, 2.56
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def closure():
    paths = [p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py')]
    paths += ['experiments/relative_tempo/PROTOCOL-v1.md', ALIGNMENT,
              'experiments/tempo_pairs/results-v1.json', 'experiments/tempo_pairs/protocol.py',
              'experiments/tempo_pairs/timeline.py', 'experiments/tempo_alignment/run.py',
              'experiments/coupled_clock/inputs-v1.json', 'experiments/clock_readout/inputs-v1.json',
              'experiments/separated_evidence_fit/results-v1.json',
              'experiments/separated_evidence/model.py', 'experiments/clock_readout/model.py',
              'evaluation/parity/prehead_capture.py']
    # Pin transitive in-repository implementations, not only the runner entrypoint.
    for directory in ('coupled_clock', 'separated_evidence', 'tempo_attribution', 'tempo_alignment'):
        paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments'/directory).glob('*.py')]
    paths += [r['truth'] for r in json.loads((ROOT/'experiments/clock_readout/inputs-v1.json').read_bytes())['cases']]
    return {p: sha(ROOT/p) for p in sorted(set(paths))}


def point_record(case, point):
    timeline = Timeline(SOURCE_EDGES, PROFILES[case['profile']])
    center = point['natural']['center_s']
    source = float(timeline.to_source(center))
    segment = min(int(np.searchsorted(timeline.output_edges, center, side='right')-1), 2)
    reason = None
    if not point['feature_capture_supported']:
        reason = 'alignment_not_supported'
    elif point['region'] != 'interior':
        reason = 'noninterior'
    elif min(abs(center-timeline.output_edges)) < CLEARANCE or min(abs(source-np.asarray(SOURCE_EDGES))) < CLEARANCE:
        reason = 'cnn_context_near_seam_or_edge'
    return dict(source_id=case['source_id'], profile=case['profile'], index=point['index'],
                output_s=center, source_s=source, rate=timeline.rates[segment],
                pair_role='fit' if case['profile'] in FIT_PROFILES else 'schedule_diagnostic',
                admitted=reason is None, exclusion=reason)


def population():
    require(sha(ROOT/ALIGNMENT) == ALIGNMENT_SHA, 'alignment outcome changed')
    selected_sources()  # Revalidate original fit roles, licenses and truth identities.
    report = json.loads((ROOT/ALIGNMENT).read_bytes())
    require(report['completed'] and report['negative_gate_passed'] and not report['training_eligible'],
            'alignment source contract changed')
    require([(c['source_id'], c['profile']) for c in report['cases']] ==
            [(i, p) for i in IDS for p in PROFILES], 'pair population changed')
    rows = [point_record(c, p) for c in report['cases'] for p in c['points'] if p['kind'] == 'grid']
    require(len(rows) == 915, 'grid denominator changed')
    for identity in IDS:
        for profile in PROFILES:
            require(any(r['admitted'] and r['source_id'] == identity and r['profile'] == profile for r in rows),
                    'source/profile lost every point')
    return rows


def query_indices(times, length):
    times = np.atleast_1d(np.asarray(times, np.float64))
    position = times * FPS - .5  # Scalar predictions live at outgoing-cell centers.
    require(np.isfinite(position).all() and np.all(position >= 0) and np.all(position <= length-1),
            'query outside predicted cell support')
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower+1, length-1)
    return lower, upper, position-lower


def query_numpy(values, times):
    values = np.asarray(values)
    require(values.ndim == 1 and len(values) > 1 and np.isfinite(values).all(), 'invalid predicted field')
    a, b, f = query_indices(times, len(values))
    return values[a]*(1-f) + values[b]*f


def relative_metrics(prediction, target):
    prediction, target = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    require(prediction.ndim == 1 and prediction.shape == target.shape and len(target) > 0
            and np.isfinite(prediction).all() and np.isfinite(target).all(), 'invalid relative population')
    changed = target != 0
    return dict(points=len(target), changed_points=int(changed.sum()),
                rmse=float(np.sqrt(np.mean((prediction-target)**2))),
                zero_change_rmse=float(np.sqrt(np.mean(target**2))),
                mean_abs_prediction=float(np.mean(np.abs(prediction))),
                slope=float(np.dot(prediction[changed], target[changed])/np.dot(target[changed], target[changed])) if changed.any() else None,
                sign_correct=float(np.mean(prediction[changed]*target[changed] > 0)) if changed.any() else None)
