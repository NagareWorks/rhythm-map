"""Immutable prior closure and explicit exposed-versus-fresh pair ownership."""
import json
from pathlib import Path

import numpy as np

from experiments.relative_tempo import data as old
from experiments.tempo_pairs.timeline import Timeline

ROOT, IDS, require, sha, write_json = old.ROOT, old.IDS, old.require, old.sha, old.write_json
HERE = Path(__file__).resolve().parent
FRESH = {'fresh_slow_fast_normal': (.8, 1.25, 1.), 'fresh_fast_slow_normal': (1.25, .8, 1.)}
FFMPEG_SHA = '36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45'
OLD_REPORT = 'experiments/relative_tempo/results-v1.json'
OLD_REPORT_SHA = '599086fdbdb0134b9fe758814add0b785884f6d2eb2e038a54340ef763950ddc'
OLD_PLAN_SHA = '95f1aea338a585a691453cfeaadc7d809e24e7cd1b9cf9888d2cd0077b531677'


def prior():
    require(sha(ROOT/OLD_REPORT) == OLD_REPORT_SHA and sha(old.HERE/'plan-v1.json') == OLD_PLAN_SHA,
            'old relative experiment changed')
    report = json.loads((ROOT/OLD_REPORT).read_bytes())
    require(report['completed'] and report['source_sha256'] == old.closure(), 'old code closure changed')
    return report


def closure():
    result = dict(prior()['source_sha256'])
    paths = [p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py') if not p.name.startswith('test_')]
    paths += ['experiments/local_tempo/PROTOCOL-v1.md', OLD_REPORT,
              'experiments/relative_tempo/plan-v1.json', 'experiments/tempo_pairs/render.py',
              'experiments/tempo_pairs/run.py', 'experiments/tempo_pairs/PROTOCOL-v1.md']
    result.update({p: sha(ROOT/p) for p in paths})
    return result


def fit_points():
    rows = old.population()
    require(sum(r['admitted'] for r in rows) == 559, 'old sparse population changed')
    return [dict(r, previous_pair_role=r['pair_role'], pair_role='fit_exposed') for r in rows]


def fresh_point(identity, profile, index, natural, delayed, supported, region):
    timeline = Timeline(old.SOURCE_EDGES, FRESH[profile])
    center = natural['center_s']
    source = float(timeline.to_source(center))
    segment = min(int(np.searchsorted(timeline.output_edges, center, side='right')-1), 2)
    reason = None
    if not supported:
        reason = 'alignment_not_supported'
    elif region != 'interior':
        reason = 'noninterior'
    elif min(abs(center-timeline.output_edges)) < old.CLEARANCE or min(abs(source-np.asarray(old.SOURCE_EDGES))) < old.CLEARANCE:
        reason = 'cnn_context_near_seam_or_edge'
    return dict(source_id=identity, profile=profile, index=index, output_s=center, source_s=source,
                rate=timeline.rates[segment], pair_role='fresh_schedule_diagnostic', admitted=reason is None,
                exclusion=reason, region=region, natural=natural, delayed=delayed)


def groups(points, profiles):
    result = [[r for r in points if r['admitted'] and (r['source_id'], r['profile']) == (i, p)]
              for i in IDS for p in profiles]
    require(all(result), 'empty source/profile group')
    return result
