"""Pinned closed outcome and immutable state ownership."""
import json
from pathlib import Path

from experiments.local_tempo import data as old

ROOT, require, sha, write_json = old.ROOT, old.require, old.sha, old.write_json
HERE = Path(__file__).resolve().parent
PREVIOUS = 'experiments/local_tempo/results-v1.json'
PREVIOUS_SHA = 'e4c61280383e8c56fd0378eb9d178fc9e4a6eeb3e649ce64fe0683e85f1580ee'


def prior():
    require(sha(ROOT/PREVIOUS) == PREVIOUS_SHA, 'closed local-tempo result changed')
    report = json.loads((ROOT/PREVIOUS).read_bytes())
    require(report['completed'] and report['source_sha256'] == old.closure(), 'predecessor source changed')
    require(sha(old.HERE/'plan-v1.json') == report['plan_sha256'], 'predecessor plan changed')
    return report


def closure():
    result = dict(prior()['source_sha256'])
    paths = [p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py') if not p.name.startswith('test_')]
    paths += [PREVIOUS, 'experiments/local_tempo/plan-v1.json', 'experiments/tempo_conflict/PROTOCOL-v1.md']
    result.update({p: sha(ROOT/p) for p in paths})
    return result


def slots(report):
    require([f['arm'] for f in report['fits']] == ['local-only', 'local-retained'], 'arm population changed')
    result = []
    for fit in report['fits']:
        require([r['epoch'] for r in fit['history']] == list(range(21)), 'missing saved epoch')
        for row in fit['history']:
            result.append(dict(arm=fit['arm'], epoch=row['epoch'],
                key='initial' if row['epoch'] == 0 else f"{fit['arm']}.epoch-{row['epoch']:02}",
                file=f"{fit['arm']}.epoch-{row['epoch']:02}.weights.npz", weight_sha256=row['weight_sha256'],
                state_sha256=row['state_sha256']))
    require(result[0]['state_sha256'] == result[21]['state_sha256'], 'epoch-zero identities differ')
    return result
