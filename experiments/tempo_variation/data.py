"""Closed inputs and explicit exposure labels; no new data admission."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.tempo_conflict import data as audit

old = audit.old
ROOT, require, sha, write_json = old.ROOT, old.require, old.sha, old.write_json
HERE = Path(__file__).resolve().parent
AUDIT_SHA = 'ec0321b9992221b2ab7008af50f8516914bd2f001944b5aed0536c4bc0762b99'


def prior():
    path = audit.HERE/'results-v1.json'
    require(sha(path) == AUDIT_SHA, 'closed conflict audit changed')
    report = json.loads(path.read_bytes())
    require(report['completed'] and report['source_sha256'] == audit.closure(), 'audit source changed')
    return audit.prior()


def closure():
    prior()
    result = dict(audit.closure())
    paths = [p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py') if not p.name.startswith('test_')]
    paths += ['experiments/tempo_variation/PROTOCOL-v1.md', 'experiments/tempo_conflict/results-v1.json']
    result.update({p: sha(ROOT/p) for p in paths})
    return result


def points():
    previous = prior()
    path = old.HERE/'plan-v1.json'
    require(sha(path) == previous['plan_sha256'], 'previous point ledger changed')
    rows = json.loads(path.read_bytes())['points']
    require(len(rows) == 1281, 'point population changed')
    return [dict(r, previous_pair_role=r['pair_role'],
                 pair_role='exposed_schedule_diagnostic' if r['profile'] in old.FRESH else 'fit_exposed') for r in rows]


def load_exposed(local_run, device):
    report = prior()
    require(sha(local_run/'report.json') == audit.PREVIOUS_SHA, 'private local result changed')
    cases = report['capture']['cases']
    require([(r['source_id'], r['profile']) for r in cases] == [(i, p) for i in old.IDS for p in old.FRESH],
            'exposed capture population changed')
    captures = {}
    for row in cases:
        path = local_run/f"{row['source_id']}.{row['profile']}.features.npz"
        require(sha(path) == row['feature_sha256'], 'exposed feature changed')
        with np.load(path, allow_pickle=False) as archive:
            x = archive['hidden'].copy()
        require(x.shape == (row['frames'], 512) and x.dtype == np.float32 and np.isfinite(x).all(), 'bad feature tensor')
        captures[row['source_id'], row['profile']] = torch.from_numpy(x).to(device)
    return captures
