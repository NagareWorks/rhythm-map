"""Failure-preserving reports for training-exposed and new-schedule points."""
import numpy as np
import torch

from experiments.coupled_clock.run import state_hash, work_mean
from experiments.relative_tempo.run import natural_metrics
from . import data
from .data import require, sha

NAMES = ('initial', 'previous-global', 'local-only-selected', 'local-only-final',
         'local-retained-selected', 'local-retained-final')
PRIMARY = 'local-retained-selected'


def evaluate(models, records, captures, points, device, output, check):
    require(tuple(models) == NAMES, 'evaluation model population changed')
    natural, pairs = [], []
    before = {k: state_hash(v) for k, v in models.items()}
    for row in records:
        check()
        metrics, predictions = {}, {}
        for name, model in models.items():
            with torch.inference_mode():
                p = model(row['features'].to(device)[None])[0].cpu().numpy()
            require(np.isfinite(p).all(), 'nonfinite natural field')
            predictions[name] = p
            metrics[name] = natural_metrics(p, row)
        path = output/(row['id']+'.natural.npz')
        with path.open('xb') as stream:
            np.savez(stream, **predictions)
        natural.append(dict(id=row['id'], role=row['role'], work=row['work'], metrics=metrics, prediction_sha256=sha(path)))
    profiles = (*data.old.PROFILES, *data.FRESH)
    for identity in data.IDS:
        fields = {}
        for profile in ('source', *profiles):
            check()
            x = captures[identity, profile][None]
            with torch.inference_mode():
                for name, model in models.items():
                    fields[name, profile] = model(x)[0].cpu().numpy()
                fields['primary-zero', profile] = models[PRIMARY](torch.zeros_like(x))[0].cpu().numpy()
        path = output/(identity+'.paired-fields.npz')
        with path.open('xb') as stream:
            np.savez(stream, **{k[0]+'__'+k[1]: v for k, v in fields.items()})
        for profile in profiles:
            all_rows = [r for r in points if (r['source_id'], r['profile']) == (identity, profile)]
            admitted = [r for r in all_rows if r['admitted']]
            require(admitted and len({r['pair_role'] for r in all_rows}) == 1, 'invalid evaluation group')
            observed = {}
            for name in (*models, 'primary-zero'):
                observed[name] = -(data.old.query_numpy(fields[name, profile], [r['output_s'] for r in admitted])
                                   - data.old.query_numpy(fields[name, 'source'], [r['source_s'] for r in admitted]))
            target = np.log2([r['rate'] for r in admitted])
            changed = target != 0
            metrics = {name: dict(all=data.old.relative_metrics(p, target),
                                  changed=data.old.relative_metrics(p[changed], target[changed]) if changed.any() else None)
                       for name, p in observed.items()}
            pairs.append(dict(source_id=identity, profile=profile, role=admitted[0]['pair_role'],
                total_grid_points=len(all_rows), admitted_points=len(admitted), metrics=metrics, prediction_sha256=sha(path),
                observations=[dict(index=r['index'], target_log2_rate=float(target[j]),
                                   prediction={k: float(v[j]) for k, v in observed.items()}) for j, r in enumerate(admitted)]))
    require(before == {k: state_hash(v) for k, v in models.items()}, 'evaluation changed weights')
    return natural, pairs


def decision(natural, pairs, selected_epoch):
    require(len(natural) == 40 and len({r['id'] for r in natural}) == 40, 'incomplete natural evaluation')
    require([(r['source_id'], r['profile']) for r in pairs] ==
            [(i, p) for i in data.IDS for p in (*data.old.PROFILES, *data.FRESH)], 'incomplete pair evaluation')
    summary = {}
    for role in ('fit', 'development', 'diagnostic'):
        summary[role] = {}
        for name in NAMES:
            summary[role][name] = {}
            for metric in ('mse', 'tempo_median_error_percent', 'tempo_p95_error_percent'):
                groups = {}
                for row in natural:
                    if row['role'] == role:
                        groups.setdefault(row['work'], []).append(row['metrics'][name][metric])
                summary[role][name][metric] = work_mean(groups)
    identities, transfer = [], []
    for pair in pairs:
        p = pair['metrics'][PRIMARY]
        if pair['profile'] == 'identity_seams':
            identities.append(dict(source_id=pair['source_id'], passed=p['all']['mean_abs_prediction'] <= .05))
        if pair['profile'] in data.FRESH:
            p, b = p['changed'], pair['metrics']['previous-global']['changed']
            require(p is not None and b is not None, 'fresh changed population missing')
            checks = dict(against_zero=p['rmse'] <= .75*p['zero_change_rmse'], against_previous=p['rmse'] <= .8*b['rmse'],
                          slope=.5 <= p['slope'] <= 1.5, sign=p['sign_correct'] >= .8)
            transfer.append(dict(source_id=pair['source_id'], profile=pair['profile'], checks=checks, passed=all(checks.values())))
    retention = [dict(role=role, metric=k, passed=summary[role][PRIMARY][k] <= 1.05*summary[role]['initial'][k])
                 for role in ('development', 'diagnostic') for k in ('mse', 'tempo_median_error_percent')]
    require(len(identities) == 3 and len(transfer) == 6, 'decision population incomplete')
    gate = dict(selected_epoch=selected_epoch, learned_checkpoint=selected_epoch > 0, identities=identities,
                fresh_transfer=transfer, natural_retention=retention, identity_passed=all(r['passed'] for r in identities),
                fresh_pairs_passed=sum(r['passed'] for r in transfer), fresh_transfer_passed=all(r['passed'] for r in transfer),
                natural_retention_passed=all(r['passed'] for r in retention))
    gate['passed'] = all(gate[k] for k in ('learned_checkpoint', 'identity_passed', 'fresh_transfer_passed', 'natural_retention_passed'))
    return summary, gate
