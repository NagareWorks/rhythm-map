"""Development-constrained, training-pair-ranked checkpoint selection."""
import numpy as np

from .data import require

METRICS = ('mse', 'tempo_median_error_percent')


def admissibility(current, baseline):
    require(len(current) == len(baseline) == 5, 'expected five development records')
    require([(r['id'], r['work']) for r in current] == [(r['id'], r['work']) for r in baseline],
            'development population changed')
    works = sorted({r['work'] for r in baseline})
    require(len(works) == 3, 'development works changed')
    require(all(np.isfinite(r[k]) and r[k] >= 0 for rows in (current, baseline) for r in rows for k in METRICS),
            'invalid development metric')
    checks = []
    for work in works:
        for metric in METRICS:
            now = float(np.mean([r[metric] for r in current if r['work'] == work]))
            old = float(np.mean([r[metric] for r in baseline if r['work'] == work]))
            checks.append(dict(work=work, metric=metric, value=now, baseline=old, passed=now <= 1.05*old))
    regressions = sum(a['tempo_median_error_percent'] > 1.05*b['tempo_median_error_percent']
                      for a, b in zip(current, baseline))
    return dict(passed=all(c['passed'] for c in checks) and regressions <= 2,
                work_checks=checks, recording_regressions=regressions)


def choose(history):
    require(history and history[0]['epoch'] == 0 and history[0]['admissibility']['passed'], 'missing initial fallback')
    require([r['epoch'] for r in history] == list(range(len(history))), 'incomplete epoch history')
    require(all(np.isfinite(r['training_pair_mse']) and r['training_pair_mse'] >= 0 for r in history),
            'invalid training pair score')
    # Python min preserves the first exact tie. No diagnostic metric is read.
    return min((r for r in history if r['admissibility']['passed']), key=lambda r: r['training_pair_mse'])
