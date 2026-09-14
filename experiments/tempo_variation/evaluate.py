"""Reuse frozen absolute evaluation, explicitly relabel exposed schedule roles."""
import numpy as np
import torch

from experiments.local_tempo import evaluate as legacy
from .data import require
from .loss import components

# Keep the old evaluator byte-pinned, including its saved NPZ field names.
# Public metrics use new arm names. Artifact aliases are explicit provenance,
# not extra strategies or an inference-time correction.
ALIASES = {
    'initial': 'initial', 'previous-global': 'previous-global',
    'local-only-selected': 'native-retained-selected', 'local-only-final': 'native-retained-final',
    'local-retained-selected': 'variation-retained-selected', 'local-retained-final': 'variation-retained-final',
}


def rename(value):
    if isinstance(value, dict):
        return {ALIASES.get(k, k): rename(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rename(v) for v in value]
    return value


def evaluate(models, records, captures, points, device, output, selected_epoch, check):
    require(tuple(models) == tuple(ALIASES.values()), 'model population changed')
    old_models = {old: models[new] for old, new in ALIASES.items()}
    natural, pairs = legacy.evaluate(old_models, records, captures, points, device, output, check)
    summary, gate = legacy.decision(natural, pairs, selected_epoch)
    # The numeric checks are identical, but no schedule is fresh in this run.
    for key in ('fresh_transfer', 'fresh_pairs_passed', 'fresh_transfer_passed'):
        gate[key.replace('fresh', 'exposed_schedule', 1)] = gate.pop(key)
    gate['independent_acceptance'] = False
    decomposition = []
    for row, observed in zip(records, natural):
        require(row['id'] == observed['id'], 'native decomposition population changed')
        check()
        with np.load(output/(row['id']+'.natural.npz'), allow_pickle=False) as archive:
            metrics = {}
            for old, new in ALIASES.items():
                terms = components(torch.from_numpy(archive[old].copy())[None], row['target'].cpu()[None], row['mask'].cpu()[None])
                result = {k: float(v) for k, v in terms.items()}
                require(abs(result['full']-observed['metrics'][old]['mse']) <= 1e-7, 'absolute native metric differs')
                require(abs(result['full']-result['offset']-result['variation']) <= 1e-10, 'native decomposition differs')
                metrics[new] = result
            decomposition.append(dict(id=row['id'], role=row['role'], work=row['work'], metrics=metrics))
    return rename(natural), rename(pairs), rename(summary), gate, decomposition
