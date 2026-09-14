"""Reuse the trained tempo branch; differentiable queries never warp features."""
import numpy as np
import torch

from experiments.clock_readout.model import HALO
from experiments.separated_evidence.model import SharedEvidence, SeparatedEvidence
from .data import WEIGHTS_SHA, require, sha, query_indices


def load_model(path, device):
    require(sha(path) == WEIGHTS_SHA, 'starting export changed')
    model = SeparatedEvidence(SharedEvidence())
    with np.load(path, allow_pickle=False) as archive:
        model.load_state_dict({k: torch.from_numpy(archive[k].copy()) for k in archive.files}, strict=True)
    return model.tempo.to(device)


def query(model, features, times):
    """Two adjacent output cells, exact local context, then scalar interpolation."""
    lower, upper, fractions = query_indices(times, len(features)-1)
    require(np.all(upper == lower+1), 'paired query requires two physical adjacent cells')
    patches = []
    for index in lower:
        start, end = int(index)-HALO, int(index)+HALO+3
        require(start >= 0 and end <= len(features), 'incomplete CNN query context')
        patches.append(features[start:end])
    prediction = model(torch.stack(patches))[:, HALO:HALO+2]
    f = torch.as_tensor(fractions, dtype=prediction.dtype, device=prediction.device)
    return prediction[:, 0]*(1-f) + prediction[:, 1]*f


def paired_loss(model, source, transformed, rows):
    left = query(model, source, [r['source_s'] for r in rows])
    right = query(model, transformed, [r['output_s'] for r in rows])
    target = torch.as_tensor([-np.log2(r['rate']) for r in rows], dtype=right.dtype, device=right.device)
    return (right-left-target).square().mean()
