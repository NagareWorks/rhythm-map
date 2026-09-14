"""Remove only the recording-wide residual offset inside FIT supervision."""
import torch

from experiments.separated_evidence.supervision import task_loss, validate_reference, FPS
from experiments.local_tempo.data import require


def components(prediction, reference, mask):
    require(reference.ndim == 2 and reference.shape[0] == 1,
            'decompose one recording at a time')
    full = task_loss('tempo', prediction, reference, mask)
    cells = validate_reference(reference, mask)
    rate = (reference[:, 1:].double()-reference[:, :-1].double())[cells]*FPS
    residual = prediction[cells].double()+torch.log2(rate)
    mean = residual.mean()
    return dict(full=full, offset=mean.square(), variation=(residual-mean).square().mean())


def active_loss(arm, terms):
    require(arm in ('native-retained', 'variation-retained'), 'unknown loss arm')
    return terms['full' if arm == 'native-retained' else 'variation']
