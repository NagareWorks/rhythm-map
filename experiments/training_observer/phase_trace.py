"""Observation-only adapter for the frozen phase scan; no gradient replacement."""
import math

import torch

from experiments.phase_sync.model import PhaseSyncReadout
from experiments.phase_sync.scan import synchronize
from experiments.training_observer.recorder import cpu_copy, tensor_summary


class TracedPhaseSyncReadout(PhaseSyncReadout):
    def __init__(self):
        super().__init__()
        self.last_trace = None

    def forward(self, features, chunk_frames=None):
        self.last_trace = dict(stage='fields', complete=False)
        fields = self.fields(features, chunk_frames)
        return traced_synchronize(self, fields[:, :-1, 0], fields[:, 1:, 1:3], fields[:, 0, 3],
                                  math.log(2.) * torch.sigmoid(self.gain_logit.double()))


def traced_synchronize(owner, periods, observations, initial, gain):
    """Capture the one known frozen scan's saved Jacobian and actual upstream.

    The hook is scoped to synchronize, not the CNN or arbitrary saved tensors.
    A new implementation must supply its own explicit tracing contract.
    """
    trace = dict(stage='scan', complete=False, source='experiments/phase_sync/scan.py',
                 inputs=cpu_copy(dict(periods=periods, observations=observations, initial=initial, gain=gain)),
                 input_devices=[str(x.device) for x in (periods, observations, initial, gain)],
                 upstream=None, jacobian=None)
    owner.last_trace = trace
    captured = []

    def pack(tensor):
        if tensor.shape == (5, *periods.shape) and tensor.dtype == torch.float64:
            captured.append(cpu_copy(tensor))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        result = synchronize(periods, observations, initial, gain)
    if result.cycles.requires_grad:
        if len(captured) != 1:
            raise ValueError('phase trace contract: expected one saved scan Jacobian')
        trace['jacobian'] = captured[0]

        def observe_upstream(gradient):
            trace['upstream'] = cpu_copy(gradient)
            # Returning None leaves the actual gradient untouched.

        result.cycles.register_hook(observe_upstream)
    trace.update(stage='forward_complete', complete=True,
                 cycles_summary=tensor_summary(result.cycles),
                 state_jacobian_summary=tensor_summary(captured[0][0]) if captured else None)
    return result


def replay_backward(trace, device='cpu'):
    """Explicit optimizer-free scan replay, never a continuation of model fitting.

    This replays the saved *field-level* operation, not the CNN backward. CPU
    replay of a CUDA record is not a promise of identical device numerics.
    """
    if not trace['complete'] or trace['upstream'] is None or trace['jacobian'] is None:
        raise ValueError('complete forward and upstream capture required')
    values = {k: v.to(device).detach().clone().requires_grad_() for k, v in trace['inputs'].items()}
    result = synchronize(values['periods'], values['observations'], values['initial'], values['gain'])
    result.cycles.backward(trace['upstream'].to(device))
    return {k: v.grad for k, v in values.items()}
