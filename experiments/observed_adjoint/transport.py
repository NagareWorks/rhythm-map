"""Staged transport orchestration over the frozen scaled-adjoint primitives.

The earlier record_gradient remains a frozen numerical/cost reference. This
adapter exposes real in-flight state, rather than scraping locals or replacing
autograd at runtime. It does not invent a second recurrence or clipping rule.
"""
import math

import numpy as np
import torch

from experiments.training_observer.recorder import cpu_copy
from experiments.training_observer.phase_trace import traced_synchronize
from experiments.scaled_adjoint.scaled import Gradient, accumulate, require
from experiments.scaled_adjoint.scan import scaled_vjp
from experiments.scaled_adjoint.transport import BAND_BITS, MAX_BANDS, INPUT_NAMES


def pack_gradient(gradient):
    if gradient is None:
        return None
    return dict(schema='binary-gradient-v1', exponent=gradient.exponent,
                alignment_underflows=gradient.alignment_underflows,
                blocks={name: torch.from_numpy(value.copy()) for name, value in gradient.blocks.items()})


def unpack_gradient(packet):
    require(isinstance(packet, dict) and set(packet) == {'schema', 'exponent', 'alignment_underflows', 'blocks'}
            and packet['schema'] == 'binary-gradient-v1', 'invalid gradient packet schema')
    require(isinstance(packet['blocks'], dict) and all(isinstance(x, torch.Tensor) and x.dtype == torch.float64
            and x.device.type == 'cpu' for x in packet['blocks'].values()), 'CPU float64 gradient blocks required')
    return Gradient({name: x.detach().numpy() for name, x in packet['blocks'].items()},
                    packet['exponent'], packet['alignment_underflows'])


def replay_field_adjoint(snapshot):
    """Only replay the captured field adjoint; no CNN, optimizer or fit resume."""
    require(snapshot.get('schema') == 'observed-adjoint-v1', 'invalid transport snapshot')
    trace = snapshot.get('clock_trace')
    require(trace is not None and trace.get('complete') and trace.get('upstream') is not None
            and trace.get('jacobian') is not None, 'complete clock trace/upstream required')
    return scaled_vjp(trace['jacobian'].numpy(), trace['upstream'].double().numpy())


class ObservedTransport:
    def __init__(self, notify=None):
        self.notify = notify
        self.reset()

    def reset(self):
        self.stage, self.last_trace = 'idle', None
        self.field_gradient, self.parameter_gradient, self.loss = None, None, None
        self.bands, self.pieces, self.band_index = [], [], None

    def mark(self, stage, band_index=None):
        self.stage, self.band_index = stage, band_index
        if self.notify is not None:
            self.notify(stage, band_index)

    def snapshot(self):
        return dict(schema='observed-adjoint-v1', stage=self.stage, loss=self.loss,
                    band_index=self.band_index, bands=self.bands.copy(), clock_trace=cpu_copy(self.last_trace),
                    field_gradient=pack_gradient(self.field_gradient),
                    completed_bands=[pack_gradient(g) for g in self.pieces],
                    parameter_gradient=pack_gradient(self.parameter_gradient))

    def run(self, model, payload, loss_fn):
        self.reset()
        self.mark('fields')
        parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        require(parameters and all(p.dtype in (torch.float32, torch.float64) for p in parameters.values()),
                'native float32/float64 trainable parameters required; no AMP')
        fields = model.fields(payload['features'])
        inputs = (fields[:, :-1, 0], fields[:, 1:, 1:3], fields[:, 0, 3],
                  math.log(2.) * torch.sigmoid(model.gain_logit.double()))
        require(all(v.requires_grad for v in inputs), 'all four clock fields must be trainable')
        self.mark('clock_forward')
        clock = traced_synchronize(self, *inputs)
        self.mark('loss')
        loss = loss_fn(clock, payload['reference'], payload['valid'])['total']
        require(loss.numel() == 1 and torch.isfinite(loss).item(), 'invalid scalar clock loss')
        self.loss = float(loss.detach())
        self.mark('upstream')
        upstream, = torch.autograd.grad(loss, clock.cycles)
        self.mark('field_adjoint')
        self.field_gradient = scaled_vjp(self.last_trace['jacobian'].numpy(), upstream.detach().cpu().double().numpy())
        require(self.field_gradient.alignment_underflows == 0, 'field alignment lost nonzero components')
        exponents = {name: np.frexp(value)[1] for name, value in self.field_gradient.blocks.items()}
        self.bands = sorted({int(band) for name, value in self.field_gradient.blocks.items()
                             for band in ((exponents[name][value != 0] - 1) // BAND_BITS)})
        self.mark('band_plan')
        require(len(self.bands) <= MAX_BANDS, 'neural VJP band budget exceeded')
        for index, band in enumerate(self.bands):
            self.mark('band_cast', index)
            shift = BAND_BITS * (band + 1)
            gradients = []
            for name, tensor in zip(INPUT_NAMES, inputs):
                value = self.field_gradient.blocks[name]
                mask = (value != 0) & ((exponents[name] - 1) // BAND_BITS == band)
                value = np.ldexp(np.where(mask, value, 0.), -shift)
                native = torch.from_numpy(np.asarray(value).copy()).to(tensor)
                require(torch.isfinite(native).all().item() and
                        not np.any((value != 0) & (native.detach().cpu().double().numpy() == 0)),
                        'native field-gradient cast lost range')
                gradients.append(native)
            self.mark('neural_vjp', index)
            values = torch.autograd.grad(inputs, tuple(parameters.values()), tuple(gradients),
                                         retain_graph=index + 1 < len(self.bands), allow_unused=False)
            require(all(torch.isfinite(v).all().item() for v in values), 'nonfinite native CNN VJP')
            self.pieces.append(Gradient.create({name: v.detach().cpu().double().numpy()
                                               for name, v in zip(parameters, values)},
                                              self.field_gradient.exponent + shift))
            self.mark('band_complete', index)
        self.mark('parameter_accumulation')
        self.parameter_gradient = (accumulate(self.pieces) if self.pieces else
                                   Gradient.create({name: np.zeros(tuple(p.shape)) for name, p in parameters.items()}))
        require(self.parameter_gradient.alignment_underflows == 0, 'parameter-band alignment lost nonzero components')
        self.mark('record_complete')
        return self.parameter_gradient, self.loss
