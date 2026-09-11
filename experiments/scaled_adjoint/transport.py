"""Full-clock VJP -> exponent bands -> unchanged native CNN -> batch clipping.

No optimizer, dataset loader or automatic retry lives in this numerical module.
Only losses depending on the returned clock are supported by this chain rule.
"""
import math
from types import SimpleNamespace

import numpy as np
import torch

from experiments.coupled_clock.supervision import clock_loss
from experiments.training_observer.phase_trace import traced_synchronize
from experiments.scaled_adjoint.scaled import Gradient, accumulate, require
from experiments.scaled_adjoint.scan import scaled_vjp

# Numerical transport bounds, not learned policies or per-song settings.
BAND_BITS = 64
MAX_BANDS = 32
INPUT_NAMES = ('periods', 'observations', 'initial', 'gain')


def record_gradient(model, features, reference, valid, *, loss_fn=clock_loss):
    """Return an un-clipped parameter gradient with an explicit binary scale.

    The caller must apply its registered work/batch weight and accumulate EVERY
    record before install_clipped. No detach of recurrent state or window reset.
    """
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    require(parameters and all(p.dtype in (torch.float32, torch.float64) for p in parameters.values()),
            'native float32/float64 trainable parameters required; no AMP')
    fields = model.fields(features)
    inputs = (fields[:, :-1, 0], fields[:, 1:, 1:3], fields[:, 0, 3],
              math.log(2.) * torch.sigmoid(model.gain_logit.double()))
    require(all(v.requires_grad for v in inputs), 'all four clock fields must be trainable')
    owner = SimpleNamespace(last_trace=None)
    clock = traced_synchronize(owner, *inputs)
    result = loss_fn(clock, reference, valid)
    loss = result['total']
    require(loss.numel() == 1 and torch.isfinite(loss).item(), 'invalid scalar clock loss')
    # Stop at q to obtain the complete dL/dq. Do not invoke the old overflowing
    # scan backward. Its identical local Jacobian is evaluated below instead.
    upstream, = torch.autograd.grad(loss, clock.cycles)
    trace = owner.last_trace
    field_gradient = scaled_vjp(trace['jacobian'].numpy(), upstream.detach().cpu().double().numpy())
    require(field_gradient.alignment_underflows == 0, 'field alignment lost nonzero components')
    exponents = {name: np.frexp(value)[1] for name, value in field_gradient.blocks.items()}
    bands = sorted({int(band) for name, value in field_gradient.blocks.items()
                    for band in ((exponents[name][value != 0] - 1) // BAND_BITS)})
    require(len(bands) <= MAX_BANDS, 'neural VJP band budget exceeded')
    pieces = []
    for index, band in enumerate(bands):
        shift = BAND_BITS * (band + 1)
        gradients = []
        for name, tensor in zip(INPUT_NAMES, inputs):
            value = field_gradient.blocks[name]
            mask = (value != 0) & ((exponents[name] - 1) // BAND_BITS == band)
            # Each nonzero injected component has magnitude in [2**-64, 1).
            value = np.ldexp(np.where(mask, value, 0.), -shift)
            native = torch.from_numpy(np.asarray(value).copy()).to(tensor)
            require(torch.isfinite(native).all().item() and
                    not np.any((value != 0) & (native.detach().cpu().double().numpy() == 0)),
                    'native field-gradient cast lost range')
            gradients.append(native)
        values = torch.autograd.grad(inputs, tuple(parameters.values()), tuple(gradients),
                                     retain_graph=index + 1 < len(bands), allow_unused=False)
        require(all(torch.isfinite(v).all().item() for v in values), 'nonfinite native CNN VJP')
        pieces.append(Gradient.create({name: v.detach().cpu().double().numpy()
                                       for name, v in zip(parameters, values)},
                                      field_gradient.exponent + shift))
    parameter_gradient = (accumulate(pieces) if pieces else
                          Gradient.create({name: np.zeros(tuple(p.shape)) for name, p in parameters.items()}))
    require(parameter_gradient.alignment_underflows == 0, 'parameter-band alignment lost nonzero components')
    return dict(gradient=parameter_gradient, loss=float(loss.detach()), clock=clock.cycles.detach(),
                trace=trace, neural_vjp_passes=len(bands), field_exponent=field_gradient.exponent,
                field_gradient_norm_log2=field_gradient.norm_log2())


def install_clipped(model, gradient, max_norm=1.):
    """Install one globally clipped accumulated gradient; never call step here."""
    require(gradient.alignment_underflows == 0, 'accumulation lost nonzero components')
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    require(parameters.keys() == gradient.blocks.keys(), 'parameter ownership differs')
    clipped = gradient.clipped(max_norm=max_norm)
    prepared = {}
    for name, parameter in parameters.items():
        require(clipped[name].shape == tuple(parameter.shape), 'parameter gradient shape differs')
        value = torch.from_numpy(np.asarray(clipped[name]).copy()).to(parameter)
        require(torch.isfinite(value).all().item(), 'nonfinite installed parameter gradient')
        prepared[name] = value
    # Conversion/validation succeeds for every parameter before mutating any.
    for name, parameter in parameters.items():
        parameter.grad = prepared[name]
    return dict(gradient_norm_log2=gradient.norm_log2(), binary_exponent=gradient.exponent,
                alignment_underflows=gradient.alignment_underflows, max_norm=max_norm)
