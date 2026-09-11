"""Experimental positive clock with distributed, bounded phase feedback.

The host float64 scan has an O(B*T) analytic FIRST-order adjoint. It is not an
ONNX operator, a GPU-native kernel, or a production inference implementation.
"""
import math

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from experiments.coupled_clock.clock import Clock, FPS, require

MAX_GAIN = math.log(2.)
TAU = 2 * math.pi


def _host(value):
    return value.detach().to(device='cpu', dtype=torch.float64).numpy()


class _PhaseScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, periods, observations, initial, gain, fps):
        ell, obs, origin, strength = map(_host, (periods, observations, initial, gain))
        strength = float(strength)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            advance = np.exp2(-ell) / fps
        require(np.isfinite(60 * advance * fps).all() and (advance > 0).all(),
                'unrepresentable positive advancement')
        batch, cells = ell.shape
        result = np.empty((batch, cells + 1), dtype=np.float64)
        # q, period, observation-u, observation-v, gain local Jacobians.
        jac = np.empty((5, batch, cells), dtype=np.float64)
        for b in range(batch):
            q = float(origin[b])
            result[b, 0] = q
            for t in range(cells):
                a = float(advance[b, t])
                predicted = q + a
                require(math.isfinite(predicted), 'unrepresentable predicted clock')
                u, v = map(float, obs[b, t])
                scale = math.hypot(1., math.hypot(u, v))
                # hypot itself can overflow for two near-max float64 inputs.
                require(math.isfinite(scale), 'unrepresentable observation norm')
                x, y = u / scale, v / scale
                angle = TAU * (predicted % 1.)
                sn, cs = math.sin(angle), math.cos(angle)
                error = y * cs - x * sn
                multiplier = math.exp(strength * error)
                delta = a * multiplier
                next_q = q + delta
                require(math.isfinite(next_q) and math.isfinite(60 * delta * fps)
                        and next_q > q, 'unrepresentable accumulated clock')
                require(math.isclose(next_q - q, delta, rel_tol=1e-9, abs_tol=1e-12),
                        'accumulation lost cell precision')
                h = -TAU * (y * sn + x * cs)
                feedback = delta * strength
                jac[:, b, t] = (1. + feedback * h,
                               (multiplier + feedback * h) * (-MAX_GAIN * a),
                               feedback * (-sn - error * x) / scale,
                               feedback * (cs - error * y) / scale,
                               delta * error)
                q = next_q
                result[b, t + 1] = q
        require(np.isfinite(jac).all(), 'unrepresentable local clock derivative')
        ctx.save_for_backward(torch.from_numpy(jac))
        ctx.input_meta = [(v.dtype, v.device) for v in (periods, observations, initial, gain)]
        return torch.from_numpy(result).to(device=periods.device)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        jac, = ctx.saved_tensors
        jq, jp, ju, jv, jk = jac.numpy()
        upstream = _host(grad_output)
        batch, cells = jp.shape
        period_grad = np.empty_like(jp)
        observation_grad = np.empty((batch, cells, 2), dtype=np.float64)
        # Never modify an incoming gradient or one of its NumPy views in-place.
        adjoint = upstream[:, -1].copy()
        gain_grad = np.float64(0.)
        with np.errstate(over='ignore', invalid='ignore'):
            for t in range(cells - 1, -1, -1):
                period_grad[:, t] = adjoint * jp[:, t]
                observation_grad[:, t, 0] = adjoint * ju[:, t]
                observation_grad[:, t, 1] = adjoint * jv[:, t]
                gain_grad += (adjoint * jk[:, t]).sum()
                adjoint = upstream[:, t] + adjoint * jq[:, t]
        converted = []
        for value, (dtype, device) in zip(
                (period_grad, observation_grad, adjoint, np.asarray(gain_grad)), ctx.input_meta):
            gradient = torch.from_numpy(np.asarray(value)).to(device=device, dtype=dtype)
            require(torch.isfinite(gradient).all().item(), 'nonfinite clock gradient')
            converted.append(gradient)
        return *converted, None


def synchronize(log2_period, observations, initial, gain, fps=FPS):
    """Right-endpoint phase observations correct positive cell advancements.

    Inputs: [B,C] prior log2 periods/s, [B,C,2] uncalibrated phase vectors,
    [B] initial cycles, and one scalar gain in [0, ln(2)]. Output: C+1 clock
    points and C corrected cell periods, BOTH derived from that same clock.
    There is no availability mask, timestamp list, reference, or reset input.
    State-carry chunking passes the preceding output's final cycle as initial.
    """
    inputs = (log2_period, observations, initial, gain)
    require(all(isinstance(v, torch.Tensor) and v.is_floating_point() for v in inputs),
            'clock inputs must be floating tensors')
    require(log2_period.ndim == 2 and min(log2_period.shape) > 0,
            'invalid period geometry')
    batch, cells = log2_period.shape
    require(observations.shape == (batch, cells, 2) and initial.shape == (batch,)
            and gain.ndim == 0, 'invalid phase clock geometry')
    require(all(v.device == log2_period.device for v in inputs), 'clock devices differ')
    require(isinstance(fps, (int, float)) and not isinstance(fps, bool)
            and math.isfinite(fps) and fps > 0, 'invalid frame rate')
    require(all(torch.isfinite(v).all().item() for v in inputs), 'nonfinite clock input')
    require(0. <= gain.item() <= MAX_GAIN, 'gain outside [0, ln(2)]')
    cycles = _PhaseScan.apply(*inputs, float(fps))
    # The prior period is NOT the returned tempo after feedback.
    corrected = -torch.log2(cycles.diff(dim=1) * fps)
    require(torch.isfinite(corrected).all().item(), 'unrepresentable corrected period')
    return Clock(cycles=cycles, log2_period=corrected, fps=float(fps))
