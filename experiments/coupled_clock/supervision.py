"""Native beat-count supervision, separate from observation availability."""
from dataclasses import dataclass
import math

import numpy as np
import torch

from experiments.coupled_clock.clock import FPS, require

LAGS = (1, 25, 100, 200)  # 20 ms, 0.5 s, 2 s, 4 s at 50 Hz.


@dataclass(frozen=True)
class Reference:
    cycles: np.ndarray
    valid: np.ndarray


def reference_clock(beats, frames, duration_s, support=None):
    """q(t)=i+(t-b[i])/(b[i+1]-b[i]) on bracketed native reference support.

    The optional support mask describes missing REFERENCE annotation, not weak
    attacks or absent detector events. No extrapolated or hard-change labels.
    Unknown targets are NaN so accidentally supervising them fails loudly.
    """
    beats = np.asarray(beats, dtype=np.float64)
    require(beats.ndim == 1 and len(beats) >= 2 and np.isfinite(beats).all()
            and (beats >= 0).all() and (np.diff(beats) > 0).all(), 'invalid reference beats')
    require(type(frames) is int and frames >= 2 and np.isfinite(duration_s)
            and duration_s > 0, 'invalid reference extent')
    times = np.arange(frames, dtype=np.float64) / FPS
    indices = np.searchsorted(beats, times, side='right') - 1
    valid = (indices >= 0) & (indices < len(beats) - 1) & (times < duration_s)
    if support is not None:
        support = np.asarray(support)
        require(support.dtype == bool and support.shape == (frames,), 'invalid reference support')
        valid &= support
    cycles = np.full(frames, np.nan, dtype=np.float64)
    i = indices[valid]
    cycles[valid] = i + (times[valid] - beats[i]) / (beats[i + 1] - beats[i])
    return Reference(cycles=cycles, valid=valid)


def supported_pairs(valid, lag):
    """Require every reference frame on [t,t+lag], not merely both endpoints."""
    require(valid.ndim == 2 and valid.dtype == torch.bool, 'invalid support tensor')
    require(type(lag) is int and 0 < lag < valid.shape[1], 'invalid reference lag')
    missing = torch.cat((torch.zeros_like(valid[:, :1], dtype=torch.int64),
                         (~valid).to(torch.int64).cumsum(dim=1)), dim=1)
    return (missing[:, lag + 1:] - missing[:, :-(lag + 1)]) == 0


def clock_loss(clock, reference, valid):
    """One-recording objective; a future fitter must balance works externally.

    Circular phase alone can accept octave-related clocks at coincident ticks.
    Positive UNWRAPPED advances are also compared in native beat units at four
    fixed lags. No half/double minimum, per-song option, change classifier or
    smoothing penalty. Averaging over available lags retains short tails;
    absent support is not a zero-error term. Returned counts expose coverage.
    """
    q = clock.cycles
    require(reference.shape == valid.shape == q.shape and valid.dtype == torch.bool
            and reference.is_floating_point(), 'reference geometry mismatch')
    require(reference.device == valid.device == q.device, 'reference devices differ')
    require(clock.fps == FPS, 'loss lags require the 50 Hz clock contract')
    require(valid.any().item() and torch.isfinite(reference[valid]).all().item(),
            'missing or nonfinite reference')
    error = q[valid] - reference[valid].double()
    phase = (1 - torch.cos(2 * math.pi * torch.remainder(error, 1.))).mean()
    terms, counts = [], {}
    for lag in LAGS:
        if lag >= q.shape[1]:
            counts[lag] = 0
            continue
        pairs = supported_pairs(valid, lag)
        count = int(pairs.sum().item())
        counts[lag] = count
        if not count:
            continue
        expected = (reference[:, lag:] - reference[:, :-lag])[pairs].double()
        predicted = (q[:, lag:] - q[:, :-lag])[pairs]
        require(torch.isfinite(expected).all().item() and (expected > 0).all().item(),
                'reference advancement must be positive')
        # Subtract logs rather than divide first, avoiding ratio overflow.
        terms.append((torch.log2(predicted) - torch.log2(expected)).square().mean())
    require(bool(terms), 'no fully supported reference interval')
    advance = torch.stack(terms).mean()
    total = phase + advance
    require(torch.isfinite(total).item(), 'nonfinite supervised loss')
    return dict(total=total, phase=phase, advance=advance,
                reference_frames=int(valid.sum().item()), pairs_by_lag=counts)
