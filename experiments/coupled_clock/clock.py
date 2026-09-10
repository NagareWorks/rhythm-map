"""One positive musical clock, with explicit frame-point/cell semantics."""
from dataclasses import dataclass
import math

import torch

FPS = 50


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Clock:
    # T frame points, T-1 cells. The last point has no outgoing tempo cell.
    cycles: torch.Tensor
    log2_period: torch.Tensor
    fps: float

    @property
    def phase_vector(self):
        angle = 2 * math.pi * torch.remainder(self.cycles, 1.)
        return torch.stack((angle.cos(), angle.sin()), dim=-1)

    @property
    def cell_bpm(self):
        return 60 * torch.exp2(-self.log2_period)


def integrate(log2_period, phase0, fps=FPS):
    """[B,T-1] cell periods + one [B] phase anchor -> a [B,T] clock.

    Each cell has positive beat advancement exp2(-log2_period)/fps.
    Frame-point phase and cell-average tempo cannot disagree by construction.
    This is not instantaneous tempo inside a cell, rhythm confidence, or
    observed beat detection. Invalid numerics fail; no BPM clamp repairs them.
    Accumulation uses float64 even when the neural fields use float32.
    """
    require(isinstance(log2_period, torch.Tensor) and isinstance(phase0, torch.Tensor),
            'clock inputs must be tensors')
    require(log2_period.is_floating_point() and phase0.is_floating_point(),
            'clock inputs must be floating point')
    require(log2_period.ndim == 2 and min(log2_period.shape) > 0
            and phase0.shape == (log2_period.shape[0],), 'invalid clock geometry')
    require(log2_period.device == phase0.device, 'clock devices differ')
    require(isinstance(fps, (int, float)) and not isinstance(fps, bool)
            and math.isfinite(fps) and fps > 0, 'invalid frame rate')
    periods, anchor = log2_period.double(), phase0.double()
    require(torch.isfinite(periods).all().item() and torch.isfinite(anchor).all().item(),
            'nonfinite clock input')
    rate = torch.exp2(-periods)
    advance = rate / fps
    require(torch.isfinite(60 * rate).all().item() and torch.isfinite(advance).all().item()
            and (advance > 0).all().item(), 'unrepresentable positive advancement')
    cycles = torch.cat((anchor[:, None], anchor[:, None] + advance.cumsum(dim=1)), dim=1)
    require(torch.isfinite(cycles).all().item() and (cycles.diff(dim=1) > 0).all().item(),
            'unrepresentable accumulated clock')
    require(torch.allclose(cycles.diff(dim=1), advance, rtol=1e-9, atol=1e-12),
            'accumulation lost cell precision')
    return Clock(cycles=cycles, log2_period=periods, fps=float(fps))
