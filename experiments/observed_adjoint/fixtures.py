"""Authored numerical/cost inputs only, never music or a production readout."""
import math

import numpy as np
import torch

from experiments.phase_sync.model import PhaseSyncReadout
from experiments.coupled_clock.clock import FPS


class OffsetWitness(PhaseSyncReadout):
    """Full native CNN plus fixed authored field offsets, not a learned repair.

    Offsets are constructed once from initial fields and never recomputed after
    an optimizer step. The derivative still traverses every CNN parameter.
    This adversarial fixture is not an accuracy or natural-music distribution.
    """
    def __init__(self, frames):
        super().__init__()
        self.register_buffer('fixed_offsets', torch.zeros(1, frames, 4))

    def fields(self, features, chunk_frames=None):
        return super().fields(features, chunk_frames) + self.fixed_offsets


def fixture(kind='natural', cells=3000, seed=777, device='cpu'):
    if kind not in ('natural', 'opposed') or type(cells) is not int or cells < 2:
        raise ValueError('authored natural/opposed fixture and positive extent required')
    torch.manual_seed(seed)
    features = (torch.randn(1, cells + 1, 512) * .1).to(device)
    torch.manual_seed(seed)
    model = (OffsetWitness(cells + 1) if kind == 'opposed' else PhaseSyncReadout()).to(device)
    if kind == 'opposed':
        with torch.no_grad():
            raw = model.fields(features).detach().cpu().numpy().copy()
        offsets = np.zeros_like(raw)
        offsets[:, :, 0] = np.float32(-1) - raw[:, :, 0]
        offsets[:, 0, 3] = -raw[:, 0, 3]
        periods = (raw + offsets)[0, :-1, 0].astype(np.float64)
        advance = np.exp2(-periods) / FPS
        gain = float((math.log(2.) * torch.sigmoid(model.gain_logit.double())).detach())
        q = 0.
        for t in range(cells):
            angle = 2 * math.pi * ((q + float(advance[t])) % 1.)
            desired = np.array([-math.cos(angle), -math.sin(angle)], dtype=np.float32)
            offsets[0, t + 1, 1:3] = desired - raw[0, t + 1, 1:3]
            # Carry the ACTUAL rounded field, not an ideal opposite vector. This
            # avoids relying on cross-device CNN identity or cancelling float32
            # subtraction/addition as if they were exact real arithmetic.
            u, v = map(float, raw[0, t + 1, 1:3] + offsets[0, t + 1, 1:3])
            scale = math.hypot(1., math.hypot(u, v))
            error = v / scale * math.cos(angle) - u / scale * math.sin(angle)
            q += float(advance[t]) * math.exp(gain * error)
        model.fixed_offsets.copy_(torch.from_numpy(offsets).to(device))
    reference = torch.arange(cells + 1, device=device, dtype=torch.float64)[None] * .04
    rows = [dict(id=f'authored-{kind}-{index}', scale=weight,
                 payload=dict(features=features, reference=reference + phase,
                              valid=torch.ones_like(reference, dtype=torch.bool)))
            for index, (weight, phase) in enumerate(((.3, .125), (.7, .25)))]
    return model, rows
