"""Small frozen-feature CNN feeding one phase-synchronized clock."""
import math

import torch
from torch import nn

from experiments.coupled_clock.model import CoupledClockReadout
from experiments.phase_sync.scan import synchronize

PARAMETERS = 24037


class PhaseSyncReadout(CoupledClockReadout):
    def __init__(self):
        super().__init__()
        # Reuse field chunk ownership and the unchanged convolution primitive;
        # no checkpoint is loaded and no old experiment source is modified.
        self.output = nn.Conv1d(self.output.in_channels, 3, 1)
        self.origin = nn.Conv1d(self.output.in_channels, 1, 1)
        nn.init.zeros_(self.origin.weight)
        nn.init.zeros_(self.origin.bias)
        self.gain_logit = nn.Parameter(torch.tensor(0.))
        assert sum(p.numel() for p in self.parameters()) == PARAMETERS

    def _local(self, x):
        x = torch.relu(self.projection(x))
        for block in self.blocks:
            x = block(x)
        return torch.cat((self.output(x), self.origin(x)), dim=1).transpose(1, 2)

    def fields(self, features, chunk_frames=None):
        """[B,T,512] -> period, phase-u, phase-v, origin fields.

        Only physical frame zero's origin is consumed. Each subsequent frame's
        phase vector corrects its incoming cell; its magnitude is NOT calibrated
        confidence. Chunking partitions CNN work, not clock state.
        """
        return super().fields(features, chunk_frames)

    def forward(self, features, chunk_frames=None):
        fields = self.fields(features, chunk_frames)
        return synchronize(fields[:, :-1, 0], fields[:, 1:, 1:3], fields[:, 0, 3],
                           math.log(2.) * torch.sigmoid(self.gain_logit.double()))
