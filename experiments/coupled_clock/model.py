"""Frozen-feature CNN fields followed by one globally integrated clock."""
import torch
from torch import nn
from torch.nn import functional as F

# Reuse the unchanged, source-pinned v1 convolution primitive, not its weights,
# three independent outputs, runner, candidate paths, or learned predictions.
from experiments.clock_readout.model import Block, CHANNELS, DILATIONS, HALO, WIDTH
from experiments.coupled_clock.clock import integrate, require

PARAMETERS = 23970


class CoupledClockReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv1d(WIDTH, CHANNELS, 1)
        self.blocks = nn.ModuleList(Block(d) for d in DILATIONS)
        self.output = nn.Conv1d(CHANNELS, 2, 1)
        assert sum(p.numel() for p in self.parameters()) == PARAMETERS

    def _local(self, x):
        x = torch.relu(self.projection(x))
        for block in self.blocks:
            x = block(x)
        return self.output(x).transpose(1, 2)

    def fields(self, features, chunk_frames=None):
        """[B,T,512] -> [B,T,2], preserving physical-edge padding and ownership.

        Column 0 is log2(cell period/s). Only column 1 at physical frame zero
        is used as the sequence's phase anchor (cycles). No label or timestamp
        input. Chunking partitions convolution work, NOT the musical clock.
        """
        require(isinstance(features, torch.Tensor) and features.is_floating_point(),
                'features must be a floating tensor')
        require(features.ndim == 3 and features.shape[0] > 0 and features.shape[1] >= 2
                and features.shape[2] == WIDTH, 'expected [B,T>=2,512] features')
        require(torch.isfinite(features).all().item(), 'nonfinite features')
        length = features.shape[1]
        size = length if chunk_frames is None else chunk_frames
        require(type(size) is int and size > 0, 'invalid convolution chunk size')
        padded = F.pad(features.transpose(1, 2), (HALO, HALO))
        pieces = []
        for start in range(0, length, size):
            owned = min(size, length - start)
            local = self._local(padded[:, :, start:start + owned + 2 * HALO])
            pieces.append(local[:, HALO:HALO + owned])
        return torch.cat(pieces, dim=1)

    def forward(self, features, chunk_frames=None):
        fields = self.fields(features, chunk_frames)
        # The final frame has no complete outgoing cell. Do not pad a fabricated
        # final tempo or reset phase at convolution ownership boundaries.
        return integrate(fields[:, :-1, 0], fields[:, 0, 1])
