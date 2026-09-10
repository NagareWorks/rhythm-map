"""Direct phase/period readout. No candidate clocks enter this network."""
import torch
from torch import nn

WIDTH = 512
CHANNELS = 32
DILATIONS = (1, 2, 4, 8, 16, 32)
HALO = 2 * sum(DILATIONS)
PARAMETERS = 24003


class Block(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.depthwise = nn.Conv1d(CHANNELS, CHANNELS, 5, padding=2 * dilation,
                                   dilation=dilation, groups=CHANNELS)
        self.pointwise = nn.Conv1d(CHANNELS, CHANNELS, 1)

    def forward(self, x):
        return x + self.pointwise(torch.relu(self.depthwise(x)))


class ClockReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv1d(WIDTH, CHANNELS, 1)
        self.blocks = nn.ModuleList(Block(d) for d in DILATIONS)
        self.output = nn.Conv1d(CHANNELS, 3, 1)
        assert sum(p.numel() for p in self.parameters()) == PARAMETERS

    def forward(self, features):
        """[batch,time,512] -> [batch,time,3]: log2(period/s), cos, sin.

        The phase vector norm is not calibrated confidence. These three fields
        need not constitute a coherent clock; that is measured, not imposed by
        silently replacing detector events. No softmax/candidate ranking.
        """
        if features.ndim != 3 or features.shape[-1] != WIDTH:
            raise ValueError('expected [batch,time,512] features')
        x = torch.relu(self.projection(features.transpose(1, 2)))
        for block in self.blocks:
            x = block(x)
        return self.output(x).transpose(1, 2)
