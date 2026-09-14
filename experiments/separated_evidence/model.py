"""Matched initial functions with shared versus independently owned CNN trunks."""
import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from experiments.clock_readout.model import Block, WIDTH, CHANNELS, DILATIONS, HALO
from experiments.coupled_clock.clock import require

SHARED_PARAMETERS = 24003
SEPARATED_PARAMETERS = 47907


def validate_features(features, module):
    parameter = next(module.parameters())
    require(isinstance(features, torch.Tensor) and features.is_floating_point()
            and features.ndim == 3 and features.shape[0] > 0 and features.shape[1] >= 2
            and features.shape[2] == WIDTH, 'expected floating [B,T>=2,512] features')
    require(features.device == parameter.device and features.dtype == parameter.dtype,
            'feature/model device or dtype differs')
    require(torch.isfinite(features).all().item(), 'nonfinite features')


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv1d(WIDTH, CHANNELS, 1)
        self.blocks = nn.ModuleList(Block(d) for d in DILATIONS)

    def forward(self, features, chunk_frames=None):
        validate_features(features, self)
        length = features.shape[1]
        size = length if chunk_frames is None else chunk_frames
        require(type(size) is int and size > 0, 'positive integer chunk size required')
        padded = F.pad(features.transpose(1, 2), (HALO, HALO))
        pieces = []
        for start in range(0, length, size):
            owned = min(size, length - start)
            x = torch.relu(self.projection(padded[:, :, start:start + owned + 2 * HALO]))
            for block in self.blocks:
                x = block(x)
            pieces.append(x[:, :, HALO:HALO + owned])
        return torch.cat(pieces, dim=2)


@dataclass(frozen=True)
class Evidence:
    # T-1 outgoing cell periods; T frame-point phase vectors. NOT a Clock.
    log2_cell_period: torch.Tensor
    phase_vector: torch.Tensor


class SharedEvidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = Trunk()
        self.tempo_head = nn.Conv1d(CHANNELS, 1, 1)
        self.phase_head = nn.Conv1d(CHANNELS, 2, 1)
        assert sum(p.numel() for p in self.parameters()) == SHARED_PARAMETERS

    def forward(self, features, chunk_frames=None):
        x = self.trunk(features, chunk_frames)
        return Evidence(self.tempo_head(x)[:, 0, :-1], self.phase_head(x).transpose(1, 2))


class TaskBranch(nn.Module):
    def __init__(self, task, trunk, head):
        super().__init__()
        require(task in ('tempo', 'phase'), 'unknown evidence task')
        self.task, self.trunk, self.head = task, trunk, head

    def forward(self, features, chunk_frames=None):
        x = self.head(self.trunk(features, chunk_frames)).transpose(1, 2)
        return x[:, :-1, 0] if self.task == 'tempo' else x


class SeparatedEvidence(nn.Module):
    def __init__(self, initial_shared):
        super().__init__()
        require(type(initial_shared) is SharedEvidence, 'fresh shared architecture required')
        # Deep copies consume no RNG and share no trainable storage with the
        # other task OR the shared control. Frozen audio features alone are common.
        self.tempo = TaskBranch('tempo', copy.deepcopy(initial_shared.trunk),
                                copy.deepcopy(initial_shared.tempo_head))
        self.phase = TaskBranch('phase', copy.deepcopy(initial_shared.trunk),
                                copy.deepcopy(initial_shared.phase_head))
        assert sum(p.numel() for p in self.parameters()) == SEPARATED_PARAMETERS

    def forward(self, features, chunk_frames=None):
        return Evidence(self.tempo(features, chunk_frames), self.phase(features, chunk_frames))
