"""Direct task supervision on the existing native reference; no learned clock scan."""
import math

import torch

from experiments.coupled_clock.clock import FPS, require


def validate_reference(reference, valid):
    require(isinstance(reference, torch.Tensor) and reference.is_floating_point()
            and reference.ndim == 2 and reference.shape[0] > 0 and reference.shape[1] >= 2,
            'expected floating reference [B,T>=2]')
    require(isinstance(valid, torch.Tensor) and valid.dtype == torch.bool
            and valid.shape == reference.shape and valid.device == reference.device,
            'reference support geometry or device differs')
    require(valid.any().item() and torch.isfinite(reference[valid]).all().item(),
            'missing or nonfinite supported reference')
    cells = valid[:, :-1] & valid[:, 1:]
    advancement = (reference[:, 1:].double() - reference[:, :-1].double())[cells]
    require(torch.isfinite(advancement).all().item() and (advancement > 0).all().item(),
            'supported native reference must advance')
    return cells


def task_loss(task, prediction, reference, valid):
    cells = validate_reference(reference, valid)
    require(isinstance(prediction, torch.Tensor) and prediction.is_floating_point()
            and prediction.device == reference.device, 'invalid task prediction')
    expected_shape = (reference.shape[0], reference.shape[1] - 1) if task == 'tempo' else (*reference.shape, 2)
    require(task in ('tempo', 'phase') and prediction.shape == expected_shape, 'task prediction geometry differs')
    # Missing reference is not negative supervision. Bad model outputs never
    # disappear by changing support, even outside the annotated part.
    require(torch.isfinite(prediction).all().item(), 'nonfinite task prediction')
    if task == 'tempo':
        require(cells.any().item(), 'no supported native tempo cell')
        target_rate = (reference[:, 1:].double() - reference[:, :-1].double())[cells] * FPS
        require(torch.isfinite(target_rate).all().item() and (target_rate > 0).all().item(),
                'unrepresentable native rate')
        return (prediction[cells].double() + torch.log2(target_rate)).square().mean()
    angle = 2 * math.pi * torch.remainder(reference[valid].double(), 1.)
    target = torch.stack((angle.cos(), angle.sin()), dim=-1)
    return (prediction[valid].double() - target).square().mean()


def losses(evidence, reference, valid):
    return dict(tempo=task_loss('tempo', evidence.log2_cell_period, reference, valid),
                phase=task_loss('phase', evidence.phase_vector, reference, valid))
