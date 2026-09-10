"""Whole-span native count error, modulo ONE integer origin per annotation span."""
import torch

from experiments.coupled_clock.clock import FPS, require


def trajectory_loss(clock, reference, valid):
    """min over integer k: mean((q - reference - k)^2), separately per span.

    The optimal integer k is round(mean(error)), not round(error) at each
    frame. Thus integer beat numbering is a gauge, but accumulated cycle slips
    remain errors. Round-to-even resolves exact ties; the discrete registration
    is detached. Away from ties this is the exact derivative of the minimized
    objective, not a straight-through estimator. At ties one minimizing branch
    is used. Nonconvex model training and global-origin ambiguity remain.

    Unknown annotation gaps separate label gauges, not the predicted clock.
    No label enters inference, and no physical/prediction boundary is reset.
    One recording's valid frames are equally weighted; works balance externally.
    """
    q = clock.cycles
    require(q.ndim == 2 and reference.shape == valid.shape == q.shape and valid.dtype == torch.bool
            and reference.is_floating_point(), 'reference geometry mismatch')
    require(reference.device == valid.device == q.device and clock.fps == FPS, 'clock/reference contract mismatch')
    require(valid.any().item() and torch.isfinite(reference[valid]).all().item()
            and torch.isfinite(q[valid]).all().item(), 'missing or nonfinite required values')
    total, centered, origin = [], [], []
    count, spans = 0, 0
    for predicted, target, mask in zip(q, reference, valid):
        padding = torch.zeros(1, device=mask.device, dtype=torch.int8)
        boundaries = torch.diff(torch.cat((padding, mask.to(torch.int8), padding)))
        starts = torch.nonzero(boundaries == 1).flatten().tolist()
        ends = torch.nonzero(boundaries == -1).flatten().tolist()
        for start, end in zip(starts, ends):
            expected = target[start:end].double()
            require((torch.diff(expected) > 0).all().item(), 'reference advancement must be positive')
            error = predicted[start:end].double() - expected
            mean = error.mean()
            integer = torch.round(mean.detach())
            total.append((error - integer).square().sum())
            centered.append((error - mean).square().sum())
            origin.append((mean - integer).square() * len(error))
            count += len(error)
            spans += 1
    loss = torch.stack(total).sum() / count
    require(torch.isfinite(loss).item(), 'nonfinite trajectory loss')
    return dict(total=loss, centered=torch.stack(centered).sum() / count,
                origin=torch.stack(origin).sum() / count, reference_frames=count, reference_spans=spans)
