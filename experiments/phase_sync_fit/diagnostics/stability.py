"""Authored adversarial clock witness, NOT a replay of the failed music state."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.coupled_clock.prepare import ROOT, sha, write_json
from experiments.phase_sync.scan import MAX_GAIN, synchronize


def opposed_observations(cells=3000):
    """Build fixed float32 observations nearly opposite each incoming phase.

    Construction carries its own numerical clock to keep opposition even after
    float32 vector rounding. Once built, observations are fixed inputs, not
    recomputed by inference or by autograd. This is deliberately adversarial,
    not music, a reference-driven product correction, or new training data.
    """
    result = np.empty((1, cells, 2), dtype=np.float32)
    q, advance, gain = 0., 2. / 50, MAX_GAIN / 2
    for t in range(cells):
        predicted = q + advance
        angle = 2 * math.pi * (predicted % 1.)
        result[0, t] = -math.cos(angle), -math.sin(angle)
        u, v = map(float, result[0, t])
        scale = math.hypot(1., math.hypot(u, v))
        error = (v / scale) * math.cos(angle) - (u / scale) * math.sin(angle)
        q += advance * math.exp(gain * error)
    return result


def check(cells=3000):
    observations = opposed_observations(cells)
    evidence = {}
    for dtype, label in ((torch.float64, 'float64'), (torch.float32, 'float32')):
        period = torch.full((1, cells), -1., dtype=dtype, requires_grad=True)
        vector = torch.from_numpy(observations).to(dtype).requires_grad_()
        origin = torch.zeros(1, dtype=dtype, requires_grad=True)
        gain = torch.tensor(MAX_GAIN / 2, dtype=torch.float64, requires_grad=True)
        saved = []

        def pack(tensor):
            if tensor.shape == (5, 1, cells) and tensor.dtype == torch.float64:
                saved.append(tensor.detach().clone())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            clock = synchronize(period, vector, origin, gain)
        if len(saved) != 1:
            raise ValueError('unexpected saved scan Jacobian geometry')
        jacobian = saved[0][0, 0].numpy()
        # d(final q)/d(initial q) is the product of the one-step state Jacobians.
        expected_log10 = float(np.log10(np.abs(jacobian)).sum())
        error_message, actual_log10 = None, None
        try:
            clock.cycles[:, -1].sum().backward()
            actual_log10 = math.log10(abs(origin.grad.item()))
        except ValueError as error:
            error_message = str(error)
        evidence[label] = dict(
            forward_finite=bool(torch.isfinite(clock.cycles).all()),
            clock_strictly_increasing=bool((clock.cycles.diff(dim=1) > 0).all()),
            min_bpm=float(clock.cell_bpm.detach().min()), max_bpm=float(clock.cell_bpm.detach().max()),
            local_state_jacobian_min=float(jacobian.min()), local_state_jacobian_max=float(jacobian.max()),
            analytical_origin_gradient_log10=expected_log10,
            observed_origin_gradient_log10=actual_log10, backward_error=error_message)
    return dict(schema='rhythm-map.phase-sync-stability-witness.v1', cells=cells, fps=50,
        prior_bpm=120, gain=MAX_GAIN / 2, float32_max_log10=math.log10(np.finfo(np.float32).max),
        inputs_finite=bool(np.isfinite(observations).all()), observations_dtype='float32',
        observation_sha256=hashlib.sha256(observations.tobytes()).hexdigest(),
        torch_version=torch.__version__, numpy_version=np.__version__, device='cpu',
        evidence=evidence, musical_data=False, optimizer_steps=0, failed_music_state_reproduced=False,
        interpretation='positive finite clocks can have unrepresentable first-order float32 gradients; '
                       'this authored witness does not identify the actual failed recording or weights',
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in (
            Path(__file__).resolve(), ROOT / 'experiments/phase_sync/scan.py',
            ROOT / 'experiments/coupled_clock/clock.py')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(ROOT):
        raise ValueError('new private output required')
    torch.set_num_threads(1)
    result = check()
    write_json(args.output, result)
    print(json.dumps(result['evidence'], allow_nan=False))


if __name__ == '__main__':
    main()
