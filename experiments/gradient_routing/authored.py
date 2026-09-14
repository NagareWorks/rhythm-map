"""Fixed small, synthetic mechanism checks; no music or saved weights."""
import copy

import numpy as np
import torch

from experiments.coupled_clock.supervision import clock_loss
from experiments.observed_adjoint.transport import ObservedTransport
from experiments.phase_sync.model import PhaseSyncReadout
from experiments.scaled_adjoint.scaled import Gradient
from experiments.gradient_routing.route import displacement_check, route


def record_terms(model, payload):
    gradients = {}
    for term in ('phase', 'advance'):
        def selected(clock, reference, valid, key=term):
            return {'total': clock_loss(clock, reference, valid)[key]}
        gradients[term], _ = ObservedTransport().run(model, payload, selected)
    return gradients['phase'], gradients['advance']


def teacher_case(device='cpu'):
    # Fork RNG so a regression helper does not mutate the calling experiment.
    devices = [] if device == 'cpu' else [torch.device(device).index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(1729)
        model = PhaseSyncReadout().to(device)
        features = torch.randn(1, 61, 512, device=device) * .1
        teacher = copy.deepcopy(model)
        with torch.no_grad():
            teacher.output.bias[0] -= .08
            teacher.origin.bias[0] += .035
            reference = teacher(features).cycles.detach()
    return model, dict(features=features, reference=reference,
                       valid=torch.ones_like(reference, dtype=torch.bool))


def learnability(device='cpu'):
    model, payload = teacher_case(device)
    def losses():
        with torch.no_grad():
            values = clock_loss(model(payload['features']), payload['reference'], payload['valid'])
            return {k: float(values[k]) for k in ('phase', 'advance', 'total')}
    before = losses()
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    for _ in range(40):
        phase, count = record_terms(model, payload)
        gradient, _ = route(phase, count)
        clipped = gradient.clipped(1., 1e-6)
        for name, parameter in model.named_parameters():
            parameter.grad = torch.from_numpy(np.asarray(clipped[name]).copy()).to(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    after = losses()
    return dict(seed=1729, frames=61, optimizer='SGD', learning_rate=.03, updates=40,
                before=before, after=after,
                both_losses_reduced=all(after[k] < before[k] for k in ('phase', 'advance')))


def adamw_witness(momentum=False):
    """Real optimizer steps, not a guessed sign-normalization approximation."""
    parameter = torch.nn.Parameter(torch.zeros(1 if momentum else 2, dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=.01, weight_decay=0.)
    if momentum:
        for _ in range(10):
            parameter.grad = -torch.ones_like(parameter)
            optimizer.step()
    c = np.array([1.]) if momentum else np.array([2., 1.])
    p = np.array([0.]) if momentum else np.array([-3., 6.])
    from experiments.gradient_routing.route import project
    count = Gradient.create({'x': c})
    gradient, _, receipt = project(Gradient.create({'x': p}), count)
    before = {'x': parameter.detach().numpy().copy()}
    parameter.grad = torch.from_numpy(np.ldexp(gradient.blocks['x'].copy(), gradient.exponent))
    optimizer.step()
    after = {'x': parameter.detach().numpy().copy()}
    check = displacement_check(count, before, after)
    # Local quadratic with the given count gradient at the inspected point.
    delta = after['x'] - before['x']
    return dict(momentum_warmup_updates=10 if momentum else 0, raw_gradient=receipt,
                actual_displacement=delta.tolist(), check=check,
                quadratic_count_loss_change=float(c @ delta + .5 * delta @ delta))


if __name__ == '__main__':
    import argparse
    import platform
    from pathlib import Path
    from experiments.gradient_audit.register import write_json
    from experiments.gradient_routing.packet_audit import ROOT, sources
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(ROOT):
        parser.error('a fresh private output path is required')
    torch.set_num_threads(2)
    identity = sources()
    result = dict(schema='rhythm-map.gradient-routing-authored.v1', source_sha256=identity,
                  runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
                  learning=learnability(), preconditioning=adamw_witness(), momentum=adamw_witness(True),
                  musical_updates=0, admission=False)
    assert identity == sources()
    write_json(args.output, result)
    print(result['learning'], flush=True)
