"""Read-only gradient comparisons using the frozen observed transport."""
import math

import numpy as np
import torch

from experiments.coupled_clock.clock import FPS, require
from experiments.coupled_clock.supervision import clock_loss, supported_pairs
from experiments.observed_adjoint.transport import ObservedTransport, pack_gradient
from experiments.scaled_adjoint.scaled import Gradient, accumulate

TERMS = ('phase', 'count', 'total', 'prior')
GROUPS = ('period_head', 'shared_cnn', 'all_parameters')


def cosine(a, b):
    require(a.blocks.keys() == b.blocks.keys() and all(a.blocks[k].shape == b.blocks[k].shape for k in a.blocks),
            'cosine geometry differs')
    an = math.sqrt(math.fsum(float(np.sum(v * v)) for v in a.blocks.values()))
    bn = math.sqrt(math.fsum(float(np.sum(v * v)) for v in b.blocks.values()))
    if not an or not bn:
        return None
    return float(np.clip(math.fsum(float(np.sum((v / an) * (b.blocks[k] / bn)))
                                  for k, v in a.blocks.items()), -1., 1.))


def comparisons(gradients):
    out = {name + '_norm_log2': g.norm_log2() for name, g in gradients.items()}
    for a, b in (('phase', 'count'), ('count', 'prior'), ('total', 'prior'), ('phase', 'prior')):
        out[a + '_' + b + '_cosine'] = cosine(gradients[a], gradients[b])
    p, c = out['phase_norm_log2'], out['count_norm_log2']
    out['phase_count_norm_log2_ratio'] = None if p is None or c is None else p - c
    return out


def group(gradient, name):
    if name == 'period_head':
        blocks = {'output.weight': gradient.blocks['output.weight'][0],
                  'output.bias': gradient.blocks['output.bias'][:1]}
    elif name == 'shared_cnn':
        blocks = {k: v for k, v in gradient.blocks.items() if k.startswith(('projection.', 'blocks.'))}
    else:
        require(name == 'all_parameters', 'unknown parameter group')
        blocks = dict(gradient.blocks)
    return Gradient.create(blocks, gradient.exponent, gradient.alignment_underflows)


def linearity(gradients):
    p, c, t = (gradients[k] for k in ('phase', 'count', 'total'))
    residual = accumulate([t, p.multiply(-1), c.multiply(-1)])
    require(residual.alignment_underflows == 0, 'linearity comparison lost alignment')
    pn, cn, tn, rn = (g.norm_log2() for g in (p, c, t, residual))
    present = [v for v in (pn, cn) if v is not None]
    if not present:
        require(tn is None, 'nonzero total from two zero components')
        return dict(residual_relative_to_component_norm_sum=0., total_to_component_norm_sum=None)
    peak = max(present)
    denom = peak + math.log2(sum(2. ** (v - peak) for v in present))
    return dict(residual_relative_to_component_norm_sum=0. if rn is None else 2. ** (rn - denom),
                total_to_component_norm_sum=0. if tn is None else 2. ** (tn - denom))


def period_signs(gradient, direct, supported):
    g, d = gradient.blocks['periods'][supported], direct.blocks['periods'][supported]
    informative = d != 0
    g, d = g[informative], d[informative]
    if not len(d):
        return dict(informative_cells=0, agreement_fraction=None, disagreement_fraction=None,
                    zero_gradient_fraction=None, weighted_disagreement_fraction=None)
    opposite = np.sign(g) == -np.sign(d)
    return dict(informative_cells=len(d), agreement_fraction=float(np.mean(np.sign(g) == np.sign(d))),
                disagreement_fraction=float(opposite.mean()), zero_gradient_fraction=float(np.mean(g == 0)),
                weighted_disagreement_fraction=float(np.sum(abs(d[opposite])) / np.sum(abs(d))))


def direct_prior(model, payload):
    fields = model.fields(payload['features'])
    ell = fields[:, :-1, 0].double()
    valid, reference = payload['valid'], payload['reference']
    cells = supported_pairs(valid, 1)
    require(cells.any().item(), 'direct prior has no reference cells')
    rates = (reference[:, 1:] - reference[:, :-1])[cells] * FPS
    require(torch.isfinite(rates).all().item() and (rates > 0).all().item(), 'invalid native rate')
    error = ell[cells] + torch.log2(rates)
    loss = error.square().mean()
    require(torch.isfinite(loss).item(), 'nonfinite direct prior diagnostic')
    parameters = dict(model.named_parameters())
    values = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
    blocks = {}
    for (name, parameter), value in zip(parameters.items(), values):
        if value is None:
            require(name == 'gain_logit' or name.startswith('origin.'), 'unexpected unused prior parameter')
            blocks[name] = np.zeros(tuple(parameter.shape))
        else:
            require(torch.isfinite(value).all().item(), 'nonfinite native direct-prior gradient')
            blocks[name] = value.detach().cpu().double().numpy()
    periods = np.zeros(tuple(ell.shape))
    periods[cells.cpu().numpy()] = (2 * error / cells.sum()).detach().cpu().numpy()
    template = dict(periods=periods, observations=np.zeros((*periods.shape, 2)),
                    initial=np.zeros(periods.shape[0]), gain=np.asarray(0.))
    return Gradient.create(blocks), Gradient.create(template), float(loss.detach()), fields.detach().cpu().numpy()


def audit_record(model, payload, saved_fields=None, saved_cycles=None, observer=None):
    """Four fixed-field CNN passes, three unchanged clock transports, no step."""
    transport = ObservedTransport()
    field, parameter, loss, passes = {}, {}, {}, {}
    for name, key in (('phase', 'phase'), ('count', 'advance'), ('total', 'total')):
        transport.reset()  # A term-entry deadline must not retain the previous term's trace.
        if observer:
            observer(name, transport)
        def selected(clock, reference, valid, key=key):
            result = clock_loss(clock, reference, valid)
            if saved_cycles is not None:
                require(np.array_equal(clock.cycles.detach().cpu().numpy()[0], saved_cycles),
                        'clock does not reproduce selected prediction')
            return dict(total=result[key])
        parameter[name], loss[name] = transport.run(model, payload, selected)
        field[name] = transport.field_gradient
        passes[name] = len(transport.bands)
        if saved_fields is not None:
            inputs = transport.last_trace['inputs']
            for value, expected in ((inputs['periods'][0], saved_fields[:-1, 0]),
                                    (inputs['observations'][0], saved_fields[1:, 1:3]),
                                    (inputs['initial'][0], saved_fields[0, 3])):
                require(np.array_equal(value.numpy(), expected), 'clock fields do not reproduce saved fields')
    if observer:
        observer('prior', None)
    parameter['prior'], field['prior'], loss['prior'], fields = direct_prior(model, payload)
    if saved_fields is not None:
        require(np.array_equal(fields[0], saved_fields), 'CNN does not reproduce complete saved fields')
    require(all(p.grad is None for p in model.parameters()), 'audit mutated parameter grad buffers')
    require(all(g.alignment_underflows == 0 for g in (*field.values(), *parameter.values())),
            'audit lost gradient alignment')
    field_linear, parameter_linear = linearity(field), linearity(parameter)
    require(field_linear['residual_relative_to_component_norm_sum'] <= 1e-10, 'field gradient additivity failed')
    require(parameter_linear['residual_relative_to_component_norm_sum'] <= 5e-5, 'CNN gradient additivity failed')
    supported = supported_pairs(payload['valid'], 1).cpu().numpy()
    periods = {k: Gradient.create(dict(periods=g.blocks['periods'][supported]), g.exponent) for k, g in field.items()}
    result = dict(loss=loss, reference_points=int(payload['valid'].sum()), reference_cells=int(supported.sum()),
                  pairs_by_lag={str(lag): int(supported_pairs(payload['valid'], lag).sum())
                                if lag < payload['valid'].shape[1] else 0 for lag in (1, 25, 100, 200)},
                  native_cnn_vjp_passes=passes, field_linearity=field_linear, parameter_linearity=parameter_linear,
                  period_fields=comparisons(periods), parameter_groups={n: comparisons({k: group(g, n)
                      for k, g in parameter.items()}) for n in GROUPS},
                  period_signs={k: period_signs(g, field['prior'], supported)
                                for k, g in field.items() if k != 'prior'})
    packet = {kind: {name: pack_gradient(g) for name, g in values.items()}
              for kind, values in (('field', field), ('parameter', parameter))}
    return result, packet, parameter
