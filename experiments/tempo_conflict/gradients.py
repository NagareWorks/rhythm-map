"""Autograd inspection at loaded weights; never calls backward or an optimizer."""
import numpy as np
import torch

from experiments.coupled_clock.run import work_weights, state_hash
from experiments.separated_evidence.supervision import task_loss
from experiments.relative_tempo.model import paired_loss
from experiments.relative_tempo.run import natural_metrics
from experiments.local_tempo.fit import retention_loss
from .data import require

PAIR_BATCH = 16


def geometry(model):
    return [dict(name=k, shape=list(p.shape), elements=p.numel()) for k, p in model.named_parameters()]


def flatten(values):
    result = np.concatenate([v.detach().cpu().numpy().reshape(-1).astype(np.float64) for v in values])
    require(np.isfinite(result).all(), 'nonfinite parameter/gradient')
    return result


def derivative(model, loss, retain=False):
    require(loss.ndim == 0 and torch.isfinite(loss).item(), 'invalid diagnostic loss')
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=retain, allow_unused=False)
    return flatten(gradients)


def native(model, population, teacher_targets, weights, denominator, check):
    require(population and all(r['role'] == 'fit' for r in population), 'only fit rows may form training-direction diagnostics')
    require(denominator == len(population), 'natural mean denominator differs')
    n = sum(p.numel() for p in model.parameters())
    gradients = dict(native=np.zeros(n), retention=np.zeros(n))
    losses = dict(native=0., retention=0.)
    for row in population:
        check()
        prediction = model(row['features'][None])
        scale = weights[row['work']]/denominator
        a = task_loss('tempo', prediction, row['target'][None], row['mask'][None])*scale
        b = retention_loss(prediction, teacher_targets[row['id']], row['mask'][None])*scale
        gradients['native'] += derivative(model, a, retain=True)
        gradients['retention'] += derivative(model, b)
        losses['native'] += float(a.detach())
        losses['retention'] += float(b.detach())
    return gradients, losses


def pair_kind(profile):
    if profile == 'identity_seams':
        return 'pair_identity'
    if profile in ('slow', 'fast'):
        return 'pair_global'
    require(profile in ('local_slow_fast', 'local_fast_slow'), 'unexpected training profile')
    return 'pair_local'


def paired(model, captures, groups, check):
    require(groups and all(g and all(r['pair_role'] == 'fit_exposed' and r['admitted'] for r in g) for g in groups),
            'invalid sparse training support')
    n = sum(p.numel() for p in model.parameters())
    gradients = {k: np.zeros(n) for k in ('pair_identity', 'pair_global', 'pair_local')}
    losses, details = {k: 0. for k in gradients}, []
    for group in groups:
        identity, profile = group[0]['source_id'], group[0]['profile']
        require(all((r['source_id'], r['profile']) == (identity, profile) for r in group), 'mixed source/profile group')
        kind, total = pair_kind(profile), 0.
        for start in range(0, len(group), PAIR_BATCH):
            check()
            rows = group[start:start+PAIR_BATCH]
            # Each group, not each point, has weight 1 / number of groups.
            loss = paired_loss(model, captures[identity, 'source'], captures[identity, profile], rows)
            loss = loss*(len(rows)/len(group)/len(groups))
            gradients[kind] += derivative(model, loss)
            total += float(loss.detach())
        losses[kind] += total
        details.append(dict(source_id=identity, profile=profile, points=len(group), weighted_loss=total))
    return gradients, losses, details


def inspect(model, records, captures, groups, teacher_targets, check):
    model.eval()
    before = state_hash(model)
    population = [r for r in records if r['role'] == 'fit']
    weights = work_weights(population)
    gradients, losses = native(model, population, teacher_targets, weights, len(population), check)
    pair_gradients, pair_losses, details = paired(model, captures, groups, check)
    gradients.update(pair_gradients)
    losses.update(pair_losses)
    development = []
    for row in records:
        if row['role'] != 'development':
            continue
        check()
        prediction = model(row['features'][None])
        loss = task_loss('tempo', prediction, row['target'][None], row['mask'][None])
        gradients['dev_'+str(len(development))] = derivative(model, loss)
        metrics = natural_metrics(prediction[0].detach().cpu().numpy(), row)
        development.append(dict(id=row['id'], work=row['work'], **metrics))
    gradients['theta'] = flatten(model.parameters())
    require(state_hash(model) == before and all(p.grad is None for p in model.parameters()), 'diagnostic mutated state or grad buffers')
    return gradients, dict(development=development, losses=losses, pair_groups=details, state_sha256=before)
