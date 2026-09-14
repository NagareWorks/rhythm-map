"""Two matched retained fits, differing only in the active native loss."""
import copy
import time

import numpy as np
import torch

from experiments.coupled_clock.run import journal, state_hash, work_weights
from experiments.local_tempo import selection
from experiments.local_tempo.fit import retention_loss, checkpoint_score, save_weights, read_weights
from experiments.relative_tempo.model import paired_loss
from . import data, recorder
from .data import require, write_json
from .loss import components, active_loss

ARMS = ('native-retained', 'variation-retained')
EPOCHS, STEPS, SEED = 20, 100, 142


def fit(initial, records, captures, points, arm, output, check):
    require(arm in ARMS, 'unknown experiment arm')
    model = copy.deepcopy(initial)
    teacher = copy.deepcopy(initial).eval().requires_grad_(False)
    teacher_hash = state_hash(teacher)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    population = [r for r in records if r['role'] == 'fit']
    weights = work_weights(population)
    require(len(population) == 20 and len(weights) == 9, 'natural fit split changed')
    groups = data.old.groups(points, data.old.old.PROFILES)
    require(len(groups) == 15 and all(r['pair_role'] == 'fit_exposed' for g in groups for r in g), 'invalid pair roles')
    with torch.no_grad():
        targets = {r['id']: teacher(r['features'][None]).detach().clone() for r in population}
    rng, history, losses, updates = np.random.default_rng(8142), [], [], 0
    started = time.monotonic()

    def budget():
        check()
        require(time.monotonic()-started < 1800, 'arm budget exceeded')

    def checkpoint(epoch):
        development, score = checkpoint_score(model, records, captures, groups, budget)
        baseline = history[0]['development'] if history else development
        item = dict(epoch=epoch, updates=updates, development=development, training_pair_mse=score,
                    admissibility=selection.admissibility(development, baseline),
                    **save_weights(model, output/f'{arm}.epoch-{epoch:02}.weights.npz'))
        history.append(item)
        journal(output, dict(event='epoch', arm=arm, **item))

    checkpoint(0)
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.default_rng(SEED+epoch).permutation(len(population))
        for offset in range(0, len(order), 4):
            budget()
            optimizer.zero_grad(set_to_none=True)
            batch = [population[i] for i in order[offset:offset+4]]
            prefix = f'{arm}.update-{updates+1:03}'
            before = recorder.snapshot(output/(prefix+'.before.pt'), model, optimizer, rng)
            chosen_groups = [[g[j] for j in rng.integers(len(g), size=4)] for g in groups]
            samples = [dict(source_id=g[0]['source_id'], profile=g[0]['profile'], indices=[r['index'] for r in g])
                       for g in chosen_groups]
            journal(output, dict(event='backward_entered', arm=arm, update=updates+1,
                                 natural_ids=[r['id'] for r in batch], pair_samples=samples, before_sha256=before))
            totals = dict(full=0., offset=0., variation=0., active=0., retention=0., pair=0.)
            for row in batch:
                prediction = model(row['features'][None])
                scale = weights[row['work']]/len(batch)
                terms = components(prediction, row['target'][None], row['mask'][None])
                a = active_loss(arm, terms)*scale
                b = retention_loss(prediction, targets[row['id']], row['mask'][None])*scale
                (a+b).backward()
                for name in ('full', 'offset', 'variation'):
                    totals[name] += float(terms[name].detach())*scale
                totals['active'] += float(a.detach())
                totals['retention'] += float(b.detach())
            for chosen in chosen_groups:
                identity, profile = chosen[0]['source_id'], chosen[0]['profile']
                loss = paired_loss(model, captures[identity, 'source'], captures[identity, profile], chosen)/len(groups)
                require(torch.isfinite(loss).item(), 'nonfinite pair loss')
                loss.backward()
                totals['pair'] += float(loss.detach())
            before_clip = recorder.gradient_vector(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            grad_sha = recorder.gradients(output/(prefix+'.gradients.npz'), before_clip, recorder.gradient_vector(model))
            journal(output, dict(event='update_entered', arm=arm, update=updates+1, gradient_sha256=grad_sha))
            optimizer.step()
            updates += 1
            after = recorder.snapshot(output/(prefix+'.after.pt'), model, optimizer, rng)
            item = dict(update=updates, epoch=epoch+1, losses=totals, unclipped_grad_norm=float(norm),
                        before_sha256=before, gradient_sha256=grad_sha, after_sha256=after)
            journal(output, dict(event='update_returned', arm=arm, **item))
            losses.append(item)
        checkpoint(epoch+1)
    budget()
    require(updates == STEPS and len(history) == EPOCHS+1, 'incomplete training')
    require(state_hash(teacher) == teacher_hash == state_hash(initial) and all(p.grad is None for p in teacher.parameters()),
            'teacher/initial changed')
    selected = selection.choose(history)
    chosen = read_weights(initial, output/f"{arm}.epoch-{selected['epoch']:02}.weights.npz", selected['weight_sha256'])
    report = dict(arm=arm, updates=updates, selected_epoch=selected['epoch'], selected_update=selected['updates'],
                  initial_state_sha256=teacher_hash, teacher_unchanged=True, teacher_fit_ids=list(targets),
                  selected=save_weights(chosen, output/(arm+'-selected.weights.npz')),
                  final=save_weights(model, output/(arm+'-final.weights.npz')), history=history, losses=losses,
                  elapsed_s=time.monotonic()-started)
    write_json(output/(arm+'.fit.json'), report)
    return chosen, model.eval(), report
