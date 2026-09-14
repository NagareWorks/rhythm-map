"""Matched local pair updates with an optional detached initial-output penalty."""
import copy
import time

import numpy as np
import torch

from experiments.coupled_clock.run import journal, state_hash, work_weights
from experiments.separated_evidence.supervision import task_loss
from experiments.relative_tempo.model import paired_loss
from experiments.relative_tempo.run import natural_metrics
from . import data, selection
from .data import require, sha, write_json

ARMS = ('local-only', 'local-retained')
EPOCHS, STEPS, SEED = 20, 100, 142


def save_weights(model, path):
    require(all(torch.isfinite(v).all().item() for v in model.state_dict().values()), 'nonfinite state')
    with path.open('xb') as stream:
        np.savez(stream, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    return dict(weight_sha256=sha(path), state_sha256=state_hash(model))


def read_weights(initial, path, expected_sha):
    require(sha(path) == expected_sha, 'tempo weight identity changed')
    model = copy.deepcopy(initial)
    with np.load(path, allow_pickle=False) as archive:
        model.load_state_dict({k: torch.from_numpy(archive[k].copy()) for k in archive.files}, strict=True)
    require(all(torch.isfinite(v).all().item() for v in model.state_dict().values()), 'nonfinite saved state')
    return model.eval()


def retention_loss(prediction, teacher, mask):
    require(prediction.shape == teacher.shape == (mask.shape[0], mask.shape[1]-1), 'retention geometry differs')
    require(mask.dtype == torch.bool and prediction.device == teacher.device == mask.device, 'retention device/support differs')
    cells = mask[:, :-1] & mask[:, 1:]
    require(cells.any().item() and torch.isfinite(prediction).all().item() and torch.isfinite(teacher).all().item(),
            'invalid retention population')
    return (prediction[cells].double()-teacher.detach()[cells].double()).square().mean()


def checkpoint_score(model, records, captures, groups, check):
    development, losses = [], []
    model.eval()
    with torch.inference_mode():
        for row in records:
            if row['role'] == 'development':
                check()
                metrics = natural_metrics(model(row['features'][None])[0].cpu().numpy(), row)
                development.append(dict(id=row['id'], work=row['work'], **metrics))
        # Score scalar fields, never synthesize time-warped feature tensors.
        fields = {key: model(value[None])[0].cpu().numpy() for key, value in captures.items()}
        for group in groups:
            check()
            identity, profile = group[0]['source_id'], group[0]['profile']
            difference = (data.old.query_numpy(fields[identity, profile], [r['output_s'] for r in group])
                          - data.old.query_numpy(fields[identity, 'source'], [r['source_s'] for r in group]))
            losses.append(float(np.mean((difference+np.log2([r['rate'] for r in group]))**2)))
    require(np.isfinite(losses).all(), 'invalid checkpoint pair score')
    return development, float(np.mean(losses))


def fit(initial, records, captures, points, arm, output, check):
    require(arm in ARMS, 'unknown experiment arm')
    model = copy.deepcopy(initial)
    teacher = copy.deepcopy(initial).eval().requires_grad_(False)
    teacher_hash = state_hash(teacher)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    population = [r for r in records if r['role'] == 'fit']
    weights = work_weights(population)
    require(len(population) == 20 and len(weights) == 9, 'natural fit split changed')
    groups = data.groups(points, data.old.PROFILES)
    require(len(groups) == 15 and all(r['pair_role'] == 'fit_exposed' for g in groups for r in g), 'invalid pair roles')
    # Only natural FIT records are teacher targets. Cache detached outputs once.
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
            native, retained, relative, sampled = 0., 0., 0., []
            for row in batch:
                prediction = model(row['features'][None])
                scale = weights[row['work']]/len(batch)
                a = task_loss('tempo', prediction, row['target'][None], row['mask'][None])*scale
                b = retention_loss(prediction, targets[row['id']], row['mask'][None])*scale if arm == 'local-retained' else prediction.new_zeros(())
                (a+b).backward()
                native, retained = native+float(a.detach()), retained+float(b.detach())
            for group in groups:
                chosen = [group[j] for j in rng.integers(len(group), size=4)]
                identity, profile = chosen[0]['source_id'], chosen[0]['profile']
                loss = paired_loss(model, captures[identity, 'source'], captures[identity, profile], chosen)/len(groups)
                require(torch.isfinite(loss).item(), 'nonfinite pair loss')
                loss.backward()
                relative += float(loss.detach())
                sampled.append(dict(source_id=identity, profile=profile, indices=[r['index'] for r in chosen]))
            require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()), 'invalid gradient')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            journal(output, dict(event='update_entered', arm=arm, update=updates+1,
                                 natural_ids=[r['id'] for r in batch], pair_samples=sampled))
            optimizer.step()
            updates += 1
            item = dict(update=updates, epoch=epoch+1, native_loss=native, retention_loss=retained,
                        relative_loss=relative, unclipped_grad_norm=float(norm))
            journal(output, dict(event='update_returned', arm=arm, **item))
            losses.append(item)
        checkpoint(epoch+1)
    budget()
    require(updates == STEPS and len(history) == EPOCHS+1, 'incomplete training')
    require(state_hash(teacher) == teacher_hash == state_hash(initial) and all(p.grad is None for p in teacher.parameters()),
            'teacher/initial changed')
    selected = selection.choose(history)
    chosen = read_weights(initial, output/f"{arm}.epoch-{selected['epoch']:02}.weights.npz", selected['weight_sha256'])
    selected_export = save_weights(chosen, output/(arm+'-selected.weights.npz'))
    final_export = save_weights(model, output/(arm+'-final.weights.npz'))
    report = dict(arm=arm, updates=updates, selected_epoch=selected['epoch'], selected_update=selected['updates'],
                  initial_state_sha256=teacher_hash, teacher_unchanged=True, teacher_fit_ids=list(targets),
                  selected=selected_export, final=final_export, history=history, losses=losses,
                  elapsed_s=time.monotonic()-started)
    write_json(output/(arm+'.fit.json'), report)
    return chosen, model.eval(), report
