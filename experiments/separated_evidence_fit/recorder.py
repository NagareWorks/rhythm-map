"""Owned update boundaries and symmetric task-specific checkpoint selection."""
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch

from experiments.coupled_clock.clock import require
from experiments.coupled_clock.run import state_hash
from experiments.separated_evidence.model import SharedEvidence, SeparatedEvidence, TaskBranch
from experiments.separated_evidence.supervision import losses
from experiments.separated_evidence.training import TaskTrainer

TASKS = ('tempo', 'phase')


def owned(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: owned(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(owned(v) for v in value)
    return copy.deepcopy(value)


def branch(model, task):
    # A read-only view for snapshotting; never optimize this shared view alone.
    return (TaskBranch(task, model.trunk, getattr(model, task + '_head'))
            if type(model) is SharedEvidence else getattr(model, task))


class Selection:
    def __init__(self):
        self.best, self.epoch = {}, 0

    def observe(self, epoch, scores, model):
        require(epoch == self.epoch + 1 and set(scores) == set(TASKS)
                and all(math.isfinite(v) and v >= 0 for v in scores.values()),
                'complete finite development epoch required')
        changed = []
        for task in TASKS:
            if task not in self.best or scores[task] < self.best[task]['loss']:
                self.best[task] = dict(epoch=epoch, loss=scores[task],
                    state=owned(branch(model, task).state_dict()), state_sha256=state_hash(branch(model, task)))
                changed.append(task)
        self.epoch = epoch
        return changed

    def export(self, model):
        require(set(self.best) == set(TASKS), 'no complete task selection')
        pair = SeparatedEvidence(model) if type(model) is SharedEvidence else copy.deepcopy(model)
        for task in TASKS:
            getattr(pair, task).load_state_dict(self.best[task]['state'])
            require(state_hash(getattr(pair, task)) == self.best[task]['state_sha256'], 'selected branch differs')
        return pair.eval()


class SharedTrainer:
    """Native summed-loss control with the same returned-call failure semantics."""
    def __init__(self, model, max_updates):
        require(type(model) is SharedEvidence and type(max_updates) is int and max_updates > 0,
                'shared model and fixed budget required')
        self.model, self.max_updates = model, max_updates
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        self.returned_updates, self.failed, self.stage = 0, False, 'idle'

    def update(self, rows):
        require(not self.failed, 'failed shared update is closed')
        try:
            require(self.returned_updates < self.max_updates and bool(rows)
                    and len({r['id'] for r in rows}) == len(rows), 'invalid shared budget or batch')
            require(all(isinstance(r['id'], str) and r['id'] and math.isfinite(r['scale']) and r['scale'] > 0
                        for r in rows), 'invalid shared record')
            self.optimizer.zero_grad(set_to_none=True)
            values = {k: [] for k in TASKS}
            for row in rows:
                self.stage = 'forward'
                result = losses(self.model(row['features']), row['reference'], row['valid'])
                self.stage = 'backward'
                (sum(result.values()) * row['scale']).backward()
                for k, v in result.items():
                    values[k].append(float(v.detach()))
            self.stage = 'clip'
            require(all(p.grad is not None for p in self.model.parameters()), 'missing shared gradient')
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            self.stage = 'optimizer_entered'
            self.optimizer.step()
            self.returned_updates += 1
            self.stage = 'optimizer_returned'
            require(all(torch.isfinite(p).all().item() for p in self.model.parameters())
                    and all(not isinstance(v, torch.Tensor) or torch.isfinite(v).all().item()
                        for state in self.optimizer.state.values() for v in state.values()), 'nonfinite returned shared state')
            return dict(returned_updates=self.returned_updates, record_losses=values, gradient_norm=float(norm))
        except BaseException:
            self.failed = True
            raise


class Recorder:
    def __init__(self, output, model, metadata, *, updates, deadline):
        require(not output.exists() and math.isfinite(deadline), 'new output and finite deadline required')
        output.mkdir()
        self.output, self.model, self.metadata, self.deadline = output, model, metadata, deadline
        self.trainers = ({'shared': SharedTrainer(model, updates)} if type(model) is SharedEvidence else
                         {k: TaskTrainer(getattr(model, k), max_updates=updates) for k in TASKS})
        self.selection, self.paired_updates, self.failed = Selection(), 0, False
        self.cursor, self.rows = dict(stage='created'), []
        self.timings = {k: [] for k in self.trainers}

    def check(self):
        require(not self.failed, 'failed arm is closed')
        now = time.monotonic()
        if now >= self.deadline:
            raise TimeoutError('arm wall budget exhausted')
        return now

    def counts(self):
        return {k: t.returned_updates for k, t in self.trainers.items()}

    def event(self, name, **values):
        item = dict(event=name, cursor=self.cursor, paired_updates=self.paired_updates,
                    optimizer_calls=self.counts(), **values)
        with (self.output / 'journal.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(item, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def snapshot(self, name):
        numpy_rng = np.random.get_state()
        payload = dict(metadata=self.metadata, cursor=self.cursor, paired_updates=self.paired_updates,
            failed=self.failed, model=self.model.state_dict(), selection=self.selection.best,
            selection_epoch=self.selection.epoch, weighted_records=self.rows,
            gradients={n: p.grad for n, p in self.model.named_parameters()},
            trainers={k: dict(returned_updates=t.returned_updates, failed=t.failed, stage=t.stage,
                             optimizer=t.optimizer.state_dict()) for k, t in self.trainers.items()},
            rng=dict(python=random.getstate(), numpy=(numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                     torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()
                     if next(self.model.parameters()).is_cuda else []), automatic_retry=False)
        temporary = self.output / (name + '.tmp')
        with temporary.open('wb') as stream:
            torch.save(owned(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.output / name)

    def update(self, epoch, batch, rows):
        self.check()
        self.rows = rows
        self.model.train()
        result = {}
        for name, trainer in self.trainers.items():
            self.cursor = dict(stage='before_update', epoch=epoch, batch=batch, task=name)
            self.check()
            started = time.monotonic()
            self.snapshot('before.pt')
            self.check()
            self.cursor['stage'] = 'update_entered'
            self.event('update_entered')
            self.check()
            receipt = trainer.update(rows)
            self.cursor['stage'] = 'update_returned'
            self.snapshot('after.pt')
            self.event('update_returned', receipt=receipt,
                       records=[dict(id=r['id'], work=r['work'], scale=r['scale']) for r in rows])
            self.timings[name].append(time.monotonic() - started)
            self.check()
            if name == 'shared':
                result = receipt['record_losses']
            else:
                result[name] = receipt['record_losses']
        self.paired_updates += 1
        self.event('pair_complete')
        return result

    def inference_stage(self, stage, rows=()):
        self.cursor, self.rows = dict(stage=stage), list(rows)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.grad = None
        self.check()

    def checkpoint(self, epoch, scores):
        self.inference_stage('checkpoint')
        changed = self.selection.observe(epoch, scores, self.model)
        for task in changed:
            self.snapshot('best-' + task + '.pt')
            self.check()
        self.event('epoch_complete', epoch=epoch, development=scores,
                   selected_epochs={k: v['epoch'] for k, v in self.selection.best.items()})

    def abort(self, error):
        self.failed = True
        for capture in (lambda: self.event('failed', exception_type=type(error).__name__, automatic_retry=False),
                        lambda: self.snapshot('failure.pt')):
            try:
                capture()
            except BaseException as secondary:
                error.add_note('Failure evidence unavailable: ' + type(secondary).__name__)
