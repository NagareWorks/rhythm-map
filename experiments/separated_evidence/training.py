"""A single task owns its gradients, clipping and native AdamW state."""
import math

import torch

from experiments.coupled_clock.clock import require
from experiments.separated_evidence.model import TaskBranch
from experiments.separated_evidence.supervision import task_loss


class TaskTrainer:
    def __init__(self, branch, *, max_updates):
        require(type(branch) is TaskBranch, 'owned task branch required')
        require(type(max_updates) is int and max_updates > 0, 'positive fixed budget required')
        self.branch, self.max_updates = branch, max_updates
        self.optimizer = torch.optim.AdamW(branch.parameters(), lr=.001, weight_decay=.0001)
        self.returned_updates, self.failed, self.stage = 0, False, 'idle'

    def update(self, records):
        require(not self.failed, 'failed task is closed; no automatic retry')
        try:
            require(self.returned_updates < self.max_updates, 'task update budget exhausted')
            require(bool(records) and len({r['id'] for r in records}) == len(records),
                    'nonempty distinct complete records required')
            for row in records:
                require(isinstance(row['id'], str) and bool(row['id'])
                        and math.isfinite(row['scale']) and row['scale'] > 0,
                        'invalid record identity or weight')
            self.optimizer.zero_grad(set_to_none=True)
            values = []
            for row in records:
                self.stage = 'forward'
                prediction = self.branch(row['features'])
                loss = task_loss(self.branch.task, prediction, row['reference'], row['valid'])
                require(torch.isfinite(loss).item(), 'nonfinite task loss')
                self.stage = 'backward'
                (loss * row['scale']).backward()
                values.append(float(loss.detach()))
            self.stage = 'clip'
            require(all(p.grad is not None for p in self.branch.parameters()), 'missing task gradient')
            norm = torch.nn.utils.clip_grad_norm_(self.branch.parameters(), 1., error_if_nonfinite=True)
            self.stage = 'optimizer_entered'
            self.optimizer.step()
            self.returned_updates += 1
            self.stage = 'optimizer_returned'
            require(all(torch.isfinite(p).all().item() for p in self.branch.parameters()),
                    'nonfinite returned task state')
            require(all(not isinstance(v, torch.Tensor) or torch.isfinite(v).all().item()
                        for state in self.optimizer.state.values() for v in state.values()),
                    'nonfinite returned optimizer state')
            return dict(task=self.branch.task, returned_updates=self.returned_updates,
                        record_losses=values, weighted_loss=math.fsum(r['scale'] * v for r, v in zip(records, values)),
                        gradient_norm=float(norm))
        except BaseException:
            self.failed = True
            raise
