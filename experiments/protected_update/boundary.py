"""A real AdamW proposal, groupwise displacement projection, then owned commit."""
import copy
import math

import numpy as np
import torch
from torch.optim import optimizer as optimizer_module

from experiments.gradient_routing.route import GROUPS, model_shapes, partition
from experiments.scaled_adjoint.scaled import Gradient, require
from experiments.training_observer.recorder import cpu_copy


def arrays(parameters):
    return {n: p.detach().cpu().numpy().copy() for n, p in parameters.items()}


def exact_sign(count, before, after):
    """Exact sign of sum(c_i * (after_i-before_i)) for STORED binary floats.

    The positive common gradient exponent cannot affect this sign. Integer
    accumulation avoids cancellation, product underflow and subtraction rounding;
    this does not make the preceding neural gradient an exact real derivative.
    """
    require(before.keys() == after.keys() == count.blocks.keys(), 'displacement keys differ')
    require(count.alignment_underflows == 0, 'count alignment loss')
    terms = {}
    for name, c in count.blocks.items():
        x, y = np.asarray(before[name]), np.asarray(after[name])
        require(x.shape == y.shape == c.shape and np.isfinite(x).all() and np.isfinite(y).all(),
                'invalid displacement values or geometry')
        for cv, xv, yv in zip(c.flat, x.flat, y.flat):
            if cv == 0 or xv == yv:
                continue
            cn, cd = float(cv).as_integer_ratio()
            for value, sign in ((yv, 1), (xv, -1)):
                vn, vd = float(value).as_integer_ratio()
                shift = cd.bit_length() + vd.bit_length() - 2
                terms[shift] = terms.get(shift, 0) + sign * cn * vn
    common = max(terms, default=0)
    total = sum(value << (common - shift) for shift, value in terms.items())
    return (total > 0) - (total < 0)


def masks():
    parts = partition(Gradient.create({n: np.ones(s) for n, s in model_shapes().items()}))
    return {g: {n: x != 0 for n, x in part.blocks.items()} for g, part in parts.items()}


def protect(count, before, proposal):
    """Closest halfspace projection in float64, then native rounding and audit.

    An adverse rounded projection vetoes ONLY its group to the old native values.
    No line search, epsilon tolerance, hidden retry or finite-loss claim.
    """
    require(count.alignment_underflows == 0, 'count alignment loss')
    groups, owned = partition(count), masks()
    require(before.keys() == proposal.keys() == count.blocks.keys(), 'proposal ownership differs')
    for n, x in before.items():
        require(x.dtype in (np.dtype('float32'), np.dtype('float64')) and x.dtype == proposal[n].dtype,
                'matching native float32/float64 required')
    candidate = {n: x.copy() for n, x in proposal.items()}
    receipts = {}
    for group in GROUPS:
        c, mask = groups[group], owned[group]
        sign = exact_sign(c, before, proposal)
        defined = any(np.any(x != 0) for x in c.blocks.values())
        action = 'unchanged_proposal'
        if sign > 0:
            # Normalize delta before dot products to avoid squaring its full scale.
            with np.errstate(over='raise', invalid='raise'):
                d = Gradient.create({n: np.where(mask[n], proposal[n].astype(np.float64) -
                                                x.astype(np.float64), 0.) for n, x in before.items()})
            require(d.alignment_underflows == 0, 'displacement alignment loss')
            dot = math.fsum(float(np.sum(d.blocks[n] * c.blocks[n])) for n in before)
            square = math.fsum(float(np.sum(x * x)) for x in c.blocks.values())
            require(square > 0 and dot > 0, 'projection lost the adverse direction')
            with np.errstate(over='raise', invalid='raise', under='ignore'):
                for n, x in before.items():
                    delta = np.ldexp(d.blocks[n] - dot / square * c.blocks[n], d.exponent)
                    value = (x.astype(np.float64) + delta).astype(x.dtype)
                    candidate[n] = np.where(mask[n], value, candidate[n])
            action = 'projected'
            if exact_sign(c, before, candidate) > 0:
                candidate = {n: np.where(mask[n], x, candidate[n]) for n, x in before.items()}
                action = 'rounded_projection_veto'
        final_sign = exact_sign(c, before, candidate)
        require(final_sign <= 0, 'native displacement violated protection')
        receipts[group] = dict(count_direction_defined=defined, proposal_sign=sign if defined else None,
            final_sign=final_sign if defined else None, action=action,
            moved=any(np.any(mask[n] & (before[n] != candidate[n])) for n in before))
    return candidate, receipts


def finite_state(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(v) for v in value)
    return value is None or type(value) in (bool, int, str) or (type(value) is float and math.isfinite(value))


class ProtectedAdamW:
    """Single-caller adapter, NOT an Optimizer subclass or general optimizer API.

    Exclusive ownership of model/optimizer is required. The real AdamW runs on
    cloned parameters/state. A committed proposal advances its moments/step even
    if projection or rounding leaves all weights unchanged. No scheduler, hooks,
    distributed/AMP/captured/differentiable optimizer, tied storage or resume API.
    """
    def __init__(self, model, optimizer, notify=None):
        self.model, self.raw, self.notify = model, optimizer, notify
        self.parameters = dict(model.named_parameters())
        self.failed, self.committed_steps, self.moved_steps = False, 0, 0
        self.reset()
        self._validate()

    def reset(self):
        require(not self.failed, 'failed optimizer boundary is closed')
        self.stage, self.count, self.proposal, self.candidate = 'idle', None, None, None
        self.receipt, self.proposal_state, self.rollback = None, None, None

    def _validate(self):
        require(type(self.raw) is torch.optim.AdamW and 'step' not in self.raw.__dict__,
                'unmodified native AdamW required')
        require(not any(v for k, v in self.raw.__dict__.items() if 'hook' in k) and
                not optimizer_module._global_optimizer_pre_hooks and
                not optimizer_module._global_optimizer_post_hooks, 'optimizer hooks unsupported')
        pairs = list(self.model.named_parameters(remove_duplicate=False))
        require(len(pairs) == len(self.parameters) and
                all(self.parameters.get(n) is p for n, p in pairs), 'parameter identity or alias changed')
        require({n: tuple(p.shape) for n, p in pairs} == model_shapes(), 'exact model geometry required')
        require(not list(self.model.buffers()), 'buffer-mutating models unsupported')
        require(all(p.requires_grad and p.dtype in (torch.float32, torch.float64) and p.is_contiguous()
                    and p.device.type in ('cpu', 'cuda') and torch.isfinite(p).all() for _, p in pairs),
                'finite contiguous trainable native parameters required')
        require(len({(p.device, p.dtype) for _, p in pairs}) == 1 and
                len({p.untyped_storage().data_ptr() for _, p in pairs}) == len(pairs),
                'single native device/dtype with disjoint storage required')
        require(len(self.raw.param_groups) == 1 and
                [id(p) for p in self.raw.param_groups[0]['params']] == [id(p) for p in self.parameters.values()],
                'one ordered complete optimizer group required')
        options = self.raw.param_groups[0]
        require(not any(options.get(k) for k in ('capturable', 'differentiable', 'maximize', 'fused')) and
                options.get('foreach') in (False, None), 'unsupported AdamW execution mode')
        require(all(type(options[k]) is float and math.isfinite(options[k]) and options[k] >= 0
                    for k in ('lr', 'eps', 'weight_decay')) and
                all(type(x) is float and 0 <= x < 1 for x in options['betas']), 'invalid AdamW scalar options')
        require(all(id(p) in {id(v) for v in self.parameters.values()} for p in self.raw.state),
                'foreign optimizer state')
        require(finite_state(self.raw.state), 'nonfinite optimizer state')

    def state_dict(self):
        return self.raw.state_dict()

    def zero_grad(self, *, set_to_none=True):
        self.raw.zero_grad(set_to_none=set_to_none)

    def snapshot(self):
        from experiments.observed_adjoint.transport import pack_gradient
        return dict(schema='protected-adamw-v1', stage=self.stage, failed=self.failed,
            committed_steps=self.committed_steps, moved_steps=self.moved_steps, rollback=self.rollback,
            count=pack_gradient(self.count), receipt=copy.deepcopy(self.receipt),
            proposal=cpu_copy(self.proposal), candidate=cpu_copy(self.candidate),
            proposal_state=cpu_copy(self.proposal_state))

    def arm(self, count):
        require(not self.failed and self.stage == 'idle' and self.count is None, 'fresh boundary required')
        require(count.alignment_underflows == 0, 'count alignment loss')
        partition(count)
        self.count = count

    def _mark(self, stage):
        self.stage = stage
        if self.notify is not None:
            self.notify(stage)

    def _propose(self):
        # Copy the tuple together so optimizer references point at these clones.
        parameters, optimizer = copy.deepcopy((self.parameters, self.raw))
        for n, p in parameters.items():
            p.grad = self.parameters[n].grad.detach().clone()
        optimizer.step()
        return parameters, optimizer

    def _copy_parameters(self, values):
        for n, p in self.parameters.items():
            p.copy_(values[n])

    @torch.no_grad()
    def step(self):
        require(not self.failed and self.stage == 'idle' and self.count is not None, 'armed boundary required')
        try:
            self._validate()
            require(all(p.grad is not None and not p.grad.is_sparse and torch.isfinite(p.grad).all()
                        for p in self.parameters.values()), 'complete finite dense gradients required')
            self._mark('proposal_started')
            before = {n: p.detach().clone() for n, p in self.parameters.items()}
            old_state = self.raw.state
            shadow, proposed = self._propose()
            self.proposal, self.proposal_state = cpu_copy(shadow), cpu_copy(proposed.state_dict())
            require(finite_state(self.proposal) and finite_state(self.proposal_state), 'nonfinite AdamW proposal')
            values, self.receipt = protect(self.count, arrays(before), arrays(shadow))
            candidate = {n: torch.from_numpy(np.asarray(x).copy()).to(self.parameters[n]) for n, x in values.items()}
            self.candidate = cpu_copy(candidate)
            new_state = type(proposed.state)(dict)
            for n, p in self.parameters.items():
                if shadow[n] in proposed.state:
                    new_state[p] = proposed.state[shadow[n]]
            # Persist candidate, exact count packet and proposed moments before
            # any live write. Callback failures here leave live state untouched.
            self._mark('prepared')
            self._mark('commit_started')
            try:
                self._copy_parameters(candidate)
                require(all(torch.equal(p, candidate[n]) for n, p in self.parameters.items()),
                        'native commit differs from checked candidate')
                self.raw.state = new_state
            except BaseException as original:
                self.rollback = 'attempted'
                try:
                    for n, p in self.parameters.items():
                        p.copy_(before[n])
                    self.raw.state = old_state
                    require(all(torch.equal(p, before[n]) for n, p in self.parameters.items()),
                            'rollback could not verify native weights')
                    self.rollback = 'restored'
                except BaseException as secondary:
                    self.rollback = 'failed_unknown'
                    original.add_note('Rollback failed: ' + type(secondary).__name__)
                raise
            self.committed_steps += 1
            self.moved_steps += int(any(r['moved'] for r in self.receipt.values()))
            self._mark('committed')
        except BaseException:
            self.failed = True
            raise
