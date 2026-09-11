"""Record bounded updates without changing their loss, gradients or selection.

Snapshots are diagnostic artifacts, not admission to resume a failed experiment.
Only load your own trusted snapshot with torch.load(..., weights_only=True).
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f'unsupported snapshot value: {type(value).__name__}')


def rng_state(model):
    # Do not initialize or inspect unrelated GPUs on a shared host.
    devices = sorted({p.device.index for p in (*model.parameters(), *model.buffers()) if p.is_cuda})
    state = np.random.get_state()
    return dict(python=random.getstate(), torch_cpu=torch.get_rng_state().clone(),
                numpy=dict(kind=state[0], keys=state[1].tolist(), position=state[2],
                           has_gauss=state[3], cached_gaussian=state[4]),
                torch_cuda={str(d): torch.cuda.get_rng_state(d).cpu().clone() for d in devices})


def tensor_summary(value):
    x = value.detach().cpu()
    finite = torch.isfinite(x)
    usable = x[finite]
    return dict(shape=list(x.shape), dtype=str(x.dtype), nonfinite=int((~finite).sum()),
                finite_min=float(usable.min()) if usable.numel() else None,
                finite_max=float(usable.max()) if usable.numel() else None)


def atomic_save(path, payload):
    """A partial write cannot replace the last complete checkpoint."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FitRecorder:
    """One explicitly bounded fit's state recorder, not a training orchestrator.

    The caller owns the pre-registered model/loss/population/control protocol.
    This helper owns journaling, single-update accumulation and crash evidence.
    There is deliberately no resume, retry, model-selection sweep or data loader.
    """
    def __init__(self, output, model, optimizer, identity, *, max_updates, trace=None):
        if type(max_updates) is not int or max_updates <= 0:
            raise ValueError('positive update budget required')
        if not identity or not isinstance(identity, dict):
            raise ValueError('explicit execution identity required')
        json.dumps(identity, allow_nan=False)
        self.output = Path(output).resolve()
        if self.output.is_relative_to(ROOT):
            raise ValueError('snapshots require a new private output outside the repository')
        self.output.mkdir(exist_ok=False)
        self.model, self.optimizer, self.trace = model, optimizer, trace
        self.identity = copy.deepcopy(identity)
        self.max_updates, self.completed_updates = max_updates, 0
        self.best, self.current_record, self.before_record_rng = None, None, None
        self.cursor = dict(stage='created', epoch=None, batch=None, recording=None,
                           record_index=None, batch_recordings=[], optimizer_step_status='not_started')
        self.failed = False
        self._event('created', max_updates=max_updates, identity=self.identity)
        self._save('initial.pt')

    def _event(self, event, **fields):
        value = dict(event=event, completed_updates=self.completed_updates,
                     cursor=copy.deepcopy(self.cursor), **fields)
        with (self.output / 'journal.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def _payload(self):
        return dict(schema='rhythm-map.training-snapshot.v1', identity=self.identity,
                    runtime=dict(torch=str(torch.__version__), numpy=np.__version__,
                                 model_class=type(self.model).__module__ + '.' + type(self.model).__qualname__,
                                 optimizer_class=type(self.optimizer).__module__ + '.' + type(self.optimizer).__qualname__,
                                 deterministic=torch.are_deterministic_algorithms_enabled(),
                                 deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                 cudnn_tf32=torch.backends.cudnn.allow_tf32,
                                 parameter_devices={k: str(p.device) for k, p in self.model.named_parameters()}),
                    model=cpu_copy(self.model.state_dict()), model_training=self.model.training,
                    module_training={name: module.training for name, module in self.model.named_modules()},
                    optimizer=cpu_copy(self.optimizer.state_dict()),
                    gradients={name: cpu_copy(p.grad) for name, p in self.model.named_parameters()},
                    gradient_summaries={name: tensor_summary(p.grad) if p.grad is not None else None
                                        for name, p in self.model.named_parameters()},
                    best=cpu_copy(self.best), completed_updates=self.completed_updates,
                    max_updates=self.max_updates, cursor=copy.deepcopy(self.cursor),
                    current_record=cpu_copy(self.current_record), before_record_rng=self.before_record_rng,
                    rng=rng_state(self.model), automatic_retry=False)

    def _save(self, filename, **extra):
        return atomic_save(self.output / filename, dict(self._payload(), **extra))

    @contextmanager
    def _guard(self):
        if self.failed:
            raise RuntimeError('failed fit is closed; no automatic retry')
        try:
            yield
        except BaseException as original:
            self.failed = True
            # Keep the primary exception even if a full disk / failed device also
            # prevents diagnostics. A killed process cannot enter this handler.
            problems, snapshot_hash = [], None
            diagnostic = None
            if self.trace is not None:
                try:
                    diagnostic = cpu_copy(self.trace())
                except BaseException as secondary:
                    problems.append('trace: ' + type(secondary).__name__)
            try:
                snapshot_hash = self._save('failure.pt', diagnostic=diagnostic,
                    exception=dict(type=type(original).__name__, message=str(original)))
            except BaseException as secondary:
                problems.append('snapshot: ' + type(secondary).__name__)
            try:
                self._event('failed', exception_type=type(original).__name__,
                            snapshot_sha256=snapshot_hash, capture_errors=problems,
                            update_count_exact=self.cursor['optimizer_step_status'] != 'entered')
            except BaseException as secondary:
                problems.append('journal: ' + type(secondary).__name__)
            if problems:
                original.add_note('Diagnostic persistence incomplete: ' + ', '.join(problems))
            raise

    def update(self, epoch, batch_index, records, loss_fn, *, clip_norm=1.):
        """Accumulate full records, apply one ordinary clip and optimizer step.

        Each record has id, payload (the exact tensor inputs), and scale (the
        registered work weight / actual batch size). loss_fn(model, payload)
        returns an unscaled scalar loss. Callback source belongs in identity.
        """
        with self._guard():
            if self.completed_updates >= self.max_updates:
                raise ValueError('update budget exhausted')
            if type(epoch) is not int or epoch <= 0 or type(batch_index) is not int or batch_index < 0:
                raise ValueError('invalid update location')
            if not records or len({r['id'] for r in records}) != len(records):
                raise ValueError('nonempty distinct recording IDs required')
            if not math.isfinite(clip_norm) or clip_norm <= 0:
                raise ValueError('positive finite clipping norm required')
            for row in records:
                if not isinstance(row['id'], str) or not row['id'] or not math.isfinite(row['scale']) or row['scale'] <= 0:
                    raise ValueError('invalid recording identity or loss weight')
            self.current_record, self.before_record_rng = None, None
            self.cursor.update(stage='update_start', epoch=epoch, batch=batch_index,
                               recording=None, record_index=None, batch_recordings=[r['id'] for r in records],
                               optimizer_step_status='not_started')
            # Captures model, optimizer and RNG before zero_grad / any record.
            # This durable state survives process termination during a later step.
            before_sha = self._save('before-update.pt')
            self._event('update_started', before_update_sha256=before_sha)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            for index, row in enumerate(records):
                self.cursor.update(stage='forward', recording=row['id'], record_index=index)
                self.current_record = cpu_copy(row)
                self.before_record_rng = rng_state(self.model)
                self._event('recording_started')
                loss = loss_fn(self.model, row['payload'])
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not torch.isfinite(loss).item():
                    raise ValueError('finite scalar loss required')
                self.cursor['stage'] = 'backward'
                self._event('backward_started')
                (loss * row['scale']).backward()
                losses.append(float(loss.detach()))
                self._event('recording_backward_completed')
            self.cursor.update(stage='clip', recording=None, record_index=None)
            self._event('clip_started')
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm, error_if_nonfinite=True)
            self.cursor['stage'] = 'optimizer_step_prepare'
            self._event('optimizer_step_prepared')
            # The durable record is intent, not proof of entry. If its write
            # fails, no optimizer code ran. A hard kill after it is ambiguous.
            self.cursor.update(stage='optimizer_step', optimizer_step_status='entered')
            self.optimizer.step()
            # Count only a returned optimizer call; an exception inside step may
            # have partially mutated parameters and MUST remain explicitly unknown.
            self.completed_updates += 1
            self.cursor.update(stage='update_complete', optimizer_step_status='returned')
            after_sha = self._save('after-update.pt')
            self._event('update_completed', checkpoint_sha256=after_sha, gradient_norm=float(norm))
            return losses

    def development_record(self, epoch, record, score_fn):
        """Observe one development forward, without updating or selecting weights."""
        with self._guard():
            self.cursor.update(stage='development', epoch=epoch, batch=None, recording=record['id'],
                               record_index=None, batch_recordings=[], optimizer_step_status='not_started')
            self.current_record, self.before_record_rng = cpu_copy(record), rng_state(self.model)
            self._event('development_started')
            self.model.eval()
            with torch.inference_mode():
                value = float(score_fn(self.model, record['payload']))
            if not math.isfinite(value):
                raise ValueError('finite development score required')
            return value

    def checkpoint(self, epoch, development_loss):
        """Persist lowest complete-epoch development score; earlier wins ties."""
        with self._guard():
            if type(epoch) is not int or epoch <= 0 or not math.isfinite(development_loss):
                raise ValueError('invalid checkpoint score/location')
            self.cursor.update(stage='checkpoint', epoch=epoch, recording=None, record_index=None)
            selected = self.best is None or development_loss < self.best['development_loss']
            if selected:
                self.best = dict(epoch=epoch, development_loss=float(development_loss),
                                 completed_updates=self.completed_updates, model=cpu_copy(self.model.state_dict()))
                digest = self._save('best.pt')
                self._event('best_checkpoint', checkpoint_sha256=digest)
            return selected
