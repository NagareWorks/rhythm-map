"""Fit-specific deadlines and diagnostic receipts; frozen math stays in its owner."""
import math
import time

from experiments.coupled_clock.supervision import clock_loss
from experiments.observed_adjoint.recorder import ScaledFitRecorder


class MusicalRecorder(ScaledFitRecorder):
    def __init__(self, *args, deadline, **kwargs):
        self.deadline = deadline
        super().__init__(*args, **kwargs)

    def check_deadline(self):
        now = time.monotonic()
        if now >= self.deadline:
            raise TimeoutError('registered fit wall budget exhausted')
        return now

    def _begin_record(self):
        super()._begin_record()
        self.check_deadline()

    def _backward_record(self, row, loss_fn):
        self.check_deadline()
        value = super()._backward_record(row, loss_fn)
        gradient = self.transport.parameter_gradient
        self._event('record_gradient', recording=row['id'], scale=row['scale'],
                    field_norm_log2=self.transport.field_gradient.norm_log2(),
                    parameter_norm_log2=gradient.norm_log2(), parameter_exponent=gradient.exponent,
                    neural_vjp_passes=len(self.transport.pieces), alignment_underflows=gradient.alignment_underflows)
        self.check_deadline()
        return value

    def _clip_accumulated(self, clip_norm):
        self.check_deadline()
        return super()._clip_accumulated(clip_norm)

    def _after_optimizer_return(self):
        super()._after_optimizer_return()
        self.check_deadline()

    def development_record(self, epoch, record, score_fn):
        # Reset only diagnostics, not the model, optimizer, phase or input.
        # The inherited guard owns a failure during the new development forward.
        self.transport.reset()
        self.records, self.batch_gradient, self.clip_report = [], None, None
        self.cursor.update(transport_stage='idle', band_index=None)
        def checked(model, payload):
            self.check_deadline()
            value = score_fn(model, payload)
            self.check_deadline()
            return value
        return super().development_record(epoch, record, checked)

    def checkpoint(self, epoch, development_loss):
        self.check_deadline()
        return super().checkpoint(epoch, development_loss)

    def abort_external(self, error, stage):
        if not self.failed:
            self.cursor.update(stage=stage, recording=None, record_index=None)
            with self._guard():
                raise error


def score(model, payload):
    return clock_loss(model(payload['features']), payload['reference'], payload['valid'])['total']


def conditioning(events):
    records = [e for e in events if e['event'] == 'record_gradient']
    updates = [e for e in events if e['event'] == 'update_completed']
    def extent(key):
        values = [e[key] for e in records if e[key] is not None]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('nonfinite conditioning diagnostic')
        return dict(min=min(values) if values else None, max=max(values) if values else None)
    return dict(record_gradients=len(records), update_receipts=len(updates),
                zero_parameter_gradients=sum(e['parameter_norm_log2'] is None for e in records),
                field_norm_log2=extent('field_norm_log2'), parameter_norm_log2=extent('parameter_norm_log2'),
                maximum_vjp_passes=max((e['neural_vjp_passes'] for e in records), default=0),
                multi_band_records=sum(e['neural_vjp_passes'] > 1 for e in records),
                alignment_underflows=sum(e['alignment_underflows'] for e in records)
                    + sum(e['alignment_underflows'] for e in updates),
                batches_norm_above_clip=sum(e['gradient_norm_log2'] is not None
                    and e['gradient_norm_log2'] > math.log2(e['max_norm']) for e in updates))
