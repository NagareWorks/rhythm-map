"""Fit deadlines, protected-step receipts and observational finite-loss audits."""
import copy
import math

import torch

from experiments.coupled_clock.supervision import clock_loss
from experiments.coupled_clock.clock import require
from experiments.protected_update.recorder import ProtectedFitRecorder
from experiments.scaled_phase_fit.recorder import MusicalRecorder as PreviousRecorder
from experiments.scaled_phase_fit.recorder import conditioning as previous_conditioning, score
from experiments.training_observer.recorder import cpu_copy, rng_state


class MusicalRecorder(ProtectedFitRecorder):
    check_deadline = PreviousRecorder.check_deadline
    abort_external = PreviousRecorder.abort_external

    def __init__(self, *args, deadline, **kwargs):
        self.deadline, self.before_losses, self.after_losses = deadline, None, None
        super().__init__(*args, **kwargs)

    def _begin_accumulation(self):
        self.before_losses, self.after_losses = None, None
        super()._begin_accumulation()

    def _begin_record(self):
        super()._begin_record()
        self.check_deadline()

    def _transport_event(self, term, stage, band):
        super()._transport_event(term, stage, band)
        self.check_deadline()

    def _backward_record(self, row, loss_fn):
        value = super()._backward_record(row, loss_fn)
        for term, transport in self.terms.items():
            gradient = transport.parameter_gradient
            self._event('record_gradient', recording=row['id'], term=term, scale=row['scale'],
                field_norm_log2=transport.field_gradient.norm_log2(),
                parameter_norm_log2=gradient.norm_log2(), parameter_exponent=gradient.exponent,
                neural_vjp_passes=len(transport.pieces), alignment_underflows=gradient.alignment_underflows)
        self.check_deadline()
        return value

    def _clip_accumulated(self, clip_norm):
        self.check_deadline()
        self.before_losses = {term: math.fsum(r['scale'] * r['terms'][term]['loss'] for r in self.records)
                              for term in ('phase', 'advance')}
        require(all(math.isfinite(x) for x in self.before_losses.values()), 'nonfinite weighted batch loss')
        report = super()._clip_accumulated(clip_norm)
        self.check_deadline()
        return report

    def _boundary_event(self, stage):
        self.cursor['boundary_stage'] = stage
        self.check_deadline()
        super()._boundary_event(stage)
        self.check_deadline()

    def _after_optimizer_return(self):
        self.check_deadline()
        for transport in self.terms.values():
            transport.reset()
        self.cursor.update(stage='post_step_loss', loss_term=None, transport_stage='idle', band_index=None)
        totals = {key: [] for key in ('phase', 'advance')}
        device = next(self.model.parameters()).device
        for index, record in enumerate(self.records):
            self.cursor.update(recording=record['id'], record_index=index)
            self.current_record, self.before_record_rng = cpu_copy(record['record']), rng_state(self.model)
            self._event('post_step_loss_started')
            self.check_deadline()
            payload = {k: v.to(device) for k, v in record['record']['payload'].items()}
            with torch.inference_mode():
                values = clock_loss(self.model(payload['features']), payload['reference'], payload['valid'])
                losses = {k: float(values[k]) for k in totals}
            require(all(math.isfinite(x) for x in losses.values()), 'nonfinite post-step loss')
            for key in totals:
                totals[key].append(record['scale'] * losses[key])
            self._event('post_step_record_loss', losses=losses, scale=record['scale'])
            self.check_deadline()
        self.after_losses = {k: math.fsum(v) for k, v in totals.items()}
        require(all(math.isfinite(x) for x in self.after_losses.values()), 'nonfinite weighted post-step loss')
        self.cursor.update(stage='update_complete', recording=None, record_index=None)
        self._event('protected_step', before_losses=self.before_losses, after_losses=self.after_losses,
            finite_count_increased=self.after_losses['advance'] > self.before_losses['advance'],
            groups=copy.deepcopy(self.optimizer.receipt), committed_steps=self.optimizer.committed_steps,
            moved_steps=self.optimizer.moved_steps)
        self.check_deadline()

    def _payload(self):
        result = super()._payload()
        result['finite_loss_audit'] = dict(before=self.before_losses, after=self.after_losses)
        return result

    def development_record(self, epoch, record, score_fn):
        for transport in self.terms.values():
            transport.reset()
        self.records, self.phase, self.count, self.routed, self.routing = [], None, None, None, None
        self.before_losses, self.after_losses = None, None
        self.optimizer.reset()
        self.cursor.update(loss_term=None, transport_stage='idle', band_index=None, boundary_stage='idle')
        def checked(model, payload):
            self.check_deadline()
            value = score_fn(model, payload)
            self.check_deadline()
            return value
        return super().development_record(epoch, record, checked)

    def checkpoint(self, epoch, development_loss):
        self.check_deadline()
        return super().checkpoint(epoch, development_loss)


def conditioning(events):
    result = previous_conditioning(events)
    steps = [e for e in events if e['event'] == 'protected_step']
    gradients = [e for e in events if e['event'] == 'record_gradient']
    actions = ('unchanged_proposal', 'projected', 'rounded_projection_veto')
    receipts = [r for step in steps for r in step['groups'].values()]
    require(all(math.isfinite(x) for s in steps for field in ('before_losses', 'after_losses')
                for x in s[field].values()), 'nonfinite loss audit')
    result.update(complete_records=sum(e['event'] == 'recording_backward_completed' for e in events),
        phase_gradients=sum(e['term'] == 'phase' for e in gradients),
        count_gradients=sum(e['term'] == 'advance' for e in gradients),
        protected_steps=len(steps), post_step_records=sum(e['event'] == 'post_step_record_loss' for e in events),
        committed_steps=steps[-1]['committed_steps'] if steps else 0,
        moved_steps=steps[-1]['moved_steps'] if steps else 0,
        stalled_steps=sum(not any(g['moved'] for g in s['groups'].values()) for s in steps),
        group_actions={a: sum(r['action'] == a for r in receipts) for a in actions},
        undefined_groups=sum(not r['count_direction_defined'] for r in receipts),
        adverse_defined_groups=sum(r['final_sign'] is not None and r['final_sign'] > 0 for r in receipts),
        group_receipts=len(receipts), finite_count_increases=sum(s['finite_count_increased'] for s in steps))
    return result
