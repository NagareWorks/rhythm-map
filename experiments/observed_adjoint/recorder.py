"""The existing update lifecycle with an explicit scaled-gradient accumulator."""
import torch

from experiments.coupled_clock.supervision import clock_loss
from experiments.training_observer.recorder import FitRecorder
from experiments.scaled_adjoint.scaled import accumulate, require
from experiments.scaled_adjoint.transport import install_clipped
from experiments.observed_adjoint.transport import ObservedTransport, pack_gradient


class ScaledFitRecorder(FitRecorder):
    def __init__(self, output, model, optimizer, identity, *, max_updates):
        self.transport = ObservedTransport(self._transport_event)
        self.records, self.batch_gradient, self.clip_report = [], None, None
        super().__init__(output, model, optimizer, identity, max_updates=max_updates)

    def _transport_event(self, stage, band_index):
        self.cursor.update(transport_stage=stage, band_index=band_index)
        if stage == 'upstream':
            self.cursor['stage'] = 'backward'
            self._event('backward_started')
        self._event('transport_stage', transport_stage=stage, band_index=band_index)

    def _begin_accumulation(self):
        self.records, self.batch_gradient, self.clip_report = [], None, None
        self._begin_record()

    def _begin_record(self):
        self.transport.reset()
        self.cursor.update(transport_stage='idle', band_index=None)

    def _backward_record(self, row, loss_fn):
        gradient, loss = self.transport.run(self.model, row['payload'], loss_fn)
        self._transport_event('record_weighting', None)
        weighted = gradient.multiply(row['scale'])
        self.records.append(dict(id=row['id'], scale=row['scale'], gradient=weighted,
                                 neural_vjp_passes=len(self.transport.pieces),
                                 parameter_gradient_norm_log2=gradient.norm_log2()))
        return loss

    def _clip_accumulated(self, clip_norm):
        self._transport_event('batch_accumulation', None)
        self.batch_gradient = accumulate([row['gradient'] for row in self.records])
        self._transport_event('gradient_install', None)
        self.clip_report = install_clipped(self.model, self.batch_gradient, clip_norm)
        return self.clip_report

    def _after_optimizer_return(self):
        self.cursor['stage'] = 'post_step_validation'
        self._transport_event('post_step_validation', None)
        require(all(torch.isfinite(p).all().item() for p in self.model.parameters()),
                'nonfinite returned model state')
        require(all(torch.isfinite(value).all().item() for state in self.optimizer.state.values()
                    for value in state.values() if isinstance(value, torch.Tensor)),
                'nonfinite returned optimizer state')
        self.cursor['stage'] = 'update_complete'

    def _payload(self):
        result = super()._payload()
        result['scaled_accumulation'] = dict(schema='observed-scaled-update-v1',
            transport=self.transport.snapshot(),
            weighted_records=[dict(id=row['id'], scale=row['scale'], gradient=pack_gradient(row['gradient']),
                                   neural_vjp_passes=row['neural_vjp_passes'],
                                   parameter_gradient_norm_log2=row['parameter_gradient_norm_log2'])
                              for row in self.records],
            batch_gradient=pack_gradient(self.batch_gradient), clip_report=self.clip_report)
        return result

    def update(self, epoch, batch_index, records, loss_fn=clock_loss, *, clip_norm=1.):
        """Clock-only loss callback; payload owns features/reference/valid tensors.

        Complete-record weights, one global clip and the existing durable update
        boundaries are retained. The caller must register this numerical path;
        this subclass does not alter any frozen fit runner or grant fit admission.
        """
        return super().update(epoch, batch_index, records, loss_fn, clip_norm=clip_norm)
