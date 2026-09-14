"""Original recorder lifecycle with separately observed phase/count gradients."""
from experiments.coupled_clock.supervision import clock_loss
from experiments.gradient_routing.route import route
from experiments.observed_adjoint.transport import ObservedTransport, pack_gradient
from experiments.protected_update.boundary import ProtectedAdamW
from experiments.phase_sync.model import PhaseSyncReadout
from experiments.scaled_adjoint.scaled import accumulate
from experiments.scaled_adjoint.scaled import require
from experiments.scaled_adjoint.transport import install_clipped
from experiments.training_observer.recorder import FitRecorder, cpu_copy


class ProtectedFitRecorder(FitRecorder):
    def __init__(self, output, model, optimizer, identity, *, max_updates):
        require(type(model) is PhaseSyncReadout, 'unchanged deterministic PhaseSyncReadout required')
        self.terms = {key: ObservedTransport(lambda stage, band, key=key: self._transport_event(key, stage, band))
                      for key in ('phase', 'advance')}
        self.records, self.phase, self.count, self.routed, self.routing = [], None, None, None, None
        adapter = ProtectedAdamW(model, optimizer, self._boundary_event)
        super().__init__(output, model, adapter, identity, max_updates=max_updates)

    def _transport_event(self, term, stage, band):
        self.cursor.update(loss_term=term, transport_stage=stage, band_index=band)
        if stage == 'upstream':
            self.cursor['stage'] = 'backward'
            self._event('backward_started')
        self._event('transport_stage')

    def _boundary_event(self, stage):
        self.cursor['boundary_stage'] = stage
        if stage == 'prepared':
            digest = self._save('proposal.pt')
            self._event('proposal_persisted', proposal_sha256=digest)
        self._event('optimizer_boundary')

    def _begin_accumulation(self):
        self.optimizer.reset()
        self.records, self.phase, self.count, self.routed, self.routing = [], None, None, None, None
        self.cursor['boundary_stage'] = 'idle'
        self._begin_record()

    def _begin_record(self):
        for transport in self.terms.values():
            transport.reset()
        self.cursor.update(loss_term=None, transport_stage='idle', band_index=None)

    def _backward_record(self, row, loss_fn):
        # current_record/RNG were captured BEFORE this forward by the owner.
        # Keep every completed record, not just the inherited last-record slot.
        captured, before_rng = cpu_copy(self.current_record), cpu_copy(self.before_record_rng)
        values = {}
        for term, transport in self.terms.items():
            def selected(clock, reference, valid, key=term):
                return {'total': loss_fn(clock, reference, valid)[key]}
            gradient, loss = transport.run(self.model, row['payload'], selected)
            values[term] = dict(gradient=pack_gradient(gradient.multiply(row['scale'])), loss=loss,
                                neural_vjp_passes=len(transport.pieces))
        self.records.append(dict(id=row['id'], scale=row['scale'], terms=values,
                                 record=captured, before_record_rng=before_rng))
        return sum(v['loss'] for v in values.values())

    def _clip_accumulated(self, clip_norm):
        from experiments.observed_adjoint.transport import unpack_gradient
        self.phase, self.count = [accumulate([unpack_gradient(r['terms'][term]['gradient']) for r in self.records])
                                  for term in ('phase', 'advance')]
        self.routed, self.routing = route(self.phase, self.count)
        report = install_clipped(self.model, self.routed, clip_norm)
        self.optimizer.arm(self.count)
        return report

    def _payload(self):
        result = super()._payload()
        result['protected_update'] = dict(schema='protected-recorded-update-v1', weighted_records=self.records,
            terms={k: v.snapshot() for k, v in self.terms.items()},
            phase=pack_gradient(self.phase), count=pack_gradient(self.count), routed=pack_gradient(self.routed),
            routing=self.routing, boundary=self.optimizer.snapshot())
        return result

    def update(self, epoch, batch_index, records, *, clip_norm=1.):
        return super().update(epoch, batch_index, records, clock_loss, clip_norm=clip_norm)
