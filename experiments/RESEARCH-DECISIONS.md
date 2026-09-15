# Closed research directions and reopening conditions

This is a navigation index, not a replacement for frozen protocols/results.
Failed runs, excluded cases and unfavorable corpora remain in their original
reports. Chronological earlier algorithm/model work is preserved in
[Roadmap](../ROADMAP.md#phase-1-timing-accuracy-one-shipping-estimator).
Do not interpret a useful component result as an accepted complete model.

| Direction | Recorded outcome | Do not repeat without new evidence |
| --- | --- | --- |
| Direct phase/period readout | [Fixed gate failed; phase may reverse](../evaluation/baselines/direct-clock-learning-v1.md) | More epochs or favorable-corpus promotion is not a fix. |
| Coherent integrated clock | [Positive advancement does not establish alignment](../evaluation/baselines/coupled-clock-learning-v1.md) | A coherence invariant is not musical accuracy. |
| Native-trajectory loss | [ART improves while development regresses](../evaluation/baselines/trajectory-clock-learning-v1.md) | Do not alternate scalar losses using the same exposed scores. |
| Phase feedback fit | [Nonfinite adjoint; original failed state was unsaved](phase_sync_fit/README.md) | Numerical representation fixes do not prove conditioning or accuracy; retain full failure/update packets in new runs. |
| Global paired-speed continuation | [Learns trained relations; native retention and reliable local transfer fail](relative_tempo/README.md) | No longer-training rescue or blame solely on the added pair term. |
| Local pairs plus FIT-only teacher | [Both arms retain only epoch zero](local_tempo/README.md) | Do not sweep the teacher coefficient or weaken development gates. |
| Frozen gradient attribution | [Native and local-pair conflicts both visible](tempo_conflict/README.md) | Raw gradient cosine is not the actual AdamW displacement or a causal percentage. |
| Native offset removal | [Less development regression, but all learned epochs still rejected; ART worsens](tempo_variation/README.md) | No offset-weight sweep, oracle output centering, or shipping a rejected terminal model. |

Common boundaries:

- Keep the original holdout sealed. Development is repeatedly selection/exploration
  exposed; the old six schedule diagnostics are also exposed. Neither is fresh
  independent acceptance, and public role labels must reflect that.
- Keep rejected runs frozen and the one-call product unchanged. Experimental
  arms are not public strategies or user tuning parameters.
- A new attempt needs a distinct falsifiable hypothesis, fixed comparisons,
  failure-preserving artifacts and unchanged absolute-timing checks.
- The next proposed direction is broader **work-disjoint training-source
  coverage**, with provenance/license/split checks first. Do not move development
  or holdout examples into fit, or assume a larger encoder is required.

Source intake has now begun: the [BRID annotation/directory inventory](../evaluation/baselines/training-source-intake-v1.md)
finds 93 CC BY 4.0 acoustic-mixture candidates, not 367 independently labeled
works. No audio, fit admission or new evaluation claim yet. The next bounded
step is the preregistered four-style audio/annotation smoke check, followed by
full-pool admission and a matched source-expansion comparison if it passes.

Follow-up: the [preregistered four-style audio smoke](../evaluation/baselines/brid-audio-intake-v1.md)
is acquired and structurally verified (123.24 seconds, 209 beats). Full 93-mixture
acquisition was attempted but stopped at source HTTP 429 on 0005; completed
partial files and the failure are retained. This is a transport interruption,
not evidence against source coverage or permission to fit on the smoke alone.
Musical annotation review, full-pool lock and matched expansion training remain
pending. Old rejected model directions and all product defaults stay frozen.

The [full BRID acquisition and structural audit](../evaluation/baselines/brid-audio-pool-v1.md)
now completes all 93 mixtures (44:23.981, 4,342 beats) with a polite single-range
resume path. No new rate-limit failure or structural rejection occurred; the
earlier interrupted run is not rewritten. The next boundary is source-reference
alignment/adaptation and a fixed source-expansion comparison, not another loss
sweep. No encoder, fit, holdout access or default change accompanied acquisition.

The [reference adapter and alignment review](../evaluation/baselines/brid-reference-v1.md)
now retain all 93 recordings with 126,270 supported native tempo cells, without
timestamp shifts or extrapolation. The 0092 acoustic-offset anomaly survives
equal-beat and temporal-split review; it is recorded, not silently corrected.
The [source-expansion design](tempo_source_expansion/PROTOCOL-v1.md) freezes a
matched old-source/BRID extra slot and one conservative BRID group weight.
Source disposition, frozen feature packets and executable-runner registration
remain before fitting. Prepared references are not an accuracy breakthrough.

The [full-record frozen feature capture](../evaluation/baselines/brid-features-v1.md)
now completes source disposition and all 93 input/feature packets. Keep 0092
with explicit label uncertainty, not an onset-aligned repair. The T4 capture
repeats hidden tensors exactly and leaves encoder weights unchanged; no tempo
optimizer ran. Stop source-offset probing here. Next is old-input replay and
the executable matched training/replay runner, not another loss or policy sweep.
