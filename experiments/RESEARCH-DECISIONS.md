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
