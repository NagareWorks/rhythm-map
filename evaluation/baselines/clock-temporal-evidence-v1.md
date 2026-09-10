# Fixed checkpoints: temporal phase evidence survives in the direct head

Date: 2026-09-10. Decision: **advance a distributed-phase coherent-clock design,
not another scalar-loss swap; no model is admitted by this diagnostic**.

The [runner and control contract](../../experiments/clock_evidence/README.md),
[execution](../../experiments/clock_evidence/execution-v1.json) and
[every-recording report](../../experiments/clock_evidence/results-v1.json)
retain the read-only experiment. All three previous fit reports remain frozen.

## What the previous zero control did and did not establish

The previous comparisons used separately trained audio-input and zero-input
models. They test the whole training treatment, including different learned
weights. Failure of that gate is NOT proof that an audio checkpoint ignores its
input or that the frozen representation contains no timing information.

This run fixes each selected audio checkpoint and intervenes on input only:
natural features, all zeros, the recording's channel-wise temporal mean, and
one half-recording circular shift. The shift is floor(frame count / 2), chosen
from geometry without looking at reference beats or scores. The original fitted
zero-input checkpoints are separately replayed, not confused with zeroing the
input to an audio-trained model. No checkpoint, offset or favorable subset is
selected after looking at these results.

The same 20 RUBATO fit, 5 development and 15 ARTBeaT diagnostic recordings are
used. Features/weights/packets/reports are hash-pinned. There is no optimizer,
encoder inference, new audio, annotation acquisition, holdout access or export.
The CPU run completed 480 audio-checkpoint and 120 fitted-zero complete-sequence
evaluations in 33.441 seconds, excluding preparation, verification and CI.
These are not 600 independent recordings or an end-to-end audio benchmark.

## Main result: dense phase depends on correct temporal alignment

Phase error is mean absolute circular error in cycles; lower is better.
All musical scores retain identical native/raw common-support denominators and
work-balanced recording summaries. No oracle shift or seam trimming is applied
to the musical clocks.

| Cohort / checkpoint | Natural | Same weights, zero | Temporal mean | Half roll | Separately fitted zero |
| --- | ---: | ---: | ---: | ---: | ---: |
| development / direct | 0.1393 | 0.2497 | 0.2500 | 0.2384 | 0.2503 |
| development / coupled | 0.2545 | 0.2443 | 0.2507 | 0.2475 | 0.2419 |
| development / trajectory | 0.2687 | 0.2498 | 0.2367 | 0.2485 | 0.2402 |
| ARTBeaT / direct | 0.1432 | 0.2501 | 0.2494 | 0.2543 | 0.2499 |
| ARTBeaT / coupled | 0.2475 | 0.2495 | 0.2548 | 0.2638 | 0.2522 |
| ARTBeaT / trajectory | 0.2475 | 0.2486 | 0.2291 | 0.2489 | 0.2455 |

The direct head's natural input has lower phase error than EACH of the four
controls on all 5/5 development and 15/15 ARTBeaT recordings. The consistent
temporal-mean/roll response is evidence that this checkpoint uses frame-aligned
information for phase, not merely a track-level average. Around 0.25 cycles is
also the expected absolute error of a uniformly random phase, but these controls
are not claimed to be random or statistically independent.

Both integrated heads still depend on input: their tempo and clock predictions
change under interventions. Their phase alignment benefit is not similarly
reliable. In particular, the trajectory model's temporal-mean input improves
development phase from 0.2687 to 0.2367 and ARTBeaT from 0.2475 to 0.2291. This
is a failure-mode clue, NOT a proposal to remove time variation in production.

## Tempo is not thereby solved

The same cell-average BPM error convention as the coupled/trajectory reports is
used. The direct head retains its explicit trapezoidal cell-rate projection;
these are not its old instantaneous-target scores.

| Cohort / checkpoint | Natural median BPM error % | Temporal mean | Half roll |
| --- | ---: | ---: | ---: |
| development / direct | 37.2602 | 36.4397 | 39.4166 |
| development / coupled | 33.9147 | 34.9630 | 35.2977 |
| development / trajectory | 45.3405 | 36.4984 | 44.0832 |
| ARTBeaT / direct | 31.0812 | 31.2136 | 31.7361 |
| ARTBeaT / coupled | 38.0601 | 39.2070 | 40.5995 |
| ARTBeaT / trajectory | 28.1967 | 24.5929 | 34.8459 |

Good phase intervention response is not good tempo output. The direct head still
regresses against raw on ARTBeaT: median BPM error 31.0812% versus 14.4462%, phase
0.1432 versus 0.1078, and 4-second count drift 2.5825 versus 1.7287 native beats.
Its backwards-phase problem remains. P95 and all count/control/fit-case results
are retained in JSON; no favorable phase subset erases those regressions.

## A real structural difference, not a proven single cause

The independent CNN outputs log period and a phase vector at every frame. The
coupled CNN outputs a period field and a phase field, but its clock consumes
only the latter's frame-zero value:

```text
direct:     features[t] -> local period[t], local phase vector[t]
integrated: features[t] -> local period[t] -> cumulative positive advancement
            features near frame 0 -> one origin ---------------------------^
```

The new forward/backward tests verify that later phase-field values have no
effect on the integrated clock and zero gradient through it. Later features DO
affect local period and the cumulative clock, so it would be wrong to say the
model ignores all late audio. Frozen upstream features already contain context;
the readout's 126-frame halo is not the raw-audio receptive field.

This, together with the retained direct phase signal, motivates placing dense
phase evidence in the clock's INFERENCE path. It does not prove the only reason
the previous fits failed: losses, initialization, optimization, limited training
coverage and native beat-level ambiguity can all matter. No encoder absence
claim, Transformer necessity claim or general CNN failure claim follows.

## Next implementation contract, before another musical fit

One workstream: a phase-observation plus positive-rate clock synchronizer. Reuse
the direct head's local phase representation and test its information flow; do
not combine predictions from whichever old checkpoint happens to win a song.

The next implementation must:

1. consume phase observations THROUGHOUT the recording along with rate evidence,
   and return one continuous, strictly increasing cumulative beat coordinate;
2. derive reported phase and tempo from that SAME coordinate, with no independent
   inconsistent product head or invented observed beats;
3. allow a late phase observation to influence alignment locally; the effect must
   exist in forward and backward tests, not just in an auxiliary training loss;
4. distinguish a circular phase residual from an unknown integer beat advance:
   phase alone cannot choose half/double time or recover how many beats a gap
   contains. Rate/count evidence must remain in the model and its supervision;
5. avoid re-anchoring to truth, reference-mask input, chunk resets or fabricated
   observations in rests. Low phase-vector amplitude is not calibrated rhythm
   confidence and cannot silently become an availability label;
6. pass authored constant/drifting/missed-observation/real-change/gap and chunk
   consistency tests, finite-gradient tests, and a measured complete-crop cost
   check before spending another musical training budget.

Then pre-register a bounded comparison with the SAME diagnostic roles and failed
baselines, matched fitted-zero AND same-checkpoint temporal controls, native
tempo/count coverage and no relaxation of the old nonregression gate. A decoder
that preserves phase while losing tempo is still not admitted. Independent data,
perceived level, and representative weak-attack/change/no-rhythm supervision are
separate unsolved requirements, not waived by this narrower design decision.

This report closes the three-checkpoint diagnostic; it is not a fourth fitted
model or a reason to rerun arbitrary perturbations, seeds or loss coefficients.

## Numerical and interpretation boundaries

All natural predictions replay their pinned predecessors (maximum absolute
difference 2.833e-6, tolerance atol 1e-5 / rtol 2e-5). Small CPU/CUDA differences
in re-derived BPM metrics do not rewrite old reports. On the seam-safe LOCAL
field interior, rolled fields equal rolled natural fields exactly in this run.
This checks padding/ownership and shift direction, not musical correctness.
All field-response variants use that same interior; clock scores never do,
because an earlier integration error can persist after a seam.

Intervened inputs may be out of distribution; a rolled frozen vector is not a
re-encoded real shifted audio context. These are post-fit observations on exposed
data, not causal training-mechanism proof, a significance test, independent
acceptance or a calibrated confidence estimate. No production default, user
parameter, package or release changed.

Report SHA256: b40d7c5af1f8b492256e0a4bdb52e0004eff351df0c8c0dde726c2e61e529e38.
Execution SHA256: 69511a2cd2b8549402b05fad4f4d6e043642ed8c6c2336add8b3a49a942790f9.
