# Fixed-checkpoint temporal evidence diagnostic

Status, 2026-09-10: complete, 600 sequence evaluations, no new fit. See the
[musical result and next implementation contract](../../evaluation/baselines/clock-temporal-evidence-v1.md),
[execution](execution-v1.json) and [all-case report](results-v1.json). The direct
head retains temporal phase evidence; none of the old models is promoted.

This is a read-only follow-up to three closed learning experiments. It does not
fit, select an epoch, replace features, search a lag, admit a model or modify a
prior report. Existing exposed 20/5 RUBATO and 15 ARTBeaT recordings are reused;
no encoder inference, new labels/audio or holdout access occurs.

## Question and fixed comparisons

A fitted zero-input model and an audio-input model have DIFFERENT weights.
Their comparison tests the full training treatment, not within-checkpoint input
dependence. For each of the three selected audio checkpoints, evaluate:

1. natural features, verified against retained predictions;
2. zeros, keeping that same checkpoint;
3. each recording's channel-wise time mean, removing frame variation;
4. a circular shift of floor(frame count / 2), preserving every feature vector.

Also replay the original separately fitted zero-input checkpoint. None of the
interventions is a real alternate music recording; frozen upstream features can
contain contextual information. Zero and mean inputs may be out of distribution,
and roll introduces seams. These controls can describe dependence on feature
timing but cannot prove a training failure's cause or independence/generalization.

All musical scores use the identical native/raw common support, the previous
cell-rate conventions, and work-balanced summaries. No trim or oracle alignment
improves a musical score. Missing values invalidate required metrics rather than
dropping difficult points. Fixed paired differences are descriptive, not a new
promotion gate. The direct head retains its trapezoidal cell projection; do not
compare these tempo/count scores with its original instantaneous-target report.

Only LOCAL FIELD response/equivariance uses a seam-safe interior: both destination
and source frames must retain their entire 126-frame readout halo. All response
variants use that same interior. A coupled CLOCK can still carry early rate or
origin changes past those seams, so this trim does not isolate global clock
effects. Verify rolled fields equal rolled natural fields there; do not inverse
roll predictions for scoring.

## Mechanistic distinction to test

The independent head predicts a local phase vector at every frame but can move
backwards. Both integrated heads use the phase channel only at frame zero;
later phase-channel fields have exactly zero gradient through the clock and do
not enter its forward output. Later features can still change period and hence
the integrated clock: this is NOT an input-disconnection proof. The early frozen
features themselves can contain upstream context; the 126-frame readout halo is
not a claim that only the first seconds of raw audio affect its origin.

If dense phase responds to proper temporal alignment while integrated phase does
not reliably improve, a next architecture should test distributed phase evidence
inside one coherent clock, rather than train another global-offset correction or
rotate scalar losses. This diagnostic alone does not authorize/promote that model
or prove that explicit feedback is the only remedy. Representative continuation,
change, weak-attack and unavailable-rhythm supervision remains unresolved.

## Reproduction

`python -m experiments.clock_evidence.run --inputs PRIVATE_PINNED_PACKETS
--direct-results PRIVATE_DIRECT_RUN --coupled-results PRIVATE_COUPLED_RUN
--trajectory-results PRIVATE_TRAJECTORY_RUN --output NEW_PRIVATE_DIRECTORY`

The CPU runner checks all input/parent-report/weight identities, creates an
exclusive output directory, records source hashes before replay, and saves every
intervened field and prediction privately. It processes one recording at a time.
Report JSON is shareable; inputs, audio/features, weights and private paths are not.
No optimizer is constructed. Do not repeat the old training runners to reproduce
this diagnostic.
