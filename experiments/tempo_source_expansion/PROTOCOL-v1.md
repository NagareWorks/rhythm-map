# Matched source-expansion design v1

2026-09-15. Design frozen before new encoder outputs or optimizer updates.
This is **not an executable fit admission**: source review, exact feature-packet
identities, capture replay, runner closure and private execution registration
must be complete before either arm starts. Do not fit on a successful subset.

## Hypothesis and unchanged boundaries

Test whether one additional natural-percussion source group improves local
tempo transfer without sacrificing existing absolute timing. This is a source
intervention, not another loss/architecture sweep. BRID is 93 recordings from
one conservative `brid-corpus-v1` group, not 93 independent works or an unseen
evaluation corpus. Preserve its CC BY 4.0 attribution in the source lock.

Reuse the existing frozen **Beat This normalized pre-head features** (512 wide,
50 Hz), 23,937-parameter, 32-channel dilated CNN tempo branch and pre-relative
separated-evidence starting export:
`1b25f43474ba780d845dccd032a8e4c150ebabc77c50f31cc975e9419c4e214a`.
This is not MusicFM, encoder fine-tuning, a new transformer, or a product option.
Retain old 20 FIT recordings / nine works, five development recordings / three
works, 15 already exposed ART variants and the six exposed local schedules in
their existing roles. The original holdout remains sealed.

## Native reference adapter

All 93 complete source recordings retain original timestamps and 1/2 positions.
For native beat times b[i], interpolate beat count only inside bracketed support:
`q(t) = i + (t-b[i])/(b[i+1]-b[i])`. At 50 Hz, cell i's target is
`-log2(50 * (q[i+1]-q[i]))`. This is an annotation-derived cell-average rate,
not a claim that microtiming is an underlying tempo change. A cell crossing a
beat boundary integrates the two rates; it does not pick whichever interval
contains its center.

Match the old reference-clock contract exactly: first beat inclusive, last beat
exclusive, no extrapolation, both endpoints required per cell, NaN outside
support. A missing reference hole invalidates adjoining cells. Never hide a
weak attack or model error by changing support. Preserve the first position
even if it is 2: q's arbitrary zero origin is **not** downbeat phase. No global
`.bpm` replication, half/double relabeling, smoothing, or change-point labels.

The planned frame count uses the pinned Rust decoded sample count and centered
Beat This mel geometry. Actual capture must reproduce this exact extent and
50 Hz origin; mismatch fails, never crops/pads references to make them fit.
Full audio, not hand-selected sections, supplies encoder context.

Freeze byte identities for beats, positions, q, frame validity and cell validity.
Do not cache/hash a NumPy log2 target as a cross-platform identity: native log2
implementations can differ in the last float64 bit. The unchanged PyTorch loss
derives its target from q and validity. No rounding, target shift or relaxed
reference-hash check is needed. Runtime-derived numeric targets are diagnostics,
not canonical packet arrays.

## Two matched arms, fixed source-group weight

Both arms take exactly 20 epochs / 100 AdamW updates. Every update shares the
same original four-record natural batch (seed 142 + epoch), the same four
replacement draws from each of 15 old pair groups (seed 8142), and the same
FIT-only initial teacher on those four old records. All 559 admitted pair
points and all 915 original point/exclusion records remain unchanged.

Each arm has **one additional natural-supervision slot per update**:

- `old-source-control`: sample an old FIT work uniformly, then a recording
  uniformly within that work, with replacement (seed 24142).
- `brid-source-primary`: use all 93 BRID IDs in a shuffled no-replacement cycle
  (seed 34142); after all 93, take seven IDs from the next shuffled cycle.

Let N be the existing work-balanced native MSE on the shared four records,
A the one-record native MSE on the additional slot, P the unchanged pair loss,
and R the unchanged teacher retention on the shared **old** four records:

`loss = 0.9*N + 0.1*A + P + R`.

The primary therefore assigns BRID one tenth of natural-supervision weight,
against nine old work groups. The control's extra slot has the same expected
old-work distribution as N. No teacher is created for BRID; no new labels enter
development, pair ranking or diagnostics. No renormalization by file count,
style, duration, native error, or acoustic-diagnostic score. Each recording's
native MSE averages its supported cells before external group weighting.

The two arms have equal update, forward-slot and pair-query budgets, **not
identical FLOPs or elapsed time**: BRID clips are shorter than old 60 s crops.
Both now have five natural forwards per update instead of the old experiment's
four. This explicit matched control is not a bit-exact replay of the previous
native-retained trajectory. Its epoch-zero model/predictions must replay the
old initial state; do not falsely require later hashes to match that old run.
No third arm, coefficient sweep or data-dependent resampling is authorized.

AdamW lr 0.0003, weight decay 0.0001, clip norm 1, deterministic FP32 model,
float64 loss arithmetic, TF32 off, two CPU threads and at most 40% GPU memory
remain unchanged. Keep 1,800 s per arm / 5,400 s outer wall-time cap. Failure or
budget exhaustion preserves artifacts and stops; no automatic fit retry.

## Selection, observation and interpretation

Reuse `local_tempo.selection` unchanged: epoch zero plus 20 complete epoch
states, old development work/record 1.05 admissibility, lowest old training-pair
MSE among admissible states, earliest exact tie. BRID scores do not select an
epoch. Persist selected and terminal states before accessing exposed schedules
or ART metrics. Keep all entered/returned update receipts, before/after weights,
moments and RNG, unclipped/clipped gradients, every epoch export and private
input hashes. Independently replay predictions, selection and NumPy AdamW with
the existing weight/moment tolerances. Neither arm may mutate the teacher.

Retain the previous exploratory gates unchanged: selected epoch > 0, identity
MAE <= 0.05, each of six changed-schedule RMSE <= 0.75 of zero-change and <= 0.8
of previous-global, slope [0.5,1.5], sign >= 0.8; development AND ART work-macro
native MSE and median BPM error <= 1.05 of initial. Report both arms on all
records and schedules, selected and final, including every failure.

A passed primary is only exploratory evidence. Claim a source-expansion benefit
only if primary passes and its selected changed-point RMSE is strictly lower
than the selected matched control on all six schedules, while neither retained
development nor ART aggregate metric exceeds 1.05 of the matched control.
If both choose epoch zero, record no retained learning. If both improve similarly,
do not attribute the gain to BRID. One seed and already exposed evaluation cannot
establish generalization; fresh independent acceptance is still required before
shipping. No rejected old weights, per-song strategies, release or default change.

## Remaining admission checks (not new research sweeps)

The all-record RMS review repeats a +25 ms peak for 0092, versus +5 ms for the
other 92. Equal-beat normalization and both time halves preserve that result.
This is not evidence authorizing a timestamp shift: percussion attack shape,
microtiming and annotation convention remain alternatives. Keep 0092 and all
other IDs in the frozen population. Before fitting, complete a source-alignment
disposition (including this case) and explicitly record any residual label
uncertainty. An actual annotation defect requires a versioned correction and
new input freeze, not an invisible shift or dropped example.

Then capture/verify frozen features for all 93, join exact frame counts and
array hashes to reference packets, verify old packet roles and no known audio
identity overlap, and register the executable runner/hash closure. A technical
preflight may check forward/gradient numerics without committing a musical
optimizer update. Do not treat these prepared references as trained weights.
