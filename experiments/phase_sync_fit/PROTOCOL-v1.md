# Distributed phase clock: bounded musical fit v1

Freeze this protocol, code, parent identities, population and initial-state hash
in a private plan BEFORE the first optimizer step. One audio fit plus one matched
zero-feature fit; no automatic retry, seed/epoch/loss sweep or holdout opening.
This validates a research component, not a shipping model or independent accuracy.

## What changes and what does not

Use the source-pinned 24,037-parameter PhaseSyncReadout that passed component and
complete-crop cost checks. Distributed local phase evidence changes positive cell
advance in forward inference; reported phase and BPM derive from one clock.
No truth re-anchoring, phase/chunk resets, detector-event replacement, learned
encoder update, old weight loading or per-song policy. Use its already-fixed
origin-head zero initialization and gain-logit zero initialization. The entire
registered architecture/initialization is tested; this is not an isolated causal
ablation distinguishing feedback from every other head/initialization difference.

Keep coupled-v1's circular phase plus equally averaged native log-count loss at
lags 1/25/100/200, optimizer, work weights, full-crop loop and checkpoint criterion.
Do not combine this structural change with the failed trajectory-loss variant.
The fit loop is a source-tested copy with only the model constructor changed;
generic loading, development scoring and measurement helpers remain unchanged.

## Population, optimization and selection

- Exact coupled-v1 manifest SHA256
  b4f05c10ebede84c394b0b8e06f6723b096d1f9a0a48e63024eb81d913fffd9c;
  same 40 private packets, frozen features, native labels, raw and v1 predictions.
- Fit: 20 RUBATO recordings / 9 works, existing first-60-second crops. Develop:
  5 recordings / 3 works. Diagnose only: 15 existing complete ARTBeaT recordings.
  No role changes, new audio/annotation, frozen-encoder forward or holdout access.
- Both fresh fits use seed 142 and exactly the same registered initial weights.
  AdamW lr 0.001, decay 0.0001, global gradient norm clip 1. No mixed precision,
  deterministic PyTorch, TF32 off, 2 CPU threads, one inter-op thread.
- 20 epochs / 100 updates maximum, four complete recordings per accumulated
  update, NumPy shuffle seed 142 + epoch index. Per-recording weight is
  N/(G*n_work)/actual_batch_size. No truncated backward or phase-window resets.
- 1,800 seconds per fit; incomplete epoch/update or nonfinite computation is a
  failure, not a reason to restart. Retain partial journals and failure type.
- Development total loss averages recordings within work, then works. Select its
  lowest complete-epoch value, keeping the earlier epoch on exact ties. Diagnostic
  and intervention results are evaluated only AFTER both checkpoints are fixed.

Both cohorts are previously exposed; RUBATO has expressive native beats, not
audited weak-attack/continuation/hard-change/no-rhythm region labels. ARTBeaT is
not representative training coverage or independent acceptance. Native convention
is not proof of a uniquely perceived beat level. These limitations survive a pass.

## Fixed measurements and within-checkpoint controls

Reuse exact coupled-v1 cell/phase/4-second-count measurements and all support
denominators. Report reference-only and native/raw common support. Do not trim
seams or align clocks using labels. Missing required predictions invalidate a
metric instead of shrinking its denominator. Raw is interpolated detector events,
not the shipping estimator. The independent v1 head retains its declared
trapezoidal cell-rate projection. Compare source-pinned previous coupled AND
trajectory reports too, including every-recording regressions and coverage.

After both selected checkpoints are fixed, evaluate each recording using:

1. natural input with the audio checkpoint;
2. all-zero input with that SAME audio checkpoint;
3. each channel's temporal mean with that SAME audio checkpoint;
4. a circular input shift by floor(frame_count/2), fixed by geometry, with that
   SAME audio checkpoint (not a random shuffle or searched phase correction);
5. zero input with the separately fitted matched zero-feature checkpoint.

Keep all clocks and local fields privately, and hashes/metrics publicly. Do not
inverse-roll outputs, choose an intervention as product output or fit a new epoch
from intervention scores. These controls can be out of distribution and half-roll
introduces seams. They test operational timing dependence, not causal proof that
the head understands music. Report work-macro deltas and per-recording signs.

## Necessary research-continuation gates (not release gates)

Preserve every prior nonregression requirement and add the registered timing
control. All must pass:

1. Both fits complete 20 epochs / 100 updates within 1,800 seconds, with finite
   required natural/control outputs and unchanged source/parent/population hashes.
2. Reference-only development median BPM error AND phase error improve >=10%
   over the separately fitted zero-feature control.
3. On both development and ARTBeaT common support, work-macro median/P95 tempo,
   phase and 4-second count drift are no worse than EACH of raw, independent v1,
   previous coupled and previous trajectory baselines. No favorable corpus alone.
4. In each cohort, no more than half the recordings regress versus raw on either
   median BPM error or phase error. Count strict increases; no new tolerance band.
5. On both cohorts, natural-input common-support phase error is strictly lower
   than each same-checkpoint zero/mean/half-roll control. Complete all control
   metrics and keep per-recording signs, not only a favorable mean.

A failure closes this fixed experiment; report which criteria failed without an
automatic second fit or new strategy. A pass only supports proposing independent
validation and representative-label/rhythm-availability work. Neither outcome
changes defaults, publishes weights, packages a model or invents observed beats.
