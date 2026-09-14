# Local-tempo supervision and retention v1

Freeze code and this contract before new audio preparation or fitting. Preserve
all prior results, weights, nominal eligibility flags, masks and source roles.
No product option, larger model, encoder fine-tuning, phase update or release.
This is a new experiment, not a retry or revised verdict for relative_tempo v1.

## Training and newly reserved schedules

Reuse all 18 actual-audio feature captures from relative_tempo v1, byte-pinned
to its result. All 559 admitted grid points across its 15 source/profile groups
now supply pair supervision. The previously exposed local_slow_fast and
local_fast_slow points are explicitly **training-exposed**, not fresh validation.
Keep every one of the old 915 grid points and exclusion reasons in the ledger.
No beat-query points or interpolation of support into a dense mask.

Retain the original 20 RUBATO fit recordings / 9 works, 5 development recordings
/ 3 different works, and 15 exposed ART diagnostic variants / 1 source. Neither
development nor ART contributes gradients. Development now selects checkpoints;
ART never selects. All claims remain exploratory; the original holdout is sealed.

Create exactly two new schedules on the same three retained 60-second source
prefixes, at the same 0/20/40/60 source boundaries:

- fresh_slow_fast_normal: rates (.8, 1.25, 1.)
- fresh_fast_slow_normal: rates (1.25, .8, 1.)

These six audio files are new **schedule diagnostics**, not new songs, performers,
rates or renderer families. They cannot prove source/domain generalization or
absence of renderer-artifact shortcuts. Encode them and inspect learned outputs
only after BOTH arms' selected AND terminal weight files have been frozen.
Preparation may inspect PCM correspondence, never model predictions.

Use the same byte-pinned external FFmpeg/Rubber Band renderer, three independent
pieces concatenated with no timing repair. First pass the existing authored
click, click+tone and tone gates for both new schedules (six witnesses). Require
exact expected segment sample counts for musical PCM. Reuse unchanged 512/2048
spectral alignment, +120 ms actual-PCM delay control and wrong-map negative
control. Measure every .5,1.5,... grid point, retaining all 366 points. Admit only
supported interior points with source/output CNN clearance >=2.56 s. Require
the old wrong-map gate in every pair and at least 10 admitted changed points per
pair. A failed preparation gate stops without fitting, no threshold rescue.
Do not certify dense annotations, seam localization or all context samples.

## Matched optimization and retained output

Both arms start from the same pre-relative separated-natural tempo branch:
23,937 parameters, 32-channel kernel-5 CNN, dilations 1/2/4/8/16/32. Reuse the
unchanged native-cell supervision and interpolated scalar-field pair query.
Keep old relative v1 global-only final weights as a frozen descriptive reference;
do not retrain them. Comparisons against that reference also change checkpoint
policy and must NOT isolate a pure local-data effect.

Arms: local-only and local-retained. Each completes exactly 20 epochs / 100
AdamW steps, LR .0003, weight decay .0001, norm cap 1. Seed 142 for natural
record order, independent seed 8142 for pair sampling. Four natural fit records
per update with unchanged expected equal-work weighting. Draw four admitted
points with replacement from EACH of the 15 old groups; average group losses
equally, pair coefficient 1. Both arms use identical draws and natural batches.

Both use native-label MSE + pair MSE. Only local-retained adds coefficient-1
MSE to the initial frozen teacher's outgoing-cell log2-period on the SAME valid
natural fit cells, with the same natural-work weights. Teacher targets are
detached and never updated. They are a movement penalty, not new ground truth;
the real native labels remain supervised. This may also preserve old mistakes.
No teacher term, gradient, or optimizer input on development/ART/fresh schedules.

## Fixed development-conditioned selection

At epoch 0 and each complete epoch, evaluate the five natural development
records and the 559 **training** pair points. A checkpoint is admissible only
when EACH development work's mean native MSE and BPM median error is <=1.05
times the initial value, and no more than two of five recordings exceed 1.05
times their initial BPM median error. Select the admissible checkpoint with
lowest equally weighted 15-group training-pair MSE; strict comparison keeps
earlier ties. Epoch 0 is an explicit no-update fallback. Always finish all 100
steps; retain selected and terminal weights and full selection history.

No new-schedule/ART score enters selection. A selected epoch 0 cannot count as
learning success merely because it preserved the teacher. Both arms have the
same selection policy; do not select the winning arm after observing outcomes.
The primary candidate is local-retained. Report both candidates and both final
states to expose terminal regressions hidden by checkpoint selection.

## Fixed outcome and budget

Use initial, frozen previous-global, both selected and both terminal models on
all 40 natural records and all old/new sparse pair points. Add zero-input
queries for the selected primary. Use changed-only RMSE/slope/sign for changes,
retain identity MAE and all denominators. Fresh primary gate requires, in EACH
of six new pairs: RMSE <=.75 * zero-change RMSE, <=.8 * previous-global RMSE,
slope in [.5,1.5] and correct sign >=.8. Old processed identities must each keep
mean absolute log2 difference <=.05. Require primary selected epoch >0.

Natural primary retention: work-macro MSE AND BPM median error <=1.05 * initial
on development AND ART; report every recording too. Development is selection-
exposed, not independent confirmation. Overall exploratory gate additionally
requires complete preparation/execution, fresh transfer and identity gates.
Even a pass cannot promote defaults or prove perceptual/local-onset accuracy.

FP32 deterministic Torch, TF32 off, GPU allocation cap 40%; CPU preparation
900 seconds; fitting/capture/evaluation 5400-second outer watchdog, 1800 seconds
per arm. No retries or post-outcome parameter adjustment. Save source/runtime
pins, all entered/returned update receipts, teacher identities, epoch weights,
selection freeze, actual fresh PCM/features and final per-frame predictions in
fresh private directories. Publish metrics, provenance and exclusion ledgers,
not audio/weights. Keep original natural packets and old experiments unchanged.
