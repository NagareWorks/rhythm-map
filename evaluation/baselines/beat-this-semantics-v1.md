# Beat This score semantics: not an absence likelihood

## Decision

Do not connect raw logits, sigmoid scores, or weight-offset scores to the
[three-density interface](presence-likelihood-v1.md). The pinned backend does
not provide that contract. This closes the direct-conversion route, not the
training-free approach: no default changes, extra user settings, neural training,
fresh inference, experimental decoder replay, or holdout evaluation occurred.

## Verified source and checkpoint

The [source record](../parity/beat-this-semantics-source-v1.json) pins six Git
objects at the existing reference revision and the existing `final0` SHA-256.
The verifier checks bytes before `torch.load(weights_only=True)`; only selected
hyperparameters leave the private environment. It never instantiates the model.

Actual checkpoint values are 50 Hz, `shift_tolerant_weighted_bce`, positive
weights **19 for beat and 86 for downbeat**. `sum_head` is **absent**, not false
or a checkpoint-provided true. The pinned loader filters available metadata;
the pinned `BeatThis` constructor consequently supplies `sum_head=True`.
This establishes current loaded behavior, not an unknown historical training
revision. See the pinned [loader](https://github.com/CPJKU/beat_this/blob/b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c/beat_this/inference.py).

Training targets round annotation times to frames. The beat target includes
downbeats; the downbeat target selects annotation value 1. Missing downbeat
annotations are masked out, not trained as negatives. These are metrical
annotations, not labels of audible drum hits, energy accents, or acoustic
silence. Positive weights are rounded frame/count ratios with an excluded
neighborhood; they are not exported class priors. Crop sampling, augmentation,
masking, and oversampling further matter. See pinned
[targets and weight calculation](https://github.com/CPJKU/beat_this/blob/b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c/beat_this/dataset/dataset.py)
and [training setup](https://github.com/CPJKU/beat_this/blob/b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c/launch_scripts/train.py).

## Why exact-frame low scores are not absence evidence

For tolerance 3, a positive target uses the largest prediction within three
frames (60 ms at 50 Hz). Negative loss centers within six frames of a positive
are ignored; six edge centers on each side are cropped. The negative loss also
acts on pooled predictions. The independent scalar equation is checked against
the pinned [upstream loss](https://github.com/CPJKU/beat_this/blob/b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c/beat_this/model/loss.py).

In a 41-frame control, target frame 20 has one +4 prediction displaced by any
offset from -3 to +3; every other prediction is -4. All seven placements have
the same loss. At offset +1, the exact target has logit -4 and gradient zero.
That is an objective-level counterexample, not just a failed learned prediction.
It does not imply all low scores are uninformative; it rules out their automatic
interpretation as exact-frame absence probabilities.

## Why two heads do not form three classes

The pinned [SumHead](https://github.com/CPJKU/beat_this/blob/b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c/beat_this/model/beat_tracker.py)
emits `b=u+v, d=v`. It neither outputs an independent plain-beat probability
nor guarantees `sigmoid(d) <= sigmoid(b)`. For `u=-4, v=2`, subtracting the
two sigmoid outputs gives a negative supposed plain-beat mass. Clipping and
renormalizing would hide this defect, not establish calibration. Likewise,
adding both exported logits double-counts `v` algebraically; it is not justified
by independent beat/accent evidence.

For ordinary pointwise weighted BCE at its population optimum only,
`z=logit(P(Y=1|x))+log(w)`. Subtracting `log(w)` undoes that weighting under those
assumptions. It is not an inverse for the actual pooled, masked objective.
Even a calibrated posterior needs the appropriate reference class priors:

```
f_c(x) / f_0(x) = [P(c|x) / P(0|x)] * [pi_0 / pi_c]
```

This recovers a likelihood ratio, not an absolute density. Normalizing over
classes at a fixed observation does not normalize over observations per class.
A non-symmetric finite-channel test verifies the distinction. Adjacent neural
outputs and overlapping pooled windows are also dependent; a product across
frames needs an explicit observation model or a declared approximation.

## Complete frozen-head check

All 15 ARTBeaT and 25 RUBATO calibration captures pass the existing identity,
complete-cohort and default-event reconstruction gates. No timestamps or dense
arrays are published. The [report](../parity/beat-this-semantics-v1.json) retains
every track and denominator; the comparison uses logits to avoid saturation.

| Diagnostic | ARTBeaT | RUBATO |
| --- | ---: | ---: |
| Frames per head | 12,328 | 324,515 |
| `d > b` | 1,435 | 60,040 |
| `d-log(86) > b-log(19)` | 56 | 2,758 |
| Beat logit <= 0 | 11,630 | 288,388 |

These are algebraic incompatibility counts, **not** false downbeats, absence
labels, beat errors or accuracy improvements. Even the inapplicable pointwise
weight offset leaves invalid nested masses. Large nonpositive-frame counts
cannot measure missing music. Earlier ideal-template win rates and candidate
AUCs rank selected positions; neither calibrates a shared-frame density.

## Next bounded gate

Define a temporally tolerant **metrical-event observation contract** before
choosing its likelihood. Keep latent clock ticks, annotated beat/downbeat
events, audible attacks, detector peaks, and unavailable data separate. In
particular, a downbeat is a bar position, not necessarily a stronger sound;
ARTBeaT's beat-only truth must not become all-negative downbeat supervision.

The next audit should reuse all existing calibration captures to inventory
joint-head evidence in the fixed training-tolerance neighborhood, including
weak/empty constant-tempo beats and genuine changes. Labels must distinguish
annotated metrical events, annotation-relative off-grid controls, and unknown
acoustic presence. Half-beat controls are not silence labels. Freeze extraction
and coverage rules before inspecting results; retain all tracks and missing
annotation denominators. Do not fit a mapping, select a threshold, or multiply
overlapping window scores as independent evidence in that audit.

Only a justified adapter passing matched omission/constant/change controls can
advance to decoder replay. This audit neither accepts an adapter nor proves
that neural retraining is necessary. Public one-call/zero-tuning behavior stays
unchanged. Reproduction and the seven CI controls are in the
[parity guide](../parity/README.md#backend-score-semantics).
