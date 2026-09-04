# Downbeat-supported beat insertion: rejected calibration screen

Date: 2026-09-04. Training-free, one fixed observation-stage experiment. No
production change, new public policy, threshold sweep, holdout inference, or
additional neural inference.

## Question and fixed rule

The adapter currently decodes beat and downbeat logits separately, then snaps
downbeats to existing beats. A strong downbeat prediction cannot create a beat
that the beat decoder omitted. Since a true downbeat is also a beat, can the
existing second output head safely recover omissions?

Before examining outcomes, the experiment fixed this candidate-space rule:

1. Preserve every raw decoded beat, including its exact timestamp.
2. Consider only existing radius-one beat candidate peaks whose same-frame
   downbeat confidence is strictly greater than 0.5 (the existing zero-logit
   decision boundary). Do not lower the beat threshold or fabricate a grid time.
3. Exclude candidates within three 50 Hz frames of a raw beat, reusing the
   decoder's local-maximum radius. For comparisons, reconstruct candidate frame
   indices and raw half-frame indices from their float32 timestamps; emit the
   original timestamps unchanged.
4. Process remaining candidates by descending downbeat confidence, then beat
   confidence, then earliest timestamp. Suppress an insertion within three
   frames of an already accepted insertion.

Decisions see observations only, not annotations, case IDs, tempo labels,
corpus identity, or scores. Truth is attached after every decision. There are
no genre exceptions, fitted coefficients, inferred meter, or alternative rules
selected after inspecting results.

This is **not** decoding local maxima of the dense downbeat head, a union of two
dense peak sequences, or independent-model consensus. The cache contains the
downbeat value at each beat candidate, not all dense downbeat frames. This
screen tests exactly the rule above, not every possible use of downbeats.

## Scope and result

All 25 frozen RUBATO calibration recordings and 15 shipping-preprocessing
ARTBeaT cases were used. The separate ARTBeaT reference-resampler probe was
excluded. Existing raw/truth matching pairs replayed exactly on all 40 cases;
the chronological one-to-one matching and 70 ms tolerance were unchanged.
No recording, observation cache, or historical report was rewritten.

| Raw beat observations | RUBATO (25) | ARTBeaT (15) |
| --- | ---: | ---: |
| Baseline mean beat F1 | 0.5214049901 | 0.8079605744 |
| Candidate mean beat F1 | 0.5205976919 | 0.8103401597 |
| Inserted events | 336 | 5 |
| Previously missed truth beats recovered | 62 | 3 |
| Previously matched truth beats lost | 0 | 0 |
| Additional unmatched events | 274 | 2 |
| Cases with improved / regressed F1 | 7 / 16 | 3 / 1 |
| Cases with decreased precision | 19 | 2 |
| Cases with increased median / P95 timing error | 4 / 6 | 1 / 0 |

The RUBATO baseline here is the **raw observation sequence**, not the final
estimator baseline of 0.5212633988. These are not end-to-end product scores.
Insertions were not run through the BPM/section estimator: the raw-observation
screen already failed. No new downbeat, tempo, section, or change-point accuracy
claim follows from this experiment.

ARTBeaT's aggregate improvement hides two relevant failures. `180-to-120`
recovers one beat but adds one unmatched event, reducing precision despite a
higher F1. `piano-rubato` adds an unmatched event without recovering a beat and
reduces F1. Both fail the no-regression intent. On RUBATO, only 62 of 336 added
events produce additional matches. Keeping all original timestamps therefore
does not make automatic insertion safe.

Truth identities were compared, not just match counts. No previously covered
truth identity was lost in this run. Timing quantiles still depend on which
events are matched; an increased quantile does not mean an original timestamp
was moved. Quantiles use the evaluator's `ceil((n - 1) * q)` index.

## Decision and reproducibility

Reject this rule; do not add another inference option or retain it as a fallback
policy. The downbeat head supplies some complementary signal, but a high value
is not a verified musical downbeat and does not reliably resolve the beat level.
This does not establish that all joint beat/downbeat decoding is impossible.
Further work needs a distinct evidence or inference hypothesis, not extra
guards fitted to these insertions. The held-out recordings remain sealed.

The immutable private script contains six authored tests for the strict gate,
timestamp preservation, frame-boundary exclusion, competition/tie ordering,
invalid confidence rejection, and chronological matching/quantile semantics.
All pass. Dense inputs and timestamp-level output stay outside Git. Identities:

- Frozen rule script SHA-256:
  `5dae877bbb1776d3d0f1ab57d10dc5498d1955039c4a754a2def1ffeaa3c5ab6`.
- Private result SHA-256:
  `249636879ee11b3f899b28976ba6a7a66dab32fec5d0b8d1b9ed59ae13a3b59d`.
- RUBATO evidence SHA-256:
  `ce5e678276888a0e430c004444dce4b27f0cfac0761767736abee2ec3fc05937`.
- ARTBeaT evidence SHA-256:
  `3f1ba43fd4f373579727a48668d8de8e00166523d2d1141e072bc3471a71ab3e`.

These are the same observation inputs used by the
[active-region replay](active-region-rust-v1.md); their distinct provenance is
preserved rather than relabeled as a new common inference contract.
