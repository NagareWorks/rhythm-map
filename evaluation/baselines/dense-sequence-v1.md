# Frozen full-frame renewal-clock experiment v1

## Pre-evaluation specification

One evaluation-only candidate, fixed before running authored controls or the
40-case calibration comparison. No parameter sweep, case routing, training,
holdout access, new production option or promotion is part of this experiment.

Input is only complete 50 Hz beat/downbeat logits and optional explicit frame
availability. Selected beats, candidates, baseline analysis, filenames and
truth cannot enter the decoder function. Retained observations remain unchanged;
output ticks are **inferred clock positions**, never relabeled detections.

The state is a beat frame, preceding integer period (10 through 75 frames,
40 through 300 BPM), bar phase and visible/missing observation component. The
period range is the integer-frame realization of the earlier 40--320 BPM
experimental range, not a claim to cover all music. Six fixed whole-component
meter hypotheses (2 through 7 beats per bar) compete by the same objective;
no truth chooses the hypothesis and meter changes are not modeled in v1.

For each proposed interval ending at frame t with period p:

- Compute local maxima in radius `clamp(floor(p/16), 1, 2)` frames around t
  and the preceding interval's quarter, half and three-quarter positions.
  Round the quarter centers by `(p*q+2)/4` with integer division. Clip windows
  to actual component boundaries, without padding or borrowing missing frames.
- Pulse contrast is the pulse endpoint maximum minus the mean of those three
  off-phase maxima. Add the larger of `log(sigmoid(peak))` (visible component)
  and `log(0.1) + log(sigmoid(-peak))` (missing component).
- At bar phase zero only, add the same endpoint-minus-quarter contrast from
  the downbeat head. Flat downbeat evidence contributes zero, not a fabricated
  bar observation. Meter, score and mixture components are uncalibrated model
  hypotheses, not confidence or verified musical labels.
- Subtract `log(100) * abs(log2(p / previous_p))` on every period transition.
  The fixed odds encode a missing-observation prior of 1:10 and an octave
  transition penalty of 100:1. They are reference assumptions, not learned or
  calibrated probabilities. Relative contrasts themselves are not probabilities.

Dynamic programming optimizes all possible frame positions, without peak
snapping or a beam. An exact two-pass L1 distance transform on log periods
replaces the quadratic predecessor search. Unit tests compare it with exhaustive
search, including deterministic lower-period tie breaking. Start phase is free
within the first period (zero initial score); the final tick's next period must
reach the component end. No inferred ticks are added beyond captured frames.
First/last partial intervals must be reported separately from supported spans.

Explicit unavailable spans split independent components. A component shorter
than 20 frames or with both heads exactly constant produces no clock. This is
an exact absence-of-variation guard, not a general silence detector. A neural
score in a quiet recording is not an independently calibrated availability
probability. Counts of positive pulse windows, mixture-missing ticks, weak
contrast and period-boundary states remain visible in diagnostic summaries.

## Acceptance fixed before observing results

Authored output-space controls cover constant 120 BPM and 120/60/120,
120/90/120 and 120/240/120, with eight-second sections. Use intact, weak central
alternating, erased central alternating, four erased central and eight erased
tail pulses, each with retained or flat downbeat heads (40 variants). Strong
pulses peak at +8 over background -8, with linear width four frames; weak peaks
are -2. Include an all-flat control and an explicit central unavailable span.
These are detector-output controls, not real audio or neural inference results.

Retain an identical-input witness: central alternating erasure at constant
120 BPM versus intact 120/60/120 with flat downbeats. An algorithm must return
the same result on identical heads; it cannot be required to recover both
incompatible truths. Downbeat-preserved and weak-evidence controls test the
additional information independently. Do not revise the candidate if it fails.

Then compare all 15 ARTBeaT and 25 RUBATO calibration recordings, verifying
private capture hashes, complete cohort/source identity, raw replay and current
default analysis before scoring. Report the frozen raw-event baseline and the
current primary-analysis baseline separately. Use identical truth-interval
midpoint tempo queries for every method, retaining unknown/prior-only coverage.
Compare beat counts, precision/recall/F1, median/P95 timestamp errors and exact
matched-truth identities; aggregate improvements cannot erase per-track losses.
For change-annotated cases, report before/after tempo and boundary localization,
not a jump count interpreted as correctness on unannotated expressive music.

A failed control, erased real change, loss of matched truth identities, timing
regression or coverage loss blocks promotion. This tests a particular sequence
model and objective, not the impossibility of training-free decoding. Keep the
[training decision gate](../../docs/TRAINING-DECISION.md) and product defaults.

## Measured result: rejected, not a product strategy

The [frozen aggregate report](../parity/dense-sequence-v1.json) contains all
42 authored controls and all 40 calibration recordings. Decoder, runner, audit
and unchanged default-estimator fingerprints are recorded there. Complete
capture/source hashes and raw observations were verified; primary beat metrics
replayed the frozen selected baseline within `1e-9`. This last comparison is
specifically beat-score replay, not an independent replay of every tempo metric.
The actual default analysis is run separately on the same retained observations.
No neural inference, training, holdout access or production change was needed.

| Calibration cohort | Primary mean beat F1 | Inferred-clock mean F1 | Tracks with a regression | Primary truth matches lost / recovered | Candidate false positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARTBeaT (15) | 0.80796 | 0.79654 | 15/15 | 5 / 79 | 168 |
| RUBATO (25) | 0.52126 | 0.46166 | 25/25 | 308 / 494 | 7,374 |

F1 alone improves on seven ARTBeaT and five RUBATO recordings, but every track
fails at least one joint regression check. All 25 RUBATO recordings lose at
least one previously matched truth identity. Recoveries do not cancel those
losses: RUBATO predicted positions also grow from 9,261 to 11,770. These are
inferred-clock evaluation metrics, not an improvement delivered to consumers.

On the shared truth-interval midpoint queries, the mean of per-track tempo P95
errors changes from 54.86% to 95.14% for ARTBeaT and from 153.36% to 163.67% for
RUBATO. These query-weighted measurements are not interchangeable with the main
evaluator's tempo summaries or earlier ideal-observation results. None of these
real-cohort queries is unavailable; three RUBATO queries use a prior-only
endpoint extension and remain explicitly counted as such.

### What the controls establish

- Constant-tempo weak alternating pulses and four erased central pulses recover
  perfect beat F1 and zero tempo P95 error, with either retained or flat downbeat
  heads. This shows a useful distinction from thresholded sparse events, but
  does not establish correct bars or acceptance on music.
- Sustained alternating erasure at constant tempo still produces 50% tempo P95
  error and two spurious jumps, even with downbeat evidence retained. Tail
  erasure also remains unresolved (50% or 66.67% tempo P95 error).
- Intact 120/60/120 and 120/90/120 preserve both real changes. Intact
  120/240/120 also preserves both, but has 10.71% tempo P95 error: integer-frame
  period quantization and path variability remain visible.
- Weak alternating pulses in 120/240/120 erase **both** genuine changes and
  decode the fast region at 120 BPM, with either downbeat factor. Both the
  baseline and candidate can fail a control; relative `no_regression` is not
  sufficient to claim the expected change was recovered.
- Explicit unavailability excludes exactly 200 frames and creates two separate
  components, without a clock crossing the gap. Of 47 tempo queries, eight are
  unavailable and one uses an endpoint prior. This control uses single-frame
  pulses; the 40 regular controls use the smooth pulse shape specified above.
- Both flat heads produce zero ticks and 47/47 unavailable tempo queries, with
  null errors, not zero error. Relative `no_regression` is true only because
  the baseline is also empty; this is abstention, not successful analysis.

The two incompatible-truth witnesses have identical input and decoded-output
hashes. This is an output-space identifiability limit, not proof that their
original audio would be identical or that training would solve the ambiguity.

### Identified objective defect: extra bars are insufficiently penalized

All 40 regular controls select meter two despite the authored four-beat bars.
Consider a fixed beat path with downbeat contrasts
`[positive, 0, 0, 0, positive, 0, 0, 0]`. Meter four collects the two positive
terms. Meter two collects the same terms plus zero-cost unsupported halfway
bars. Both scores tie and the fixed lower-meter tie break chooses two. Positive
noise at a halfway bar makes that wrong hypothesis win strictly. The pulse and
period scores are identical for this fixed path, so they cannot resolve this
objective defect. The intact 120 BPM retained-bar control therefore has perfect
beat F1 but downbeat precision 0.5 and downbeat F1 0.66667.

This diagnoses a limitation of the current scoring model, not a lack of meter
information in clean neural evidence. Do not tune v1 weights after looking at
these results, retain it as a user-selectable fallback, or conclude that a new
neural model is already necessary. The next bounded experiment must first
demonstrate a consistent full-state likelihood that accounts for unsupported
and off-phase bar evidence on ideal controls, before freezing a new candidate
for real-music comparison. It must still preserve weak-beat tempo changes,
explicit unknown spans, observed/inferred separation and the zero-tuning API.

### Cost and retained artifacts

Optimized CPU decoder time totals 1.38 seconds for ARTBeaT and 42.01 seconds for
RUBATO in this run. These exclude capture loading, scoring, process startup and
neural inference; they are not end-to-end benchmarks. The largest reported
backpointer allocation is 14,375,130 bytes, not total process memory.

Private per-case predictions retain input/capture hashes, unchanged baseline
analysis and inferred coordinates. They and dense captures stay outside Git.
Only aggregate and per-case scalar summaries are published. The decoder and
audit are frozen with this failed result; no defaults or additional user
strategy switches have been introduced.
