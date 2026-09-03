# Weak-candidate evidence: ARTBeaT calibration

Date: 2026-09-03. Descriptive analysis only; no recovery policy, fitted model,
threshold search, resampler promotion, holdout inference, or release.

## Scope and controls

The exporter replays all 15 frozen ARTBeaT v2 observation caches through the
actual engine, including shipping PCM-derived activity, onset, and harmonic
evidence. Every original raw beat and final selected score must match the
previous full calibration exactly. The separate full-recording ARTBeaT 14
reference-resampler trace must reproduce its frozen candidate score. All 16
replays pass; there are 15 cache hits, zero neural inferences, and zero cache
writes. Missing caches fail closed without a neural fallback.

The main cohort uses shipping preprocessing. The previously diagnosed 1.50 s
miss uses candidate preprocessing and is **not pooled into that cohort**.
FSLD is excluded because its truth is tempo-only. This run does not cover
RUBATO: the selected local cache root had no matching entries. Prior RUBATO
metrics under other observation contracts are not mixed into this analysis.

Input selection and constants are in
[`candidate-evidence-lock-v1.json`](../parity/candidate-evidence-lock-v1.json).
The result, feature source identity, private-export digest, per-track counts,
overlapping tag slices, feature availability, class quantiles, and fixed-
direction AUCs are retained in
[`candidate-evidence-separability-v1.json`](../parity/candidate-evidence-separability-v1.json).
No audio, dense observations, logits, absolute environment paths, or complete
candidate-row lists are checked in.

## What the labels do and do not mean

Features see observations only. Truth is attached afterward, using the existing
70 ms beat tolerance and the engine evaluator's raw/truth event matching.
Candidates within 20.001 ms of an accepted event are excluded to avoid counting
the original event or its plateau-rounding neighbor as a recovery.

- `missed_truth_support`: within tolerance of one annotated main beat not
  already matched by the raw sequence.
- `covered_truth_duplicate`: near a main beat the raw sequence already covers.
  These are not successful recoveries and are excluded from the AUC contrast.
- `offbeat_subdivision_aligned`: outside every main-beat tolerance window but
  within 20.001 ms of a fixed 1/4, 1/3, 1/2, 2/3, or 3/4 truth-interval position.
- `offbeat_other`: inside the annotated span but outside those windows/grids.
- Ambiguous multi-truth windows and candidates outside the annotated span are
  excluded from AUC rather than silently assigned a reliable negative label.

ARTBeaT supplies main-beat truth, not subdivision-note truth. Grid alignment is
only annotation-relative geometry: it does **not** prove an actual musical
subdivision was played, or identify a drum/instrument. Conversely, a supported
truth beat is not an automatically recoverable beat.

The primary cohort has confidence at most 0.5, the existing decoder gate. Four
positive-logit candidates left unselected by the wider peak decoder form a
separate secondary cohort. No new confidence cutoff is selected.

## Measurements

There are 1,180 unselected subthreshold candidates:

- 144 support 119 distinct missed truth beats; 25 are additional peaks near
  already-supported misses, not 25 further recoveries.
- 987 are anchored offbeats: 546 subdivision-grid-aligned and 441 other.
- 17 are covered-truth duplicates and 32 lie outside the annotated beat span.
- No candidate in this cohort overlaps multiple truth-tolerance windows.

The raw sequence misses 128 truth beats across the 15 tracks. Existing weak
peaks therefore support 119/128 (92.97%) of these misses under this diagnostic
labeling, leaving nine without an unselected candidate within tolerance. This
is a truth-assisted evidence-coverage count, **not recovery recall or beat F1**.
The four positive-logit unselected candidates support no additional misses.

Selected independent features are shown below. AUC means the probability that
a randomly sampled positive ranks above a negative in the declared direction,
with ties counted as half. It is not accuracy at a chosen cutoff.

| Feature (declared better direction) | All anchored offbeats: pooled / macro-track AUC | Grid-aligned offbeats: pooled AUC |
| --- | --- | --- |
| Confidence, higher | 0.798 / 0.812 | 0.791 |
| Onset strength, higher | 0.735 / 0.734 | 0.697 |
| Onset relative to anchors, higher | 0.754 / 0.783 | 0.729 |
| Midpoint error ratio, lower | 0.760 / 0.727 | 0.722 |
| Double-gap residual, lower | 0.577 / 0.641 | 0.575 |
| Context dispersion, lower | 0.351 / 0.440 | 0.365 |

The machine-readable report retains all 13 declared features, including the
weaker harmonic and band-specific onset results. These summaries use different
available-row counts: confidence has 144 positives/987 negatives; relative
onset and midpoint have 138/980; two-sided context has 118/823. Do not interpret
their AUC ordering as a matched-sample feature-ablation experiment. Macro-track
averages include the 13 tracks with both classes; the report records each
track separately. No confidence intervals or generalization claims are made
for these correlated candidates from one already-opened calibration corpus.

Distributions overlap substantially. Positive confidence spans approximately
0.000006--0.471, while anchored negatives extend to 0.496. Positive and negative
onset medians are 0.609 and 0.318, but their ranges overlap. A stable local
interval pattern is not sufficient evidence of a main beat: regular
subdivisions can be very stable, while real misses occur near tempo changes
and missing/half-time anchors. The unfavorable context-dispersion direction
must not be silently reversed after inspecting the result.

## The fixed 1.50 s probe

The missed peak remains at confidence 0.486622 with a strong onset (0.698243),
but its onset is only 0.850032 times the mean at its two accepted anchors.
Those anchors lie at 1.26 and 1.76 s; the normalized midpoint error is 0.020000.

The two intervals on each side, excluding the enclosing gap, have median
0.38 s. Thus the 0.50 s enclosing gap is not cleanly twice that period:
double-gap residual is 0.342105 and context dispersion is 0.368421. Missing
and metrical-level-mixed anchors contaminate the very context a naive local
repair would rely on.

None of 823 fully comparable shipping-cohort negative candidates is at least
as favorable as this probe simultaneously on the four declared coordinates:
confidence, relative onset, midpoint error, and double-gap residual. This
narrow dominance check is encouraging for investigating this specific event,
but is **not a zero-false-positive recovery rule**. The probe is a known,
selected regression from different preprocessing; using its coordinate values
as cutoffs would fit a rule to the answer. No such rule is implemented.

## Decision and next experiment

Keep the shipping decoder, isolated-midpoint guard, estimator, and resampler
unchanged. Existing evidence has useful ranking signal but does not yet
establish safe automatic main-beat selection.

Before defining another recovery policy, obtain the same source-locked
evidence on the already-opened RUBATO calibration slice using the same v2
contract. Reuse matching caches if available; otherwise budget an explicit
calibration inference run. Keep feature definitions and directions fixed,
report the corpus separately, and preserve the work-disjoint holdout seal.
Only after that transfer check should a single evaluation-only, abstaining
sequence rule be specified and tested across both corpora. Do not add a user
strategy switch, infer subdivisions from these labels, or train a model from
this audit.
