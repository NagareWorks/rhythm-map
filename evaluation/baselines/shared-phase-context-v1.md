# Shared-phase multi-beat context audit v1

This is a truth-assisted representation diagnostic, not a decoder or a tempo
accuracy result. Requiring five points to share one timing offset changes the
evidence in both directions. It does not pass a default-replacement gate and
does not establish a need to train a model. Production output and user options
are unchanged.

## Locked comparison

The [contract](../parity/shared-phase-context-lock-v1.json) and
[extractor](../parity/shared_phase_context_audit.py) were frozen before the first
cohort run. Ten authored controls passed before inspecting these results.

- Contract SHA-256: `ee40508d522ac6113be18fd397234edbb3a79ee72801c65566e7afea801d6f04`.
- Extractor SHA-256: `c9670d18f342b682cd766e9522addf9616c322b0530cb3288f7f992c334d8bbe`.
- Report SHA-256: `321fa50013a62da0d805ef13bb0cd690802b6c19aa9014261b096c10e35c5e55`.

At each annotated anchor, use the mean of the three preceding annotated
intervals as period P. Construct four five-point templates: the anchor and next
four annotations, constant continuation at P, half tempo at 2P, and double
tempo at P/2. Quantize to 50 Hz with nearest-integer/ties-to-even rounding.
The anchor, period and future annotated template are oracle inputs, not
predictions or locations found by this tool.

For each of three fixed pairs, both sides require the same complete closed
frame domain: the hull of the two templates expanded by three frames. Missing
frames reject the pair rather than clipping or padding it. Nonfinite available
frames fail validation. Within-template windows that share frames are excluded;
cross-template overlap is allowed. Identical grids are counted separately,
never as successful distinctions. Prefix/suffix exclusions remain queries.

For a five-point template T and beat logits b:

```text
shared(T)      = max over d in {-3,...,3} of mean over t in T of b[t+d]
independent(T) = mean over t in T of max over d in {-3,...,3} of b[t+d]
phase_penalty = independent(T) - shared(T)
pair margin   = score(left) - score(right)
```

Equal maxima choose the offset nearest zero, then the earlier offset. The
downbeat head is read at the beat-selected shared offset and reported separately;
it is not independently maximized and added to the beat score. Raw logits are
neither probabilities nor class-conditional likelihoods. The
[shift-tolerant training semantics](beat-this-semantics-v1.md) do not justify
multiplying independent window probabilities or assuming a shared offset across
five beats.

**Equal-count five-point templates are not full clock hypotheses.** A pair shares its
available domain, but unused frames contribute no density evidence. An ideal
constant pulse train can support both the continuation and its half-time subset
equally. Identical alternating evidence can mean omissions or a genuine
slowdown. Both counterexamples pass as required limitations in the authored
tests. Equal counts cancel additive logit shifts and preserve ordering under
positive scaling; this is feature invariance, not likelihood normalization.

The three pairs can have different horizons and exclusions. Their margins must
not be pooled into a three-way tempo winner. In the two ratio comparisons,
positive means continuation is favored, not necessarily that the correct tempo
was selected. The period reference may itself straddle an annotated change.

## Frozen cohort coverage

All 15 ARTBeaT and 25 RUBATO calibration recordings use the unchanged captures,
source/model hashes, truth identities, and raw/truth matching from the
[window audit](metrical-window-v1.md). There are 12,328 and 324,515 frames per
head. No inference, decoder replay, fitting, holdout access or training ran.

| Cohort and pair | Queries | Informative | Prefix / suffix | Identical | Out of capture | Overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARTBeaT annotated / continuation | 460 | 296 | 45 / 60 | 59 | 0 | 0 |
| ARTBeaT continuation / half tempo | 460 | 297 | 45 / 60 | 0 | 58 | 0 |
| ARTBeaT continuation / double tempo | 460 | 333 | 45 / 60 | 0 | 0 | 22 |
| RUBATO annotated / continuation | 6726 | 6500 | 75 / 100 | 4 | 4 | 43 |
| RUBATO continuation / half tempo | 6726 | 6519 | 75 / 100 | 0 | 32 | 0 |
| RUBATO continuation / double tempo | 6726 | 6541 | 75 / 100 | 0 | 4 | 6 |

The first applicable exclusion is retained; exclusions are not independent
counts. Every query's acoustic presence is unknown. ARTBeaT's downbeat fields
remain unknown placeholders, not negative labels. RUBATO's measure labels are
preserved without treating stored per-interval constant segments as constant
performance. An ARTBeaT constant context requires the entire three-interval
prefix and four-interval suffix to have continuous constant annotation and no
change point. These wider regime groups differ from the preceding audit.

## Both directions retained

Counts below are positive margins among the same informative queries, with
independent alignment followed by shared alignment. They are not recovered
beats or accuracy. The full [report](../parity/shared-phase-context-v1.json)
retains all track results, pair-specific denominators, nulls, sign transitions,
phase penalties, score quantiles and separate downbeat readouts.

| Cohort / pair | All queries | Raw-missed anchors | Candidate-absent missed anchors |
| --- | ---: | ---: | ---: |
| ARTBeaT annotated / continuation | 145 to 232 / 296 | 61 to 74 / 91 | 3 to 3 / 3 |
| ARTBeaT continuation / half tempo | 199 to 196 / 297 | 73 to 74 / 97 | 2 to 2 / 5 |
| ARTBeaT continuation / double tempo | 277 to 268 / 333 | 74 to 67 / 100 | 4 to 4 / 7 |
| RUBATO annotated / continuation | 4372 to 4465 / 6500 | 1855 to 1848 / 2368 | 959 to 933 / 1151 |
| RUBATO continuation / half tempo | 4218 to 4514 / 6519 | 1210 to 1309 / 2373 | 577 to 622 / 1152 |
| RUBATO continuation / double tempo | 4307 to 4357 / 6541 | 1237 to 1324 / 2389 | 600 to 630 / 1157 |

Important counterweights to the favorable totals:

- ARTBeaT missed annotated/continuation comparisons have six positive-to-negative
  transitions, one negative-to-positive, and 29 former ties split 18 positive /
  11 negative. Overall growth is not uniform correction.
- In the constant-context missed subset, 18 grids are identical and excluded;
  only 25 are informative, with positive counts 5 to 13. Twenty former ties
  split 12 positive / 8 negative; four former positives become negative. Small
  frame-quantization differences can make annotated and extrapolated grids
  different even in a constant region; this is not detected tempo change.
- ARTBeaT changed-context missed annotated comparisons go 20 to 24 / 28;
  ramps remain 22 / 22. Yet ramp continuation/double-tempo positives fall from
  17 to 11 / 22. No ratio winner is assumed correct in these changing regions.
- RUBATO missed annotated comparisons have 141 positive-to-negative and 102
  negative-to-positive transitions; 71 ties split 32 positive / 39 negative.
  The candidate-absent subset has 68 positive-to-negative versus 38
  negative-to-positive, with nine ties split 4 / 5.
- Track-balanced missed annotated positive fractions move from 72.83% to
  87.23% on ARTBeaT (14/15 contributing tracks), and 79.01% to 79.75% on RUBATO
  (25/25). RUBATO's pooled count nevertheless falls. Do not report only the
  aggregation that looks better.
- ARTBeaT 240-to-96 goes 5 to 4 / 11 on missed annotated comparisons. RUBATO
  Berlioz/Dennis goes 48 to 40 / 57 and Boccherini/Krux 62 to 54 / 69;
  Mussorgsky/Staab improves 85 to 98 / 111. All tracks remain in the report.

## Decision and next gate

Do not promote shared alignment alone, add a strategy switch, tune its radius
or point count on these recordings, or convert these margins to confidence.
The comparison shows real phase-coherence sensitivity, but not a universal
advantage and not resolution of the omission/tempo ambiguity. This is still a
limitation of the tested representation, not proof that the waveform lacks the
necessary information or that training is required.

The next representation gate must put complete candidate clocks on one fixed
time span, account explicitly for every predicted pulse and unmatched response,
and keep unavailable observations separate from weak responses and missing
annotations. First check matched constant-with-omissions, true changes and
observationally identical controls; equivalent evidence must remain ambiguous.
Do not merely sum these five-point margins or re-label them as a normalized
presence sensor: the earlier [presence-likelihood failures](presence-likelihood-v1.md)
and the backend's pooled loss semantics still apply. Define and validate any
new observation contract before another cohort replay or scalable search.

## Reproduce

Run `python -m unittest discover -s evaluation/parity -p test_shared_phase_context.py -v`
for the ten authored controls and two frozen report integrity tests. For private
capture replay, see the command in the [parity README](../parity/README.md).
Public files contain aggregate counts and hashes only, not audio, reconstructable
frame arrays, beat coordinates or private environment paths.
