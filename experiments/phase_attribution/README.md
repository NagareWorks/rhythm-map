# Why phase improvement did not deliver correct tempo

Status, 2026-09-11: **read-only diagnosis complete; the fitted model stays rejected**.
The [predeclared decomposition](PROTOCOL-v1.md) reads the existing 40 input packets
and 200 saved predictions. No model forward, weight loading, hidden-feature loading,
training or new music. All 320 pre-existing applicable metric blocks recompute;
all original source/input/prediction hashes match before and after analysis.
The actual numerical diagnostic took 6.59 s on CPU (excludes interpreter startup,
archive extraction and writing the final report). No GPU was needed for this run.

The [retained report](results-v1.json) contains all common-support cases and both
native/common cohort summaries. Its private-report hash identifies the complete
native/common per-case evidence; raw inputs, arrays and weights are not published.
Code/tests live outside the prior fit's frozen source closure.

## Finding 1: not just a wrong whole-track octave

All error percentages below are work-macros of recording median percentage
errors, not accuracy or pooled frame medians. Support is the unchanged intersection
of native reference and raw-event support. The two label-informed columns are
diagnostics, NOT corrected outputs or replacement acceptance scores.

| Cohort / saved clock | Original BPM error | One reference-informed recording octave | Per-cell reference-informed octave |
| --- | ---: | ---: | ---: |
| Development / audio | 62.73% | 24.09% | 17.76% |
| Development / raw | 41.59% | 9.14% | 7.76% |
| ARTBeaT diagnostic / audio | 24.58% | 24.93% | 15.59% |
| ARTBeaT diagnostic / raw | 14.45% | 2.84% | 1.78% |

The recording level is `floor(median(log2(rate/reference))+0.5)`. This is NOT a
search for the best percentage-error correction or a guaranteed lower bound;
ARTBeaT illustrates that it can worsen percentage error. Per-cell folding is
more permissive and still leaves substantial residuals. It cannot provide an
inference algorithm because it uses the answer at every cell.

Only 8.28% of development audio cells are near double-time and 0.01% near
half-time under the predeclared factor-1.1 band; 73.09% lie outside ALL integer
octave bands. ARTBeaT audio has 69.38% off-grid cells versus raw's 15.86%.
These are work-macros of within-record cell fractions, not fractions of music
proven rhythmically wrong. Native annotations are not perceptual-octave truth.
Nevertheless a blanket half/double fix does not explain this model's errors.

## Finding 2: a bad rate prior is already present before feedback

| Cohort / audio | CNN prior BPM error | Saved corrected BPM error | Native rate outside fixed local feedback bound |
| --- | ---: | ---: | ---: |
| Development | 68.95% | 62.73% | 69.32% |
| ARTBeaT diagnostic | 23.47% | 24.58% | 51.51% |

Feedback sometimes helps and sometimes worsens speed; removing it is not a
demonstrated solution. The main gain is 0.35597. For each saved local vector its
correction in log2-rate is bounded by `gain * norm(u,v)/hypot(1,norm(u,v))/ln(2)`.
In the cells counted above, even the most favorable phase angle cannot reach the
native rate while retaining that cell's prior/vector/gain. This is an algebraic
fixed-output limit, NOT a theorem about another trained model or a suggestion
to enlarge the gain. The saved incoming clock and fields reproduce every local
feedback equation to below 1e-11 log2 units across all retained variants.

The period fields already contain large error, so solving phase synchronization
alone cannot supply the missing native speed. This does not yet tell us whether
learning failed because of gradient routing, insufficient features, population
coverage, the optimizer budget, or their interaction.

## Finding 3: circular alignment and native counts are different successes

For development, the fixed audio model's common-support circular loss is 0.6737
versus fitted zero's 0.9341, but its multi-lag count loss is 0.6571 versus 0.4100.
The total is almost tied (1.3308 versus 1.3441). This accounts for the observed
score tradeoff; comparing scalar magnitudes does NOT prove gradient cancellation
or justify changing weights. The full report retains all four lag terms.

Every development recording has one common-support component. Its signed
terminal count drift, relative to that component's starting offset, is:

| Recording | BPM median error | Circular phase MAE (cycles) | Terminal count drift (cycles) |
| --- | ---: | ---: | ---: |
| Boccherini / Krux | 15.84% | 0.0862 | -0.11 |
| Mozart / Papalin | 108.26% | 0.2452 | +36.56 |
| Mozart / Gli Armonici | 179.23% | 0.2456 | +50.19 |
| Mussorgsky / Bertoglio | 35.78% | 0.2232 | +21.52 |
| Mussorgsky / Staab | 21.46% | 0.2156 | +13.88 |

These are not whole-file endpoint errors outside support. Phase improvement in
the cohort is not uniformly good phase in every recording; the Mozart cases
remain near quarter-cycle MAE as well as accumulating excess counts. ARTBeaT
also has mixed outcomes: 240-to-96 has phase MAE 0.1493 yet ends 14.35 cycles
behind. Missing support is never bridged. Rounded integer-offset transitions
are diagnostic boundary crossings, not detected missed/extra beat events.

## What this changes next

Do not add a public octave strategy, increase feedback gain, smooth the curve,
train longer or reopen this fixed run based on these results. Prioritize the
native-speed learning path: a separate fixed-checkpoint gradient audit can test
whether native count supervision reaches the local period head with a useful
direction through the recurrent phase feedback, and how the phase term changes
that direction. Check that mechanism before proposing direct period supervision
or another architecture/fit. Such an audit would measure gradients, not update
weights; this report itself performs neither operation.

Representative weak-attack/no-rhythm labels and independent acceptance are still
required. Neither this diagnosis nor the earlier timing-dependence result admits
a product model. The one-call/no-per-song-tuning contract stays unchanged.

```sh
python -m experiments.phase_attribution.run --inputs PRIVATE_INPUTS \
  --predictions PRIVATE_SAVED_PREDICTIONS --output NEW_PRIVATE_REPORT.json
python -m unittest discover -s experiments/phase_attribution -p 'test_*.py' -v
```
