# What the independent tempo head learned, and what it did not

2026-09-14: read-only diagnosis of the [closed four-arm experiment](../separated_evidence_fit/README.md).
All 40 frozen packets and ten retained variants were examined, with no model
forward, weight/hidden-feature loading, new music or training. Every one of the
720 applicable original metric blocks matched the previous report. Numerical
diagnosis took 6.63 s on CPU (excluding extraction/interpreter startup/report I/O).
The [protocol](PROTOCOL-v1.md) was fixed before reading per-cell outputs; the
[complete scalar report](results-v1.json) retains native/common support, all
recordings, every control and null-coverage accounting. Old gates stay failed.

## 1. Global bias matters, but a whole-recording octave fix is insufficient

Common-support work macros for the separated natural-input network:

| Cohort | Original BPM error % | Reference-informed recording octave diagnostic % | Per-cell octave diagnostic % | Off-octave-band cells % |
| --- | ---: | ---: | ---: | ---: |
| Fit | 16.32 | 16.53 | 14.12 | 64.04 |
| Development | 35.48 | 14.01 | 13.28 | 62.79 |
| ARTBeaT diagnostic | 28.21 | 20.71 | 15.49 | 64.98 |

The two octave columns use reference answers, not inference, repaired scores or
new acceptance thresholds. They reuse the earlier diagnostic unchanged, including
its factor-1.1 octave band. Recording-level rounding is not optimized for median
percentage error and can worsen it (as in fit). Native annotations are not
perceptual beat-level ground truth. The original columns are never replaced.

For cell log-rate error `e`, accounting uses `mean(e*e) = mean(e)**2 +
mean((e-mean(e))**2)`. The squared constant-bias term is 36.88% of aggregate
log-rate MSE on fit, 78.22% on development, and 65.18% on ARTBeaT. These are
ratios of work-macro squared terms, not mean per-record percentages, octave
fractions, causal explanations or an output correction.

Development's two Mozart performances have mean log2 rate biases +0.6885 and
+0.9528; a reference-informed recording octave helps them substantially.
The other three development recordings round to octave zero. This is why a
large aggregate bias share and substantial non-octave residual can coexist.
Raw-event oracle recording diagnostics remain much smaller: 9.14% development
and 2.84% ARTBeaT, versus this model's 14.01% and 20.71%.

## 2. Local speed response is weak even on fitting recordings

At 4 s lags, using all uninterrupted pairs where reference speed changes by at
least factor 1.1, compare log-rate changes. The zero-change comparator predicts
no change, NOT the reference absolute BPM. Lower RMSE is better; slope 1 would
be ideal tracking, slope 0 no aggregate reference-aligned response. Direction
agreement alone does not establish the correct magnitude.

| Cohort / separated natural | Change RMSE, log2 | Zero-change RMSE, log2 | Response slope | Direction agreement |
| --- | ---: | ---: | ---: | ---: |
| Fit | 0.46362 | 0.46360 | 0.12578 | 57.94% |
| Development | 0.40715 | 0.37423 | 0.03848 | 52.62% |
| ARTBeaT diagnostic | 0.63303 | 0.63244 | 0.04310 | 56.06% |

These are work-macros of per-record summaries, not pooled-pair regressions or
change-point detection accuracy. A slope of 0.04 does NOT mean four percent of
changes were found or every change has four percent amplitude. Cancellation,
incorrect direction, magnitude and annotation variability can all contribute.
Overlapping pairs are not independent samples. All three cohort RMSE macros are
no better than the zero-change comparator; per-case values remain available.

At the predeclared 1 s lag, slopes are also small: 0.03875 / 0.00127 / 0.01348
for fit/development/ARTBeaT. This is not solely an unseen-domain phenomenon.
It does not prove that another training budget or representation would fail,
but it rejects interpreting the present audio-versus-zero gain as successful
local tempo tracking. Raw events are not a perfect change reference either:
their development/ARTBeaT 4 s slopes are -0.00399 / 0.21529 and change RMSEs
0.63984 / 0.66262. Their better absolute native BPM does not guarantee good local
change response under this measurement.

Keeping the natural model fixed and replacing each sequence with its temporal
mean barely changes development/ARTBeaT absolute BPM error (indeed, 35.19% /
26.58% are lower), but removes most rate variation. Its residual log-rate MSE
is 0.04922 / 0.14524 versus natural's 0.07037 / 0.14950. Natural phase remains
useful under the previous experiment, but its independent tempo branch does not
provide a demonstrated native clock. Time-mean/zero/roll inputs can be out of
distribution; these are controlled associations, not an identified cause.

## 3. Coverage differs, but that alone does not explain failure

Reference-only common-support duration fractions, with unchanged work weights:

| Cohort | Below 60 BPM | At least 180 BPM |
| --- | ---: | ---: |
| Fit | 43.48% | 0.076% |
| Development | 48.27% | 0% |
| ARTBeaT diagnostic | 1.99% | 7.01% |

The full report retains all five fixed speed bands, per-record quantiles and
large/small/all change-pair counts. These descriptive populations are 20/9 fit
recordings/works, 5/3 development, and 15 diagnostic variants of one source.
They are reused research data, not independent acceptance. Broadening speed
coverage might help, but poor local response on fit means a domain-gap-only
explanation is insufficient. Do not silently relabel small-change pairs as
missing/weak beats, silence or constant-tempo regions.

## Next mechanism boundary

Do not reopen the four-arm fit, tune octave/smoothing thresholds, promote a
coherent fusion layer or add a public strategy. First make native speed-change
learnability an explicit falsifiable target instead of relying only on aggregate
absolute BPM versus zero inputs. A concrete next proposal is a same-source,
known-audio-tempo-transform paired supervision/validation contract: speed should
transform by the known factor, while mapped beat positions retain their meaning.
It must transform licensed audio and obtain its real encoder evidence; stretching
cached hidden-frame axes is NOT assumed equivalent. Preserve baseline and natural
recording performance and include a zero/change-insensitive control. No such new
audio inference, augmentation or fit has been performed or authorized by this
diagnostic alone; scope, labels, compute budget and rejection rule come first.

Shared/separated details, controls and every adverse case remain in the report.
This narrows the missing capability to measured native-speed response; it does
not prove a particular loss, CNN capacity, Transformer or data intervention will
solve it. The shipping high-accuracy, one-call/no-per-song-tuning goal is unchanged.

## Reproduction and tests

```sh
python -m unittest discover -s experiments/tempo_attribution -p 'test_*.py' -v
python -m experiments.tempo_attribution.run --inputs PINNED_PRIVATE_INPUTS \
  --predictions RETAINED_FOUR_ARM_OUTPUT --output NEW_PRIVATE_JSON
```

`diagnostic.py` owns offset/shape, 1/4 s response and reference coverage;
`run.py` verifies frozen identities and original metrics before producing the
complete scalar report. Existing rate/octave and measurement functions are reused
without changes. Authored witnesses cover exact tracking, pure octave bias,
missed/rate-reversed/under-amplitude changes, constant reference, gaps, empty
support, invalid required cells, nonuniform work weighting and nonmutation.
Outcome checks independently recompute all macros and old metrics and retain
failure decisions. Report SHA256:
`4ef7f051af28c8bdf38dd4b34d983298aad80df540fd013aadd732a09d2a3814`.
