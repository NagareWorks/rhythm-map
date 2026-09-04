# Acoustic phase evidence for missing-beat clock states

## Decision

Do not add this onset-phase score to the default decoder or fit a threshold to
these results. It has useful ranking signal on ARTBeaT but does not transfer
reliably to RUBATO. Five-interval context does not fix that limitation. This
rejects this particular observation likelihood, not all acoustic evidence or
the possibility of a training-free clock decoder.

The next missing evidence is the **complete neural frame sequence**, which the
40 retained Beat This observation exports do not contain. The adapter already
exposes `BeatThisInference` logits, but its normal observations currently set
`activations: None`. Candidate peaks and candidate-centered harmonic features
must not be interpolated and described as dense neural/harmonic evidence.

## Frozen experiment

The [script](../parity/clock_phase_evidence.py) consumes only the two exact,
hash-locked calibration evidence files used in previous replay experiments.
All 15 ARTBeaT and 25 RUBATO cases are included; ARTBeaT's separate preprocessing
probe is excluded. It reruns neither audio decoding nor model inference and
does not touch caches, holdout, product output or user options.

For every consecutive raw-event interval, before reading truth:

1. Sample the existing continuous PCM-derived onset envelope at quarter,
   midpoint and three-quarter phase. Each sample is a local maximum within
   `min(50 ms, interval / 16)`, keeping the three windows disjoint.
2. Measure midpoint strength and midpoint minus the mean of the two quarter
   strengths, independently for full-band, low, mid and high onset strength.
3. Average each contrast over exactly five raw intervals (two on each side).
   Incomplete contexts and absent evidence remain null. No extrapolation,
   silence substitution, fitted weights or feature-direction reversal occurs.

Larger values are declared favorable to the extra-beat interpretation. This
is an evidence audit, not a calibrated probability, selected path or recovery
algorithm. Quarter-phase onsets can be real subdivisions or other musical
events; their interpretation is not supplied by these annotations.

After extracting features, reproduce every frozen chronological raw/truth
matching pair exactly. Two correctly matched anchors define the labels:

- one truth interval between them: negative for an additional main beat;
- two truth intervals: one missed main beat, the positive class;
- larger advancement: a separate multiple-miss category;
- either raw anchor unmatched: unknown, not a negative label.

Positive labels do not require the missing truth to fall near the midpoint.
That geometric reachability is reported separately; it cannot be used to
discard harder positives. Context features do not inspect neighboring labels.

## Domain coverage matters

| Count | ARTBeaT | RUBATO |
| --- | ---: | ---: |
| Raw intervals inspected | 360 | 9,248 |
| One-beat negative intervals | 218 | 2,030 |
| One-miss positive intervals | 72 | 127 |
| Multiple-miss intervals | 0 | 17 |
| Intervals with an unmatched anchor | 70 | 7,074 |
| All missed truth beats under the frozen raw matching | 128 | 2,514 |
| Positive midpoint within the existing 70 ms truth tolerance | 69 | 76 |
| Positive midpoint outside that tolerance | 3 | 51 |
| Positives with no real model candidate near the missed truth | 0 | 31 |

In RUBATO, 76.49% of raw intervals have at least one unmatched anchor. The
127 isolated misses represent only 5.05% of all raw-sequence misses. An
anchor-dependent rule cannot claim whole-corpus recovery from its small clean-
anchor contrast. The raw-pair missed count is not interchangeable with earlier
selected-output, candidate-label or alternate-runtime counts.

## Ranking results, not product accuracy

The full-band results are below. AUC is the probability that a randomly selected
positive ranks above a negative, with half credit for ties. It is not accuracy,
beat F1 or an operating threshold. Macro AUC covers only tracks with both
classes: seven ARTBeaT and ten RUBATO tracks, not all 40 recordings.

| Feature | ARTBeaT pooled / macro AUC | RUBATO pooled / macro AUC |
| --- | --- | --- |
| Midpoint onset | 0.879 / 0.949 | 0.385 / 0.501 |
| Phase contrast | 0.730 / 0.736 | 0.421 / 0.465 |
| Five-interval contrast | 0.711 / 0.681 | 0.405 / 0.544 |

Point features include 72/218 positive/negative ARTBeaT rows and 127/2,030
RUBATO rows. Sequence features retain 60/177 and 127/2,003 respectively. On
**identical complete-context rows**, point contrast versus sequence contrast
is 0.729 versus 0.711 on ARTBeaT and 0.420 versus 0.405 on RUBATO. Dropping edge
rows therefore does not explain away the lack of sequence improvement.

Among the 31 RUBATO positive intervals whose missing truth lacks any nearby
model candidate, full-band contrast has AUC 0.340 against the same 2,030
one-beat negatives. This does not establish useful recovery of the candidate-
absent misses. All four band results, missing counts, matched-sample comparisons
and per-track statistics are in the [aggregate report](../parity/clock-phase-evidence-v1.json).
Do not flip feature direction or select whichever band looks best after viewing
the labels. These correlated calibration intervals do not establish unseen-
music performance; no confidence intervals or generalization claim are made.

## Next evidence boundary

Capture complete beat/downbeat logits through the existing inference API,
without changing the product observation contract or adding a user strategy.
The capture should retain exact model, full PCM, preprocessing, source and
frame-time identities, verify default decoded events against the frozen
calibration baseline, and fail closed on mismatches. Existing 60-second parity
traces are not full-recording RUBATO evidence. Store compact logits privately,
not unnecessary PCM/mel dumps, and never reconstruct them from peaks.

Then evaluate clock alternatives without requiring all selected raw events to
be correct phase anchors, while accounting for false-positive observations,
real tempo changes, missing evidence and uncertainty. No new training decision
is justified until that genuinely available neural evidence has been tested.

## Reproduction

```sh
python evaluation/parity/clock_phase_evidence.py \
  --artbeat <locked-artbeat-private-evidence.json> \
  --rubato <locked-rubato-private-evidence.json> \
  --output <new-aggregate-report.json>
python -m unittest discover -s evaluation/parity -p test_clock_phase_evidence.py -v
```

The output contains aggregate/per-track statistics, not audio, private paths,
dense features or individual event timestamps. Twelve authored/frozen-summary
tests cover phase windows, missing evidence, context edges, truth separation,
event identities, fixed AUC direction and complete denominators.

Report SHA-256: `081ce706eaa0f87d6654093a0bdef0cbdc5583a7b271e267af303b888be0d390`.
Script SHA-256: `6302fef2747292142d10901475acd4e934aa711d8742a9ba781fb8140d915117`.
Input and AUC implementation digests are also retained in the report.
