# Full neural-frame evidence and ideal pulse templates

## Question and scope

The earlier [onset-phase audit](clock-phase-evidence-v1.md) could label only
intervals with two correctly matched raw anchors. In RUBATO, 76.49% of raw
intervals lacked that context. Instead of adding another repair threshold,
retain the complete existing neural output and ask whether it favors the
annotated pulse over a half-phase alternative, including raw misses.

This is a **truth-assisted representation diagnostic**, not a new automatic
decoder or a product-accuracy measurement. It cannot demonstrate either that a
truth-free sequence decoder will recover those beats or that training is
necessary. No new model, weights, production strategy, cache fallback, public
observation field, tuning option or release is introduced. Holdout stays sealed.

## Identity before interpretation

The [Rust exporter](../../crates/rhythm-map-eval/examples/dense_beat_evidence.rs)
accepts only the exact existing calibration manifests and immutable evidence
hashes: all 15 ARTBeaT and all 25 RUBATO cases, in their frozen order. ARTBeaT's
separate preprocessing probe is excluded. It decodes complete recordings with
the shipping audio path, checks sample count/rate and every PCM bit by SHA-256,
verifies the pinned model pack, and runs the existing inference implementation.

Both 50 Hz frame heads are retained without peak filtering, interpolation,
cropping, or padding. The unchanged default decoder must reproduce all raw
beats/candidates, timestamps, confidences, duration and display metadata
exactly. The historical model display name remains unchanged; the actual model
identity is separately established by the verified manifest and artifact hashes.

The private per-case capture records source hashes, lockfile hash, runtime and
thread settings, full PCM identity, common time origin, frame count and measured
inference time. It is written before the next recording starts. The summary
hashes every case file. Partial cohorts, replay failures or implementation/hash
changes fail the subsequent audit. Nothing writes or relabels a production
observation cache; full PCM and mel tensors are never exported.

The [independent Python audit](../parity/dense_clock_evidence.py) also compares
the raw observations and rebuilds the default beat timestamps from actual
retained pulse logits (strict zero threshold, radius-three maxima, original
plateau deduplication and float32 timestamp conversion). This second gate does
not rely solely on a producer's `replay.exact` flag. The downbeat head is retained
and checked for shape/finiteness and provenance, but is not treated as a pulse
likelihood at every main beat in this experiment.

## Frozen comparison

The diagnostic and its authored tests are fixed before running the complete
comparison. For every consecutive annotated pair `t[i], t[i+1]`:

1. Query the actual beat-head maximum around `t[i]`.
2. Query the same head around `(t[i] + t[i+1]) / 2` as a half-phase control.
3. Use the same radius `min(50 ms, (t[i+1] - t[i]) / 16)` at both positions.
4. Keep a pair unavailable if either complete window lies outside captured
   frames or contains no frame. Do not substitute zero or discard its coverage
   denominator. The last annotated beat has no following interval and is
   separately counted as excluded.

Larger canonical-minus-control logits are declared favorable to the annotated
pulse. Report wins, ties, losses, logit margins and counts above the existing
zero-logit decoder threshold. No threshold fitting, favorable-direction flip,
selected-case prefix or best-band selection is performed.

Replay the original 70 ms chronological raw/truth matching **including pair
identities**, then stratify into raw matched, raw missed, and misses with/without
a real model candidate inside that tolerance. These categories do not control
where the templates are queried. Thus even a track with no usable raw anchors
still contributes ideal-template queries. The narrower template window is not
the scoring tolerance: a template loss cannot be counted as a scoring failure.

Per-track counts, pooled query counts and macro-track win fractions preserve
their own denominators. A stratum without complete pairs has a null fraction,
not zero. Macro statistics identify how many tracks contribute. Recordings and
adjacent beats are not independent samples; no unseen-music generalization or
confidence interval is claimed.

## Measured result

Both producer replay and independent pulse-event reconstruction pass **40/40**
recordings. ARTBeaT retains 12,328 frames per head and RUBATO 324,515 frames per
head. The two heads share a timeline; these counts must not be described as
the sum of both heads' scalar values. Private captures including summaries use
732,222 and 17,193,581 bytes respectively. Measured inference totals are 63.11 s
and 2,561.21 s (42.69 minutes), with two RTen CPU threads in the local virtualized
environment. RUBATO contains 6,490.03 s of audio. This is a one-time diagnostic
capture cost, not an added product stage or a portable performance benchmark.

The [aggregate report](../parity/dense-clock-evidence-v1.json) has SHA-256
`e4826ec6996b58e404a9773f49fe21c46126b960ee1ca83aa719cce6fe18fd12`.
Its source and capture-summary hashes link every statistic to immutable inputs.

| Ideal pulse-template comparison | ARTBeaT | RUBATO |
| --- | ---: | ---: |
| All annotated beats | 460 | 6,726 |
| Interval queries / complete pairs | 445 / 435 | 6,701 / 6,682 |
| All raw missed truth beats | 128 | 2,514 |
| Missed-beat queries / complete pairs | 124 / 123 | 2,489 / 2,478 |
| Missed-beat canonical wins / complete pairs | 91 / 123 (73.98%) | 1,552 / 2,478 (62.63%) |
| Missed-beat mean logit margin | +1.895 | +1.089 |
| Candidate-absent missed queries / complete pairs | 9 / 9 | 1,196 / 1,193 |
| Candidate-absent canonical wins / complete pairs | 5 / 9 (55.56%) | 773 / 1,193 (64.79%) |
| Candidate-absent mean logit margin | -2.364 | +1.316 |

Excluded final truth beats remain accounted for: 15 ARTBeaT final beats
(11 raw matched, four raw missed), and 25 RUBATO final beats (all raw missed).
Of the remaining interval queries, 10 ARTBeaT and 19 RUBATO pairs lack full
frame-window coverage. Among missed queries specifically, one and 11 pairs
are unavailable. These are not silently dropped from coverage statistics.

Macro-track missed-beat win fractions are 74.19% on 14/15 ARTBeaT tracks and
63.35% on 25/25 RUBATO tracks. Candidate-absent macro fractions are 52.38% on
only 3/15 ARTBeaT tracks and 65.95% on 25/25 RUBATO tracks. For RUBATO's
candidate-absent stratum, 20 tracks have a win fraction above one half and five
below it. This is not uniformly reliable across music, and a tiny ARTBeaT
candidate-absent sample cannot establish cross-domain consistency.

For context, all-query canonical win fractions are 92.41% and 84.03%, while
raw-matched fractions are 99.68% and 96.65%. Those easier, selected strata must
not replace the missed-beat result. These fractions compare two ideal templates;
they are neither AUC nor beat F1 nor automatic recovery rates.

None of ARTBeaT's 123 complete missed-beat canonical windows is above the
existing zero-logit threshold, despite 91 wins over the half-phase control.
In RUBATO, 256/2,478 missed canonical windows are above zero; 185/1,193
candidate-absent canonical windows are above zero. A positive frame is not
necessarily a local maximum, so positive values do not contradict the absence
of a nearby candidate peak and do not authorize automatic beat insertion.

## Decision and next experiment

There is useful **relative pulse evidence at some known missed-beat positions**,
including positions without a nearby selected candidate. Consequently, missing
peaks alone do not establish that the model contains no usable timing evidence.
Do not lower a peak threshold or conclude that model training is already needed.
Equally, this truth-assisted comparison does not establish a usable decoder.

The next bounded experiment is a truth-free whole-frame sequence interpretation
with explicit clock phase/tempo and missing-observation states, using the
retained beat/downbeat heads. It must not require two correct raw anchors or
receive annotated template positions. Freeze one candidate before evaluation;
test constant-tempo weak/empty-beat controls against genuine half/double-speed
changes, then all 40 calibration recordings with per-track event-identity,
timing, coverage and tempo-change guards. Do not promote a fitted collection of
case-specific rules. Use the existing training decision gate if that experiment
exposes an evidence limitation; do not train automatically.

## Interpretation boundaries

- Positive margins on missed beats would establish retained local evidence at
  known annotated positions, not a way to discover those positions unaided.
- Negative margins can reflect a different plausible metrical level or a real
  subdivision. Half-phase controls are not annotations of nonmusical events.
- Matched/missed strata condition on the existing peak decoder and therefore
  are not randomized groups. A strong matched-only score is expected selection
  behavior, not evidence that the same rule recovers misses. Preserve the
  missed and candidate-absent denominators instead of reporting only a pooled
  whole-recording win fraction.
- A pointwise template score does not evaluate whole-track tempo, meter or
  phase-state inference. Its success/failure is not a model-training verdict.
- Preserve the [practical training gate](../../docs/TRAINING-DECISION.md): only
  advance toward training after controlled evidence separates model evidence,
  state inference, coverage, labels and implementation problems.

Reproduction commands and private-artifact handling are in the
[parity guide](../parity/README.md#complete-dense-neural-evidence).
