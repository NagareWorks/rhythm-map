# FSLD tempo v1 public baseline

Measured on 2026-08-22 with an optimized build, the checked-in FSLD member
lock, and `beat-this-full-v1.json`. The verified model manifest SHA-256 was
`ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`.
Thresholds were not changed for this run.

This is a tempo-only calibration suite. FSLD supplies expert-agreed global BPM
and cut quality, but no timestamped beat phase. The report therefore declares
`end_to_end_only` and contains no oracle or oracle delta. It cannot attribute a
failure to the Beat This observation path versus the deterministic estimator.

The end-to-end path passed 6 of 15 cases. Nine cases had median tempo error
below 5 percent, but three of those still failed the P95 gate because their
tempo curves switched metrical level within the clip.

| Case | BPM | Result | Tempo median error | Tempo P95 error | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| fsld-476866-41-bpm | 41 | fail | 99.57% | 101.40% | 3.51 s |
| fsld-404840-60-bpm | 60 | fail | 96.13% | 101.43% | 16.62 s |
| fsld-210835-70-bpm | 70 | pass | 0.33% | 0.47% | 2.68 s |
| fsld-131423-80-bpm | 80 | pass | 0.45% | 1.35% | 1.85 s |
| fsld-219021-90-bpm | 90 | fail | 0.98% | 100.08% | 4.43 s |
| fsld-360687-100-bpm | 100 | pass | 0.04% | 2.36% | 1.62 s |
| fsld-19069-110-bpm | 110 | fail | 1.01% | 33.48% | 1.53 s |
| fsld-418991-120-bpm | 120 | pass | 1.25% | 1.94% | 0.92 s |
| fsld-124542-128-bpm | 128 | fail | 95.31% | 113.07% | 1.28 s |
| fsld-486302-130-bpm | 130 | fail | 50.19% | 50.89% | 1.69 s |
| fsld-271070-140-bpm | 140 | fail | 44.08% | 128.57% | 1.68 s |
| fsld-330889-150-bpm | 150 | fail | 1.61% | 49.57% | 6.37 s |
| fsld-322315-160-bpm | 160 | pass | 0.48% | 2.31% | 5.74 s |
| fsld-348652-180-bpm | 180 | pass | 0.04% | 4.17% | 1.90 s |
| fsld-439993-200-bpm | 200 | fail | 50.00% | 50.00% | 0.79 s |

Total measured analysis time was 52.60 seconds on the development VDI. Runtime
is machine-specific and diagnostic rather than an acceptance gate.

The dominant failure is not random BPM noise. The 41, 60, and 128 BPM cases
settle near double time; the 130 and 200 BPM cases settle near half time; and
several otherwise-correct clips change metrical level locally. This is useful
calibration evidence for product-level metrical selection, but the absence of
beat timestamps prevents bottleneck attribution. Changes to the default must
still be selected here and confirmed on a separate timestamped holdout rather
than treating these global BPM labels as event-level truth.

## Metrical consistency candidate

On 2026-08-23, `metrical-consistency-v1` was evaluated with the upstream decoder.
It repaired the bounded three-interval half-time run in
`fsld-330889-150-bpm`, reducing tempo P95 error from 49.57 percent to about 2.61
percent and raising the suite from 6 to 7 passing cases. No other FSLD case
changed. The same estimator candidate produced no beat, median-tempo, or
P95-tempo metric change on any of the 15 timestamped ARTBeaT calibration cases.

The supported-midpoint decoder also reached 7 of 15 FSLD cases: it repaired the
130 BPM median and the 150 BPM P95, but the 130 BPM tail remained at half time.
On ARTBeaT it reproduced the earlier mean beat-F1 gain from 0.8052 to 0.8235 and
the known regression on `artbeat-15-85-to-127-5`. Combining both candidates did
not exceed 7 of 15 FSLD cases. Both therefore remain explicit opt-in policies;
neither changes the shipping default before independent holdout evidence.

## Sequence and phase candidate

On 2026-08-23, `sequence-phase-v1` was evaluated with the unchanged upstream
decoder. It includes `metrical-consistency-v1`, adds bar-phase validation to
whole-track half-time selection, removes one-sided edge midpoint extras only
when a stable grid and PCM/model evidence support the retained phase, and
repairs paired fixed-frame quantization jitter. The suite rose from 7 to 9 of
15 passing cases:

- `fsld-404840-60-bpm` changed from 96.13 percent median and 101.43
  percent P95 error to 0.00 and 0.99 percent. The analyzed event count changed
  from 62 to 50 after weak midpoint extras extending to the track edge were
  rejected.
- `fsld-439993-200-bpm` changed from 50.00 percent median and P95 error to
  approximately 0.00 and 2.29 percent. Its raw cadence was already near 200
  BPM; the earlier salience-only fold produced downbeat evidence on nearly
  every retained beat, so the sequence policy rejected that inconsistent
  half-time choice and corrected only opposing frame-quantization jitter.

No other FSLD case changed pass status. Against all 15 timestamped ARTBeaT
calibration cases, analyzed event counts and end-to-end beat F1, tempo median,
and tempo P95 were identical to `metrical-consistency-v1`; the corresponding
oracle tempo metrics were also identical. Exact timestamp observations are
explicitly excluded from fixed-frame jitter repair.

The remaining failures are not safe targets for another global fold. The 41
BPM clip has a clean approximately 81 BPM raw cadence, but its alternating
salience pattern is not unique among correctly decoded material. The 128 BPM
clip is irregular at both roughly 250 and 125 BPM. The 130 BPM clip has
supported subdivisions in its interior but loses peaks at the edge; the core
estimator will not invent the missing timestamps. These need independent
timestamp truth and, for missing events, an opt-in logits sequence decoder or
an alternate observation backend.

## Edge-connected Viterbi candidate

On 2026-08-23, `viterbi-edge-logit-minus-3.0-bias-2.0` was combined with
`sequence-phase-v1`. The decoder requires a connected run of at least six weak
model peaks plus local and observed-edge support; it never emits a bare path
grid timestamp. The suite rose from 9 to 10 of 15 passing cases:

- `fsld-19069-110-bpm` gained the supported alternating model peaks needed to
  keep its tempo curve at the annotated level; P95 tempo error fell from 33.48
  percent to approximately 2.92 percent while median error remained 1.01
  percent.
- `fsld-271070-140-bpm` improved but remained below the suite gate. Its short
  recovered edge run is tempo-only evidence and is not accepted as timestamp
  truth.
- The 128 BPM four-candidate run is rejected, preserving the upstream result.
  The 130 BPM clip is also unchanged because Beat This has no sufficiently long
  qualifying edge sequence.

All 15 timestamped ARTBeaT cases remain identical at the raw event-metric level
to upstream decoding. FSLD does not provide beat timestamps, so the extra 110
BPM events remain a calibration result rather than permission to change the
shipping default.
