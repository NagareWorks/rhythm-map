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
