# ARTBeaT v1 public baseline

Measured on 2026-08-21 with an optimized build, the checked-in ARTBeaT lock,
and `beat-this-full-v1.json`. The verified model manifest SHA-256 was
`ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`.
Thresholds were not changed for this run.

All 15 oracle cases passed. The worst oracle tempo P95 error was 5.97 percent,
so the deterministic estimator clears this slice when given the official beat
timestamps. The end-to-end path passed 1 of 15 cases and was attributed to
`observation_path`.

| Case | Raw / analyzed beats | End to end | Beat F1 | Tempo median error | Tempo P95 error | Change recall | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| artbeat-05-75-to-150 | 17 / 17 | fail | 0.7907 | 49.99% | 50.41% | 0.00 | 1.83 s |
| artbeat-06-150-to-75 | 18 / 18 | fail | 0.7727 | 48.70% | 50.00% | 0.00 | 2.30 s |
| artbeat-07-75-to-112-5 | 25 / 25 | fail | 0.8696 | 1.28% | 185.71% | 0.00 | 2.07 s |
| artbeat-08-112-5-to-75 | 24 / 24 | fail | 0.8696 | 1.19% | 90.48% | 0.00 | 1.91 s |
| artbeat-09-90-to-80 | 25 / 25 | fail | 0.9600 | 0.98% | 3.78% | 0.00 | 2.75 s |
| artbeat-10-90-to-120 | 29 / 29 | pass | 1.0000 | 1.01% | 1.42% | 1.00 | 3.44 s |
| artbeat-11-60-to-80 | 20 / 20 | fail | 0.9268 | 0.45% | 39.52% | 1.00 | 3.19 s |
| artbeat-12-80-to-150 | 22 / 22 | fail | 0.8000 | 1.29% | 50.41% | 0.00 | 2.27 s |
| artbeat-13-180-to-120 | 31 / 31 | fail | 0.7500 | 1.42% | 77.78% | 0.00 | 1.71 s |
| artbeat-14-240-to-96 | 31 / 31 | fail | 0.7778 | 2.47% | 51.92% | 1.00 | 2.79 s |
| artbeat-15-85-to-127-5 | 34 / 34 | fail | 0.6486 | 31.79% | 34.02% | 0.00 | 3.54 s |
| artbeat-18-piano-rubato | 30 / 30 | fail | 0.7568 | 35.87% | 51.74% | 1.00 | 3.90 s |
| artbeat-19-ramp-80-to-200 | 23 / 23 | fail | 0.6667 | 50.11% | 69.23% | 0.00 | 2.30 s |
| artbeat-20-ramp-200-to-80 | 25 / 25 | fail | 0.7077 | 50.27% | 50.75% | 0.00 | 2.76 s |
| artbeat-21-polyrhythm-70-to-105 | 22 / 22 | fail | 0.7805 | 0.85% | 51.28% | 0.00 | 1.49 s |

Median per-clip runtime was 2.30 seconds and mean beat F1 was 0.8052. Runtime
is machine-specific and remains diagnostic rather than an acceptance gate.

The main failure mode is sustained missing beats or a sustained half-time
metrical choice, not an isolated event that deterministic repair can safely
correct. For example, the model stays near 75 BPM after the official beat level
changes from 75 to 150 BPM. The ramp and rubato cases likewise omit enough
events that reconstructing exact beat timestamps would require new acoustic
evidence. This baseline therefore supports evaluating an alternate observation
backend or decoder before adding more ungrounded post-processing heuristics.

