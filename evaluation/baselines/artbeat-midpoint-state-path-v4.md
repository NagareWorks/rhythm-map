# ARTBeaT midpoint-state path rejection v4

This calibration tested whether a two-state gap path could distinguish a local
double-time beat run from ordinary subdivisions. The path used only selected
beats, real backend candidate timestamps, confidence, PCM activity, phase
alignment, and a fixed transition penalty. Case IDs, tags, and truth were not
available until after the path was complete.

The source report is retained outside Git at
`D:/rhythm-map-eval/reports/artbeat-midpoint-state-path-v4.json`. It uses model
manifest SHA-256
`ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`.

| Case | Selected F1 | State-path F1 | Delta | Active gaps | Boundaries |
| --- | ---: | ---: | ---: | ---: | ---: |
| 05 75 to 150 | 0.7907 | 0.8475 | +0.0568 | 16 | 2 |
| 06 150 to 75 | 0.7727 | 0.8525 | +0.0797 | 17 | 2 |
| 08 112.5 to 75 | 0.8696 | 0.6774 | -0.1921 | 16 | 2 |
| 10 90 to 120 | 1.0000 | 0.6988 | -0.3012 | 25 | 2 |
| 11 60 to 80 | 0.9268 | 0.6333 | -0.2935 | 19 | 2 |
| 12 80 to 150 | 0.8000 | 0.8421 | +0.0421 | 21 | 2 |
| 15 85 to 127.5 | 0.6486 | 0.4800 | -0.1686 | 26 | 2 |
| 18 piano rubato | 0.7568 | 0.7586 | +0.0019 | 13 | 2 |
| 19 ramp 80 to 200 | 0.6667 | 0.8831 | +0.2165 | 14 | 2 |
| 20 ramp 200 to 80 | 0.7077 | 0.9250 | +0.2173 | 15 | 2 |
| 21 polyrhythm 70 to 105 | 0.7805 | 0.5926 | -0.1879 | 13 | 2 |

The path activated in 11/15 cases: six improved and five regressed. Across the
activated cases, mean F1 fell from 0.7927 to 0.7446; suite top-1 mean F1 fell
from 0.8052 to 0.7699. Every activation formed one almost whole-track run. The
path therefore did not locate local tempo changes: a stable subdivision peak
has the same midpoint phase continuity as a missing double-time beat.

This rejects midpoint phase continuity as a runtime classifier and rejects the
state path itself. The implementation is not retained as another policy. Its
only positive result was raising the truth-assisted top-K construction ceiling
from 0.8370 to 0.8465, which is not deployable. Further automatic selection
requires evidence that can distinguish musical beat level from subdivision,
not another threshold or transition-cost sweep on ARTBeaT.
