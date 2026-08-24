# ARTBeaT missed-candidate evidence v3

This follow-up calibration removes a misleading aggregate from the first
candidate baseline. Instead of asking only whether every annotated beat has a
nearby candidate, it isolates annotated beats missed by the selected backend
sequence and measures candidate evidence only for those misses.

The source report is kept outside Git at
`D:/rhythm-map-eval/reports/artbeat-missed-candidate-evidence-v3.json`. It uses
the 15-case `artbeat-v1` calibration suite and model manifest SHA-256
`ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`.

| Case | Selected misses | Candidate-covered | Miss recall | Median candidate confidence | Top-1 F1 | Best top-K F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 05 75 to 150 | 9 | 9 | 1.0000 | 0.0002 | 0.7907 | 0.8475 |
| 06 150 to 75 | 9 | 9 | 1.0000 | 0.0726 | 0.7727 | 0.8525 |
| 07 75 to 112.5 | 1 | 1 | 1.0000 | 0.0009 | 0.8696 | 0.8696 |
| 08 112.5 to 75 | 2 | 2 | 1.0000 | 0.3769 | 0.8696 | 0.8696 |
| 09 90 to 80 | 1 | 0 | 0.0000 | — | 0.9600 | 0.9600 |
| 10 90 to 120 | 0 | 0 | 1.0000 | — | 1.0000 | 1.0000 |
| 11 60 to 80 | 2 | 2 | 1.0000 | 0.0526 | 0.9268 | 0.9268 |
| 12 80 to 150 | 11 | 11 | 1.0000 | 0.0323 | 0.8000 | 0.8421 |
| 13 180 to 120 | 9 | 8 | 0.8889 | 0.0047 | 0.7500 | 0.7500 |
| 14 240 to 96 | 13 | 12 | 0.9231 | 0.0888 | 0.7778 | 0.7778 |
| 15 85 to 127.5 | 16 | 9 | 0.5625 | 0.0000 | 0.6486 | 0.6486 |
| 18 piano rubato | 16 | 16 | 1.0000 | 0.2627 | 0.7568 | 0.7647 |
| 19 ramp 80 to 200 | 19 | 19 | 1.0000 | 0.0201 | 0.6667 | 0.8471 |
| 20 ramp 200 to 80 | 17 | 17 | 1.0000 | 0.0017 | 0.7077 | 0.8182 |
| 21 polyrhythm 70 to 105 | 3 | 3 | 1.0000 | 0.0011 | 0.7805 | 0.7805 |

Across the suite, 118 of 128 selected-sequence misses have a real candidate in
tolerance, for 0.9219 micro recall. However, the median of the per-case median
miss confidences is only 0.0323. A global threshold reduction would therefore
admit many unrelated local maxima. The useful signal is repeated weak evidence
at a coherent intermediate phase over a run, not the strength of one peak.

The evidence-retention ranker fixes the first baseline's half-time deletion
bias: mean top-1 F1 rises from 0.5224 to 0.8052 and the truth-best member is
ranked first in 9/15 cases. It nevertheless ranks `selected` first in all 15
cases, so it is safe but not a useful double-time selector and is not eligible
for product promotion.

The next evaluation candidate should be one sequence decoder with local states
for preserving the selected pulse and activating a coherent midpoint-supported
run. It must accumulate weak candidate evidence across adjacent gaps, penalize
state transitions, preserve backend timestamps, and remain independent of case
IDs, tags, and truth. This is local sequence inference inside one default
algorithm, not a user-facing Strategy pattern.
