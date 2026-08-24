# ARTBeaT spectral-flux evidence calibration v5

This calibration asked whether deterministic PCM onset strength can distinguish
real missing double-time beats from stable subdivision peaks. The engine used a
centered approximately 40 ms Hann window, an approximately 10 ms hop, positive
spectral flux, logarithmic compression, and per-track peak normalization. The
first frame established the spectrum baseline and contributed zero flux.

The source report is retained outside Git at
`D:/rhythm-map-eval/reports/artbeat-spectral-flux-v5-final.json`. It uses model
manifest SHA-256
`ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`.
Onset evidence did not change the selected sequence or hypothesis rank.

| Case | Selected F1 | Midpoint F1 | Delta | Added onset strength |
| --- | ---: | ---: | ---: | ---: |
| 05 75 to 150 | 0.7907 | 0.8475 | +0.0568 | 0.6912 |
| 06 150 to 75 | 0.7727 | 0.8525 | +0.0797 | 0.7296 |
| 07 75 to 112.5 | 0.8667 | 0.6250 | -0.2417 | 0.4262 |
| 08 112.5 to 75 | 0.8696 | 0.6364 | -0.2332 | 0.5376 |
| 09 90 to 80 | 0.9565 | 0.6500 | -0.3065 | 0.2905 |
| 10 90 to 120 | 1.0000 | 0.6667 | -0.3333 | 0.3641 |
| 11 60 to 80 | 0.9268 | 0.6667 | -0.2602 | 0.4507 |
| 12 80 to 150 | 0.8000 | 0.8421 | +0.0421 | 0.6408 |
| 13 180 to 120 | 0.7500 | 0.6400 | -0.1100 | 0.4266 |
| 14 240 to 96 | 0.7826 | 0.7692 | -0.0134 | 0.3795 |
| 15 85 to 127.5 | 0.6486 | 0.4615 | -0.1871 | 0.6903 |
| 18 piano rubato | 0.7568 | 0.7586 | +0.0019 | 0.6513 |
| 19 ramp 80 to 200 | 0.6667 | 0.8511 | +0.1844 | 0.4057 |
| 20 ramp 200 to 80 | 0.7077 | 0.8182 | +0.1105 | 0.2755 |
| 21 polyrhythm 70 to 105 | 0.7805 | 0.6000 | -0.1805 | 0.4593 |

The five cases improving by more than 0.02 F1 averaged approximately 0.55
onset strength, compared with approximately 0.45 for the other ten. This is a
weak correlation, not a classifier. The strongest no-false-positive threshold
on this already-opened calibration set recovered only two of five improvements;
the regressive case 15 was almost as strong as the highest-scoring positives,
while the two useful ramp cases were below several negatives.

The onset envelope is retained because it is backend-neutral evidence useful
for future meter, accent, boundary, and alternative-pulse analysis. It is not
added to the hypothesis score, does not trigger midpoint augmentation, and is
not exposed as a product strategy. Automatic metrical selection still requires
independent meter- or structure-level evidence and validation on a fresh,
precommitted holdout.
