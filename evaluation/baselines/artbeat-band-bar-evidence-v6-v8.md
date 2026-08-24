# ARTBeaT band and bar evidence calibration v6-v8

These calibration runs tested two independent additions to midpoint evidence:
frequency-band composition of deterministic spectral flux, and periodic
downbeat structure. None of the experiments changed the shipping beat sequence
or used truth while constructing a hypothesis.

The reports are retained outside Git below the configured evaluation report
root:

- `artbeat-onset-bands-v6.json`;
- `artbeat-bar-periodicity-v7.json`; and
- `artbeat-local-bar-midpoints-v8.json`.

All use Beat This model manifest SHA-256
`ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`.

## Frequency-band onset evidence

The existing positive spectral flux was divided into low (below 250 Hz), mid
(250 Hz through 2 kHz), and high (above 2 kHz) contributions. Contributions
sum to the normalized full-band onset strength at every frame and reuse the
same FFT.

| Case | Midpoint F1 delta | Low | Mid | High |
| --- | ---: | ---: | ---: | ---: |
| 05 75 to 150 | +0.0568 | 0.110 | 0.201 | 0.380 |
| 06 150 to 75 | +0.0797 | 0.099 | 0.231 | 0.400 |
| 07 75 to 112.5 | -0.2417 | 0.098 | 0.087 | 0.241 |
| 08 112.5 to 75 | -0.2332 | 0.160 | 0.118 | 0.260 |
| 09 90 to 80 | -0.3065 | 0.075 | 0.063 | 0.153 |
| 10 90 to 120 | -0.3333 | 0.080 | 0.091 | 0.193 |
| 11 60 to 80 | -0.2602 | 0.057 | 0.060 | 0.334 |
| 12 80 to 150 | +0.0421 | 0.109 | 0.213 | 0.319 |
| 13 180 to 120 | -0.1100 | 0.080 | 0.098 | 0.248 |
| 14 240 to 96 | -0.0134 | 0.109 | 0.123 | 0.147 |
| 15 85 to 127.5 | -0.1871 | 0.321 | 0.169 | 0.200 |
| 18 piano rubato | +0.0019 | 0.204 | 0.413 | 0.034 |
| 19 ramp 80 to 200 | +0.1844 | 0.030 | 0.020 | 0.356 |
| 20 ramp 200 to 80 | +0.1105 | 0.000 | 0.012 | 0.263 |
| 21 polyrhythm 70 to 105 | -0.1805 | 0.137 | 0.030 | 0.293 |

The five cases improving by more than 0.02 F1 averaged approximately 0.35
high-frequency contribution, versus 0.21 for the other ten. That direction did
not provide separation: the regressive 60-to-80 case scored 0.33. The band
split is retained as general backend-neutral onset metadata, but no band value
changes hypothesis ranking.

## Downbeat periodicity and local bar support

A whole-track diagnostic searched for the strongest downbeat phase at event
periods from two through eight. It did not discriminate the hypotheses:
useful and regressive midpoint sequences both commonly changed from four to
eight events per detected period or remained at eight. A variable-tempo track
cannot safely be reduced to one whole-track bar period, so the diagnostic
implementation was removed.

A stricter local candidate then added real midpoint peaks only inside two
consecutive spans where three decoded downbeats were each separated by two
selected events. It activated in 10 of 15 cases:

| Case | Selected F1 | Local-bar F1 | Delta |
| --- | ---: | ---: | ---: |
| 06 150 to 75 | 0.7727 | 0.9600 | +0.1873 |
| 08 112.5 to 75 | 0.8696 | 0.6897 | -0.1799 |
| 11 60 to 80 | 0.9268 | 0.8421 | -0.0847 |
| 12 80 to 150 | 0.8000 | 0.8750 | +0.0750 |
| 13 180 to 120 | 0.7500 | 0.6222 | -0.1278 |
| 14 240 to 96 | 0.7826 | 0.7391 | -0.0435 |
| 18 piano rubato | 0.7568 | 0.8500 | +0.0932 |
| 19 ramp 80 to 200 | 0.6667 | 0.8163 | +0.1497 |
| 20 ramp 200 to 80 | 0.7077 | 0.8571 | +0.1495 |
| 21 polyrhythm 70 to 105 | 0.7805 | 0.6545 | -0.1259 |

Five activations improved and five regressed. Activated-case mean F1 moved
from 0.7809 to 0.7910, which is not a safe selector. After opening the results,
mean onset strength above 0.6 happened to retain four improvements and no
regressions. That threshold is post-hoc and therefore cannot be validated on
the same calibration set. The existing ARTBeaT holdout has already been opened
for an earlier decoder experiment, and the corpus has only one otherwise unused
case, so it cannot provide a fresh holdout of useful size.

Both bar algorithms were removed rather than retained as strategies. A future
selector needs an untouched timestamped corpus or independent meter evidence
from another observation backend. Until then, the product should preserve
metrical ambiguity instead of choosing from this calibration history.
