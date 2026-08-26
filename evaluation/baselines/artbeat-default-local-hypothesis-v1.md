# ARTBeaT default locally varying hypothesis v1

Date: 2026-08-27

This baseline promotes the frozen harmonic-aware locally varying path from an
evaluation-only candidate into the single shipping estimator. It remains an
additional ambiguity result: no user strategy is added and it cannot replace
selected beats, the BPM curve, tempo segments, change points, or rhythm
sections.

## Reproducible input

- suite: `evaluation/suites/artbeat-v1.json` (`calibration`)
- model pack: `models/beat-this-full-v1.json`
- model manifest SHA-256:
  `ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`
- backend: Beat This through RTen, immutable upstream decoder
- estimator: shipping default
- analysis schema: 3
- report schema: 10
- observation cache: 15/15 hits
- report:
  `D:/rhythm-map-eval/reports/artbeat-default-local-hypothesis-v10.json`
- comparison default:
  `D:/rhythm-map-eval/reports/artbeat-cache-default-v9-final.json`
- comparison frozen candidate:
  `D:/rhythm-map-eval/reports/artbeat-cache-local-metrical-v8.json`

## Result

The new default emitted a distinct `locally_varying` hypothesis on 10/15 cases.
All primary end-to-end metrics, deltas, raw observations, and selected beat
timestamps exactly match the previous default report. Every emitted hypothesis
sequence and score exactly matches the frozen `local-metrical-path-v1` report;
the only intentional diagnostic addition is
`locally_varying_metrical_hypothesis_available`.

Relative scores are normalized across all returned alternatives. Therefore the
selected hypothesis's relative score may decrease when a newly published local
path has a stronger truth-free evidence score; this changes ambiguity metadata,
not the selected sequence or time map.

| Cached deterministic work | Total | Mean per case |
| --- | ---: | ---: |
| Previous shipping default | 364.70 ms | 24.31 ms |
| New default with harmonic local path | 637.70 ms | 42.51 ms |
| Frozen candidate reference run | 770.92 ms | 51.39 ms |

The measured incremental cost over the previous default is 18.20 ms per short
evaluation track. These are hot-cache deterministic post-processing timings,
not neural-inference benchmarks.

## Promotion evidence

The candidate had already passed a corpus-disjoint generalization check before
this promotion:

- ARTBeaT with BeatNet: best-top-K mean beat F1 rose from 0.8620 to 0.9180;
  7/15 cases improved, none regressed, and 13/15 emitted a distinct path.
- Frozen Vienna 4x22 holdout: 6/12 emitted a path and all six improved; the
  primary/local coverage ceiling rose from 0.4018 to 0.4772.

Those results establish value as additional coverage, not as a selector. On
both calibration history and the holdout, the local path is not uniformly
better and its truth-free relative score cannot identify the canonical musical
pulse safely.

## Decision

Ship the local path automatically when the existing evidence gates pass. Keep
all constants internal, synthesize no timestamps, preserve the selected time
map, and expose ambiguity through the existing schema-v3 `beat_hypotheses`
field. Retain `local-metrical-path-v1` only as an evaluation-report compatibility
alias. Do not tune on the opened Vienna recordings and do not introduce a public
strategy switch.
