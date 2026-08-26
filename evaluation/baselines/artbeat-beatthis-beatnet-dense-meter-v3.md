# ARTBeaT Beat This / BeatNet dense meter evidence v3

Date: 2026-08-27

This calibration removes a selection bias from
`artbeat-beatthis-beatnet-meter-gated-v2.md`. The earlier report sampled
BeatNet downbeat confidence only at events already retained by BeatNet's
decoder. This follow-up retains BeatNet's complete 50 Hz pulse and downbeat
activation series before peak picking, then evaluates the same weight-free
Pareto gate at every timestamp in each Beat This hypothesis.

## Reproducible inputs

- suite: `evaluation/suites/artbeat-v1.json` (`calibration`)
- primary report:
  `D:/rhythm-map-eval/reports/artbeat-cache-default-v9-final.json`
- dense secondary report:
  `D:/rhythm-map-eval/reports/artbeat-beatnet-dense-meter-v5.json`
- agreement tolerance: 70 ms, one-to-one matching
- meter gate policy: `pareto-beat-agreement-dense-downbeat-meter-v2`
- diagnosis report schema: 3
- local diagnosis:
  `D:/rhythm-map-eval/reports/artbeat-beatthis-beatnet-dense-meter-v3.json`

Backend report schema 10 stores the optional uniform activation series under
observation diagnostics. It remains observation-layer evidence and is not
copied into product `Analysis` output.

## Result

| Measure | Beat This primary | Decoded-event gate v1 | Dense-activation gate v2 |
| --- | ---: | ---: | ---: |
| Mean beat F1 | 0.80516 | 0.82097 | 0.80516 |
| Delta | - | +0.01581 | 0.00000 |
| Improved cases | - | 2/15 | 0/15 |
| Regressed cases | - | 0/15 | 0/15 |
| Calibration gate | - | apparent pass | fail |

All four alternatives favored by global backend agreement have a negative
dense downbeat-periodicity margin:

| Case | Agreement margin | Dense meter margin | Annotated F1 delta if selected |
| --- | ---: | ---: | ---: |
| `75-to-150` | +0.318 | -0.204 | +0.057 |
| `60-to-80` | +0.048 | -0.247 | -0.260 |
| `piano-rubato` | +0.204 | -0.167 | +0.008 |
| `ramp-80-to-200` | +0.051 | -0.132 | +0.180 |

The dense gate therefore keeps the primary on every case. This is safe but
not useful as a selector.

## Decision

Reject the decoded-event meter gate and do not spend a fresh holdout on it.
Keep dense activations as auditable evaluation evidence because they expose the
selection bias and allow future meter experiments to inspect the model before
its own decoder commits to timestamps. Do not add BeatNet to the default bundle
or expose a backend/strategy choice. Further selector work needs evidence that
can distinguish locally changing metrical level, not another threshold over
the same whole-track 2/3/4 periodic score.
