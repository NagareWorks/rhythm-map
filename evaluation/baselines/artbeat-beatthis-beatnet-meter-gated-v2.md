# ARTBeaT Beat This / BeatNet meter-gated consensus v2

Date: 2026-08-26

Status: superseded and rejected by the dense-activation follow-up in
`artbeat-beatthis-beatnet-dense-meter-v3.md`. The apparent gain below depended
on measuring downbeat evidence only at events already selected by BeatNet.

This calibration follows the rejected global-agreement selector in
`artbeat-beatthis-beatnet-consensus-v1.md`. It tests whether BeatNet's downbeat
channel can veto relative beat-sequence agreement when that agreement chooses
the wrong absolute metrical level. It does not change product output, train a
model, synthesize timestamps, or open either timestamped holdout.

## Reproducible inputs

- suite: `evaluation/suites/artbeat-v1.json` (`calibration`)
- primary report:
  `D:/rhythm-map-eval/reports/artbeat-cache-default-v9-final.json`
- secondary report:
  `D:/rhythm-map-eval/reports/artbeat-beatnet-local-metrical-path-v4.json`
- agreement tolerance: 70 ms, one-to-one matching
- meter gate policy: `pareto-beat-agreement-downbeat-meter-v1`
- diagnosis report schema: 2
- local report:
  `D:/rhythm-map-eval/reports/artbeat-beatthis-beatnet-consensus-v2.json`

For each Beat This hypothesis, the diagnosis maps BeatNet downbeat confidence
onto its real timestamps. It evaluates every phase of 2-, 3-, and 4-pulse bar
cycles and records the best class-balanced mean log likelihood: half of the
score comes from expected downbeat positions and half from expected ordinary
beat positions. A missing BeatNet event receives probability 0.01.

The frozen gate has no learned weight or threshold. It retains the Beat This
primary unless another existing hypothesis strictly improves both:

1. beat-sequence F1 against BeatNet's selected sequence; and
2. BeatNet downbeat periodic log likelihood.

If more than one hypothesis dominates the primary, the existing agreement rank
selects among those eligible hypotheses. Ground truth is read only afterward.

## Result

| Measure | Beat This primary | Global agreement v1 | Meter-gated v2 |
| --- | ---: | ---: | ---: |
| Mean beat F1 | 0.80516 | 0.80416 | 0.82097 |
| Delta | - | -0.00100 | +0.01581 |
| Improved cases | - | 3/15 | 2/15 |
| Regressed cases | - | 1/15 | 0/15 |
| Calibration gate | - | fail | pass |

The gate changes only two cases:

| Case | Agreement margin | Meter margin | Annotated F1 delta |
| --- | ---: | ---: | ---: |
| `artbeat-05-75-to-150` | +0.318 | +0.139 | +0.057 |
| `artbeat-19-ramp-80-to-200` | +0.051 | +0.069 | +0.180 |

It rejects both other global-agreement switches:

| Case | Agreement margin | Meter margin | Decision |
| --- | ---: | ---: | --- |
| `artbeat-11-60-to-80` | +0.048 | -0.003 | retain primary |
| `artbeat-18-piano-rubato` | +0.204 | -0.270 | retain primary |

This specifically prevents the v1 `60-to-80` regression of -0.260 beat F1.
Downbeat evidence is useful only as a veto combined with independent beat
agreement. Selecting by meter likelihood alone yields mean F1 0.71893, improves
three cases, and regresses seven.

## Decision

Freeze the conjunction as one evaluation candidate. Do not expose it as a
strategy or put BeatNet in the default product bundle yet. The result is
promising but still assumes a 2/3/4-pulse bar vocabulary and observes downbeat
confidence only at decoded BeatNet events. Promotion requires a newly
precommitted, meter-diverse timestamped holdout that was not used to design this
gate. Failure there means retaining explicit metrical ambiguity, not adding
calibration-specific thresholds.
