# ARTBeaT Beat This / BeatNet consensus diagnosis v1

Date: 2026-08-26

This calibration asks a narrow question: can BeatNet's independently inferred
primary beat sequence choose among Beat This's already published pulse
hypotheses without labels? It does not train a model, add timestamps, change a
shipping result, expose a strategy, or reopen either timestamped holdout.

## Reproducible inputs

- suite: `evaluation/suites/artbeat-v1.json` (`calibration`)
- primary: Beat This shipping-default report schema 9,
  `D:/rhythm-map-eval/reports/artbeat-cache-default-v9-final.json`
- secondary: BeatNet guarded-graph/local-path report schema 7,
  `D:/rhythm-map-eval/reports/artbeat-beatnet-local-metrical-path-v4.json`
- agreement tolerance: 70 ms, one-to-one matching
- diagnosis report schema: 1
- local report:
  `D:/rhythm-map-eval/reports/artbeat-beatthis-beatnet-consensus-v1.json`
- selection rule: choose the Beat This hypothesis with greatest whole-track F1
  against BeatNet's top-ranked sequence; ties retain the earlier Beat This rank

The choice sees only backend timestamps and stable hypothesis ranks. Annotated
F1 is read after selection for attribution and never enters the choice.

## Result

| Measure | Beat This primary | Naive consensus | Delta |
| --- | ---: | ---: | ---: |
| Mean beat F1 | 0.80516 | 0.80416 | -0.00100 |
| Improved cases | - | 3/15 | - |
| Regressed cases | - | 1/15 | - |
| Calibration gate | - | fail | - |

All four changed choices selected the real-timestamp midpoint-augmented Beat
This hypothesis:

| Case | Agreement margin | Window support reversals | Annotated F1 delta |
| --- | ---: | ---: | ---: |
| `artbeat-05-75-to-150` | +0.318 | 0 | +0.057 |
| `artbeat-11-60-to-80` | +0.048 | 1 | -0.260 |
| `artbeat-18-piano-rubato` | +0.204 | 0 | +0.008 |
| `artbeat-19-ramp-80-to-200` | +0.051 | 2 | +0.180 |

The regression explains why global consensus is not an absolute metrical
anchor. In `60-to-80`, BeatNet follows roughly double-time during the first half
and converges to Beat This's level later. Whole-track F1 rewards the denser Beat
This alternative because it covers both regions, even though the annotated
musical pulse follows the sparse hypothesis. The quarter-track agreement margin
changes from positive to negative and records one material support reversal.

## Decision

Reject the naive selector. A second model is useful corroborating evidence, but
two beat sequences can identify their relative octave relationship without
proving which level is the musically canonical beat. Use backend agreement to
increase confidence when level and phase remain stable; otherwise retain
explicit ambiguity. Any automatic resolution now needs independent meter or
downbeat semantics, or a locally varying real-timestamp path, and must be frozen
before a new timestamped holdout is opened.
