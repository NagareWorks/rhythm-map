# ARTBeaT BeatNet locally varying metrical path v4

Date: 2026-08-25

This calibration keeps the guarded candidate-graph v2 primary sequence and adds
one core-owned, locally varying metrical hypothesis. It does not add a user
strategy, change the primary tempo map, or inspect the Vienna holdout.

## Reproducible input

- suite: `evaluation/suites/artbeat-v1.json` (`calibration`)
- model pack: `models/beatnet-v1.json`
- model manifest SHA-256:
  `dcc6aeb313fda31ab862d287976cc7d7bc996e1ee78fc77b028e8b0a9d69b1e5`
- backend: native Rust frontend, RTen ONNX inference, guarded candidate graph
- estimator policy: `local-metrical-path-v1`
- analysis schema: 3
- calibration report schema: 7
- local report: `D:/rhythm-map-eval/reports/artbeat-beatnet-local-metrical-path-v4.json`
- untouched holdout: not opened

## Result

The primary selected sequence is unchanged in all 15 cases. Its mean beat F1
remains 0.8536, so the candidate cannot silently improve its own calibration
score by replacing a shipping result.

| Coverage measure | Explicit hypotheses v3 | Local path v4 | Delta |
| --- | ---: | ---: | ---: |
| Mean primary beat F1 | 0.8536 | 0.8536 | 0.0000 |
| Mean best-top-K beat F1 | 0.8620 | 0.9180 | +0.0561 |
| Cases improved in best-top-K | - | 7/15 | - |
| Cases regressed in best-top-K | - | 0/15 | - |
| Cases emitting a distinct local path | - | 13/15 | - |

The seven added coverage gains are:

| Case | Previous best-top-K F1 | New best-top-K F1 | Local-path F1 |
| --- | ---: | ---: | ---: |
| `artbeat-05-75-to-150` | 0.8667 | 0.9615 | 0.9615 |
| `artbeat-06-150-to-75` | 0.8525 | 0.9804 | 0.9804 |
| `artbeat-07-75-to-112-5` | 0.8333 | 0.9268 | 0.9268 |
| `artbeat-08-112-5-to-75` | 0.8462 | 1.0000 | 1.0000 |
| `artbeat-12-80-to-150` | 0.8462 | 0.9851 | 0.9851 |
| `artbeat-13-180-to-120` | 0.9851 | 1.0000 | 1.0000 |
| `artbeat-18-piano-rubato` | 0.5873 | 0.8041 | 0.8041 |

The path uses only real backend candidates. Harmonic change is a deterministic
chroma-distance feature around those supported events; no model is trained and
no beat timestamp is synthesized. Ordinary onset autocorrelation was tested
first and rejected because it continued to prefer the dense full-time pulse in
the early piano passage.

## Decision

Retain this candidate as a necessary additional hypothesis, not as another
public strategy and not as the primary selector. Its distinct path is worse
than the selected sequence on several calibration cases, so choosing it
unconditionally would regress the product. The next valid experiment is a
single precommitted run on the untouched Vienna holdout with this exact
candidate; a truth-free selector may be designed only after generalization is
confirmed.
