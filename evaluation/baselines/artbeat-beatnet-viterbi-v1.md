# ARTBeaT BeatNet Viterbi calibration

Date: 2026-08-24

This calibration compares a pinned BeatNet ONNX graph with the existing Beat
This observation baseline. It does not open the Vienna 4x22 holdout and does
not promote another product strategy.

## Reproducible input

- suite: `evaluation/suites/artbeat-v1.json` (`calibration`)
- model pack: `models/beatnet-v1.json`
- model manifest SHA-256:
  `dcc6aeb313fda31ab862d287976cc7d7bc996e1ee78fc77b028e8b0a9d69b1e5`
- backend: native Rust BeatNet frontend, RTen ONNX inference, one
  evidence-snapped variable-tempo Viterbi path
- comparator: `artbeat-candidate-coverage-v1` Beat This upstream sequence

## Result

| Metric | Beat This | BeatNet candidate |
| --- | ---: | ---: |
| mean selected-sequence beat F1 | 0.8052 | 0.8080 |
| candidate truth-beat coverage | 450 / 460 | 459 / 460 |
| candidate micro recall | 0.9783 | 0.9978 |
| mean local-max candidate count | 102.33 | 110.20 |
| cases improved / unchanged / regressed | - | 7 / 1 / 7 |
| end-to-end cases passing locked gates | - | 1 / 15 |
| mean end-to-end runtime on the development VDI | - | 1.92 s |

The first single-frame activation-gate baseline reached only 0.7801 mean beat
F1 because adjacent strong peaks were accepted independently. Replacing that
gate with the one sequence path restored precision and slightly exceeded the
Beat This aggregate, while still emitting only real local model maxima.

The result is useful but not promotable. Candidate coverage shows that BeatNet
adds evidence, especially on several metrical and tempo-change cases, but the
current path loses recall on fast changes, ramps, and piano rubato. The nearly
zero aggregate gain hides equal numbers of gains and regressions. Continue with
one improved sequence decoder on this calibration set; do not retain the
single-frame gate, expose a backend strategy list, or inspect Vienna until one
complete candidate has been selected.

Follow-up: `artbeat-beatnet-guarded-graph-v2.md` records the single merged
decoder that supersedes this calibration candidate without exposing v1 as a
runtime strategy.
