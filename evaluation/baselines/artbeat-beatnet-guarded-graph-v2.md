# ARTBeaT BeatNet guarded candidate graph v2

Date: 2026-08-24

This calibration replaces the initial BeatNet virtual-grid selector with one
internal guarded candidate graph. It does not add a user strategy, inspect the
Vienna 4x22 holdout, or promote BeatNet as the shipping default.

## Reproducible input

- suite: `evaluation/suites/artbeat-v1.json` (`calibration`)
- model pack: `models/beatnet-v1.json`
- model manifest SHA-256:
  `dcc6aeb313fda31ab862d287976cc7d7bc996e1ee78fc77b028e8b0a9d69b1e5`
- backend: native Rust frontend, RTen ONNX inference, guarded candidate graph
- previous candidate: `artbeat-beatnet-viterbi-v1`
- untouched holdout: not opened

## Result

| Metric | BeatNet v1 | Guarded graph v2 |
| --- | ---: | ---: |
| mean selected-sequence beat F1 | 0.8080 | 0.8536 |
| mean precision | 0.8418 | 0.8513 |
| mean recall | 0.8264 | 0.8894 |
| candidate truth-beat coverage | 459 / 460 | 459 / 460 |
| cases improved / unchanged / regressed | - | 4 / 10 / 1 |
| end-to-end cases passing locked gates | 1 / 15 | 3 / 15 |
| mean end-to-end runtime on the development VDI | 1.92 s | 0.88 s |

The graph traverses only real local model maxima. A state contains the latest
inter-beat interval and 2/3/4-beat bar phase. Beat/downbeat/non-beat evidence
may override a soft virtual-grid prior across a sustained tempo change, but
metrical-level transitions have an explicit cost. Continuity guards reject a
fragmented path, and edge completion can restore only a real model maximum.

`artbeat-10-90-to-120` and `artbeat-14-240-to-96` reached beat F1 1.0. The
accelerating and decelerating ramps rose from approximately 0.66 and 0.78 to
0.91 and 0.82. All other v1 results were preserved except piano rubato, which
fell from approximately 0.62 to 0.59.

The piano regression is not safely thresholdable. The neural activation score
strongly prefers the dense half/double-time alternative, so forcing a BPM band
would conceal unresolved ambiguity. Keep v2 experimental, retain explicit
metrical alternatives, and do not open the holdout until independent meter or
accent evidence can preselect one complete result.
