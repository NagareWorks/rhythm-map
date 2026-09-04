# Active-region hypothesis adoption audit

Status: **do not promote the experimental automatic selector**. The shipping
estimator, model, observation contracts, API and default options are unchanged.

## What was tested

Private calibration diagnostics reused the PCM-backed RUBATO cache replay and
the validated ARTBeaT v2 evidence export. No model training, new neural inference,
parameter search or holdout evaluation was performed. ARTBeaT was used in earlier
development and is a separate calibration corpus, not untouched holdout.

The existing locally varying path requires a chain spanning the file, with
candidate intervals between 0.1875 and 1.5 seconds under the default BPM limits.
On RUBATO, silence filtering makes that chain impossible in 22/25 recordings.
Private prototypes instead constructed paths within active intervals, preserving
silence and unsupported gaps. All events remained real backend candidates;
the existing path weights were unchanged. Candidate availability increased from
3/25 to 25/25 recordings, but availability is not evidence of correctness.

An experimental selector applied the existing evidence/continuity/retention
weights locally. A subsequent domain guard abstained when the original sequence
had fewer than three events. The experiments used raw observation confidences,
not final repaired downbeat confidences; they are not exact shipping-score replay.

## Cross-corpus result

These are **mean per-recording Beat F1** values, not release acceptance scores.

| Corpus / method | Mean Beat F1 | Improved / regressed recordings |
| --- | ---: | ---: |
| RUBATO selected baseline | 0.5212634 | baseline |
| RUBATO automatic selector with domain guard | 0.5322282 | 11 / 3 |
| ARTBeaT selected baseline | 0.8079606 | baseline |
| ARTBeaT same automatic selector with domain guard | 0.7982098 | 1 / 4 |

On RUBATO the accepted edits recover 66 previously missed truth identities but
lose 65 previously matched ones. On ARTBeaT the selector accepts 21 deletions:
eight unmatched events are removed, but 13 true beats are lost. The ARTBeaT
240-to-96 case falls from F1 0.810811 to 0.666667, with nine fewer matched beats.
A reduction in false positives must not conceal lost beat coverage.

Replacing segment-median continuity with adjacent-interval continuity was also
rejected: RUBATO regressions increase from three to nine. Common-anchor-relative
acoustic features provide some descriptive signal, but no reliable adoption gate.
In particular, ARTBeaT relative-onset deletion-protection macro AUC falls from
0.911 to 0.607 on the same eligible samples; only four recordings contain both
outcome classes. Relative harmonic evidence improves on that small subset, but
this is insufficient to establish transfer. Undefined AUC stays unknown.

## Safety and uncertainty boundaries

- A higher relative hypothesis score is not calibrated correctness, and does
  not by itself authorize replacing primary beat timestamps.
- A candidate that spans active audio need not span padding or rests. Conversely,
  missing full-file hypotheses do not authorize fabricating beats to bridge gaps.
- Insufficient data and low-quality evidence are different. The private score
  function returns zero below three events; the public short-input branch returns
  one selected hypothesis with relative score **1.0**, no global BPM, no tempo
  curve, and `too_few_beats_for_tempo_curve`. That rank is not reliability.
- A timestamp shared by two hypotheses is not necessarily a correct beat. Edited
  events cannot be their own anchors; absent context cannot be replaced by zeros.
- No threshold derived from these calibration failures is added to the product.
  Active-region generation and automatic primary selection are separate decisions.

## Automated regression coverage

`crates/rhythm-map-core/tests/hypothesis_adoption.rs` exercises the public default
API using hand-authored observations, with no audio, model pack or corpus access:

1. alternating genuine beat intervals with harmonic accents on every other beat:
   a sparse alternative scores higher but must not replace the primary sequence;
2. leading/trailing padding: preserve primary timestamps without inventing events;
3. an internal rest: do not fill the gap to force candidate connectivity;
4. two primary events plus dense candidates: retain the insufficient-data result
   rather than treating a relative rank as confidence or fabricating a tempo map.

The padding/rest absence checks characterize the **current** full-file path
limitation. A validated active-region generator may intentionally revise them;
the no-invented-beats and no-ranking-only-adoption protections must remain.
These are synthetic failure-mode tests, not replays of the real recordings and
not a claim that the default algorithm solves the observed metrical errors.

Run with:

```sh
cargo test -p rhythm-map-core --test hypothesis_adoption
```

The existing workspace-test CI matrix includes this integration test on Linux,
Windows and macOS. Tests live outside the hash-locked estimator implementation,
so the existing PCM/cache replay source identities remain valid.

## Evidence and next gate

The detailed path, event and context experiments remain private diagnostics;
this document summarizes them rather than claiming a public real-corpus replay
command. They verified 25/25 RUBATO and 15/15 ARTBeaT historical path/absence
results, 2,062 identical RUBATO context-feature records before transfer, and
110,130 indexed-nearest queries against exhaustive lookup during transfer.
Existing public input/provenance audits remain:

- [RUBATO scoped cache replay](beat-this-rubato-cache-replay-v1.md).
- [Frozen candidate evidence definitions](../parity/candidate-evidence-lock-v1.json).
- [ARTBeaT preprocessing baseline](../parity/reference-resampler-calibration-v1.json).

Do not promote this selector or add per-recording exceptions. Future adoption
requires explicit evidence for metrical level, abstention when comparison is
unsupported, exact Rust/final-confidence parity, and cross-corpus beat/BPM/
downbeat/change-point checks. This negative result does not imply that all
active-region analysis is invalid or that training a new model is required.
