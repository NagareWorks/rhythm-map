# ARTBeaT BeatNet explicit sequence hypotheses v3

Date: 2026-08-25

This calibration keeps the guarded candidate-graph v2 primary sequence and
makes metrical uncertainty a product result. It does not add a user strategy,
change any selected ARTBeaT path, or inspect the Vienna holdout.

## Reproducible input

- suite: `evaluation/suites/artbeat-v1.json` (`calibration`)
- model pack: `models/beatnet-v1.json`
- model manifest SHA-256:
  `dcc6aeb313fda31ab862d287976cc7d7bc996e1ee78fc77b028e8b0a9d69b1e5`
- backend: native Rust frontend, RTen ONNX inference, guarded candidate graph
- analysis schema: 2
- calibration report schema: 6
- untouched holdout: not opened

## Primary result

Primary selected-sequence metrics are byte-for-byte equivalent in meaning to
v2: mean beat F1 0.8536, precision 0.8513, recall 0.8894, candidate truth-beat
coverage 459/460, and 3/15 complete strict-gate passes. No case changed its
selected beat sequence.

Each analysis now includes truth-free, backend-supported `beat_hypotheses`:

- the selected sequence;
- both alternating half-time phases when their implied tempo remains in range;
- a double-time sequence only when at least three real midpoint candidates
  exist and its implied tempo remains in range.

Scores combine event evidence, interval continuity, and retained selected
evidence, then normalize the strongest returned sequence to 1.0. They are
relative scores, not probabilities. Every timestamp is an accepted observation
or a backend candidate.

For `artbeat-18-piano-rubato`, the selected dense sequence scores 1.00 while the
two half-time phases score approximately 0.81 and 0.87. Neither fixed phase is
the correct locally changing pulse, so v3 does not claim to solve the piano
case; it makes the ambiguity explicit and machine-readable.

## Rejected unique-selection attempts

A PCM-only adaptive half-time beam reached beat F1 0.45 on piano and measured
only a 1.09 retained/discarded accent ratio. Removing the candidate graph's
per-event bias left piano unchanged, lowered aggregate F1 to 0.8364, and
regressed three real variable-tempo cases. Neither experiment is retained.

The next candidate must represent a locally varying metrical path and add
independent long-range meter or harmonic evidence. Do not tune another global
BPM band or open the holdout merely because uncertainty is now serialized.
