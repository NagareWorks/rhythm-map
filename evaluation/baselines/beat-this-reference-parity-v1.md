# Beat This reference parity v1

Date: 2026-09-03

This is a numerical implementation audit, not a new music-accuracy baseline.
The tooling and reference identity are described in
[`../parity/README.md`](../parity/README.md). No holdout audio was opened, no
model was trained, and no production estimator or decoder behavior changed.

## Scope and identity

- Official Python: CPJKU/beat_this revision
  `b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c`, also pinned by the Rust port's
  `scripts/gen_golden.py`.
- Rust port: beat-this 1.0.0, revision
  `089b509247e6fdcec666511c0dcf0d5f39c21e73`.
- Official `final0` checkpoint SHA-256:
  `8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331`.
- Shipping ONNX model-pack manifest SHA-256:
  `ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`.
- Reference: PyTorch/torchaudio 2.8.0 CPU, ONNX Runtime 1.23.2 CPU; RTen 0.24.0.
- ARTBeaT `artbeat-05-75-to-150`: complete approximately 13.6-second MP3,
  originally 44.1 kHz; 680 mel frames; 17 selected beats and five downbeats.
- RUBATO `rubato-bach-bwv1007-01-ar-macleod2011`: first 35 seconds after
  decoding, originally 22.05 kHz; 1,751 frames; 50 beats and 24 downbeats.
  This crosses the 30-second model chunk boundary but is not an end-of-track test.

Both are existing calibration cases. Audio identities and model bytes are
verified before use. Raw traces contain PCM and remain outside Git. The compact
aggregate report is `../parity/baseline-v1.json`.

## Comparisons

All 16 comparisons per case pass their predeclared numerical budgets:

1. Same PCM: official torchaudio frontend and ONNX Runtime frontend versus RTen.
2. Same mel: official checkpoint versus converted ONNX, and ONNX Runtime versus
   RTen, separately for beat and downbeat logits. Official split/aggregate code
   independently checks the Rust port's padding and keep-first chunk stitching.
3. Same logits: official minimal postprocessor versus the Rust port, and the
   Rust port versus Rhythm Map's immutable default adapter decoder.
4. Same PCM: official complete frontend/checkpoint versus RTen, including
   decoded event agreement.
5. Same encoded file: official decode/downmix/soxr path versus the shipping
   decode/resample path, checking final event counts and one-frame agreement.

The largest official-versus-RTen mel error is about `1.12e-4`; the largest
checkpoint-versus-ONNX logit error is about `3.43e-5`. Same-PCM official-versus-
RTen logits differ by at most about `5.15e-5`. Port-versus-adapter timestamps
are identical. Same-PCM official-versus-port timestamp differences are below
two microseconds, consistent with the port's f32 timestamp storage.

## A separate source-audio timing discrepancy

On the 44.1 kHz ARTBeaT file, the shipping decoder produces one fewer 22.05 kHz
sample than the reference and the first two seconds of its waveform best align
when delayed by 63 samples (about 2.86 ms). Unshifted waveform RMSE is 0.07913;
diagnostically aligning that delay reduces it to about 0.000109. The original
file's beat/downbeat sequences have identical counts but some timestamps differ
by one 20 ms frame. They pass the existing upstream-style one-frame budget,
not bit-exact file-path parity.

On the native-22.05 kHz RUBATO prefix, decoded PCM is exactly identical and no
delay is detected. The Rust port's sinc resampler returns `process(..., 0, None)`
output directly without compensating `output_delay()` or explicitly flushing
the tail. This is a concrete suspect for the rate-dependent time-axis mismatch;
the current audit has not implemented a correction or established all-rate
behavior. No actual waveform or output timestamps were shifted by this audit.

## Decision

These samples do not show a wrong model conversion, frontend definition,
RTen numerical failure, or chunk/decoder mismatch explaining the large musical
accuracy failures. That is narrower than claiming complete reference parity or
proving that every failure is inherent to the model.

Next isolate the file/PCM resampling contract using impulses, short buffers,
tail events, and 44.1/48/96 kHz input; compensate delay and preserve duration at
the resampling boundary if confirmed. Do not subtract one hard-coded offset
from returned beats. Any production preprocessing change must bump the
observation-cache contract and replay calibration metrics before promotion.
Keep the sealed holdout for a genuine accuracy candidate, not this diagnosis.
