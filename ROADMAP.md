# Roadmap

## Phase 1: training-free timing engine

- Beat This adapter and parity fixtures.
- Robust BPM curve and metrical alternatives.
- Constant, ramp, jump, and discontinuity segmentation.
- CLI, C ABI, and observation-driven WASM API.
- Synthetic constant/step/ramp tests and a licensed evaluation manifest.

## Phase 2: product surfaces

- Signed model packs and a license/provenance manifest.
- End-to-end browser inference with a measured WASM backend.
- Native GUI for waveform, beat grid, confidence, and editable tempo segments.
- Export adapters for common rhythm-game and DAW tempo-map formats.

## Phase 3: evidence-driven model work

- Evaluate game music, rubato, extreme tempo, compound meter, and drumless audio.
- Add a learned boundary/confidence head only if deterministic analysis is the
  measured bottleneck.
- Consider a custom multitask model only if the Beat This observation backend is
  the measured bottleneck.

### Bottleneck attribution protocol

Run the same evaluation cases through two paths:

1. **oracle observations** feed annotated beat/downbeat timestamps directly to
   `TempoMapEstimator` and isolate tempo-curve and segmentation quality;
2. **end-to-end observations** run audio through Beat This! before the same
   estimator and measure the complete product path.

Compare beat F1 and timing error, tempo-curve median/P95 error, change-point
precision/recall, and section-boundary error by capability slice. Keep at least
game music, rubato, extreme tempo, compound meter, drumless audio, stops, and
half/double-time ambiguity as separate slices rather than hiding them in one
aggregate score.

Use the difference between the two paths to decide where model work belongs:

- If oracle observations pass but end-to-end observations fail, treat the beat
  observation backend as the primary bottleneck. First test backend calibration,
  decoding, and metrical-level behavior; consider a custom multitask model only
  after those alternatives are exhausted.
- If both paths fail on the same tempo or boundary cases, treat the deterministic
  estimator as the primary bottleneck. Improve robust statistics, segmentation,
  or add a learned boundary/confidence head without replacing a beat tracker that
  is already adequate.
- If both paths pass, keep the training-free design and spend complexity on
  product surfaces, export fidelity, performance, and confidence calibration.

Do not train on the hidden holdout or relax suite thresholds to accommodate a
candidate. Require a material improvement across the failing slices without a
regression in constant-tempo, platform, memory, and latency gates before adding
a learned component to the default distribution.
