# Roadmap

## Phase 1: training-free timing engine

- Beat This adapter and parity fixtures.
- Robust BPM curve and metrical alternatives.
- Constant, ramp, jump, and discontinuity segmentation.
- CLI, C ABI, and observation-driven WASM API.
- Synthetic constant/step/ramp tests and a licensed evaluation manifest.
- Verified Beat This model-pack manifest and paired oracle/end-to-end evaluation.
- Raw beat/confidence diagnostics and deterministic PCM activity observations.
- Evidence-based half-time selection, silence-beat rejection, recovery of short
  model-smeared tempo jumps, and guarded duplicate/missed-event grid repair.

## Phase 2: product surfaces

- Signed model-pack authenticity layered over the existing provenance and
  SHA-256 integrity manifest.
- Async analysis with progress and cancellation across native API, C ABI, and
  GUI surfaces.
- A measured fast/accurate model-pack policy; do not make the full model the
  default merely because it is more accurate in isolation.
- Separate short model smoke tests from scheduled/release full-suite baselines,
  with per-case progress visible to developers.
- Run deterministic CI evaluation without release LTO; reserve optimized
  release builds for model-backed performance baselines.
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

- If oracle observations pass but end-to-end observations fail, treat the broad
  observation path as the primary bottleneck. First separate missed/extra model
  events from deterministic robustness to noisy observations, especially
  half/double-time ambiguity; consider a custom multitask model only after
  calibration, decoding, metrical normalization, and backend alternatives are
  exhausted.
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
