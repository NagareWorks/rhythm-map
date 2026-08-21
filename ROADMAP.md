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
- Independent downbeat/bar-phase evaluation and evidence-based half-bar
  candidate selection with recovered-grid boundary realignment.
- Content-addressed external-audio resolution, strict external truth
  validation, and a private calibration workflow that never stores audio paths
  or bytes in reports.
- Immutable public-dataset acquisition and an ARTBeaT oracle suite covering 15
  CC BY 4.0 tempo-step, ramp, rubato, and polyrhythm exercises.
- Edge-preserving metrical outlier repair that keeps sustained half/double-time
  tempo changes instead of folding an entire recording into one BPM band.

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

- Populate separate calibration and holdout manifests for legally held
  drumless-control, drumless-ramp, drumless-step, rubato, compound-meter, and
  percussive-control slices.
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

Compare beat and downbeat F1 and timing error, tempo-curve median/P95 error,
change-point precision/recall, and section-boundary error by capability slice.
Keep at least game music, rubato, extreme tempo, compound meter, drumless audio,
stops, and half/double-time ambiguity as separate slices rather than hiding
them in one aggregate score.

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

### Measured bottlenecks

- 2026-08-21: the first ARTBeaT oracle run failed 13 of 15 cases even with exact
  upstream beat timestamps. Global preferred-band folding erased sustained
  octave-related changes and ordinary smoothing blurred jump edges, proving the
  deterministic estimator was the bottleneck for those slices. Replacing that
  policy with bilateral metrical-outlier evidence and edge-preserving smoothing
  made all 15 oracle cases pass without changing suite thresholds; the worst
  tempo P95 error fell to 5.97 percent.
- 2026-08-21: the paired Beat This full-model run passed 1 of 15 ARTBeaT cases,
  with mean beat F1 0.8052, while all paired oracle paths passed. The model often
  remains at a sustained half-time level or omits beats throughout ramps and
  rubato, so the measured bottleneck is now the observation path. Prefer
  comparing alternate model decoding/backends over inventing timestamps in
  deterministic post-processing without acoustic evidence.
- 2026-08-21: a single-inference sweep compared nine Beat This peak decoders.
  The best fixed threshold/window candidate raised mean beat F1 only from
  0.8052 to 0.8179, while lower thresholds increasingly traded precision for
  extra events. A per-case policy oracle reached only 0.8371, and a narrower
  local-maximum window regressed. Keep the upstream decoder as the default;
  next evaluate sequence-aware tempo/phase decoding and an alternate
  observation backend. Require either path to beat this measured ceiling on
  held-out slices before integrating it into the product.
- 2026-08-21: missed-beat logit attribution found that only 42 of 128 ARTBeaT
  misses had an upstream-radius local peak above logit -3; 58 had weaker peaks,
  18 only appeared with a narrower radius, and 10 had no local peak. A
  conservative supported-midpoint decoder raised mean beat F1 from 0.8052 to
  0.8235 and materially improved three cases, but regressed one metrical case.
  Keep it experimental until separate calibration and holdout slices confirm a
  net improvement. The very weak step/ramp failures remain evidence for
  comparing an alternate observation backend rather than expanding grid-based
  timestamp invention.
