# Architecture

## Pipeline

```text
interleaved PCM
    -> observation backend
    -> beat/downbeat events + frame confidence + PCM activity
    -> low-activity rejection + evidence-based metrical selection
    -> preliminary normalized tempo curve
    -> guarded short-transition beat-grid recovery
    -> final normalization + robust local tempo curve
    -> piecewise constant/ramp simplification
    -> change points and rhythm-homogeneous sections
    -> versioned Analysis schema
```

`rhythm-map-core` owns every type after the observation boundary. A backend may
use Beat This!, another ONNX model, a streaming tracker, or supplied observations
without changing consumers.

## Observation boundary

Backends implement `RhythmObservationBackend` and return `RhythmObservations`.
They may expose confidence, model identity, frame rate, and an optional activity
envelope, but no backend tensor type crosses into the core schema. The engine
adds a deterministic PCM activity envelope when a backend does not provide one.

The default Beat This adapter consumes its frame logits to attach confidence to
events. The tempo estimator never depends on the upstream crate's Rust structs.

## Analysis boundary

`Analysis.schema_version` versions serialized output independently of crate
versions. Native Rust callers use typed structures. The C ABI initially returns
the same schema as owned UTF-8 JSON to avoid exposing Rust layouts. WASM returns
the same structure through `serde-wasm-bindgen`.

## Tempo inference

The initial estimator is intentionally training-free:

1. Reject invalid or unsorted event sequences.
2. Reject model events inside sustained low-activity spans.
3. Select half-time only when alternating onset salience supports it.
4. Generate a preliminary normalized tempo curve and locate short bracketed
   transitions.
5. Reconstruct a transition grid only when duplicate/missed-event evidence is
   present and both adjacent grids are stable.
6. Recompute octave-equivalent normalization around a robust metrical reference.
7. Median-filter and locally average in log-tempo space.
8. Detect direct jumps and short model-smeared transition blocks.
9. Simplify the curve into constant and ramp segments.
10. Split rhythm sections at tempo changes and beat/audio discontinuities.

This is a deterministic baseline, not the final research endpoint. Evaluation
will determine whether a learned change-point/confidence head is needed.
The formulas, default thresholds, and output interpretation are documented in
[`docs/ALGORITHM.md`](docs/ALGORITHM.md).

## Model packages

Model files are external artifacts. `rhythm-map-models` verifies a versioned
manifest containing model identity, SHA-256, license, conversion provenance,
feature contract, and artifact roles before a product surface loads the files.
No model is silently downloaded by the core library. A future signature layer
will authenticate the same content-addressed manifest rather than replace local
integrity checks.
