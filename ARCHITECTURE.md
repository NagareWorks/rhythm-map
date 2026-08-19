# Architecture

## Pipeline

```text
interleaved PCM
    -> observation backend
    -> beat/downbeat events + frame confidence
    -> metrical-level normalization
    -> robust local tempo curve
    -> piecewise constant/ramp simplification
    -> change points and rhythm-homogeneous sections
    -> versioned Analysis schema
```

`rhythm-map-core` owns every type after the observation boundary. A backend may
use Beat This!, another ONNX model, a streaming tracker, or supplied observations
without changing consumers.

## Observation boundary

Backends implement `RhythmObservationBackend` and return `RhythmObservations`.
They may expose confidence, model identity, and frame rate, but no backend tensor
type crosses into the core schema.

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
2. Generate inter-beat tempo observations.
3. Normalize octave-equivalent candidates around a robust metrical reference.
4. Median-filter and locally average in log-tempo space.
5. Detect sustained jumps with left/right robust windows.
6. Simplify the curve into constant and ramp segments.
7. Split rhythm sections at tempo changes and long beat discontinuities.

This is a deterministic baseline, not the final research endpoint. Evaluation
will determine whether a learned change-point/confidence head is needed.

## Model packages

Model files are external artifacts. A future signed manifest will contain model
identity, SHA-256, license, feature contract, tensor names, and schema
compatibility. No model is silently downloaded by the core library.
