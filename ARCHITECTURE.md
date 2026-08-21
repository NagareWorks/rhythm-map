# Architecture

## Pipeline

```text
interleaved PCM
    -> observation backend
    -> beat/downbeat events + frame confidence + PCM activity
    -> low-activity rejection + evidence-based metrical selection
    -> preliminary normalized tempo curve
    -> guarded short-transition beat-grid recovery
    -> evidence-based bar-level downbeat selection and boundary realignment
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
6. Collapse half-bar downbeat candidates only when alternating PCM accents
   identify a stronger bar phase, and realign a displaced boundary label when
   the recovered grid and activity agree.
7. Recompute octave-equivalent normalization around a robust metrical reference.
8. Median-filter and locally average in log-tempo space.
9. Detect direct jumps and short model-smeared transition blocks.
10. Simplify the curve into constant and ramp segments.
11. Split rhythm sections at tempo changes and beat/audio discontinuities.

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

## Evaluation asset boundary

External evaluation audio is never addressed by an unchecked local path in a
suite. `rhythm-map-eval` binds an explicit local directory to an
`ExternalAudioResolver`, verifies the manifest's SHA-256 over exact encoded
file bytes, and only then passes decoded mono PCM through the ordinary engine.
Filename hints are optional and non-authoritative; stale hints fall back to a
deterministic content search below the resolver root. Symbolic links and parent
path traversal are excluded.

The resolver caches digests during one suite run so each candidate file is
hashed at most once. Reports retain model identity, capability tags, verified
external-audio SHA-256, observations, and aggregate metrics, but not filenames,
resolved paths, or audio bytes. External truth is validated and run through the
same oracle estimator before backend attribution.
