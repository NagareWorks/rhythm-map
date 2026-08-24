# Architecture

## Product boundary

Rhythm Map is a modular audio metadata engine, not permanently a timing-only
library. The current `Analysis` schema is the payload of the first
`rhythm/time-map` capability pack. Future style, provenance, and MIDI-assisted
packs share decoding, time coordinates, provenance conventions, and product
surfaces, but do not become fields hidden inside the tempo estimator.

Packaging selects capabilities; ordinary users select no analysis strategies.
A CLI/GUI distribution may include several packs, while a small game-engine DLL
or browser WASM build may include only rhythm analysis. Within every included
pack, one top-level call applies the shipping policy and returns confidence and
warnings. The durable composition contract is specified in
[`docs/METADATA-PACKS.md`](docs/METADATA-PACKS.md).

## Pipeline

```text
interleaved PCM
    -> observation backend
    -> beat/downbeat events + frame confidence + PCM activity/onset evidence
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
They may expose confidence, model identity, frame rate, and optional activity
and onset envelopes, but no backend tensor type crosses into the core schema.
The engine adds deterministic PCM activity and spectral-flux onset envelopes
when a backend does not provide them. Both remain backend-neutral evidence;
neither is itself a beat decision. The onset envelope includes low, mid, and
high-frequency contributions computed from the same FFT so future consumers
can distinguish bass/body accents from high-frequency subdivisions.

`RhythmObservations.beat_candidates` carries sorted, uncommitted timestamps that
are supported by the backend but were not necessarily selected as beats. The
shipping estimator validates but deliberately ignores this field. It exists so
evaluation can distinguish absent model evidence from a wrong pulse/phase
choice without lowering the product decoder threshold or inventing grid times.
Every candidate timestamp must come from the backend itself.

The default Beat This adapter consumes its frame logits to attach confidence to
events. The tempo estimator never depends on the upstream crate's Rust structs.

## Analysis boundary

`Analysis.schema_version` versions serialized output independently of crate
versions. Native Rust callers use typed structures. The C ABI initially returns
the same schema as owned UTF-8 JSON to avoid exposing Rust layouts. WASM returns
the same structure through `serde-wasm-bindgen`.

When a second production capability pack ships, product surfaces will wrap pack
payloads in one versioned audio-metadata document. The existing timing payload
keeps its own schema version so adding or omitting another pack does not rewrite
the meaning of BPM, beat, section, or confidence fields.

## Product policy boundary

Every product surface has one zero-tuning musical-analysis path. Rust callers
construct `Engine::new(backend)`, while CLI, C ABI, and WASM route through the
same shipping estimator. BPM bands, smoothing thresholds, half/double-time
rules, and decoder candidates are not product inputs.

Research candidates live behind the `experimental-policies` Cargo feature used
by the non-published `rhythm-map-eval` crate. They are comparison instruments,
not supported modes. A candidate with no measured regression is merged into and
replaces the single shipping implementation after independent validation. A
candidate that improves one slice and regresses another remains evaluation-only
unless all of the following become true:

1. The alternatives represent a real, irreducible difference between musical
   inputs rather than an unfinished algorithm.
2. The applicable input class can be detected from runtime evidence without
   labels, filenames, truth annotations, or dataset identity.
3. A precommitted selector beats the single policy on an untouched holdout with
   no protected-slice regression.

Only then may the engine contain an internal strategy selector. The selected
strategy still does not become a user-facing rhythm parameter; uncertainty and
metrical alternatives belong in `Analysis` output.

Before that selector gate is considered, calibration measures candidate-evidence
recall and top-K pulse/phase coverage. The initial hypothesis set contains the
selected sequence, its two alternating half-time phases, and a double-time
sequence augmented only with real midpoint candidate peaks. Construction and
ranking use backend confidence, PCM activity, downbeat evidence, interval
continuity, and explicit selected-evidence retention; spectral-flux strength is
reported independently for calibration but does not yet alter the rank. Truth
is applied only afterward for evaluation. Holdout reports never expose this
oracle comparison.

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
