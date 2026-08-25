# Metadata capability packs

## Product contract

Rhythm Map derives metadata that is absent from, or too weakly represented by,
an audio file. A normal product call accepts audio plus optional auxiliary
inputs and returns every capability compiled into that distribution. Users may
choose package size and dependencies at build/install time; they do not tune
model thresholds, smoothing rules, pulse hypotheses, or classifier strategies
for each song.

The eventual aggregate result is a versioned audio-metadata document containing
independently versioned pack payloads. Every inferred claim must identify its
producer, express confidence or uncertainty, and retain warnings needed for a
consumer to decide whether manual review is appropriate. A missing pack means
“not installed or not run”, not a negative classification.

## Shared substrate

Capability packs may reuse these facilities without depending on one another's
implementation:

- one decoded, channel-aware audio representation and one seconds-based time
  coordinate system;
- content identity, source/model provenance, cancellation, progress, and cache
  keys;
- interval and point references that let one pack cite another pack's regions
  without changing their semantics;
- stable serialization across Rust, C ABI, WASM, CLI, and GUI adapters.

The shared substrate must not become a mandatory mega-model. Each pack owns its
models, licenses, evaluation suites, feature gates, and resource budget.

## Planned packs

### `rhythm/time-map`

Implemented first. It returns beat/downbeat timestamps, auditable alternative
beat sequences, BPM hypotheses and curve, constant/ramp tempo segments, timing
change points, and tempo/rhythm-homogeneous sections. Beat observations remain
an interchangeable backend input; deterministic time-map analysis is a
separate layer.

### `style`

Planned. It will return ranked whole-track style labels and time-bounded style
labels for semantic or musically homogeneous regions. Ontology/version identity
is part of the output because “genre” is dataset-dependent. Section labels must
not overwrite rhythm sections: the pack may consume or refine shared interval
references and publish its own semantic segmentation.

### `origin-signals`

Planned research capability for AI-music suspiciousness. It will return
calibrated signals, model/domain identity, applicability warnings, and an
explicit uncertainty range. It must not claim authorship, fraud, or a definitive
AI/human verdict. This pack is useful only after evaluation against unseen
generators, codecs, mastering chains, and ordinary production effects; it will
probably require separately trained models and frequent model-pack updates.

### `midi-key-sound`

Planned auxiliary-input capability for rhythm-game authoring. Audio remains the
primary artifact, while an optional MIDI file supplies note pitch, onset, and
duration hypotheses. The result will align MIDI and audio time, identify
candidate note/transient regions, and export key-sound cut suggestions with
confidence and provenance. This is metadata derived from audio plus declared
evidence, not a promise of perfect source separation.

## Packaging model

The Rust workspace will keep meaningful crate boundaries when a pack has real
behavior to own. Product crates select implemented packs with Cargo features or
distribution manifests, and model packs remain separately licensed,
content-addressed artifacts. Empty placeholder crates are deliberately avoided.

Examples of intended distributions:

| Distribution | Included capabilities | Intended use |
| --- | --- | --- |
| Timing library | `rhythm/time-map` | chart editors, DAWs, synchronization, game engines |
| Mapper toolkit | `rhythm/time-map`, `midi-key-sound` | rhythm-game chart and key-sound workflows |
| Catalog analyzer | `rhythm/time-map`, `style` | search, tagging, playlist and library enrichment |
| Full research build | all installed packs | offline inspection with explicit uncertainty |

The default CLI/GUI experience remains one input and one result. Advanced
diagnostics and candidate policies belong to evaluation tooling, not the normal
product interface.

## Admission gate for a new pack

A capability enters a distributable product only when it has:

1. a typed, versioned output contract with uncertainty semantics;
2. a legally redistributable implementation or a clearly optional external
   model pack with recorded licenses and provenance;
3. an evaluation set that measures the claims made by that pack, including
   failure slices and out-of-domain behavior;
4. measured binary-size, latency, memory, and platform impact;
5. identical semantic output across every supported surface that includes it.

This gate allows the product to broaden without turning optional capabilities
into undocumented heuristics or making the core distribution unexpectedly
large.
