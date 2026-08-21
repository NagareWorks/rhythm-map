# Rhythm Map

Rhythm Map is an offline music timing analysis engine for rhythm games, chart
editors, DAWs, and media tools. It turns beat/downbeat observations into a
versioned, confidence-aware tempo map with BPM curves, constant/ramp segments,
change points, and rhythm-homogeneous sections.

The project deliberately does **not** train or replace a beat tracker in its
first phase. The default backend reuses the MIT-licensed Beat This! model via
the `beat-this` Rust crate, while the public API remains backend-independent.

## Status

This repository is at `0.1.0` and establishes the long-term package boundaries:

- `rhythm-map-core`: stable analysis schema, backend trait, engine, and
  training-free tempo-map estimator.
- `rhythm-map-beat-this`: optional Beat This! adapter using pure-Rust `rten`.
- `rhythm-map-cli`: end-to-end audio-file analysis to JSON.
- `rhythm-map-models`: versioned provenance and SHA-256 model-pack verification.
- `rhythm-map-ffi`: versioned C ABI for `.dll`, `.so`, and static libraries.
- `rhythm-map-wasm`: WASM timing-analysis API from beat observations. End-to-end
  browser audio inference is the next WASM milestone.

The current rhythm sections are tempo/rhythm-homogeneous regions, not semantic
labels such as verse, chorus, or drop.

## Quick start

Download or convert the two Beat This! ONNX files described in
[`models/README.md`](models/README.md), then run:

```bash
cargo run -p rhythm-map-cli --release -- \
  song.mp3 \
  --mel-model models/mel_spectrogram.onnx \
  --beat-model models/beat_this.onnx
```

The command emits schema-versioned JSON containing beats, a BPM curve, tempo
segments, change points, and rhythm sections.

## Evaluation

The repository keeps unit tests separate from product acceptance suites. The
checked-in generated suite has analytic beat, tempo, ramp, jump, and silence
ground truth and is a required CI gate:

```bash
cargo xtask eval
```

With an explicitly downloaded and verified model pack, `cargo xtask
eval-backend` runs the same cases through Beat This and emits a paired
oracle/end-to-end bottleneck report. External public or private cases add an
explicit `--audio-dir`; assets are resolved by SHA-256 rather than trusted by
filename, and local paths are not included in reports.

`cargo xtask render --output <directory>` produces disposable deterministic
synthetic WAV files for end-to-end backend evaluation. Public and private
real-music cases use content-addressed external audio references, so possession
of a track never silently becomes permission to redistribute it. See
[`evaluation/README.md`](evaluation/README.md) for the data and license policy.

## Core API

```rust,no_run
use rhythm_map_core::{Engine, TempoMapEstimator};
use rhythm_map_beat_this::BeatThisBackend;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let backend = BeatThisBackend::load(
    "models/mel_spectrogram.onnx",
    "models/beat_this.onnx",
)?;
let mut engine = Engine::new(backend, TempoMapEstimator::default());
let analysis = engine.analyze_pcm(&mono_samples, sample_rate, 1)?;
# Ok(())
# }
```

The core accepts interleaved `f32` PCM. File decoding belongs to CLI/GUI
adapters, keeping the library usable from engines and browser hosts that already
own decoded audio.

## Design principles

- Beat tracking is an interchangeable observation backend, not the public API.
- A BPM curve is inferred robustly; it is not raw `60 / beat_interval` jitter.
- Metrical half/double-time ambiguity is surfaced instead of hidden.
- Models, source code, and training data have separate provenance records.
- Native, C ABI, WASM, CLI, and future GUI packages consume the same schema.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the component contracts and
[`docs/ALGORITHM.md`](docs/ALGORITHM.md) for the current deterministic timing
algorithm. [`ROADMAP.md`](ROADMAP.md) records the staged implementation and the
evidence required before adding learned components.

## License

Rhythm Map source code is licensed under Apache-2.0. Optional model files and
third-party backends retain their own licenses; see `NOTICE` and
`models/README.md`.
