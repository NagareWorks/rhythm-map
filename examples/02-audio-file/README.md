# 02 - Analyze an audio file

This example is the smallest complete native integration. It verifies a Beat
This model pack, decodes one WAV, MP3, FLAC, or OGG file, runs the normal
zero-tuning estimator, and writes the complete schema-versioned `Analysis` JSON.

It deliberately does not expose decoder, threshold, smoothing, BPM-range, or
half/double-time strategy switches. Supported alternatives and ambiguity are
returned as metadata in the same result.

## Prepare the model pack

Run `cargo run -p rhythm-map-cli --release -- models fetch` (or the packaged
`rhythm-map models fetch`) and use the returned `model_dir` below. Set
`RHYTHM_MAP_CACHE_DIR` or pass `--cache-dir` to choose the cache disk. You can
also download the two files named in
[`models/beat-this-full-v1.json`](../../models/beat-this-full-v1.json) into an
existing external directory. Model binaries must not be added to this repository.

The example verifies the manifest, expected file sizes, SHA-256 digests, safe
paths, and required artifact roles before constructing the backend. A corrupt,
renamed, incomplete, or incompatible pack fails before audio inference.
The verified pack ID and manifest SHA-256 are retained in the output's
`source.model` and `source.version` fields.

## Run

From the repository root:

```bash
cargo run -p rhythm-map-examples --release --example 02-audio-file -- \
  song.mp3 \
  --model-dir /path/to/beat-this-full-v1 \
  --output analysis.json
```

On PowerShell, the same invocation can be written on one line:

```powershell
cargo run -p rhythm-map-examples --release --example 02-audio-file -- song.mp3 --model-dir D:\models\beat-this-full-v1 --output analysis.json
```

`--model-pack` defaults to the checked-in Beat This manifest. Pass it
explicitly only when running from a different working directory.

The output schema is identical to the Rust library, CLI, C ABI, and WASM
surfaces. The input file is decoded to the backend's mono PCM contract; callers
that already own decoded PCM should use `Engine::analyze_pcm` directly.
