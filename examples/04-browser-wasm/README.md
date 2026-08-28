# 04 - Run the timing engine in a browser

This example packages `rhythm-map-wasm` for browsers, loads a complete
backend-neutral `RhythmObservations` object, and renders the same schema-versioned
analysis returned by Rust, the CLI, and the C ABI. Everything runs locally in
the browser after the static files are loaded.

The checked-in fixture models a 120 to 150 BPM transition. Edit its confidence
or timestamps in the page and run the estimator again; there are no decoder,
BPM-range, smoothing, or metrical-policy controls.

## Build the browser package

The `wasm-bindgen` CLI must match the `wasm-bindgen` crate version in
`Cargo.lock`. This checkout currently uses `0.2.127`:

```bash
cargo install wasm-bindgen-cli --version 0.2.127 --locked
cargo build -p rhythm-map-wasm --target wasm32-unknown-unknown --release
wasm-bindgen \
  --target web \
  --out-dir examples/04-browser-wasm/pkg \
  --out-name rhythm_map \
  target/wasm32-unknown-unknown/release/rhythm_map_wasm.wasm
```

If `CARGO_TARGET_DIR` is set, use its corresponding
`wasm32-unknown-unknown/release/rhythm_map_wasm.wasm` path in the final command.

Generated `pkg/` output is intentionally ignored. It should be rebuilt from the
Rust source for development and copied into a release package by distribution
tooling rather than committed.

Run the model-free Node smoke check:

```bash
cd examples/04-browser-wasm
npm run smoke
```

Then serve the directory over HTTP; ES modules and `fetch` should not be opened
through a `file://` URL:

```bash
python -m http.server 8000 --directory examples/04-browser-wasm
```

Open <http://localhost:8000/>.

## Public WASM calls

- `analyze_observations(observations)` accepts the full serialized
  `RhythmObservations` contract and preserves host confidence, candidate,
  activation, audio-evidence, and source fields.
- `analyze_pcm_with_observations(observations, samples, sampleRate, channels)`
  accepts a `Float32Array` of decoded interleaved PCM. The host still supplies
  beat observations; the shared Rust engine downmixes the PCM and adds activity,
  spectral-onset, and supported harmonic-change evidence before estimation.
- `analyze_timing(beatTimes, downbeatTimes, duration)` remains the compact
  timestamp-only facade. JavaScript should pass `Float64Array` values.
- `schema_version()` lets a host reject an unsupported serialized result before
  storing or forwarding it.

A browser host that already has audio can use `AudioContext.decodeAudioData`,
interleave the decoded channel arrays into one `Float32Array`, and call
`analyze_pcm_with_observations`. This is PCM enrichment, not Beat This inference:
the current WASM package intentionally does not download Python, a model server,
or model weights and does not claim to infer beats from audio by itself.

End-to-end in-browser model inference remains a separate packaging milestone.
When implemented, it should produce the same `RhythmObservations` boundary and
feed these calls rather than introducing a second timing algorithm.
