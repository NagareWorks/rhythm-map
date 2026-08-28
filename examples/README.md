# Examples

Examples are numbered in the order a new integrator should read them.

- A self-contained example stays in one numbered file, such as
  `01-observations.rs`.
- An example that needs assets, foreign-language build files, browser markup,
  or more than one source file gets a numbered directory, such as
  `03-c-ffi/`.
- Numbers describe the learning path, not API or schema versions. Existing
  numbers are never reused for a different topic.
- Every example uses the normal zero-tuning product API. Experimental decoder
  or estimator policies belong in `evaluation`, not here.

## 01 - Analyze beat observations

[`01-observations.rs`](01-observations.rs) supplies backend-neutral beat and
downbeat observations for a 120 to 150 BPM transition, then prints the complete
schema-versioned analysis JSON. It requires no audio or model files.

Run it from the repository root:

```bash
cargo run -p rhythm-map-examples --example 01-observations
```

## 02 - Analyze an audio file

[`02-audio-file/`](02-audio-file/) is the first complete native integration.
It verifies the external Beat This model pack, decodes a real audio file, and
runs the normal analysis path. The directory contains its own runtime and model
preparation instructions.

## 03 - Call the C ABI from C, Python, C#, and Unity

[`03-c-ffi/`](03-c-ffi/) builds the native `.dll`, `.so`, or `.dylib` once and
uses the same ABI from C, Python `ctypes`, standalone C#, and a Unity
`AudioClip`. It documents analyzer/JSON/error ownership, verified model-pack
loading, and platform library placement.

## 04 - Run the timing engine in a browser

[`04-browser-wasm/`](04-browser-wasm/) packages the shared estimator as a web
module, accepts complete backend-neutral observations, and renders the tempo
curve, segments, changes, and full JSON in a static browser page. It also
documents the optional decoded-PCM enrichment call without pretending that the
current WASM package already contains an audio-to-beat model.

Complex examples should remain independently runnable and explain their exact
runtime/model requirements in their own README.
