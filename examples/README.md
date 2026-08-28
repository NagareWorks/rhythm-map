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

The next examples should follow this sequence:

1. `03-c-ffi/`: native C ABI ownership and error handling.
2. `04-browser-wasm/`: browser-hosted observations and JSON output.

Complex examples should remain independently runnable and explain their exact
runtime/model requirements in their own README.
