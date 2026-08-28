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

The next examples should follow this sequence:

1. `02-audio-file/`: end-to-end Beat This analysis with verified model packs.
2. `03-c-ffi/`: native C ABI ownership and error handling.
3. `04-browser-wasm/`: browser-hosted observations and JSON output.

Complex examples should remain independently runnable and explain their exact
runtime/model requirements in their own README.
