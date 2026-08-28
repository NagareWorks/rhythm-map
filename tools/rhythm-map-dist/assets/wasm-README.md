# Rhythm Map browser WASM distribution

This package runs the shared deterministic time-map estimator in WebAssembly.
Open the included static demo through an HTTP server or import
`pkg/rhythm_map.js` from an existing browser application.

The WASM module accepts host-provided beat observations and can optionally add
deterministic evidence from decoded PCM. It does not contain a beat-tracking
model or download model weights. End-to-end browser beat inference must remain
an interchangeable observation backend feeding this same contract.

Run `python verify_package.py .` before deploying the package. `npm run smoke`
loads the packaged WASM in Node and checks the schema-versioned result.
