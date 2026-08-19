# Model files

Model binaries are intentionally excluded from this repository and from crates.

The initial backend expects:

- `mel_spectrogram.onnx`: Beat This-compatible log-mel frontend.
- `beat_this.onnx`: Beat This beat/downbeat model.

The original Beat This! project publishes its code and model weights under MIT:
https://github.com/CPJKU/beat_this

The Rust port documents its ONNX conversion and model download process here:
https://github.com/danigb/beat-this-rs

Converting a checkpoint does not change its license. Before distributing any
model pack, record the original model URL, immutable checksum, license text,
conversion command, and converter version.
