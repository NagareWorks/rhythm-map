# Model packs

Model binaries are external artifacts and are intentionally excluded from this
repository and all published crates. A model pack is a checked-in JSON contract
that records immutable provenance, file roles, feature compatibility, exact
sizes, SHA-256 digests, and download locations.

The core library never downloads a model. Acquisition is an explicit operator
step, and every CLI, GUI, or evaluation path must verify the pack before loading
the files.

## Beat This full model pack

[`beat-this-full-v1.json`](beat-this-full-v1.json) pins:

- the log-mel frontend from `danigb/beat-this-rs` commit
  `089b509247e6fdcec666511c0dcf0d5f39c21e73`;
- the full FP32 `beat_this.onnx` asset from the immutable `model-large` release;
- the upstream MIT license and documented `final0` conversion command; and
- the 22,050 Hz, 128-band, 50 Hz activation feature contract.

Download both artifact URLs from the manifest into an external directory. Do
not place them in Git. Verify them before inference:

```bash
cargo xtask model-verify \
  --model-pack models/beat-this-full-v1.json \
  --model-dir /path/to/beat-this-full-v1
```

Then run the paired bottleneck evaluation:

```bash
cargo xtask eval-backend \
  --model-pack models/beat-this-full-v1.json \
  --model-dir /path/to/beat-this-full-v1 \
  --report /path/to/reports/beat-this-full-v1.json \
  --no-fail
```

`--no-fail` is appropriate while recording a new baseline. It does not change
metrics or thresholds; it only allows the report to be written when the
end-to-end path misses an acceptance gate.

## Provenance notes

The original Beat This! project and the Rust port license their code and model
weights under MIT:

- https://github.com/CPJKU/beat_this
- https://github.com/danigb/beat-this-rs

The original project also warns that some files used to train the published
weights are copyrighted or carry limited Creative Commons terms, and leaves
downstream impact assessment to users. The MIT weight license is recorded here,
but it is not a substitute for legal review of a particular distribution or
commercial use case.

Converting a checkpoint does not create a new permission grant. Before adding a
different model pack, record the original model URL, immutable revision,
checksum, license, conversion command, converter version, and feature contract.
Future signed manifests will authenticate this same content-addressed contract;
signature support will not replace local SHA-256 verification.
