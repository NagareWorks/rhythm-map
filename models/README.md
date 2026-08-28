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

## Beat This small model pack

[`beat-this-small-v1.json`](beat-this-small-v1.json) pins the `small1`
checkpoint converted by the same upstream `scripts/ckpt2onnx.py` command and
committed to the `danigb/beat-this-rs` repository at the same pinned revision
`089b509247e6fdcec666511c0dcf0d5f39c21e73`. It reuses the identical mel
frontend artifact and feature contract; only the beat model differs
(10,555,592 bytes versus 83,162,650).

The pack exists for measured fast/accurate comparison and size-constrained
evaluation. It is not the shipping default: the paired measurement and its
limits are recorded in
[`evaluation/baselines/beat-this-small-v1.md`](../evaluation/baselines/beat-this-small-v1.md).

```bash
cargo xtask model-verify \
  --model-pack models/beat-this-small-v1.json \
  --model-dir /path/to/beat-this-small-v1
```

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

## Experimental BeatNet model pack

[`beatnet-v1.json`](beatnet-v1.json) pins the 1.6 MB ONNX graph at BeatNet
commit `be864a4b0f126aa90aeabdeedf23f48865e09512`, its CC BY 4.0 grant, exact
SHA-256, and the complete 22,050 Hz / 1,411-sample window / 441-sample hop /
272-feature contract. `rhythm-map-beatnet` implements the matching
log-frequency spectrogram and positive-difference frontend in Rust and runs the
graph with RTen; it does not embed Python, PyTorch, or madmom.

Download the artifact URL from the manifest into an external directory and
verify it before calibration:

```bash
cargo xtask model-verify \
  --model-pack models/beatnet-v1.json \
  --model-dir /path/to/beatnet-v1
```

The repository and model are marked [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
which permits commercial reuse with attribution. That grant does not by itself establish redistribution rights
for every recording used to train the published weights. The pinned upstream
configuration names Ballroom, Beatles, CMR, and Rock Corpus as training data
and GTZAN as test data. Keep this pack experimental until those training-corpus
provenance implications have been reviewed for the intended distribution.
