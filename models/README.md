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

Use the packaged CLI (or `cargo run -p rhythm-map-cli --release --`):

```bash
rhythm-map models fetch
rhythm-map models verify
rhythm-map song.mp3 --output timing.json
```

The CLI embeds the exact default manifest bytes, not the model weights. Only
`models fetch` accesses the network. Acquisition follows HTTPS-only redirects,
streams at most the expected size plus one byte, and checks exact length and
SHA-256. It downloads to a private staging directory, then publishes the whole
verified pack by renaming that directory. A failed request or hash check never
becomes an installed pack; retrying starts a fresh stage, not a byte-range resume.
Ordinary failures remove their staging directory. After a hard process kill,
an abandoned `.download-*` directory may remain; remove only that directory
after confirming no acquisition is using it. It is never considered a cache hit.

Cache selection is `--cache-dir`, then `RHYTHM_MAP_CACHE_DIR`, then the platform
user cache: `%LOCALAPPDATA%/rhythm-map/cache` on Windows,
`~/Library/Caches/rhythm-map` on macOS, and
`$XDG_CACHE_HOME/rhythm-map` (or `~/.cache/rhythm-map`) on Linux. For another disk:

```bash
rhythm-map models fetch --cache-dir /path/on/data-disk/rhythm-map
rhythm-map song.mp3 --cache-dir /path/on/data-disk/rhythm-map
```

Each entry is `<cache>/model-packs/<manifest-sha256>/`, containing the original
`manifest.json` and an `artifacts/` directory. `fetch` and `verify` print JSON
including the verified `model_dir`, manifest digest, identity, and declared
license. Pass that manifest and artifact directory to the existing C ABI,
Python, C#, or Unity examples; those consumers do not acquire models themselves.

Cache reuse rechecks the trusted manifest bytes and all artifact sizes/hashes.
A changed manifest gets a different entry. A corrupt existing entry causes an
error without overwriting files or contacting the network. Inspect or move aside
that exact digest directory before fetching again. Do not share a writable
cache with untrusted processes: hashes do not prevent a local writer from
changing model files after verification.

`--model-pack` accepts a trusted local manifest and uses its download URLs; it
does not fetch manifests remotely or establish trust in their author. SHA-256
proves byte identity against that manifest, not a signature, model safety, or a
new license grant. Review the manifest and upstream notices before acquisition
or redistribution. The provenance cautions below still apply.

Already downloaded both artifact URLs into an external directory? Keep using
them without copying them into the cache:

```bash
rhythm-map models verify --model-dir /path/to/beat-this-full-v1
rhythm-map song.mp3 --model-dir /path/to/beat-this-full-v1
```

The legacy `--mel-model` / `--beat-model` pair is still accepted but now verifies
both files against the selected manifest (the built-in one unless overridden).
Arbitrary converted weights therefore need their own trusted manifest rather
than silently bypassing provenance checks.

Evaluation and foreign-language integrations retain their explicit paths:

```bash
cargo xtask model-verify \
  --model-pack models/beat-this-full-v1.json \
  --model-dir /path/to/beat-this-full-v1
```

Rust hosts may use `ModelPackCache::verify` without any network dependency.
Enable `rhythm-map-models`' optional `download` feature to call
`ModelPackCache::fetch` during an explicit setup step. Neither the timing core,
C ABI, nor WASM enables acquisition. Applications retain control over consent,
cache location, and which optional packs they ship.

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
