# BRID full-record frozen features v1

2026-09-15. The [source-alignment disposition](../../experiments/tempo_source_expansion/ALIGNMENT-v1.md)
is complete: retain all 93 published annotations, including 0092, with explicit
residual uncertainty. This is not independent alignment certification. No
timestamps, masks, membership, source weight or target semantics changed.

## Input and capture result

The new research-only `brid_frontend` exporter uses the shipping Rust decoder
and the pinned mel ONNX through the same named RTen call as beat-this 1.0.0.
It loads only the mel graph, not the beat model. There is no alternate Python
resampler, changed frontend, manual cut or reference crop/pad to fit outputs.
Complete PCM/mel archives stay private and byte-addressed. The join verifies
all reference arrays exactly, including unknowns, before neural inference.

The [public audit](../datasets/brid-features-v1.json) records all 93 joined input
and frozen feature identities. Features are the same final0 normalized pre-head
512-dimensional representation at 50 Hz; this is not a newly trained encoder.

| Property | Result |
| --- | ---: |
| Full recordings / source groups | 93 / 1 |
| Frame points / supported tempo cells | 133,247 / 126,270 |
| Encoder chunks / full forward calls | 129 / 387 |
| Untapped/tapped head equality | Bit exact on every chunk |
| Repeated hidden/beat/downbeat tensors | Bit exact on every chunk |
| Final capture wall time on an idle T4 | 39.14 s |
| Peak PyTorch CUDA allocation | 188,083,712 bytes (179.37 MiB) |
| Musical optimizer steps | 0 |

Three forwards per chunk are a capture verification cost, not a requirement
for product inference. The peak is allocator-tracked tensor memory, not total
process/device memory. These short captures are not a production benchmark.
Existing 1,500/6/1,488 chunk geometry and earlier-chunk overlap ownership stay
unchanged. Every tapped head is also reconstructed from its captured hidden
tensor with the existing numerical budget. Archive replay independently checks
shape, dtype, bytes, exact input/reference join and overlap ownership.

Torch 2.10.0+cu129, NumPy 2.2.6, deterministic FP32, TF32 off, two CPU threads
and the 40% GPU allocation cap are retained. The encoder state remains
`db0c7b0d5e2b42aab3651477a058e72b6a82e236ef538cc1566131670da468ce`,
matching the earlier relative-tempo capture. No label is passed to its forwards.

## Preserved preflight failure

The first attempt stopped before registration/encoder loading because the old
Windows source archive had CRLF bytes, while the recorded upstream hashes are
Git blob LF bytes. All six checked sources match the pinned commit after
exporting with process-local `core.autocrlf=false`. Merely using `git archive`
under the inherited Windows setting still emitted CRLF and was rejected by
the local check. No source-hash tolerance was relaxed; the original archive,
failed attempt and canonical export are preserved.

The initial full capture took 40.64 s. Pre-push Clippy subsequently rejected
exporter documentation formatting and integer-count conversion. After fixing
those checks, the complete Rust frontend, reference join and neural capture
were repeated under a new source closure (39.14 s). All 93 input archives and
all 93 feature archives are byte-identical to the initial capture; registration
and report identities change to record the corrected exporter. Both complete
captures and the unpublished candidate are retained. This is a technical
recapture, not another training arm, favorable-subset retry or musical fit.

## Next boundary

Do not reopen the per-song RMS-offset investigation or tune another heuristic.
The matched [source-expansion protocol](../../experiments/tempo_source_expansion/PROTOCOL-v1.md)
and [draw schedule](../../experiments/tempo_source_expansion/plan-v1.json) are
unchanged. Old natural/pair input identities and the executable training/replay
runner closure still need registration before either 100-update arm starts.
All six old schedule diagnostics are already exposed, not fresh acceptance.
No holdout, production default, model distribution or release changes here.

## Reproduction

Use fresh private data-drive directories; preserve failure outputs.

```sh
cargo run --locked --profile evaluation -p rhythm-map-eval --example brid_frontend -- \
  --audio-dir /external/brid-mixtures-v1 \
  --mel-model /external/models/mel_spectrogram.onnx --output /external/brid-frontend-v1
python -m experiments.tempo_source_expansion.prepare \
  --frontend /external/brid-frontend-v1 --references /external/brid-reference-v1 \
  --output /external/brid-feature-inputs-v1
python -m experiments.tempo_source_expansion.capture \
  --inputs /external/brid-feature-inputs-v1 --assets /external/encoder-assets \
  --dependencies /external/encoder-dependencies --input-report-sha256 INPUT_REPORT_SHA256 \
  --output /external/brid-feature-capture-v1
python -m experiments.tempo_source_expansion.audit \
  --inputs /external/brid-feature-inputs-v1 --features /external/brid-feature-capture-v1 \
  --input-report-sha256 INPUT_REPORT_SHA256 --check evaluation/datasets/brid-features-v1.json
```

The retained-report check replays the retained private packets, not a promise
that a new hardware/runtime capture has the same elapsed time or archive hash.
Source acquisition is separate; these commands do not download audio/models.
`encoder-assets` contains the byte-pinned `final0.complete.ckpt` and canonical
`upstream` source tree. The explicit dependency tree supplies the retained
`rotary_embedding_torch` source, recorded before any encoder calls. CI covers
authored failures without music or a GPU; the real full-corpus capture is private.
