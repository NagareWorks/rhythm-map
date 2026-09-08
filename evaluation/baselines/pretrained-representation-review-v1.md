# Richer pretrained evidence: reuse review and next capture gate

Date: 2026-09-08. This is an architecture, prior-experiment and licensing-source
review, **not** a new music-accuracy result. No inference, feature extraction,
training, model download, holdout access or production change ran for this
review. The existing checkpoint was hash-verified and inspected on CPU with
`torch.load(..., weights_only=True)`; the model was not instantiated.

## Decision

Investigate the **final normalized hidden sequence immediately before Beat
This's task head**, using the existing pinned `final0` weights. Do not repeat
head-score transformations as if they were new observations. Keep MusicFM as
one independent pretrained-feature fallback, pending an artifact/runtime gate.
Do not add either path to the normal API, CLI, FFI, WASM or model bundle yet.

This is a finite next step: first prove that one fixed feature tap can be
captured without changing existing outputs. Only then test its discriminative
value under a separately frozen rule. Higher dimensionality alone cannot
establish that missing musical-clock information survives in the representation.

## What has already been tried

The following are closed candidates, not exclusions of entire mathematical
families. Historical baselines retain their original scope and denominators.

| Evidence or transformation | Existing evidence | Consequence for this next step |
| --- | --- | --- |
| Selected events plus missing-beat advancement priors | [Observation-clock audit](observation-clock-v1.md) and [training gate](../../docs/TRAINING-DECISION.md) | Identical retained events can encode incompatible clocks. Another penalty cannot supply missing observations. |
| PCM onset strength, band/phase contrasts and five-interval context | [Acoustic phase audit](clock-phase-evidence-v1.md) | RUBATO rankings do not transfer reliably; anchor-dependent coverage is small. Do not rename these features a new representation. |
| Complete continuous beat/downbeat heads and ideal pulse templates | [Dense evidence](dense-clock-evidence-v1.md) | Already inspected, including candidate-absent misses. A favorable truth-assisted rank is not an automatic clock. |
| Full-frame renewal-clock decoding over those heads | [Dense sequence](dense-sequence-v1.md) | All 40 tracks fail at least one joint regression check. Continuous heads and global sequence search are not untested alternatives. |
| Raw/sigmoid/weight-offset scores interpreted as event absence | [Pinned loss and SumHead semantics](beat-this-semantics-v1.md) | No calibrated absence/plain/accent likelihood follows from this training objective. |
| Joint heads in seven-frame windows; five-point shared-phase context | [Metrical windows](metrical-window-v1.md), [shared phase](shared-phase-context-v1.md) | Wider/head-aligned scores have mixed results and density ambiguities. No width, phase or source sweep. |
| Different ownership of existing model-chunk overlap | [Central-window stitching](beat-this-chunk-context-v1.md) | Prefix gains did not safely transfer to complete recordings. Preserve the shipping chunk context and ownership. |
| A second tracker supplying dense metrical evidence | [BeatNet dense meter gate](artbeat-beatthis-beatnet-dense-meter-v3.md) | Already explored; a second network's agreement is not itself correctness or new calibrated evidence. |
| More elaborate priors or analytic observation densities | [Boundary](clock-boundary-v1.md), [jump](jump-evidence-v1.md), [presence](presence-likelihood-v1.md) | Do not repair failed authored controls by retuning these weights on the same labels. |
| Matched response counts and paired head signs, with oracle periods | [Seven paired controls](oracle-period-paired-v1.md) | Both sources resolve density in 0/7 pairs. Close this selector, retaining every false-step/erased-step counterexample. |

The repository's recorded audits expose mel, two output logits, decoded peaks
and acoustic descriptors; this review found no recorded evaluation of the
512-channel pre-head sequence. That is a bounded repository-history finding,
not a claim about every unpublished experiment or about all encoder layers.

## Actual architecture and extraction boundary

The verified [reference lock](../parity/reference-lock.json) identifies upstream
revision `b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c`, Rust port revision
`089b509247e6fdcec666511c0dcf0d5f39c21e73`, and the 81,058,141-byte `final0`
checkpoint with SHA-256
`8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331`.
Both local source checkouts were clean at those revisions. Checkpoint inspection
using PyTorch 2.8.0 CPU confirms `transformer_dim=512`, `n_layers=6`,
`head_dim=32`, `stem_dim=32`, `spect_dim=128`, and `fps=50`. The task-head weight
has shape `[2, 512]`, with a two-element bias. The earlier semantics audit
establishes the absent `sum_head` checkpoint field and effective true default.

The pinned [model forward](https://github.com/CPJKU/beat_this/blob/b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c/beat_this/model/beat_tracker.py)
runs the spectrogram frontend, six temporal transformer blocks, then the task
head. The [transformer](https://github.com/CPJKU/beat_this/blob/b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c/beat_this/model/roformer.py)
applies its output RMS normalization before returning. The selected tap is that
returned `[batch, frames, 512]` tensor, not attention weights, the unnormalized
last residual, all layers, or a song-averaged embedding.

For a frame vector `h`, the last affine layer produces `(u, v) = W h + a` and
SumHead emits `(beat, downbeat) = (u + v, v)`. Recovering `u = beat - downbeat`
therefore only reexpresses existing information. The affine projection has at
most two independent output directions in a 512-dimensional ambient space;
its outputs cannot uniquely determine an arbitrary hidden vector. This algebra
does **not** prove that real reachable hidden states contain usable tempo cues,
or that they distinguish rests from genuine changes. The same training loss
still shapes these features; they are not independent model votes, calibrated
probabilities, acoustic-presence labels or a ready-made tempo estimator.

The pinned Rust-port [export wrapper](https://github.com/danigb/beat-this-rs/blob/089b509247e6fdcec666511c0dcf0d5f39c21e73/scripts/ckpt2onnx.py)
exports only `beat` and `downbeat`. Its
[predictor](https://github.com/danigb/beat-this-rs/blob/089b509247e6fdcec666511c0dcf0d5f39c21e73/src/inference.rs)
aggregates only those two arrays. The
[RTen wrapper](https://github.com/danigb/beat-this-rs/blob/089b509247e6fdcec666511c0dcf0d5f39c21e73/src/runtime/rten.rs)
can return named graph outputs, but the normal adapter does not expose a hidden
sequence. Thus this is not an existing runtime switch: a research tap and, if
successful, an explicitly versioned export/adapter change would be required.
This review inspected source, not the shipping ONNX graph's internal tensors.

The tap can in principle share the original forward pass. It is not free:
float32 output storage is 2,048 bytes/frame versus eight for the two heads
(256 times their payload), excluding model memory and temporary copies.
At 50 Hz a five-minute sequence alone is 30.72 MB. Runtime, peak memory and
serialization overhead must be measured; no speedup is claimed.

## Pretrained alternatives and rights screen

These are public-source observations as of the date above, not legal clearance.
Code, model weights, dependency terms and training-recording rights remain
separate. No listed license permits us to redistribute arbitrary training audio.

| Candidate | Evidence and remaining issue | Disposition |
| --- | --- | --- |
| Existing Beat This `final0` | Upstream explicitly licenses both code and published weights under MIT, while warning about restricted training recordings. Same existing weight identity; no new model acquisition. | First technical probe; retain the existing distribution/provenance cautions. |
| MusicFM | Source license assigns MIT to most files and Apache-2.0 to `modules/flash_conformer.py`; the author's model repository is marked MIT. A 25 Hz, 1,024-channel representation and FMA/MSD checkpoints are published. Exact weight bytes, dependency closure, runtime, source-recording provenance and work-disjoint evaluation are not verified here. | One independent-feature fallback, not an installed or approved pack. |
| MERT-v1-95M | Official model repository marks weights CC-BY-NC-4.0. | Exclude from this commercially usable product route under that grant. |
| MuQ-large-msd-iter | Official card explicitly separates MIT code from CC-BY-NC-4.0 weights. | Exclude under the published weight terms; porting code does not remove the restriction. |
| All-In-One | MIT repository with a frame-embedding interface, but its documented pipeline includes separation and Python/native dependencies. This review has not established a separate checkpoint/dependency licensing closure or native/WASM feasibility. | Defer this larger integration; structural segments alone are not tempo-change evidence. |

Primary sources:

- [Beat This explicit code/weight grant and training-data warning](https://github.com/CPJKU/beat_this/blob/b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c/README.md#license).
- MusicFM source revision `b83ebedb401bcef639b26b05c0c8bee1dc2dfe71`:
  [license](https://github.com/minzwon/musicfm/blob/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71/LICENSE),
  [feature implementation](https://github.com/minzwon/musicfm/blob/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71/model/musicfm_25hz.py),
  [usage](https://github.com/minzwon/musicfm/blob/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71/README.md),
  [author's model repository](https://huggingface.co/minzwon/MusicFM).
- [MERT-v1-95M model card](https://huggingface.co/m-a-p/MERT-v1-95M) and
  [MuQ model card with separate grants](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter#license).
- All-In-One revision `18e78903c0365147a2c5d4e5e57ebf88cb7d800e`:
  [usage/dependencies](https://github.com/mir-aidj/all-in-one/blob/18e78903c0365147a2c5d4e5e57ebf88cb7d800e/README.md)
  and [source license](https://github.com/mir-aidj/all-in-one/blob/18e78903c0365147a2c5d4e5e57ebf88cb7d800e/LICENSE).

The Hugging Face cards are dated discovery references, **not immutable model
pack locks**. MusicFM's 25 Hz features cannot gain new temporal information by
upsampling to 50 Hz. Its downstream-task examples do not certify a training-free
tempo-change estimator. Verify the selected checkpoint, normalization statistics,
config and safe loading path before any future acquisition or evaluation.

## Next bounded gate: capture fidelity, then discrimination

The following is the proposed next implementation, not a completed result:

1. Freeze an evaluation-only extractor for the single normalized pre-head tap.
   Use the exact checkpoint/source above, evaluation mode, float32 CPU and the
   existing frontend. No optimizer, layer sweep, fitted projection, PCA trained
   on the cohort, feature blending or public parameter is allowed in this gate.
2. Test the tap on authored spectrogram controls covering short, padded and
   overlapping chunks. Capture the hidden tensor and both heads from the same
   forward pass; apply the unchanged head to the retained hidden tensor and
   compare with that pass's logits. Check shape, finiteness, input immutability
   and unchanged hooked/unhooked outputs. Do not treat synthetic mel as music.
3. Preserve 1,500-frame chunks, six-frame borders, 1,488-frame stride,
   avoid-short-end and `keep_first` ownership. Hidden vectors and both logits
   must have the same chunk owner and global frame coordinate. Do not infer
   four-second windows in isolation or substitute the rejected central-owner
   rule. These offline transformer features use context on both sides.
4. Reuse the two existing [reference-parity traces](beat-this-reference-parity-v1.md)
   to check unchanged frontend/head/decoder behavior before any cohort feature
   inspection. Preserve their original scope: one complete short ARTBeaT case
   and one 35-second RUBATO prefix, not a full-recording accuracy test. Freeze
   numerical budgets and exact event-identity checks before executing the tap;
   a mismatch stops capture, not a reason to relax parity after seeing results.
5. Keep raw features private and outside the system drive; record source,
   checkpoint, runtime, input/mel identity, extractor hash, chunk ownership,
   shape, elapsed inference/serialization time and peak memory. Do not replace
   existing two-head captures, model manifests or observation cache identities.
6. If fidelity passes, freeze **one** label-free temporal comparison and its
   authored controls before reading real hidden features for discrimination.
   Retain all seven existing constant/step pairs, all four prefix phases,
   half/double alternatives and non-separable cases. They are calibration,
   not new independent evidence. Account for all 40 calibration IDs; RUBATO
   remains expressive/untyped rather than invented hard step/constant labels.
   Test previously failed omissions, genuine changes and edges, not merely
   annotated-beat versus offbeat separation. No threshold or per-track fallback
   may be chosen from the resulting failures.

A passed capture gate establishes instrumentation only. A useful calibration
contrast must still pass an automatic, observation-only decoder and independent
work-disjoint acceptance before default adoption. If this fixed representation
test fails, close it and assess the single MusicFM fallback's feasibility; do
not turn the same cohort into an unlimited search over layers and distances.
Training remains a proposal requiring the existing
[practical decision gate](../../docs/TRAINING-DECISION.md), not an automatic
consequence of this review. The one-call, zero-tuning metadata-pack contract and
the no-release scope are unchanged.
