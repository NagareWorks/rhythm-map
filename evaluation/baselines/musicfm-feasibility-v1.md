# MusicFM: artifact, rights and input/runtime feasibility screen

Date: 2026-09-08. This closes the source/metadata screen requested after the
[fixed pre-head recurrence test](prehead-recurrence-v1.md), not a MusicFM
inference or accuracy experiment. Source was inspected at an immutable revision;
small model-card, statistics, configuration and LFS-pointer bytes were read and
hashed. **Neither checkpoint payload was downloaded or loaded.** No dependency
was installed, music features inspected, holdout opened, model trained, public
interface changed or release created.

## Decision

Keep **one conditional research candidate: MusicFM-FMA, float32, non-Flash,
`get_latent(..., layer_ix=7)`**. Layer 7 follows the author's extraction example,
not a result on our labels. Do not compare layers or FMA/MSD and pick winners.
This is a source-derived probe choice, not an executable runtime lock.

| Gate | Result | Consequence |
| --- | --- | --- |
| Exact published identities | Pinned checkpoint pointers, statistics and external config | Full weight bytes still require size/hash verification before loading. |
| Public license screen | Permissive source terms and MIT model-repository metadata; no explicit NC weight grant found in the inspected publication | Sufficient to retain a research candidate, not commercial distribution clearance. |
| Offline/runtime closure | Not passed: dynamic config lookup, unpinned dependencies, unchecked checkpoint contents | Build a fail-closed offline loader and pin dependencies before acquisition/inference. |
| Input/time compatibility | A distinct 24 kHz frontend and nominal 25 Hz grid are required | No reuse of Beat This mel/frame indices; derive and test the coordinate bridge. |
| Musical discrimination/native integration | Not tested | No default, strategy selector, model pack, FFI/WASM promise or accuracy claim. |

FMA is selected for its published-paper correspondence and publicly described
source dataset, **not** because all FMA recordings are commercially reusable or
because its performance is known to be better. MSD stays inventoried but is not
a second experiment if FMA fails. Failure of this feasibility or later bounded
probe does not automatically prove that training is necessary; use the
[training decision gate](../../docs/TRAINING-DECISION.md).

## Immutable artifact inventory

The machine-readable [screen record](../parity/musicfm-artifact-screen-v1.json)
is deliberately **not** a shipping model manifest or completed acquisition lock.
`bytes_verified=false` distinguishes a publisher's pointer from a locally
verified checkpoint. Xet identifiers, Git blob IDs and payload SHA-256 are not
interchangeable.

- Source: `minzwon/musicfm` at
  `b83ebedb401bcef639b26b05c0c8bee1dc2dfe71`.
- Model repository: `minzwon/MusicFM` at
  `4513b38bc25ad1d227b1980819b9691ba97f4d87`.
- External configuration repository:
  `facebook/wav2vec2-conformer-rope-large-960h-ft` at
  `6b36ef01c6443c67ae7ed0822876d091ab50e4aa`.

| Artifact | Bytes | SHA-256 | What was verified |
| --- | ---: | --- | --- |
| `pretrained_fma.pt` | 1,316,802,154 | `68392eee13d34c2941b3761934abb6b1e67b2e9df498695bda2ea5c1087d4b96` | Published LFS pointer and Hub metadata agree; no payload read. |
| `fma_stats.json` | 2,281 | `5416e468018bae68c6231d4cbb2b11f0d11c04e6437881505ae427a3f8344904` | Complete small-file bytes. |
| `pretrained_msd.pt` | 1,316,802,088 | `218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded` | Pointer/metadata only; not selected. |
| `msd_stats.json` | 2,277 | `c36c61ab10ca4d2e7fdfefc3fcc15205316bec276a06a47baa3641a62c546f22` | Complete small-file bytes; not selected. |
| External `config.json` | 2,239 | `7a63cb5706c9a37483f1973a3c226d54eb504ce15cf62cb52637019540c8a75d` | Complete small-file bytes, before source overrides. |

The author's [checkpoint warning and usage](https://github.com/minzwon/musicfm/blob/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71/README.md)
say the earlier FMA checkpoint was incorrect. These pins refer to the published
replacement, not an arbitrary `main` download or previously cached filename.
The [pinned FMA file page](https://huggingface.co/minzwon/MusicFM/blob/4513b38bc25ad1d227b1980819b9691ba97f4d87/pretrained_fma.pt)
also identifies a pickle-based artifact, not a safetensors file.

## Rights: four separate boundaries

1. **Code.** The pinned [source license](https://github.com/minzwon/musicfm/blob/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71/LICENSE)
   assigns MIT to most files and Apache-2.0 to `modules/flash_conformer.py`.
   This is a file-specific allocation, not permission to pick either license
   for every file. Preserve copyright/license notices and identify modifications
   when code is eventually copied or redistributed.
2. **Weights.** The immutable [model card](https://huggingface.co/minzwon/MusicFM/blob/4513b38bc25ad1d227b1980819b9691ba97f4d87/README.md)
   consists only of 21 bytes of MIT license front matter (SHA-256 in the record).
   It is a license declaration on the actual author's model repository, not
   merely a code badge. That repository contains no separate LICENSE text,
   detailed weight-specific grant, training manifest or rights warranty.
   Preserve this evidence and resolve distribution/provenance questions before
   a commercial pack. Do not label these weights NC merely because some source
   music may have restrictions; nor declare all rights cleared by a badge.
3. **Training music and evaluation data.** The [FMA maintainers](https://github.com/mdeff/fma#acknowledgments-and-licenses)
   distinguish MIT code, CC BY metadata and audio under individual artists'
   chosen licenses. The dataset name is not a blanket audio-use permission.
   No training audio is needed or acquired for the reserved inference probe.
   Our existing evaluation recordings retain their own acquisition/license
   records; no audio or feature archive is added to Git.
4. **Dependencies.** A non-Flash probe still requires PyTorch, torchaudio,
   einops and Transformers plus their resolved dependencies. Source licensing
   does not close a wheel/native-library redistribution audit. The external
   config repository declares Apache-2.0; the speech checkpoint itself is not
   used. Exact dependency versions, artifacts, notices and platform closure are
   still outstanding, so no complete SBOM or commercial package approval is
   claimed. This is an engineering rights screen, not legal advice.

There is also a provenance inconsistency worth retaining: the usage section
names FMA-large while the paper describes roughly 8,000 hours of FMA music;
the [FMA inventory](https://github.com/mdeff/fma#data) distinguishes 30-second
large clips from untrimmed full recordings. Neither MusicFM publication supplies
the exact training-item/exclusion manifest needed to resolve that difference.
Do not invent a precise training subset from the name alone.

## Actual inference path and costs

The inspected [MusicFM model](https://github.com/minzwon/musicfm/blob/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71/model/musicfm_25hz.py)
uses two residual convolution blocks, a projection to 1,024 channels and twelve
Conformer layers. These are constructor defaults, **not verified tensor shapes
inside the unacquired checkpoint**. Its constructor obtains a speech-model
configuration with an unpinned `from_pretrained` call, changes depth to 12 and
width to 1,024, and initializes an encoder. It does not load speech-model weights.
The inspected config uses 16 attention heads, 4,096 intermediate width and rotary
position embeddings. Pin the complete config and effective defaults rather than
copying just those dimensions.

`get_latent` delegates to prediction, executes all twelve layers, collects all
hidden states and computes the 4,096-way token projection before returning the
chosen sequence. Layer 7 does **not** currently save five layers of compute.
The normal `forward` instead creates targets and random masks for the training
objective; it is the wrong extraction entrypoint even if no optimizer runs.
Use `eval()` plus inference mode: disabling dropout alone does not disable
autograd. No quantizer lookup, masking, pooled song vector or trained probing
head belongs in the proposed timing feature comparison.

The [Transformers 4.31.0 implementation inspected for tuple semantics](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py)
places encoder input at index 0 and the state after seven blocks at index 7;
the final entry includes output layer normalization. This source reading does
not select 4.31.0 as a supported runtime or prove compatibility with current
packages. The chosen runtime must verify that tap identity explicitly. Do not
call layer 7 the final normalized pre-head tensor used by Beat This.

The source tree provides no requirements lock or installable project manifest.
The existing parity environment has torch/torchaudio 2.8.0 CPU, NumPy 2.2.6 and
einops 0.8.1, but no Transformers installation. It is not yet a MusicFM runtime.
Do not disturb that established environment to try an unpinned latest package.
The non-Flash branch avoids the custom Flash imports/CUDA extension route;
actual CPU compatibility and speed still require measurement.

Source-derived storage estimates, **not measured peak memory or latency**:

- One float32 sequence costs `25 * 1024 * 4 = 102400` bytes/second: 3.072 MB
  for 30 seconds or 30.72 MB for five minutes. This equals the payload rate of
  the earlier 50 Hz by 512-channel tap despite halving its temporal resolution.
- Thirteen returned hidden-state tensors at 750 frames occupy about 39.936 MB;
  a 750 by 4,096 token-logit tensor adds 12.288 MB. Parameters, convolutional
  activations, copies and temporary attention tensors are additional.
- An explicitly materialized float32 16-head attention-score tensor is
  36 MB at 750 tokens versus 3.6 GB at 7,500 tokens, per such tensor. This is
  quadratic scaling arithmetic, not an assertion that every backend allocates
  precisely that peak or holds all layers' scores simultaneously.

A future optimization may drop unused heads/layers only after proving parity
with the fixed reference extraction. Do not benchmark that optimization against
a different checkpoint, precision, context or machine and claim a model gain.

## Input and time-coordinate bridge

The [frontend](https://github.com/minzwon/musicfm/blob/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71/modules/features.py)
accepts mono waveform samples at 24 kHz. It uses 128 mel bands, FFT 2,048 and
hop 240, converts power to dB, then drops the final spectrogram frame. FMA
normalization uses global mean `3.0710578151459664` and standard deviation
`20.999089814337626` for `melspec_2048`; these are not per-song statistics or
the MSD numbers.

The unspecified frontend arguments inherit torchaudio defaults. For the
[2.8.0 MelSpectrogram](https://docs.pytorch.org/audio/2.8.0/generated/torchaudio.transforms.MelSpectrogram.html)
and [AmplitudeToDB](https://docs.pytorch.org/audio/2.8.0/generated/torchaudio.transforms.AmplitudeToDB.html)
inspected here, freeze centered reflective padding, Hann window, power 2,
HTK mel without mel normalization, and power-to-dB with no top-dB clipping.
Test them against the selected runtime; a same-sized Beat This spectrogram is
not equivalent. The loader does not resample or downmix for the caller.

For an unpadded input of `N > 1024` samples under those STFT defaults, the
derived retained mel length is `M = floor(N / 240)`. Two time-stride-2,
kernel-3, padding-1 convolutions in the [subsampler](https://github.com/minzwon/musicfm/blob/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71/modules/conv.py)
give `T = ceil(M / 4)` (not unconditional floor division). The nominal token
grid has 960-sample/40 ms spacing and origin zero under these padding choices.
This is an index-coordinate derivation, not a verified acoustic response delay,
event timestamp or claim that a token only sees 40 ms of music. Very short
inputs need a specified policy because reflective STFT padding has a minimum
length. Test non-divisible lengths and right-edge validity explicitly.

The encoder is context-dependent and noncausal; its wrapper supplies no
attention mask for variable-length padded examples and no complete-recording
chunk/ownership policy. Do not import Beat This's 1,500-frame/6-frame-border
policy or infer the selected four-second windows in isolation. First prove
single-example, unpadded short-clip extraction; freeze a bounded whole-recording
context, tail and seam policy before real captures. Upsampling 25 Hz features
does not create 50 Hz evidence, and adaptive pooling must not accidentally
pool the 1,024-channel axis of a `[batch, time, channel]` tensor.

## What the paper and existing cohort cannot establish

The [paper's downstream evaluation](https://arxiv.org/html/2311.03318v1#S3.SS2)
trains a small probing network on frozen features and uses madmom decoding for
beat/downbeat evaluation. Thus its reported scores are not a training-free
BPM/change-point algorithm. The author does not publish those fine-tuned task
models or the downstream evaluation pipeline. We are testing an independent
pretrained representation, not installing a proven replacement beat tracker.

Do not promote any existing corpus's Beat This overlap status into MusicFM
overlap clearance. A newer dataset publication does not mean its recordings,
compositions or alternate performances were absent from FMA/MSD. No item-level
or work-level training-overlap check was completed here. In particular, RUBATO
contains performances of older works; recording and work identity are distinct.

The existing seven ARTBeaT pairs remain repeatedly exposed calibration cases.
All 40 calibration identities must still be accounted for, with RUBATO
expressive/untyped and unpaired ARTBeaT retained as exclusions. No holdout is
opened for this feasibility screen; a later positive result requires independent
acceptance with audited work/recording provenance, not relabeling these seven
pairs as unseen examples for a new backend.

## Next bounded implementation gate

1. Resolve and pin one compatible non-Flash dependency set, including artifact
   hashes and notices, in a separate private environment. Keep generated output
   off the system drive. No MusicFM dependency enters the Rust workspace.
2. Implement the offline constructor/loader against the pinned local config
   and statistics, with authored rejection tests for wrong hashes, malformed
   config, missing files, unexpected key prefixes/shapes and attempted network
   access. Require `torch.load(..., map_location="cpu", weights_only=True)`;
   do not fall back to arbitrary pickle execution or blanket global allowlists.
   [PyTorch's restricted loader](https://docs.pytorch.org/docs/2.8/notes/serialization.html#torch-load-with-weights-only-true)
   reduces code-execution risk but is not a complete sandbox against resource
   exhaustion. Use a resource-bounded process and strict state-dict checks.
3. Only after those prerequisites, acquire **one** FMA payload to private
   storage, check its full size/hash and tensor inventory, then test authored
   PCM silence/impulses/tones, awkward lengths and repeatability. Require
   exact tap/shape/coordinate identity, unchanged model state, finiteness,
   offline operation and measured wall time/peak memory. Agree numerical
   tolerances before the first comparison; failures stop the gate.
4. A fidelity pass allows freezing one separate temporal-discrimination rule
   and complete-input context policy before music-feature access. It does not
   authorize a layer/metric/precision sweep, per-track blend, holdout search,
   native export or default adoption. Preserve false changes, erased changes
   and unresolved half/double density alternatives.

Native ONNX/RTen/FFI integration would still need an exporter, operator support,
frontend/chunk parity and resource tests. Browser/WASM feasibility is separately
unverified for a roughly 1.32 GB artifact; Python extraction does not demonstrate
either deployment. The one-call, zero-tuning contract and optional
[metadata-pack product horizon](../../docs/METADATA-PACKS.md) are unchanged.
