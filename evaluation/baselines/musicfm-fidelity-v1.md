# MusicFM FMA: authored extraction fidelity gate

Date: 2026-09-08. This follows the [offline loader prerequisites](musicfm-loader-prerequisite-v1.md).
The [protocol](../parity/musicfm-fidelity-lock-v1.json) and extractor were fixed
before checkpoint acquisition. No calibration recording or holdout participates;
no model is trained and no product interface or default is changed.

## Frozen experiment

One checkpoint only: `pretrained_fma.pt`, 1,316,802,154 bytes, SHA-256
`68392eee13d34c2941b3761934abb6b1e67b2e9df498695bda2ea5c1087d4b96`, from the
[immutable author's artifact](https://huggingface.co/minzwon/MusicFM/blob/4513b38bc25ad1d227b1980819b9691ba97f4d87/pretrained_fma.pt).
Download to private off-system storage; verify full bytes before restricted
loading. No MSD, unsafe pickle, version, precision or layer fallback is allowed.
The previous loader, resource helper and runtime hashes are part of this lock.

The authored corpus retains the previous seven PCM controls: silence, centered
impulses and 440 Hz tones at 1,025, 1,199, 1,200, 1,919, 1,920, 24,001 and
48,000 samples, respectively as listed in the protocol. Inputs are unpadded,
single-example mono 24 kHz float32 arrays. There is no decoder, resampler,
complete-song context policy or music-file argument in this probe.

For every case, require all of the following with `atol=rtol=0` fixed before
any pretrained output is seen:

1. An unhooked `get_latent(..., layer_ix=7)` reference, a hooked call to the same
   unchanged entrypoint, and an unhooked repeat yield exactly equal finite
   float32 `[1, ceil(floor(N/240)/4), 1024]` tensors.
2. A snapshot of `conformer.layers[6]` output zero exactly equals hidden-state
   tuple index 7 and the returned feature. All twelve blocks execute once,
   the tuple has thirteen entries and the final 4,096-way projection executes.
   Layer 7 does not imply early-exit or a compute optimization.
3. Full named-parameter and named-buffer byte hashes (including nonpersistent
   buffers), configuration, normalization statistics and eval modes do not
   change; input samples also remain byte-identical. Hashing uses bounded byte
   views, not a second complete in-memory weight snapshot.
4. Restore the rotary-position cache and Python/CPU random state around each
   pass. The pinned Transformers implementation updates a transient position
   cache and samples LayerDrop randomness even in eval mode; this is distinct
   from parameter mutation or training. We do not assert that upstream forward
   execution itself never touches transient state. Training/masking/quantizer
   entrypoints are explicitly blocked throughout the seven comparisons.
5. Hooks are observers returning no replacement tensor and are removed even
   on failure. Attempted guarded Python networking invalidates the result.

Before the real artifact, a tiny real-Torch trace fixture checks hook identity,
all-block/projection execution, refusal of wrong-layer/missing-projection traces,
cleanup after interruption, scalar-buffer mutation detection and transient-state
restoration. These authored controls are not a MusicFM model accuracy test.

The worker requires the unchanged pinned Windows CPU runtime, two Torch threads,
a 4 GiB committed-memory job limit and a 300-second parent timeout. It refuses
to start below 2.5 GiB available physical memory, 6 GiB available commit or 5 GiB
off-system disk space. These are operational bounds, not peak inference claims.
The Python network guard and restricted pickle loader are not a hostile-code
sandbox; caller-owned immutable input/runtime storage is still required.

## Result

The **separate compatibility plus authored fidelity gate passes**, with the
original strict refusal retained. This is not musical discrimination or accuracy.

1. **Acquisition:** all 1,316,802,154 bytes match the frozen SHA-256. Transfer took
   573.68 seconds on the validation connection; that is not model runtime.
2. **Original strict gate:** stopped before state assignment and before any
   pretrained inference, because a state dtype differed. The
   [failure report](../parity/musicfm-fidelity-v1.json) retains zero completed
   pretrained cases, 20.97 seconds worker time and the original protocol/code
   identities. The original loader and fidelity protocol have not been relaxed.
3. **Read-only diagnosis:** the [complete inventory](../parity/musicfm-checkpoint-inventory-v1.json)
   contains 465 finite ordinary CPU tensors: 445 names/shapes/dtypes agree,
   eighteen scalar BatchNorm `num_batches_tracked` values use float32 instead of
   int64, and two positional-convolution weight-norm entries use legacy names.
   This inventory neither assigns state nor runs inference.
4. **Explicit compatibility gate:** a [second frozen protocol](../parity/musicfm-compatibility-lock-v1.json)
   permits only those two names and eighteen counter representations. It was
   defined and tested after metadata inspection but before any pretrained feature
   output. All 447 non-counter tensor objects are preserved, including aliased
   weight-norm tensors; there are no learned-value casts, reshapes or repairs.
5. **Actual pretrained extraction:** the [complete compatible result](../parity/musicfm-fidelity-compat-v1.json)
   passes all seven original cases and all 21 extraction calls, with bit-exact
   reference/hook/repeat comparisons and identical initial/final model-state
   hashes. The fixed controls retain 1, 1, 2, 2, 2, 25 and 50 tokens. Worker time
   was 68.64 seconds and process-lifetime peak RSS 1,783,451,648 bytes (1.66 GiB).
   Timings include construction, loading, strict validation and repeated whole-
   state hashing; they are not a per-song throughput or optimized inference
   benchmark. Nominal coordinate checks do not measure acoustic response delay.

The [PyTorch migration guide](https://docs.pytorch.org/docs/2.8/generated/torch.nn.utils.weight_norm.html)
maps `weight_g`/`weight_v` to `parametrizations.weight.original0`/`original1`.
The selected runtime's compatibility hook implements those same mappings. A
separate tiny Conv1d fixture proves identical reconstructed weights and outputs
under the old/new implementations with `dim=2` before applying the two exact
FMA key aliases. This does not authorize arbitrary prefix stripping or aliases.

Counters must be exactly the eighteen named scalar CPU float32 entries, finite,
nonnegative, integer-valued and below 2^63. Int64 conversion must round-trip to
the identical float32 value. Reject fractional, negative, nonfinite, overflow,
wrong-dtype, missing or colliding entries. The
[pinned BatchNorm implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/batchnorm.py)
uses the counter only on the training path; authored eval outputs remain exact
across restoration. This is eval-only serialization compatibility, not recovery
of historical training counts that may already have been rounded by the author.

The seven-case reference is the unchanged author's `get_latent` method in our
one pinned runtime after verified normalization, not a claimed replay of the
author's unavailable historical training/runtime environment. We did not change
versions, layers, precision, input cases or numerical tolerance to obtain a pass.

## Reproduction and interpretation

Use the separate environment from the prerequisite gate. Run model-free controls
with `python -m unittest discover -s evaluation/parity -p test_musicfm_fidelity.py -v`.
The actual probe requires the exact private source, sidecars and checkpoint:

```powershell
<private-python> evaluation/parity/musicfm_fidelity.py --upstream <pinned-source> --config <verified-config.json> --stats <verified-fma_stats.json> --checkpoint <verified-pretrained_fma.pt> --output <fresh-off-system-directory>
```

That original strict command is expected to retain the documented refusal.
`musicfm_checkpoint_inventory.py` takes the same arguments for metadata-only
diagnosis. The separate `musicfm_fidelity_compat.py` command takes those arguments
to run the explicit compatibility contract and unchanged authored comparisons;
use a different fresh output directory for each. None is a product strategy or
end-user mode, and no repaired checkpoint is written back to disk.

The first failed gate stops the worker. A failure report retains the completed
case denominator and stage; process termination can leave an incomplete output
directory instead. Neither is a pass, and failed controls must not be replaced.
Reports contain identities and summaries, not weights, music or feature arrays.

Even a complete authored pass would establish only this extraction path. A next
step must separately freeze one temporal-discrimination rule and complete-input
context/ownership policy before inspecting music features. Existing ARTBeaT
pairs stay exposed calibration, all 40 calibration identities remain accounted
for, RUBATO stays untyped and the holdout stays sealed. Positive discrimination
would still require independent acceptance, deployment feasibility and rights
clearance before becoming a product backend. The prior weight/training-provenance
and dependency-notice audit limitations remain; no commercial pack is approved.
