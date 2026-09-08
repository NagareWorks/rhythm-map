# Normalized pre-head capture fidelity v1

## Frozen scope

This implements the capture gate selected by the
[reuse review](pretrained-representation-review-v1.md), not a new decoder or a
musical-discrimination test. The source, runner, contract and 15 model-free
controls were fingerprinted before the first neural execution. No layer search,
feature fitting, new checkpoint, holdout access or product change is involved.

The [contract](../parity/prehead-capture-lock-v1.json) retains the same `final0`
checkpoint, source revision and CPU dependency versions as the original
[reference audit](beat-this-reference-parity-v1.md). All private trace bytes
must match that audit before JSON interpretation; checkpoint bytes are checked
before weights-only deserialization. Model construction uses those in-memory
verified bytes, not an upstream loader that could fall back to a download.

The only neural feature tap is the normalized transformer output immediately
before `task_heads`, with shape `[1, chunk_frames, 512]`. A pre-forward hook
clones the tensor and returns nothing. It is removed in a `finally` block;
an injected-failure module verifies removal when the tapped call throws.
No optimizer or backward pass is created. Model parameters and buffers must
have the same digest before and after the complete audit.

## What must agree

- Each original chunk runs once without the hook and once with it. The two
  beat/downbeat output arrays must be **bit-exact**, not merely within tolerance.
- The unchanged task head runs on the cloned feature tensor. Its outputs must
  match the tapped pass within `2e-5 + 1e-6 * abs(reference)` per element.
  This numerical budget was fixed before execution, not fitted to results.
- All tensors have exact dimensions, finite float32 values and CPU ownership.
  Input mel is unchanged. Failed shape, availability, budget or identity checks
  raise errors rather than trimming samples or returning partial success.
- Chunk splitting independently reproduces the pinned upstream's padded input
  bits. Geometry stays at 1,500 frames, six-frame borders, stride 1,488,
  avoid-short-end and `keep_first`. All 512 channels and both logits use the
  same owner. Per-chunk border positions never become global observed frames.
- Aggregated logits must be bit-exact with the upstream aggregator applied to
  the untapped chunk outputs. Minimal decoding must retain identical ordered
  events; there is no fresh event matching or favorable time shift.
- The reference cases also retain the old mel/logit numerical budgets against
  frozen RTen traces. Port/adapter events must have identical ordered half-frame
  identities and differ by at most `1e-5` seconds. Absolute time tolerance is
  never scaled by song duration. This is not a new Rust feature-extraction run.
- Private NPZ archives use exclusive creation, disallow pickle on reload and
  must round-trip every feature/head/owner bit. A failed run has no completion
  report; any partial private files are not accepted captures.

Seventeen NumPy-only CI controls cover budget/schema drift, byte verification
before parsing, short/full/overlapping chunk geometry, first-owner behavior,
missing/extra/malformed tensors, finiteness/dtype, exact versus approximate
checks, ordered events, private-path restrictions and archive integrity.
The final two controls also bind the published report to its code/lock and
verify that public output contains digests rather than raw feature arrays.
PyTorch is required only for the explicit private integration command below.

## Inputs and interpretation

The four neural authored controls are a 17-frame zero spectrogram, modular
deterministic spectrograms with 1,488 and 1,489 frames, and a 2,980-frame
three-chunk input. These are instrumentation controls, not synthesized music,
silence labels or tempo-change truth. Additional model-free boundary controls
exercise lengths up to the fixed 4,096-frame budget.

The only music inputs are the original two frozen reference traces:

- `artbeat-05-75-to-150`: complete short example, 680 frames;
- `rubato-bach-bwv1007-01-ar-macleod2011`: 35-second prefix, 1,751 frames.

They test single-window and overlapping-window extraction. They do not certify
whole-recording edges, all 40 calibration tracks, or independent accuracy.
The retained mel/PCM and legacy output identities are not replaced by current
decoding, cropping, resampling candidates or an alternative backend.

Runtime fields separate untapped forward, tapped forward and archive write
times. They are sequential diagnostic timings, not a warmed randomized
benchmark; do not infer a speedup or reliable overhead ratio from their order.
Gate elapsed time excludes initial imports, input verification and model load.
Peak RSS is the **process-lifetime high-water mark**, including verification,
model loading, baseline runs and temporary copies, not native product memory.
Raw features, heads and owner arrays stay in the private directory. Only
identities, shapes, counts, numerical-error aggregates and costs may enter Git.

## Measured result: capture fidelity passes

The first frozen neural execution passed all four authored inputs and both
reference inputs. The [aggregate report](../parity/prehead-capture-v1.json) is
byte-identical to the private run's summary; SHA-256 is
`f2102630d9e5b36d14cea43ea8a0121df2dfabf5c2ccfd57b7df0b8cdaf4d90a`.
The extraction source and lock did not change after inspection.

| Check | Result |
| --- | --- |
| Original chunks / neural forwards | 10 / 20 (one untapped and one tapped per chunk) |
| Both head outputs, tap versus untapped | Bit-exact on all ten chunks; maximum difference zero |
| Both outputs recomputed from retained features | Also bit-exact on all ten chunks, stricter than the frozen tolerance |
| Shared ownership and upstream aggregation | All six inputs pass; no padded or uncovered global frames |
| Reference events preserved | ARTBeaT 17 beats / 5 downbeats; RUBATO prefix 50 / 24 |
| Maximum reference-logit error versus frozen RTen | Approximately `1.91e-5` |
| Maximum frontend-mel error versus frozen trace | Approximately `1.12e-4` |
| Maximum event-time difference versus frozen port/adapter | Approximately `1.84e-6` seconds, with the same ordered event identities |
| Private archives | All six round-trip exactly; no public feature payload |
| Model state and hook failure handling | Unchanged parameter/buffer digest; injected exception removes hook |

The CPU gate took 167.97 seconds from the start of the authored cases through
the final checks. Its process-lifetime peak RSS was 563,953,664 bytes (537.83 MiB).
These figures cover duplicate forward comparisons and instrumentation, not
shipping latency, a runtime improvement or native/WASM feasibility. No layer,
threshold or numerical-budget adjustment was made to obtain this pass.

## Reproduction

Use the existing reference environment and a new private output directory on
a non-system drive. There are no automatic installs or model downloads:

```bash
python -m unittest discover -s evaluation/parity -p test_prehead_capture.py -v
python evaluation/parity/prehead_capture_audit.py \
  --upstream /data/reference/beat_this \
  --checkpoint /data/reference/final0.ckpt \
  --trace /data/private/artbeat-05.final.trace.json \
  --trace /data/private/rubato-bach.final.trace.json \
  --output-dir /data/private/prehead-capture-new
```

The two traces must be supplied in that frozen order. The output parent must
already exist; the output directory must not. Do not publish the `.private.npz`
files or use this short-reference tool as a whole-corpus feature exporter.

## Next boundary

A passed capture gate only makes the feature tap available for research. Before
inspecting real hidden features for musical discrimination, freeze one
label-free temporal comparison with authored omission, true-change, metrical
density and edge controls. Keep the previous seven pairs/four phases and all
failures; account for all 40 calibration IDs without inventing hard tempo labels
for RUBATO. This gate introduces no such comparison and reports no accuracy
improvement. Native export, automatic clock inference and independent product
acceptance remain separate work.
