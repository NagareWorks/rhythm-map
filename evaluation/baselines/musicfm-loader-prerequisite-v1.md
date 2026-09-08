# MusicFM: offline loader prerequisites, not pretrained fidelity

Date: 2026-09-08. This implements the runtime/construction prerequisites from
the [feasibility screen](musicfm-feasibility-v1.md). The
[complete result](../parity/musicfm-loader-prerequisite-v1.json) records real
PyTorch authored controls, source identities and the installed notice inventory.
No MusicFM checkpoint was acquired or loaded, no Conformer inference was run,
and no music features, calibration recordings or holdout were accessed.

## Decision and measured scope

The isolated Windows CPU runtime can construct the pinned architecture and
exercise the restricted loading primitives. That is enough to prepare the next
bounded checkpoint gate, **not** to claim pretrained compatibility, better BPM
accuracy, native/WASM feasibility, or commercial distribution clearance.

- CPython 3.13.5, torch/torchaudio 2.8.0+cpu, NumPy 2.2.6, einops 0.8.1,
  Transformers 4.46.3 and tokenizers 0.20.3; 28 resolved wheels in the
  [runtime inventory](../parity/musicfm-runtime-win64.json) and matching
  [hash requirements](../parity/musicfm-runtime-win64.txt). This is a separate
  private environment; the established Beat This environment is unchanged.
- Construction produced 328,932,480 meta parameter elements and 465 state
  entries, with only 133,248 real frontend buffer elements. The effective
  twelve-layer/1,024-wide configuration and full shape/dtype schema are hashed.
  These are constructor observations, not a tensor inventory of the FMA file.
- Seven authored silence/impulse/tone clips from 1,025 to 48,000 samples passed
  actual frontend defaults, global normalization, awkward-length geometry,
  finite output, exact repeatability and unchanged input checks. The subsampler
  uses tiny authored channel dimensions, not trained MusicFM encoder weights;
  its parameters and BatchNorm buffers remain exactly unchanged.
- The successful development run used about 289 MiB process-lifetime peak RSS.
  The worker has a 4 GiB committed-memory cap, two Torch threads and a parent
  timeout of 180 seconds. RSS and committed-memory limits are different
  quantities. This is neither a full-model inference benchmark nor its resource
  estimate; meta parameters deliberately occupy no real parameter storage.

Transformers 4.46.3 was selected before any music outputs, for its
[Conformer implementation](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py)
and compatible binary dependencies. Source inspection retains the intended
hidden-state tuple interpretation; actual layer-7 extraction fidelity is still
untested. There was no version/layer/metric sweep against musical labels.

## Loading and refusal contract

`musicfm_loader.py` hashes the complete sidecars and four explicitly enumerated
upstream source files before source execution. Only source/checked-in runtime
text has documented LF canonicalization; downloaded config, statistics and
checkpoint bytes must match exactly. Arbitrary checkout imports, occupied module
namespaces and upstream `torch.load` calls are refused. The online configuration
lookup is replaced by one verified local configuration; no speech weights load.

Checkpoint loading hashes the regular, non-link file first, then rewinds and
passes **the same open handle** to `torch.load(map_location='cpu',
weights_only=True)`. There is no unrestricted fallback or custom-global allowlist.
The state must use exactly the `model.` prefix and expected keys, shapes and
dtypes, containing ordinary finite CPU tensors. Strict `assign=True` loading
replaces meta tensors without allocating a second full random model. The real
FMA checkpoint may still fail these requirements; never relax them silently.

Small authored Torch checkpoints verify exact state round-trip and rejection of
bad shapes, dtypes, nonfinite values and an unsupported pickle callable. Unit
tests separately verify refusal before deserialization for bad hashes, sizes,
links and missing files; malformed JSON, sidecars, source changes and runtime
drift; namespace cleanup; geometry; and swallowed network failures. Fake tensor
unit tests are not counted as pretrained model tests.

Offline flags, isolated Python import mode and a Python socket/DNS guard stop
accidental downloads. Any attempted guarded operation invalidates the result,
even when a dependency swallows the immediate error. Initial runs were rejected
because urllib3's IPv6 capability probe binds localhost during import. Setting
`socket.has_ipv6=False` inside the guard suppresses that irrelevant probe;
connections and binds are still prohibited and restoration is tested. Hugging
Face import-time cache bookkeeping stays in the private output directory.

This is not an OS network sandbox or hostile-native-code defense. Runtime files
and checkpoint storage are caller-owned and immutable during a run; hashing an
open handle does not lock out concurrent hostile writes. Restricted pickle is
not a resource-exhaustion sandbox. The memory/time envelope is mandatory for the
probe; standalone loader helpers do not apply it for callers.

## Dependency notices are inventoried, not cleared

The report preserves relative installed notice paths, byte counts and hashes
for 27 packages. The tokenizers wheel has an Apache license classifier in
METADATA but no separate notice file in RECORD; that exact absence is retained
as a known distribution-audit item, not filled with an invented notice. The
probe rejects any change in the known missing-notice set. This inventory does
not audit every embedded native/Rust dependency or satisfy redistribution by
itself. MusicFM's weight/training provenance limitations from the earlier screen
also remain. No dependency, source copy or model payload becomes a shipping pack.

## Reproduce without changing the existing evaluation environment

Use a fresh CPython 3.13.5 Windows x64 environment and off-system-drive paths.
Provisioning wheels requires network access; the subsequent probe does not.
Do not re-resolve versions or install into the established parity environment.

```powershell
uv venv --python <python-3.13.5.exe> <private-env>
uv pip sync evaluation/parity/musicfm-runtime-win64.txt --python <private-env>/Scripts/python.exe --torch-backend cpu --default-index https://pypi.org/simple --only-binary :all: --require-hashes
uv pip check --python <private-env>/Scripts/python.exe
<private-env>/Scripts/python.exe evaluation/parity/musicfm_loader_probe.py --upstream <pinned-source> --config <verified-config.json> --stats <verified-fma_stats.json> --output <fresh-private-directory>
python -m unittest discover -s evaluation/parity -p test_musicfm_loader.py -v
```

Keep TEMP/TMP and UV_CACHE_DIR off the system drive during provisioning. Source
and sidecar pins are inherited from the screen; this probe deliberately has no
MusicFM checkpoint or audio-file argument. Normal repository CI requires neither
Torch/Transformers nor these private assets and runs the model-free controls.
The private integration result is separate from that CI result.

## Next finite gate

1. Freeze the exact FMA-only acquisition and authored extraction protocol before
   loading: full byte identity, strict state schema, fixed float32/non-Flash
   layer 7, sample lengths, reference comparison, numerical tolerance, resource
   bounds and stop conditions. Check available memory/storage first.
2. Acquire one private FMA payload and verify all bytes; compare unmodified
   reference `get_latent(..., layer_ix=7)` with any tap on authored PCM only.
   Verify unchanged complete model state, tuple/tap identity, coordinates,
   repeatability, finiteness, peak memory and wall time. No fallback to MSD,
   unsafe pickle, another layer, precision or version if it fails.
3. Only a fidelity pass allows freezing one temporal discrimination rule and
   complete-input/context policy before music features are inspected. Keep all
   40 calibration identities accounted for, RUBATO untyped and holdout sealed.

This change adds no public strategies, knobs, mandatory backend or Rust runtime
dependency. One-call, zero-tuning behavior and the optional metadata-pack horizon
remain intact. There is no training decision or release in this prerequisite gate.
