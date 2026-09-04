# RUBATO full-PCM input equivalence

## Scope and result

On 2026-09-04, the locked, already-opened 25-recording RUBATO calibration
cohort passed a model-free full-input comparison: **25/25 bit-identical**,
143,105,243 float32 mono samples, 108.167228 minutes at 22,050 Hz. There
were zero differing shared samples and zero unpaired tail samples.

The audit ran on Windows with Rust 1.98.1 and the committed Cargo.lock,
using the optimized evaluation profile. Each recording was decoded by the
actual former `beat_this::load_audio(path, 22050)` implementation from
`beat-this =1.0.0` and by the shipping `decode_audio`. It compared complete
finite, nonempty buffers by float32 bits (including signed zero), not a
preview, a rounded sample, duration alone, or floating-point tolerance.
Audio hashes were validated before and after decoding. Native-rate decoding
also verified the model-rate bypass's complete sample count and PCM hash.

The [lock](../parity/rubato-pcm-equivalence-lock-v1.json) was specified before
running the cohort. The [summary](../parity/rubato-pcm-equivalence-v1.json)
records per-case source and little-endian PCM hashes, lengths, differences,
and source/dependency identities, but no audio samples or private paths.
Tests bind it to the full ordered suite and exact auditor sources.
The manifest identity uses the repository's committed LF bytes. A stale
CRLF working copy is deliberately rejected; neither annotations nor audio
are transformed to make hashes match.

## Why this is relevant, but not cache promotion

Source comparison of the former adapter at `545a4d9` and shipping adapter
at `22d2ade` shows the inference change is input preparation before the
same tracker call. For valid, nonempty 22,050 Hz PCM, the shipping
`prepare_mono` returns a borrowed slice unchanged; the tracker receives
the same sample bits and sample rate. Duration calculation and default
peak decoding are unchanged. The upstream Rust dependency remains pinned
to 1.0.0. Diagnostic trace additions do not run on ordinary inference.

This is **input-path evidence**, not an independent rerun of inference or a
proof that every old cache was produced by the claimed model/runtime. The
audit intentionally reads/writes no cache and loads no model. It does not
relabel v1 entries, add a production cross-contract fallback, or report
old scores as new v2 measurements. The general v2 invalidation remains
necessary: other rates, stereo downmixes, and damaged files may differ.

Before reusing any old RUBATO observations, a separate read-only bridge
must bind each entry to the exact audio, decoded identity, original v1
contract, default decoder, verified model assets, and historical report;
then reproduce the old raw events and selected scores through the current
engine with the actual PCM. Missing or inconsistent evidence must fail
without inference or cache mutation. Preserve v1 provenance even if that
scoped bridge passes. Full model/runtime reproducibility is a separate
question from deterministic input identity.

Only after that gate should the frozen candidate-evidence features be
transferred to RUBATO and reported separately from ARTBeaT. No recovery
threshold, public strategy, training, holdout, or release changes here.

## Reproduce

Use the suite's hash-verified local audio layout; the destination must not
already exist. The report is small, but audio/build caches belong on a data
drive outside Git.

```bash
cargo run --locked --profile evaluation -p rhythm-map-eval \
  --example rubato_pcm_equivalence -- \
  --suite evaluation/suites/rubato-calibration-v1.json \
  --audio-dir /data/rubato-calibration-v1 \
  --output /data/reports/rubato-pcm-equivalence-new.json
```

The example rejects altered, reordered, shortened, or holdout manifests
before decoding. A valid comparison with different samples or lengths writes
a negative summary and exits unsuccessfully; decode/identity failures abort.
CI checks signed-zero differences, sample changes, truncation, nonfinite and
empty input, cohort locking, and the frozen report's identities without
downloading or redistributing any recording.
