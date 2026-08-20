# Evaluation

This directory defines reproducible product acceptance and regression suites.
It does not treat third-party music as a repository fixture merely because a
developer can play or purchase it.

## Three asset classes

- **Generated** recipes are checked in and produce exact beat, tempo, and change
  truth. Rendered WAV files are disposable build artifacts.
- **Public** cases record the audio license and annotation license separately.
  Audio is redistributed only when both the license and project policy permit
  it; otherwise the manifest points to an official source and verifies a hash.
- **Private** cases remain outside Git. Only `private.example.json` is public.
  Local manifests identify legally acquired audio by SHA-256 and a non-binding
  filename hint, while reports contain aggregate metrics rather than audio
  bytes or machine-specific absolute paths.

Every case records whether redistribution and commercial evaluation are
allowed. These fields document provenance; they are not a substitute for legal
review.

## Commands

Gate the deterministic tempo-map estimator using ideal beat observations:

```bash
cargo xtask eval
```

Render deterministic synthetic WAV files and exact truth outside the checkout
for an end-to-end Beat This run:

```bash
cargo xtask render --output D:/rhythm-map-eval/generated-v1
```

Run the generated audio directly through a verified Beat This model pack and
compare it with ideal beat observations:

```bash
cargo xtask eval-backend \
  --model-pack models/beat-this-full-v1.json \
  --model-dir D:/rhythm-map-models/beat-this-full-v1 \
  --report D:/rhythm-map-eval/reports/beat-this-full-v1.json \
  --no-fail
```

The paired report attributes a failing suite to the deterministic estimator
when the oracle path fails, or to the broader observation path when oracle
beats pass but rendered audio fails. The latter includes model errors and
deterministic robustness to noisy or metrically ambiguous observations; it is
not evidence by itself that the neural backend should be replaced. Capability
slice metrics remain the evidence used for engineering decisions.

Each end-to-end case also records the backend's raw beat timestamps and
beat/downbeat confidence values, the number retained by deterministic analysis,
the activity-envelope size and low-activity fraction, and the warnings that
identify silence rejection, half-time selection, or guarded transition-grid
recovery. These diagnostics explain a metric change without exposing model
tensors or requiring audio redistribution.

`cargo xtask` uses an optimized build because unoptimized neural inference is
not a meaningful performance baseline. Keep the generated oracle suite in
ordinary CI, use a short verified-model smoke suite for compatibility, and run
the full end-to-end suite as a scheduled or release quality gate.

After a CLI or another product surface writes one Analysis JSON file per case,
score the directory with the exact same metrics:

```bash
cargo xtask score \
  --predictions D:/rhythm-map-eval/predictions/generated-v1 \
  --report D:/rhythm-map-eval/reports/generated-v1.json
```

The report distinguishes beat matching, tempo-curve error, and same-kind
change-point matching. Acceptance thresholds belong to the suite and can be
overridden for a documented case; they should be tightened only from measured
product requirements, not adjusted to make a release green.

Synthetic recipes select an audio profile. `click` is a sparse timing signal,
`percussion` adds deterministic drums and subdivisions, and `drumless` uses
harmonic note onsets without drums. All profiles retain the same analytic beat
and tempo truth.

## Adding copyrighted evaluation music

Do not copy it below this directory. Create a local manifest below
`evaluation/private/` (ignored by Git), give the audio an immutable SHA-256,
record the source and permitted use, and keep independently created annotations
under an explicitly chosen annotation license. A future dataset resolver will
map those identities to local or access-controlled files.
