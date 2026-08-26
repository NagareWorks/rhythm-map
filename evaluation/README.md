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

Every suite also declares a `purpose`: `regression`, `calibration`, or
`holdout`. Only calibration permits truth-assisted policy sweeps and missed-beat
diagnostics. Regression and holdout suites accept fixed-policy evaluation but
reject `decoder-sweep` and `decoder-recoverability`. Reports repeat the declared
purpose so results cannot be detached from their development role.

## Commands

Fetch the public ARTBeaT rhythm-challenge slice outside the checkout. Audio is
always fetched; `--with-annotations` also retains the official SVG sources used
to audit checked-in truth:

```bash
cargo xtask dataset-fetch \
  --manifest evaluation/datasets/artbeat-v1.json \
  --output D:/rhythm-map-eval/artbeat-v1 \
  --with-annotations
```

The lock verifies exact byte sizes and SHA-256 identities. See
[`datasets/README.md`](datasets/README.md) for license, attribution, scope, and
truth derivation.

Fetch the independent FSLD fixed-tempo calibration slice without downloading
its complete 8.8 GB ZIP:

```bash
cargo xtask dataset-fetch \
  --manifest evaluation/datasets/fsld-tempo-v1.json \
  --output D:/rhythm-map-eval/fsld-tempo-v1 \
  --with-annotations
```

This suite has independently reviewed BPM but no beat phase. Run only the
end-to-end backend evaluation shown in `datasets/README.md`; its empty beat and
change labels intentionally cannot support oracle or decoder-policy claims.

Fetch the RUBATO multi-instrument real-performance calibration slice. The
ZIP64-aware fetcher transfers only 25 WAVs and their beat, measure, and
structure CSVs from the 6.3 GB archive:

```bash
cargo xtask dataset-fetch \
  --manifest evaluation/datasets/rubato-calibration-v1.json \
  --output D:/rhythm-map-eval/rubato-calibration-v1 \
  --with-annotations

cargo xtask eval \
  --suite evaluation/suites/rubato-calibration-v1.json \
  --no-fail

cargo xtask tempo-diagnose \
  --suite evaluation/suites/rubato-calibration-v1.json \
  --minimum-error-percent 25 \
  --report D:/rhythm-map-eval/reports/rubato-tempo-diagnostic.json

cargo xtask eval-backend \
  --suite evaluation/suites/rubato-calibration-v1.json \
  --model-pack models/beat-this-full-v1.json \
  --model-dir D:/rhythm-map-models/beat-this-full-v1 \
  --audio-dir D:/rhythm-map-eval/rubato-calibration-v1 \
  --report D:/rhythm-map-eval/reports/rubato-calibration-v1.json \
  --no-fail
```

RUBATO is calibration data, not a replacement holdout. Its structure labels
remain separate metadata and do not become tempo-change truth. `tempo-diagnose`
is deliberately calibration-only: it pairs every estimated point with truth,
reports the BPM implied directly by accepted beat timestamps, the regularized
BPM, their ratio to each other and to truth, and the nearest power-of-two
metrical shift. It also groups contiguous points above the requested error
threshold. It rejects regression and holdout suites so timestamp-level truth
cannot leak into policy selection.

Fetch the precommitted, case-disjoint ARTBeaT holdout and retain its auditable
SVG annotation sources:

```bash
cargo xtask dataset-fetch \
  --manifest evaluation/datasets/artbeat-holdout-v1.json \
  --output D:/rhythm-map-eval/artbeat-holdout-v1 \
  --with-annotations
```

Gate the truth and ideal-observation path before model inference, then open the
holdout once with the decoder policy selected on calibration data:

```bash
cargo xtask eval \
  --suite evaluation/suites/artbeat-holdout-v1.json

cargo xtask decoder-eval \
  --suite evaluation/suites/artbeat-holdout-v1.json \
  --policy viterbi-edge-logit-minus-3.0-bias-2.0 \
  --model-pack models/beat-this-full-v1.json \
  --model-dir D:/rhythm-map-models/beat-this-full-v1 \
  --audio-dir D:/rhythm-map-eval/artbeat-holdout-v1 \
  --report D:/rhythm-map-eval/reports/artbeat-holdout-v1.json
```

This holdout is disjoint from the ARTBeaT calibration cases but shares their
source corpus; see [`datasets/README.md`](datasets/README.md) for the exact
scope and non-claims.

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
  --observation-cache D:/rhythm-map-eval/observation-cache-v1 \
  --report D:/rhythm-map-eval/reports/beat-this-full-v1.json \
  --no-fail
```

`--observation-cache` is optional evaluation infrastructure. A cold run stores
the raw backend-neutral observations only after they pass deterministic
estimation; a hot run reloads them and still reruns PCM enrichment, the current
estimator, and every metric. Entries are addressed by audio SHA-256, exact model
manifest SHA-256, a versioned backend/decode contract, and the complete decoder
policy. Estimator policy and truth are deliberately absent from the key, so an
estimator A/B reuses identical model evidence without allowing labels into the
cache. Cache files include decoded sample rate/count validation and exact
round-trip floating-point JSON. Keep the directory outside the checkout.

Backend report schema v11 records the versioned `observation_cache_contract`,
adds per-case `observation_cache_hit`, and may retain a backend's uniform dense
pulse/downbeat activation series under observation diagnostics. Dense evidence
is not copied into product `Analysis` output. Runtime fields from a cold and hot
report are not directly comparable as end-to-end model benchmarks; all
observations, analyses, and metrics must otherwise be identical.

Schema v11 also keeps two kinds of truth-assisted diagnosis separate from product
selection. `beat_error_location` splits timestamped calibration misses and
extras into leading, interior, trailing, and completely unanchored counts, and
reports how many misses still have raw backend support. The field is absent on
holdouts and tempo-only cases. `fixed_tempo_hypothesis_coverage` scores the
already product-visible global half/selected/double alternatives only when a
calibration case has one constant truth segment. Its `best_top_k_error_percent`
is a coverage ceiling, not a deployable selector result; no truth changes the
primary tempo map or any hypothesis score.

Compare the experimental BeatNet observation path only on the already-open
ARTBeaT calibration suite:

```bash
cargo xtask eval-beatnet \
  --suite evaluation/suites/artbeat-v1.json \
  --model-pack models/beatnet-v1.json \
  --model-dir D:/rhythm-map-models/beatnet-v1 \
  --audio-dir D:/rhythm-map-eval/artbeat-v1 \
  --report D:/rhythm-map-eval/reports/artbeat-beatnet-dense-meter-v5.json \
  --no-fail
```

`eval-beatnet` rejects regression and holdout suite roles. Its guarded graph
fuses a grid prior with beat/downbeat/non-beat evidence, but every emitted
timestamp must still be a real BeatNet pulse maximum. This is developer
calibration evidence, not a second end-user strategy or a selectable shipping
backend. Analysis schema v4 exposes the selected and supported half/double-time
beat sequences with truth-free relative scores plus time-localized disagreement
regions. The shipping estimator and this command both attempt the fixed
harmonic-aware local path automatically.
`local-metrical-path-v1` remains accepted by frozen evaluation workflows as a
compatibility name for the same behavior. It adds at most one real-timestamp
path whose metrical level may vary locally. Calibration report schema v11
records those exact hypotheses, their time-localized disagreement regions, and
optional dense activations under observation diagnostics. The primary selected
sequence remains unchanged.

After freezing `local-metrical-path-v1` on calibration, open a timestamped
holdout exactly once through the fixed-candidate command:

```bash
cargo xtask eval-beatnet-holdout \
  --suite evaluation/suites/vienna4x22-holdout-v1.json \
  --policy local-metrical-path-v1 \
  --model-pack models/beatnet-v1.json \
  --model-dir D:/rhythm-map-models/beatnet-v1 \
  --audio-dir D:/rhythm-map-eval/vienna4x22-v1 \
  --report D:/rhythm-map-eval/reports/vienna4x22-beatnet-local-metrical-path-v1.json \
  --no-fail
```

This command accepts only a `holdout` suite and the single registered policy.
It emits no raw observations or policy sweep. `primary` is the selected BeatNet
sequence; `truth_free_choice` chooses between primary and local path using only
their existing relative evidence; `coverage_ceiling` uses holdout truth to show
whether either member contains a better path and is explicitly not deployable.
The one-shot result is recorded in
`baselines/vienna4x22-beatnet-local-metrical-path-v1.md`.

For a suite containing external audio, add the explicit local audio root:

```bash
cargo xtask eval-backend \
  --suite evaluation/private/real-music-v1.json \
  --model-pack models/beat-this-full-v1.json \
  --model-dir D:/rhythm-map-models/beat-this-full-v1 \
  --audio-dir D:/rhythm-map-evaluation-audio \
  --report D:/rhythm-map-eval/reports/private-real-music-v1.json \
  --no-fail
```

To isolate Beat This peak decoding from the neural model and the tempo-map
estimator, run the decoder sweep. Each audio case is inferred once; the same
50 Hz beat/downbeat logits are then decoded by every threshold and
local-maximum policy:

```bash
cargo xtask decoder-sweep \
  --suite evaluation/suites/artbeat-v1.json \
  --model-pack models/beat-this-full-v1.json \
  --model-dir D:/rhythm-map-models/beat-this-full-v1 \
  --audio-dir D:/rhythm-map-eval/artbeat-v1/audio \
  --report D:/rhythm-map-eval/reports/artbeat-decoder-sweep.json
```

The sweep scores raw decoded beats before tempo estimation. Its
`per_case_policy_oracle_mean_beat_f1` field chooses the best tested policy
separately for each case and is only a diagnostic ceiling; it is not a
deployable result and must not be compared with one fixed decoder as if it
were one.

Named policies in this crate are evaluation candidates, not product modes. A
candidate that clears promotion replaces the single product implementation. A
runtime selector is permitted only after a precommitted experiment proves both
that the alternatives are irreducibly input-dependent and that the applicable
input class can be identified from truth-free runtime evidence. Dataset IDs,
filenames, annotations, and per-case oracle scores are forbidden selector
inputs.

For calibration suites with timestamped beat truth, current reports additionally
emit `candidate_evidence`, full-band and low/mid/high
spectral-flux onset diagnostics, and `pulse_hypothesis_coverage`. The onset
envelope is backend-neutral and deterministic; its strength is reported
independently and does not alter hypothesis ranking. Candidate recall asks
whether any real backend local maximum exists near each truth beat. It
separately reports candidate recall and confidence for truth beats missed by
the selected sequence, because the all-beat median is otherwise dominated by
events the decoder already kept.
Top-K coverage normally scores a fixed, truth-free set consisting of the
selected sequence, two alternating half-time phases, and an optional real-
midpoint augmentation. `local-metrical-path-v1` adds one locally varying path
constructed from real candidates plus deterministic harmonic-change evidence.
The same fields are omitted from regression and holdout reports so their truth
cannot become an implicit per-case router.

Each hypothesis includes an evidence breakdown. Ranking charges a half-time
subset for discarded selected-event evidence in addition to measuring event
strength and interval continuity. The calibration history in
`baselines/artbeat-candidate-coverage-v1.md` records why the earlier unpenalized
confidence/continuity score was rejected.

To test whether an independent backend can rank those hypotheses without
rerunning inference, compare two reports for the same calibration suite:

```bash
cargo xtask consensus-diagnose \
  --primary D:/rhythm-map-eval/reports/artbeat-cache-default-v9-final.json \
  --secondary D:/rhythm-map-eval/reports/artbeat-beatnet-dense-meter-v5.json \
  --report D:/rhythm-map-eval/reports/artbeat-beatthis-beatnet-dense-meter-v3.json
```

`consensus-diagnose` ranks the primary report's existing hypotheses only by
one-to-one agreement with the secondary report's top-ranked sequence. It then
uses the embedded calibration scores to attribute gains and regressions. Four
equal-duration window margins expose cases where the two backends' metrical
relationship changes inside the track. Schema 3 additionally reports a
meter-gated candidate: an alternative is eligible only when its beat agreement
and its class-balanced BeatNet downbeat periodicity both strictly improve over
the primary. When available, the meter score samples the uniform pre-decoder
activation series instead of only events already selected by BeatNet. It tries
every phase of fixed 2/3/4-pulse bars and uses no fitted weight or BPM band.
The ARTBeaT dense result changes no case and rejects the apparent improvement
from decoded-event evidence as selection-biased. This command accepts
calibration reports only, requires distinct model packs and backend IDs plus
identical audio identities, and does not change either backend or add a product
strategy. A passing calibration gate would still require one separately
precommitted holdout run before any shipping selector could be considered.

After choosing one policy on calibration data, evaluate that exact registered
policy on a separate holdout manifest:

```bash
cargo xtask decoder-eval \
  --suite evaluation/private/real-music-holdout-v1.json \
  --policy supported-midpoints-logit-minus-3.0 \
  --model-pack models/beat-this-full-v1.json \
  --model-dir D:/rhythm-map-models/beat-this-full-v1 \
  --audio-dir D:/rhythm-map-evaluation-audio \
  --report D:/rhythm-map-eval/reports/holdout-supported-midpoints.json
```

`decoder-eval` runs the named candidate and the immutable `upstream-default`
baseline in one inference pass. Its report includes overall metrics, per-case
gates, stable per-tag aggregates such as `rubato`, `drumless`, or
`metric-ambiguity`, and explicit candidate-minus-baseline deltas. It deliberately
omits every other candidate and the truth-selected per-case policy oracle. The
gate requires the candidate's absolute beat budgets and rejects any case-level
F1 regression. Use `--no-fail` only for exploratory calibration reports, not for
the final holdout gate.

To determine whether a stronger decoder has model evidence to work with, inspect
every truth beat missed by the upstream decoder:

```bash
cargo xtask decoder-recoverability \
  --suite evaluation/suites/artbeat-v1.json \
  --model-pack models/beat-this-full-v1.json \
  --model-dir D:/rhythm-map-models/beat-this-full-v1 \
  --audio-dir D:/rhythm-map-eval/artbeat-v1/audio \
  --report D:/rhythm-map-eval/reports/artbeat-recoverability.json
```

This is a truth-assisted diagnostic, not a decoder score. For each missed beat
it records the strongest frame and the strongest radius-one and radius-three
local peaks inside the existing timing-tolerance window. The aggregate bins
separate moderate subthreshold evidence from very weak evidence and the absence
of a local peak. No audio bytes, paths, or model tensors are stored.

The filename hint in a manifest is non-authoritative. The resolver verifies the
SHA-256 of the exact encoded file bytes and, when the hint is stale, searches
supported audio below `--audio-dir` by content. It does not follow symbolic
links. The decoded duration must agree with truth within 100 ms.

The paired report attributes a failing suite to the deterministic estimator
when the oracle path fails, or to the broader observation path when oracle
beats pass but rendered audio fails. The latter includes model errors and
deterministic robustness to noisy or metrically ambiguous observations; it is
not evidence by itself that the neural backend should be replaced. Capability
slice metrics remain the evidence used for engineering decisions.

Each end-to-end case also records the backend's raw beat timestamps and
beat/downbeat confidence values, the number retained by deterministic analysis,
the final downbeat count, capability tags, the verified external-audio SHA-256,
the activity-envelope size and low-activity fraction, and the warnings that
identify silence rejection, metrical selection, bar-phase selection, or guarded
transition-grid recovery. These diagnostics explain a metric change without
exposing model tensors, filenames, local paths, or audio bytes.

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

The report distinguishes beat matching, downbeat/bar-phase matching,
tempo-curve error, and same-kind change-point matching. Beat and downbeat F1
are independent acceptance gates. Thresholds belong to the suite and can be
overridden for a documented case; they should be tightened only from measured
product requirements, not adjusted to make a release green.

Synthetic recipes select an audio profile. `click` is a sparse timing signal,
`percussion` adds deterministic drums and subdivisions, and `drumless` uses
harmonic note onsets without drums. All profiles retain the same analytic beat
and tempo truth.

## Adding external evaluation music

1. Copy `private.example.json` (or `private-holdout.example.json`) and
   `truth.example.json` below
   `evaluation/private/`, which is ignored by Git.
2. Inspect the exact local asset:

   ```bash
   cargo xtask audio-inspect --input D:/licensed-audio/example.flac
   ```

3. Put the reported SHA-256 in the manifest and its decoded duration in truth.
   The local filename remains only a convenience hint.
4. Replace the example beats, downbeats, tempo segments, and change points with
   independently authored annotations.
5. Run `cargo xtask eval --suite evaluation/private/<suite>.json` first to gate
   truth and the oracle estimator without requiring audio or a model.
6. Run `eval-backend` with `--audio-dir` only after the oracle path passes.

Do not copy copyrighted audio into the repository. Record the source and
permitted use, and keep annotations under an explicitly chosen license. See
[`CALIBRATION.md`](CALIBRATION.md) for corpus slices, annotation review, holdout
separation, and the model-versus-estimator decision rule.
