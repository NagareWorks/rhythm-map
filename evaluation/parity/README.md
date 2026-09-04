# Beat This numerical parity audit

This calibration-only audit separates runtime/conversion parity from musical
accuracy. No model is trained and no estimator/decoder strategy is selected.
`reference-lock.json` records the source revisions, checkpoint identity and
initial cases. The official Python revision is also the revision used by the
Rust port's pinned `scripts/gen_golden.py`.

## Boundaries

- Aggregate JSON artifacts are byte-addressed evidence. Git preserves their
  original line endings (`-text diff`) so a checkout on another OS does not
  break historical hash links. New reports use UTF-8/LF; do not normalize or
  rewrite an already-frozen report merely to restyle it.
- Raw traces contain reconstructable PCM and must remain in a private external
  directory. They inherit the audio's rights; do not commit or distribute them.
- Rust verifies the suite role, audio SHA-256 and model pack before inference.
  Holdout and regression suites are rejected. Truth annotations are not read.
- The exporter uses the actual shipping file decoder and the actual backend
  model. A feature-gated trace method captures mel, logits, port events and
  adapter observations from one inference. It always uses the default decoder.
- Python independently runs the official frontend, official chunking and
  postprocessor, the original checkpoint and CPU ONNX Runtime. No DBN is used.
- Same-PCM/same-mel comparisons isolate individual components. Original-file
  comparisons additionally cover decode/downmix/resampling. Different resampling
  filters need not be numerically identical, so their waveforms are diagnostic;
  original-file event comparison uses an absolute one-frame (20 ms) budget.
- Budgets are declared in `compare_reference.py`, not tuned per case. Missing
  checkpoint comparison is explicitly incomplete and exits nonzero. A passing
  numerical audit does not establish beat accuracy, correct musical pulse, or
  full-track/end-of-track behavior.

## Setup (evaluation machine only)

Use an isolated Python environment on a data disk; install `requirements.txt`.
For example, first install `torch==2.8.0 torchaudio==2.8.0` from
`https://download.pytorch.org/whl/cpu`, then install the remaining requirements.
Clone the official reference and check out the locked revision. The checkout
must be clean. Download `final0` explicitly from the locked URL, verify its
size/digest, and retain it outside Git. The hash records the audited bytes, not
an upstream digital signature. No comparison command downloads code or weights.

## Run

Replace the example data-disk paths with your own external directories:

```bash
cargo run --locked --profile evaluation -p rhythm-map-eval \
  --example beat_this_trace -- \
  --suite evaluation/suites/artbeat-v1.json \
  --case artbeat-05-75-to-150 --seconds 35 \
  --audio-dir /data/artbeat-v1 --model-dir /data/beat-this-full-v1 \
  --output /data/private-traces/artbeat-05.trace.json

python evaluation/parity/compare_reference.py \
  --upstream /data/reference/beat_this \
  --model-pack models/beat-this-full-v1.json --model-dir /data/beat-this-full-v1 \
  --checkpoint /data/reference/final0.ckpt \
  --checkpoint-sha256 8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331 \
  --trace /data/private-traces/artbeat-05.trace.json \
  --source-audio /data/artbeat-v1/audio/05_ARTBeaT_TempoA_75to150.mp3 \
  --output /data/reports/artbeat-parity.json
```

Repeat the exporter with the second locked case and its calibration audio
directory. Pass repeated `--trace` and matching `--source-audio` arguments to
compare multiple cases; source identities are checked against the traces.
The duration limit is applied after resampling, not before it. Existing trace
and report files are never overwritten.

Only the aggregate comparison report is suitable for checking in after a
confidentiality review: it contains digests, shapes, versions and numerical
differences, not audio samples or local file paths.

## Lightweight tests

```bash
cargo test -p rhythm-map-eval --example beat_this_trace
python -m unittest discover -s evaluation/parity -v
```

These tests need no models; the Python tests need only NumPy. Full reference
execution is an explicit evaluation job, not part of ordinary model-free CI.

## Preprocessing revisions

`baseline-v1.json` records the original untrimmed/unflushed Rust resampler.
Keep it as historical evidence, not as an expected v2 output. The comparator
accepts the explicitly audited v1 and v2 observation contracts; v2 traces must
also identify `audio.rs` by SHA-256. Unknown contracts are rejected.

The corrected two-case run is `baseline-v2.json`; the 30-case paired musical
regression summary is `resampling-v2-calibration.json`. The latter deliberately
retains mixed improvements and regressions; see the
[measured decision](../baselines/beat-this-resampling-v2.md).

The v2 resampling invariants run without models or external music:

```bash
cargo test -p rhythm-map-beat-this audio::tests
```

These cover native-rate identity, invalid PCM, generated stereo WAV versus PCM
equivalence, short/partial/exact-chunk lengths, chunk-size invariance, and
8/16/44.1/48/96/192 kHz impulses including file edges. Full reference reruns
still use the unchanged numerical/event budgets above. See
[`../../docs/ALGORITHM.md`](../../docs/ALGORITHM.md) for the sample-origin and
tail-flushing contract.

## Regression origin/tail diagnosis

`regression-lock-v2.json` freezes four calibration cases and probe times selected
from the preceding paired regression report: ARTBeaT 13/15/18 and FSLD 110 BPM.
It references the unchanged base reference lock. It does not introduce an
accuracy candidate or authorize holdout access.

Export each locked case with `--include-legacy-pcm` added to the Rust example
command above. This flag adds the exact former `beat-this = 1.0.0` decoded PCM
to the **private** trace but still runs the current backend. It is confined to
an evaluation example/dev-dependency, not a product API or a decoder choice.
New traces record the full decoded lengths so cropped tracks are rejected by
the tail audit rather than mistaken for true file endings.

Run the comparator with all four traces and corresponding original audio
files, the same pinned checkpoint/model arguments, and:

```text
--phase-tail-lock evaluation/parity/regression-lock-v2.json
--output /data/reports/regression-v2-audit.json
```

The ordinary numerical budgets are unchanged. In addition, the verified
official frontend/checkpoint/minimal decoder runs a two-factor input experiment:

| Input | Origin | Source tail |
| --- | --- | --- |
| Current v2 | Corrected | Complete |
| Tail trimmed only | Corrected | Cut at the former last available sample |
| Origin restored only | Former pre-origin prefix prepended | Complete |
| Origin and tail restored | Former prefix prepended | Cut at former endpoint |
| Actual v1 control | Exact legacy decode export | Exact legacy decode export |

The locked origin difference is 63 output samples for these 44.1 kHz inputs.
The prefix comes from actual v1, preserving any pre-ringing, not invented
silence. Both factors together must reconstruct the exact legacy waveform
within `1e-5` absolute sample error and its neural logits within `2e-3`.
Failure of this reconstruction means the decomposition is incomplete; it must
not be attributed solely to origin/tail changes.

Input lengths and therefore frontend/end-padding context intentionally change;
this is **not** a pure fractional-delay experiment with identical model padding.
Event times have the known prepended duration subtracted for diagnostic
correspondence only, retaining any negative pre-origin detection. Logit deltas
are explicitly unshifted common-frame differences. No returned product
timestamp or score is adjusted. One-to-one event correspondence allows 40 ms
to distinguish peak changes from ordinary frame quantization; this is not a
music-accuracy threshold. No truth is read and no threshold is optimized.

The report records per-factor effects and fixed candidate probes. Near-probe
confidence is a diagnostic local maximum, not necessarily a selected beat.
Original-file parity, input reconstruction, musical accuracy, and selecting a
safe decoder remain separate questions.

The first four-case result is `regression-v2-audit.json`: 63/64 ordinary checks
pass, with one original-file event mismatch retained as `passed: false`.
All four legacy reconstructions are exact. Interpretation and the next
native-PCM control are in the
[origin/tail audit](../baselines/beat-this-phase-tail-audit-v2.md); the follow-up
official-model source-input probe is `source-threshold-probes-v2.json`.

## Native decoder/resampler isolation

`native_pcm_audit.py` follows only the frozen ARTBeaT 15 failed trace. It uses
`regression-lock-v2.json` and the prior source-probe trace digest, rejecting
different sources, cropped traces, stale Rust exports, changed reference
checkouts, and incorrect checkpoints before model execution. No music truth
is loaded. The suite must still pass the shared Rust calibration/license gate.

Build the model-free exporter with the existing evaluation profile:

```bash
cargo build --profile evaluation -p rhythm-map-eval --example beat_this_pcm
cargo test -p rhythm-map-eval --examples
python -m unittest discover -s evaluation/parity -v
```

Run using the same isolated reference environment as above. The paths below
refer to locally authorized assets, not downloads. Set `PYTHONDONTWRITEBYTECODE=1`
to keep the pinned upstream checkout clean. Use the executable under your
configured data-drive `CARGO_TARGET_DIR` (`.exe` on Windows):

```bash
python evaluation/parity/native_pcm_audit.py \
  --upstream /data/parity/beat_this \
  --checkpoint /data/parity/final0.complete.ckpt \
  --trace /data/parity/artbeat-15.regression-v2.trace.json \
  --suite evaluation/suites/artbeat-v1.json \
  --source-audio /data/artbeat/audio/15_ARTBeaT_TempoA_85to127.5.mp3 \
  --rust-exporter /data/build/evaluation/examples/beat_this_pcm \
  --private-dir /data/parity/native-pcm-new-run \
  --output /data/reports/native-pcm-new-run.json
```

The private directory must be new and outside the repository. Its native PCM
JSON files inherit the recording's rights and must not be committed. The
report contains only aggregate differences, fixed probes, event deltas, and
identity digests. Output files use exclusive creation.

The four paths independently combine Rust/official native decode+downmix with
Rust/soxr resampling. Native inputs are normalized to float32; the soxr side
receives the float64 promotion. A fifth path retains the original official
float64 input, checking whether that normalization itself changes results.
Every path uses the same official frontend/checkpoint/minimal postprocessor.
The current Rust input must also reconstruct both shipping decode and the
historical v2 trace bit-exactly before neural inference proceeds.

Exit status checks experiment controls, not agreement between different
filters: `controls_passed` and `source_event_parity_passed` are separate report
fields. An original-file event mismatch is intentionally visible even if the
diagnosis completed successfully. The unchanged 20 ms event parity budget and
40 ms diagnostic correspondence have different meanings, as above. CI runs
synthetic unit tests only, without private music, models, or Python inference
dependencies beyond NumPy.

The completed [`native-pcm-v2-audit.json`](native-pcm-v2-audit.json) retains
`source_event_parity_passed: false`. Its measured interpretation and next step
are in the [native PCM baseline](../baselines/beat-this-native-pcm-v2.md).

## One reference-bandwidth candidate

The standalone `resampler_probe` example generates physical test signals and
compares the shipping preprocessor with the frozen evaluation-only
`phase-exact-bh2-256-v1` implementation. Put its large generated tensor output
on a data drive; only aggregate characterization belongs in Git:

```bash
cargo run --profile evaluation -p rhythm-map-eval --example resampler_probe -- \
  --output /data/parity/resampler-generated.trace.json
python evaluation/parity/characterize_resampler.py \
  --trace /data/parity/resampler-generated.trace.json \
  --output /data/reports/resampler-generated.json
```

For an explicit candidate neural trace, add `--reference-resampler` to the
existing `beat_this_trace` command. This is confined to evaluation and cannot
be combined with `--include-legacy-pcm`. It uses actual Rust candidate PCM,
the same verified model pack and RTen execution, and the unchanged decoder.
Trace identity has a candidate-specific contract suffix plus the candidate
source hash. The reference comparator accepts exactly this named candidate,
rejects stale candidate code or a candidate claiming the shipping contract,
and retains all existing numerical and original-file event budgets. Do not
use the origin/tail ablation flag on candidate traces.

The candidate's mathematical construction, synthetic results, and promotion
limits are recorded in the
[reference resampler baseline](../baselines/beat-this-reference-resampler-v1.md).

## Bounded coefficients and full paired calibration

The same candidate now tiles the rational phase table into at most 8 MiB of
coefficients. This is a coefficient-allocation bound, not a process-memory
claim; decoded audio, output audio, and the neural model need separate memory.
Kernel generation and each sample's accumulation order are unchanged. Native
22,050 Hz remains bit-identical. The generated-signal comparison against the
retained, hash-locked pre-optimization trace is reproducible with:

```bash
python evaluation/parity/verify_bounded_resampler.py \
  --before /data/parity/resampler-candidate-final-v1.trace.json \
  --after /data/parity/resampler-bounded-v1.trace.json \
  --output /data/reports/resampler-bounded-new-run.json
```

The old 64-check model report is historical evidence and is never rewritten
to claim a new source hash. The generated bit-identity report links both
implementations explicitly. A full rerun additionally checks that the entire
PCM inputs of those four short music traces still have the identical float32
byte hashes; this is not a fresh official-checkpoint execution.

`resampler_calibration` is a locked experiment, not a user-selectable product
strategy. It accepts only the exact 15-case ARTBeaT and 15-case FSLD calibration
suites and the retained complete shipping reports pinned by
`resampling-v2-calibration.json`. Run the two suites sequentially. For example:

```bash
cargo run --locked --profile evaluation -p rhythm-map-eval \
  --example resampler_calibration -- \
  --suite evaluation/suites/artbeat-v1.json \
  --audio-dir /data/artbeat-v1/audio --model-dir /data/beat-this-full-v1 \
  --baseline /data/reports/artbeat-resampling-after-v2.json \
  --observation-cache /data/observation-cache-v1 \
  --output-dir /data/reports/artbeat-reference-resampler-new-run
```

Repeat with `fsld-tempo-v1.json`, its matching audio and frozen baseline, and a
separate new output directory. On a constrained CPU environment, set
`RTEN_NUM_THREADS=2` before running. All sources, truth, and model assets are
verified. The shipping path replays existing verified raw-observation cache
entries through the actual engine and must exactly reproduce every old score
and oracle. All 15 entries must be cache hits; a missing entry may be populated
by the ordinary runner, but that experiment then fails closed and must be
restarted. Candidate inference always runs fresh, never reads or writes a
shipping cache entry, and uses the same default decoder, PCM evidence
extraction, and estimator. Per-case progress is saved immediately.

The runner succeeds when measurement completes, **not** when accuracy passes.
Each case retains its original acceptance failures. No threshold is adjusted.
The final paired summary requires both complete reports and the four retained
pre-optimization parity traces:

```bash
python evaluation/parity/summarize_resampler_calibration.py \
  --artbeat /data/reports/artbeat-reference-resampler-new-run/report.json \
  --fsld /data/reports/fsld-reference-resampler-new-run/report.json \
  --parity-trace /data/parity/artbeat-13.reference-resampler-v1.trace.json \
  --parity-trace /data/parity/artbeat-15.reference-resampler-v1.trace.json \
  --parity-trace /data/parity/artbeat-18.reference-resampler-v1.trace.json \
  --parity-trace /data/parity/fsld-110.reference-resampler-v1.trace.json \
  --output /data/reports/reference-resampler-calibration-new-run.json
```

The summary retains case-level deltas, gained/lost passes, and overlapping tag
slices. FSLD has tempo-only truth: its placeholder beat/downbeat scores are not
averaged into an accuracy claim. Timings are sequential VDI wall-clock samples,
not stable benchmarks; cached baseline model times cannot measure a model-speed
change. Neither this experiment nor numerical parity authorizes product
promotion, holdout access, or release.

## Frozen single-event regression diagnosis

`resampler-regression-lock-v1.json` selects only the sole ARTBeaT beat-F1
regression from that completed calibration: `artbeat-14-240-to-96`. It locks
the calibration digest, source, model, complete PCM identities, and unchanged
decoder settings before inspecting the new logits. This is not another
resampler or decoder candidate.

Export the complete case twice with `beat_this_trace`, once with shipping
defaults and once with `--reference-resampler`, to new private data-drive
files. Then run:

```bash
python evaluation/parity/resampler_event_audit.py \
  --before /data/parity/artbeat-14.shipping-v2.trace.json \
  --after /data/parity/artbeat-14.reference-resampler-bounded-v1.trace.json \
  --baseline /data/reports/artbeat-resampling-after-v2.json \
  --output /data/reports/resampler-regression-event-new-run.json
```

The audit checks exact PCM hashes, unchanged source identities, full recording
length, exact old raw events/confidences, independent replay of strict-zero
radius-three peak picking, and port/adapter event agreement. It distinguishes
a threshold crossing from local-peak competition without sweeping either
setting. The diagnostic event-matching window remains 40.001 ms, separate
from numerical parity and the existing truth-scoring tolerance. Nearest truth
is explanatory annotation only; it does not choose a peak or change a score.
Reports retain local scalar probe summaries, not PCM or dense model tensors.

For the independent official check, use `compare_reference.py` above with the
candidate trace, matching original file, and pinned checkpoint. The completed
reports are `resampler-regression-event-v1.json` and
`resampler-regression-reference-v1.json`. The latter passes numerical parity
while the former preserves a real musical regression. Neither is a decoder
fix or permission to lower the threshold. See the
[measured interpretation](../baselines/beat-this-reference-resampler-v1.md#single-event-regression-diagnosis).

## Cache-only weak-candidate evidence

The scoped `rubato_cache_replay` evaluator now completes the subsequent
historical-cache gate: 25/25 exact replays with regenerated PCM evidence,
while retaining v1 provenance and 1/25 historical musical acceptance. It is
not a production cross-contract fallback or fresh v2 inference. See its
[lock, result, and invocation](../baselines/beat-this-rubato-cache-replay-v1.md).

Before attempting to transfer this audit to RUBATO, the separate
`rubato_pcm_equivalence` example compares full former/shipping decoded PCM
without inference or cache access. Its locked 25-case summary is 25/25
bit-identical. This is input evidence only, not permission to relabel old
caches or treat old scores as v2 results. See the
[result and replay gate](../baselines/beat-this-rubato-pcm-equivalence-v1.md).

`candidate_evidence` reuses the existing cache implementation and actual PCM
engine enrichment; it does not load a neural model or write a cache entry.
It rejects every suite except the exact locked 15-case ARTBeaT calibration
manifest, verifies model assets and decoded PCM identities, and requires exact
old raw events and selected scores. Missing caches fail without inference.
The separate frozen candidate-resampler probe is never pooled into shipping
cohort statistics. All export paths below must be new files.

```bash
cargo run --locked --profile evaluation -p rhythm-map-eval \
  --example candidate_evidence -- \
  --suite evaluation/suites/artbeat-v1.json \
  --audio-dir /data/artbeat-v1/audio --model-dir /data/beat-this-full-v1 \
  --observation-cache /data/observation-cache-v1 \
  --baseline /data/reports/artbeat-resampling-after-v2.json \
  --probe-trace /data/parity/artbeat-14.reference-resampler-bounded-v1.trace.json \
  --output /data/parity/candidate-evidence-new.private.json

python evaluation/parity/candidate_evidence_audit.py \
  --evidence /data/parity/candidate-evidence-new.private.json \
  --private-rows /data/parity/candidate-evidence-rows-new.private.json \
  --output /data/reports/candidate-evidence-separability-new.json
```

The Rust export contains dense observations and truth; the optional Python row
export is also private and rejects a repository destination. Keep both off the
system drive and outside Git. Only the aggregate report may be checked in
after review. Python verifies source and historical evidence identities and
rejects changed contracts, incomplete cohorts, tempo-only labels, or changed
truth tolerances. Its feature function receives observations, never truth.

The fixed confidence gate defines the primary cohort; positive-logit peaks
suppressed by wider local competition are reported separately. Covered-beat
duplicates, ambiguous windows, and candidates outside the annotated span are
excluded from AUC. Candidate support is deduplicated by missed truth event.
No threshold or weight is searched. Per-feature missingness, per-track slices,
and macro-track AUC prevent a pooled score from being mistaken for complete
context coverage or a generalization test. See the
[measured result](../baselines/beat-this-candidate-evidence-v1.md).

## Complete dense neural evidence

`dense_beat_evidence` performs fresh, full-recording inference for exactly the
frozen 15-case ARTBeaT or 25-case RUBATO calibration cohort. It does not read or
write production observation caches. Before retaining a case, it verifies the
complete decoded PCM and the pinned model pack, then compares default raw beat
and candidate timestamps, confidences, duration and source metadata exactly
against the earlier immutable evidence. A mismatch is retained for diagnosis
and stops the cohort; it is not relabeled as an equivalent result.

The exporter keeps both unmodified 50 Hz beat/downbeat logit heads, their common
time origin and frame count, source hashes and raw observations. It exports no
PCM or mel tensors. Each case is written immediately to a new private directory
outside every Git worktree. A complete summary appears only after the loop
finishes; its `complete` field is false when replay stopped on a mismatch.
Fatal errors can leave earlier case files without a summary. Neither state is
an acceptable complete-cohort audit input. Existing captures are never replaced.

```bash
RTEN_NUM_THREADS=2 cargo run --locked --profile evaluation -p rhythm-map-eval \
  --example dense_beat_evidence -- \
  --suite evaluation/suites/artbeat-v1.json \
  --evidence /data/parity/candidate-evidence-v1.private.json \
  --audio-dir /data/artbeat-v1 --model-dir /data/beat-this-full-v1 \
  --output-dir /data/parity/dense-artbeat-new

RTEN_NUM_THREADS=2 cargo run --locked --profile evaluation -p rhythm-map-eval \
  --example dense_beat_evidence -- \
  --suite evaluation/suites/rubato-calibration-v1.json \
  --evidence /data/parity/rubato-cache-replay-final-v1.private.json \
  --audio-dir /data/rubato-calibration-v1 --model-dir /data/beat-this-full-v1 \
  --output-dir /data/parity/dense-rubato-new

python evaluation/parity/dense_clock_evidence.py \
  --artbeat-evidence /data/parity/candidate-evidence-v1.private.json \
  --artbeat-captures /data/parity/dense-artbeat-new \
  --rubato-evidence /data/parity/rubato-cache-replay-final-v1.private.json \
  --rubato-captures /data/parity/dense-rubato-new \
  --output /data/reports/dense-clock-evidence-new.json
```

The Python audit checks case-file hashes, complete cohorts, source identity,
full frame coverage, unchanged observations, and independently reconstructs
default pulse events from the retained logits. It then compares annotated beat
positions with following half-beat controls, including misses without reliable
raw anchors. This comparison **uses truth to place ideal templates**: it is not
an automatic decoder, recovered-event count, beat F1 or training decision.
The control can itself be a meaningful musical subdivision. Only aggregate and
per-track summaries may enter Git; dense captures and event coordinates remain
private. See [interpretation and limits](../baselines/dense-clock-evidence-v1.md).

## Frozen full-frame clock experiment (evaluation only)

`dense_sequence` decodes complete captured heads into private inferred clock
positions. Only frame arrays and explicit availability enter the decoder;
truth, case identity and baseline observations are confined to the evaluator.
This candidate failed acceptance and is not a product strategy or default.
The [specification and result](../baselines/dense-sequence-v1.md) explain the
objective defect, control failures and metric denominators.

```bash
cargo build --locked --profile evaluation -p rhythm-map-eval --example dense_sequence
python evaluation/parity/dense_sequence_audit.py \
  --binary target/evaluation/examples/dense_sequence \
  --private-output /data/parity/dense-sequence-new.private \
  --artbeat-evidence /data/parity/candidate-evidence-v1.private.json \
  --artbeat-captures /data/parity/dense-artbeat-new \
  --rubato-evidence /data/parity/rubato-cache-replay-final-v1.private.json \
  --rubato-captures /data/parity/dense-rubato-new \
  --output /data/reports/dense-sequence-new.json
```

Use the actual binary location if `CARGO_TARGET_DIR` is set (`.exe` on Windows).
Both output destinations must be new; private predictions must be outside Git.
The script verifies the frozen evidence, complete capture/source identities,
truth hashes and unchanged default primary beat metrics before scoring. It runs
42 authored controls and all 40 calibration recordings without fresh inference
or holdout access. The committed `dense-sequence-v1.json` is the frozen failed
run, not a report to overwrite when exploring another objective.

## Complete-frame scoring correctness

`cargo run --locked -p rhythm-map-eval --example frame_likelihood` runs authored
mask checks only, without private inputs or model inference. It compares both
positive and negative evidence over identical frame domains. Its committed
`frame-likelihood-v1.json` records ideal meter/phase checks and explicit weak
event omission diagnostics. This is not an end-to-end decoder, musical accuracy
score or successor calibration result. See the
[contract, result and remaining limitation](../baselines/frame-likelihood-v1.md).

## Normalized missing-observation reference

`cargo run --locked -p rhythm-map-eval --example dropout_likelihood` evaluates
one normalized visible/background mixture against eight authored fixed-path
controls. Its `dropout-likelihood-v1.json` keeps intact-case successes, weak
event failures, full frame denominators and identical-input witnesses. It is
not a joint clock decoder or real-music accuracy result. The
[derivation and decision](../baselines/dropout-likelihood-v1.md) explain why
changing the missing rate cannot reverse negative observation evidence.

## Contextual phase likelihood

`cargo run --locked -p rhythm-map-eval --example phase_likelihood` emits the
authored `phase-likelihood-v1.json` scoring audit. It reuses the eight dropout
control inputs, adds strong/weak bar-density checks and contrasts independent
window maximization with a normalized shared-phase model. No captures or
audio are read. The [derivation and scope](../baselines/phase-likelihood-v1.md)
explain the positive weak-evidence result and why it is not a joint decoder or
real-music accuracy claim.

## Exact joint clock reference

`cargo run --locked --profile evaluation -p rhythm-map-eval --example joint_clock`
searches unknown beat locations, per-beat durations and bar lengths from authored
dense heads. It computes exact reference/evidence partitions and a separate MAP
trace; no truth paths enter decoding. `joint-clock-v1.json` freezes the authored
run, including failed weak-change, flat-middle, noise and edge behavior.
`python evaluation/parity/joint_clock_diagnosis.py` independently decomposes the
frozen MAP and a truth-assisted diagnostic competitor on identical edge coverage;
its output is `joint-clock-diagnosis-v1.json`. These are evaluation artifacts,
not new product modes. See the [graph, controls and decision](../baselines/joint-clock-v1.md).

## Time-exposure duration intervention

`cargo run --locked --profile evaluation -p rhythm-map-eval --example time_clock`
repeats those same 15 controls with a time-consistent duration prior. It retains
the original joint graph and emissions; `time-clock-v1.json` records the complete
search results and all source identities. `python evaluation/parity/time_clock_diagnosis.py`
independently reconstructs the prior, current MAP and the original diagnostic
competitors; its result is `time-clock-diagnosis-v1.json`. A flat-middle false
slowdown is removed, but weak doubling, edges and noise still fail the authored
gate. The [derivation and full comparison](../baselines/time-clock-v1.md) explain
why this prerequisite does not justify promotion, training or a cohort replay.
