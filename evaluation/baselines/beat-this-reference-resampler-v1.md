# Reference-bandwidth resampler candidate — 2026-09-03

This is one frozen evaluation-only candidate, not a second product strategy.
The shipping adapter and its v2 observation contract are unchanged.

## Generated-signal characterization

[`resampler-characterization-v1.json`](../parity/resampler-characterization-v1.json)
compares 99 generated cases at 8/16/22.05/44.1/48/96/192 kHz against the
installed reference soxr HQ (`0.1.3-14-ga66f3ee`, Python wrapper 1.0.0).
Cases include start/center/fractional/tail impulses, DC, a step, a linear
sweep, and sinusoidal probes around the passband/transition/stopband. There
are no music, model, or beat labels in this experiment.

The existing 256-input-sample filter's transition width grows relative to the
output Nyquist frequency when downsampling more heavily. Its center-impulse
-3 dB point moves from about 0.9368 at 44.1 kHz to 0.8875 at 192 kHz,
whereas reference HQ stays near 0.9504--0.9507. Integer delay trimming also
leaves a measured passband phase delay from about -0.300 to +0.399 output
samples on the probed non-native rates. An impulse peak rounded to the right
sample does not detect this fractional phase difference.

The candidate `phase-exact-bh2-256-v1` uses standard windowed-sinc interpolation:

- Compute output position `n * input_rate / 22050` with an integer rational
  clock, without a post-hoc timestamp offset or cumulative floating-point clock.
- Scale the squared Blackman-Harris window radius to
  `ceil(128 / min(1, 22050 / input_rate))` input samples.
- Use normalized cutoff 0.95685, approximating the reference's measured -6 dB
  point on generated impulses. This value and the 256-sample lower-rate span
  were fixed **before** any candidate music inference, not fitted to beat labels.
- Normalize each polyphase kernel for DC gain, zero-extend outside the input,
  and return the exact integer-rounded number of output samples. Native
  22,050 Hz samples are returned bit-identically.

This is an approximation, not an implementation of libsoxr. Its filter shape
is not identical to reference HQ, and it does not claim reference stopband
precision. It copies no libsoxr code/tables and adds no library, native runtime,
or Python dependency to the product. The reference's documented quality and
phase semantics are available in the
[upstream header](https://github.com/chirlu/soxr/blob/0.1.3/src/soxr.h) and
[quality constructor](https://github.com/chirlu/soxr/blob/0.1.3/src/soxr.c).

All 99 output lengths agree. Relative to the reference, full-waveform RMSE
improves on 86 cases, is unchanged on 13, and regresses on none (comparison
epsilon `1e-12`). Center-impulse phase delay falls below `1e-8` output samples
on every tested rate; the native-rate signal is unchanged. These are signal
compatibility results, not evidence of musical accuracy.

## Frozen four-case neural parity

[`reference-resampler-v1-audit.json`](../parity/reference-resampler-v1-audit.json)
passes all **64/64** unchanged checks on ARTBeaT 13/15/18 and FSLD 110 BPM,
including official-checkpoint/ONNX conversion, ONNX Runtime/RTen execution,
mel frontend, decoder, and original-file event comparisons. The previous
shipping v2 audit's 63/64 report is retained unmodified as historical evidence.

| Case | Shipping v2 beat count | Candidate beat count | Official source beat count |
| --- | ---: | ---: | ---: |
| ARTBeaT 13 | 30 | 30 | 30 |
| ARTBeaT 15 | 37 | 36 | 36 |
| ARTBeaT 18 | 28 | 28 | 28 |
| FSLD 110 BPM | 18 | 18 | 18 |

Original-file beat/downbeat timestamp differences are all below one microsecond
on these runs, not just within the allowed one-frame tolerance. ARTBeaT 15's
first-two-second waveform RMSE falls from about `3.56e-4` to `6.33e-6`, with
equal lengths and zero detected lag. Native PCM, mel tensors, and model logits
remain private; checked-in reports contain aggregate differences and identities.

This restores reference agreement on the isolated failure. It does not remove
the model's known leading/interior misses, FSLD 110 BPM tempo error, or
half/double-time ambiguity. This initial four-case step did not include the
full musical calibration reported below. No accuracy score or numerical
tolerance was changed to pass.

Local Windows engineering checks pass: 184 workspace/all-target Rust tests,
26 Python tests, formatting, all-feature Clippy, doc tests, five generated core
regression cases, and the release-profile WASM build. These results are not a
remote CI/macOS run. No commit, push, package, or release was made in this step.

## Scope before promotion (initial four-case experiment)

The implementation exists only under evaluation examples. Candidate traces
use an explicit observation-contract suffix and source digest; they cannot
claim the shipping contract or enter the default cache. The unchanged mel,
logit, and event budgets still apply to candidate model-parity runs.

The initial prototype supported 8..192 kHz, but only the seven listed rates had been
characterized. Its full rational phase table can be large for unusual coprime
rates (up to roughly 400 MB of coefficients near the upper bound); a bounded
memory implementation and representative latency measurements are required
before any product adoption. Single generated-case timings include setup and
are not stable performance benchmarks. Full paired musical calibration,
embedding/platform validation, and an observation-cache identity change would
also be required for promotion. The holdout remains sealed; no release is
authorized by this experiment. The bounded implementation and broader
calibration below are a subsequent step; the original reports above remain
unchanged.

## Bounded implementation

The candidate now computes rational-phase coefficients in tiles of at most
8 MiB, instead of retaining a potentially roughly 400 MB full phase table at
unusual coprime sample rates. Each phase is generated once, and the per-output
dot-product order is unchanged. This bound covers coefficient storage only,
not native/output PCM, feature tensors, or the neural model.

[`resampler-bounded-v1.json`](../parity/resampler-bounded-v1.json) links the
frozen pre-optimization trace to the new source digest. All **99/99** generated
cases have bit-identical input PCM, shipping PCM, and candidate PCM (including
signed-zero bits). Unit tests also compare one-kernel and seven-kernel tiles
with the default budget at ordinary and coprime rates. No cutoff, support
length, clock, accumulation order, decoder, or estimator was retuned.

The old neural-parity report is not relabeled as a new-source run. The full
paired experiment additionally compares each of the four old, complete music
trace PCM hashes against the new candidate input hash. This explicit evidence
bridge preserves the old 64/64 result without claiming a fresh official-model
execution for the memory change.

## Complete 30-case paired calibration

[`reference-resampler-calibration-v1.json`](../parity/reference-resampler-calibration-v1.json)
contains both complete, hash-locked 15-case suites, per-case scores, overlapping
tag slices, timings, and four successful historical PCM links. All 30 shipping
cache replays exactly reproduce the frozen v2 scores and oracles; all 30
candidate inferences run fresh through the same model, decoder, PCM evidence
extraction, and default estimator. Candidate observations never enter the
shipping cache. Truth, thresholds, and production source files are unchanged.

| Metric | Shipping v2 | Candidate |
| --- | ---: | ---: |
| ARTBeaT mean beat F1 | 0.807961 | 0.808710 |
| ARTBeaT mean median tempo error | 19.050469% | 18.879217% |
| ARTBeaT mean P95 tempo error | 58.964548% | 55.653544% |
| ARTBeaT cases passing all existing gates | 1/15 | 1/15 |
| FSLD mean median tempo error | 22.348798% | 22.348798% |
| FSLD mean P95 tempo error | 50.187819% | 50.223669% |
| FSLD cases passing all existing gates | 7/15 | 7/15 |

No passing case is lost, but no new case passes either. FSLD provides only
tempo truth, so no beat/downbeat accuracy is claimed for it. ARTBeaT has no
downbeat labels. Change-point metrics remain unchanged.

The small ARTBeaT mean improvement does not hide the negative cases:

- `112.5-to-75`: beat F1 improves 0.869565 → 0.888889, and tempo P95 error
  falls 90.476156% → 53.846159%.
- `85-to-127.5`: beat F1 improves 0.623377 → 0.631579, tempo median error
  falls 32.754789% → 29.695860%, and P95 falls 47.058964% → 34.023898%.
- `240-to-96`: beat F1 regresses 0.810811 → 0.794521 and tempo median error
  rises 1.424154% → 1.914295%. Raw/selected beat counts fall 33 → 32 and
  correctly matched beats fall 30 → 29. The observation output has changed,
  but exact event correspondence and its logit/peak cause still need checking;
  no new repair rule is justified.
- `90-to-80`: beat F1 is unchanged, but matched-beat median timing error
  rises 17.261886 → 18.823886 ms. P95 timing error is unchanged.
- FSLD `100 BPM`: tempo P95 error rises 1.826615% → 2.364356%; the case
  still passes. All other FSLD tempo scores are unchanged within the report's
  `1e-9` display epsilon. The roughly 109.79% P95 error at `110 BPM` persists.

Timings used serial tracks, the optimized evaluation profile, and two RTen
threads on the local Windows VDI. They are observations, not stable latency
benchmarks. Mean resampling time rises 50.57 → 260.41 ms on ARTBeaT and
46.77 → 223.91 ms on FSLD: roughly five times the preprocessing work, but only
about 0.19 s extra per track across the 30 cases. Candidate model+analysis time
totals 24.84 minutes (51.41 s/ARTBeaT track and 47.94 s/FSLD track on average);
the longest, 50-second FSLD recording takes 214.06 s. The baseline model was
cached, so these measurements cannot establish a model-speed change.

**Decision: retain the candidate for evaluation, do not promote it.** Bounded
memory and closer reference agreement are useful, but the average accuracy
gain is small, one ARTBeaT beat case regresses, and FSLD gains nothing. The
shipping v2 preprocessing and cache contract remain the only default. Do not
introduce per-song resampler choices, lower a decoder threshold, retune the
filter against labels, or spend holdout on this unresolved tradeoff. Next,
inspect the newly lost `240-to-96` event against the fixed model's logits and
the old observation sequence to separate threshold crossing from local-peak
competition before considering any general correction.

Local validation passes: 194 workspace/all-target Rust tests, 14 additional
model-pack tests without default features, doc-test invocation, 35 Python
tests, formatting, all-feature Clippy, five generated core cases, and the
release-profile WASM build. This is local Windows validation, not remote CI
or macOS validation. No holdout inference, training, commit, push, package,
or release was performed.

## Single-event regression diagnosis

The subsequent frozen audit selects only ARTBeaT `240-to-96`, the sole beat-F1
regression above. Both complete 359,691-sample PCM inputs exactly reproduce the
paired calibration's byte hashes. The fresh shipping trace reproduces every
old raw timestamp, beat confidence, and downbeat confidence exactly. Independent
replay of the unchanged peak picker and agreement with the port/adapter also
pass. See [`resampler-regression-event-v1.json`](../parity/resampler-regression-event-v1.json).

There is exactly one removed raw beat at **1.50 s** (frame 75), no added beat,
and no timestamp change among the other 32 beats. All 14 downbeat timestamps
are also unchanged. The removed event is 2.865 ms from annotated truth at
1.497135 s, within the unchanged 70 ms scoring tolerance.

| Fixed frame-75 diagnostic | Shipping v2 | Candidate |
| --- | ---: | ---: |
| Beat logit | 0.195108 | -0.053524 |
| Sigmoid score | 0.548623 | 0.486622 |
| Radius-three local maximum | yes | yes |
| Margin above strongest neighboring logit | 1.588530 | 1.502342 |
| Above the strict zero-logit gate | yes | no |

This is a **threshold crossing at the same real peak**, not displacement by
a neighboring peak, an accumulated time offset, or a dropped audio tail.
Sigmoid values here are raw model scores, not calibrated probabilities of
musical correctness. The candidate's real 1.50 s peak remains present in the
backend-neutral `beat_candidates` evidence at score 0.486622; its neighboring
accepted beats are about 1.26 and 1.76 s. It is absent from the selected beat
sequence, not absent from all model evidence.

The independent pinned checkpoint/ONNX/RTen audit on this candidate passes
**16/16** unchanged checks, including the original encoded file path:
[`resampler-regression-reference-v1.json`](../parity/resampler-regression-reference-v1.json).
Official source processing likewise emits 32 beats and 14 downbeats, matching
the candidate within 0.46 microseconds. Same-mel RTen/ONNX Runtime beat-logit
differences are at most `3.44e-5`, and same-PCM official/RTen differences are at
most `2.44e-5`, far below the candidate peak's 0.0535 distance below the gate.
The new miss is therefore not explained by a runtime/conversion mismatch:
closer reference preprocessing reproduces a missed beat in the reference
pipeline itself. This rules out that explanation on this case, not every
possible model or preprocessing failure elsewhere.

**Decision remains unchanged:** no default promotion, threshold reduction,
per-song mode, or dual-preprocessor inference. The existing evaluation-only
midpoint decoder intentionally requires repeated support and has a unit test
rejecting isolated midpoint recovery. Merely weakening that guard would not
establish that this peak can be distinguished from a genuine subdivision.
The next evidence check should reuse existing real peak/PCM observations to
measure whether missed beats can be separated from subdivision false positives
across calibration slices before implementing another recovery rule. Do not
train a model, tune this filter against labels, or open holdout to do it.

Follow-up local Windows validation passes: 194 workspace/all-target Rust tests,
14 no-default-feature model-pack tests, 40 Python tests, doc-test invocation,
formatting, Clippy, five generated core cases, and the WASM release-profile
build. This diagnostic adds no production code or public strategy. The
previous 30-case scores are retained, not claimed as a new full-suite rerun;
only the locked case received fresh model diagnostics. No holdout inference,
training, commit, push, package, or release occurred.
