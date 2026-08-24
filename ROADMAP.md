# Roadmap

## Phase 1: training-free timing engine

- Beat This adapter and parity fixtures.
- Robust BPM curve and metrical alternatives.
- Constant, ramp, jump, and discontinuity segmentation.
- CLI, C ABI, and observation-driven WASM API.
- Synthetic constant/step/ramp tests and a licensed evaluation manifest.
- Verified Beat This model-pack manifest and paired oracle/end-to-end evaluation.
- Raw beat/confidence diagnostics and deterministic PCM activity observations.
- Evidence-based half-time selection, silence-beat rejection, recovery of short
  model-smeared tempo jumps, and guarded duplicate/missed-event grid repair.
- Independent downbeat/bar-phase evaluation and evidence-based half-bar
  candidate selection with recovered-grid boundary realignment.
- Content-addressed external-audio resolution, strict external truth
  validation, and a private calibration workflow that never stores audio paths
  or bytes in reports.
- Immutable public-dataset acquisition and an ARTBeaT oracle suite covering 15
  CC BY 4.0 tempo-step, ramp, rubato, and polyrhythm exercises.
- Remote ZIP/ZIP64 member acquisition and an independent 15-case FSLD
  tempo-only calibration slice spanning 41--200 BPM, drumless, sparse,
  no-kick, breakbeat, and half/double-time ambiguity without inventing beat
  phase labels.
- Enforced regression/calibration/holdout suite roles, with truth-assisted
  decoder commands restricted to calibration and a one-candidate-versus-baseline,
  per-slice holdout gate.
- Edge-preserving metrical outlier repair that keeps sustained half/double-time
  tempo changes instead of folding an entire recording into one BPM band.
- An evaluation-only sequence/phase estimator candidate that rejects bar-inconsistent
  whole-track half-time folds, removes evidence-supported one-sided edge
  midpoint extras, and repairs fixed-frame paired quantization jitter without
  modifying exact timestamp observations.
- An evaluation-only edge-connected Viterbi decoder over Beat This logits that preserves
  upstream events, recovers only long repeated model-peak sequences, and never
  emits a path-grid timestamp without a local model maximum.
- A precommitted nine-case, timestamped ARTBeaT holdout with reproducible SVG
  truth import, plus an optimized non-LTO evaluation profile for routine model
  experiments.

## Phase 2: product surfaces

- Signed model-pack authenticity layered over the existing provenance and
  SHA-256 integrity manifest.
- Async analysis with progress and cancellation across native API, C ABI, and
  GUI surfaces.
- A measured fast/accurate model-pack policy; do not make the full model the
  default merely because it is more accurate in isolation.
- Separate short model smoke tests from scheduled/release full-suite baselines,
  with per-case progress visible to developers.
- End-to-end browser inference with a measured WASM backend.
- Native GUI for waveform, beat grid, confidence, and editable tempo segments.
- Export adapters for common rhythm-game and DAW tempo-map formats.

## Phase 3: evidence-driven model work

- Add a corpus-disjoint timestamped source for drumless, rubato, compound-meter,
  and percussive slices; the current public holdout is case-disjoint ARTBeaT.
- Evaluate game music, rubato, extreme tempo, compound meter, and drumless audio.
- Add a learned boundary/confidence head only if deterministic analysis is the
  measured bottleneck.
- Consider a custom multitask model only if the Beat This observation backend is
  the measured bottleneck.

### Bottleneck attribution protocol

Run the same evaluation cases through two paths:

1. **oracle observations** feed annotated beat/downbeat timestamps directly to
   `TempoMapEstimator` and isolate tempo-curve and segmentation quality;
2. **end-to-end observations** run audio through Beat This! before the same
   estimator and measure the complete product path.

Compare beat and downbeat F1 and timing error, tempo-curve median/P95 error,
change-point precision/recall, and section-boundary error by capability slice.
Keep at least game music, rubato, extreme tempo, compound meter, drumless audio,
stops, and half/double-time ambiguity as separate slices rather than hiding
them in one aggregate score.

Use the difference between the two paths to decide where model work belongs:

- If oracle observations pass but end-to-end observations fail, treat the broad
  observation path as the primary bottleneck. First separate missed/extra model
  events from deterministic robustness to noisy observations, especially
  half/double-time ambiguity; consider a custom multitask model only after
  calibration, decoding, metrical normalization, and backend alternatives are
  exhausted.
- If both paths fail on the same tempo or boundary cases, treat the deterministic
  estimator as the primary bottleneck. Improve robust statistics, segmentation,
  or add a learned boundary/confidence head without replacing a beat tracker that
  is already adequate.
- If both paths pass, keep the training-free design and spend complexity on
  product surfaces, export fidelity, performance, and confidence calibration.

Do not train on the hidden holdout or relax suite thresholds to accommodate a
candidate. Require a material improvement across the failing slices without a
regression in constant-tempo, platform, memory, and latency gates before adding
a learned component to the default distribution.

### Measured bottlenecks

- 2026-08-21: the first ARTBeaT oracle run failed 13 of 15 cases even with exact
  upstream beat timestamps. Global preferred-band folding erased sustained
  octave-related changes and ordinary smoothing blurred jump edges, proving the
  deterministic estimator was the bottleneck for those slices. Replacing that
  policy with bilateral metrical-outlier evidence and edge-preserving smoothing
  made all 15 oracle cases pass without changing suite thresholds; the worst
  tempo P95 error fell to 5.97 percent.
- 2026-08-21: the paired Beat This full-model run passed 1 of 15 ARTBeaT cases,
  with mean beat F1 0.8052, while all paired oracle paths passed. The model often
  remains at a sustained half-time level or omits beats throughout ramps and
  rubato, so the measured bottleneck is now the observation path. Prefer
  comparing alternate model decoding/backends over inventing timestamps in
  deterministic post-processing without acoustic evidence.
- 2026-08-21: a single-inference sweep compared nine Beat This peak decoders.
  The best fixed threshold/window candidate raised mean beat F1 only from
  0.8052 to 0.8179, while lower thresholds increasingly traded precision for
  extra events. A per-case policy oracle reached only 0.8371, and a narrower
  local-maximum window regressed. Keep the upstream decoder as the default;
  next evaluate sequence-aware tempo/phase decoding and an alternate
  observation backend. Require either path to beat this measured ceiling on
  held-out slices before integrating it into the product.
- 2026-08-21: missed-beat logit attribution found that only 42 of 128 ARTBeaT
  misses had an upstream-radius local peak above logit -3; 58 had weaker peaks,
  18 only appeared with a narrower radius, and 10 had no local peak. A
  conservative supported-midpoint decoder raised mean beat F1 from 0.8052 to
  0.8235 and materially improved three cases, but regressed one metrical case.
  Keep it experimental until separate calibration and holdout slices confirm a
  net improvement. The very weak step/ramp failures remain evidence for
  comparing an alternate observation backend rather than expanding grid-based
  timestamp invention.
- 2026-08-22: the independent FSLD tempo-only slice passed 6 of 15 end-to-end
  cases. Nine clips had median BPM error below 5 percent, but three still crossed
  metrical levels locally; the 41, 60, and 128 BPM clips favored roughly double
  time, while the 130 and 200 BPM clips favored roughly half time. Because FSLD
  has no timestamped beat phase, this is evidence of a product-level metrical
  selection failure but cannot attribute the failure to Beat This versus the
  deterministic estimator. Keep its report `end_to_end_only`, use it for
  calibration, and require a separately held timestamped corpus before changing
  the default policy.
- 2026-08-22: calibration and holdout roles became executable manifest
  contracts. The original 15-case ARTBeaT slice is permanently calibration
  evidence; decoder sweeps and
  recoverability diagnostics accept only calibration suites, while `decoder-eval` runs
  one registered policy and reports overall, per-case, and capability-tag
  metrics. The remaining work is corpus population, not another decoder sweep:
  obtain multiple independent legally held sources for each required slice,
  select one policy on calibration, then open the untouched holdout once.
- 2026-08-23: an evaluation-only `metrical-consistency-v1` estimator repaired bounded
  half/double-time runs of up to three intervals without changing any ARTBeaT
  beat or tempo metric. It lowered FSLD 150 BPM P95 error from 49.57 to about
  2.61 percent and raised FSLD from 6 to 7 passes. The registered supported-
  midpoint decoder also reached 7 passes but retained its known ARTBeaT
  regression, and combining the policies added no further pass. Keep both
  candidates out of the shipping default; broader unconditional half/double
  folding remains unsafe.
- 2026-08-23: `sequence-phase-v1` resolved two previously one-sided or
  whole-track failures without a preferred-BPM-band rule. It raised FSLD from 7
  to 9 of 15 passes: the 60 BPM edge-extra case moved from 96.13/101.43 percent
  median/P95 error to 0.00/0.99, and the 200 BPM false half-time fold moved from
  50.00/50.00 to approximately 0.00/2.29. All 15 timestamped ARTBeaT cases had
  zero change in analyzed event count, end-to-end beat F1 and tempo errors, and
  oracle tempo errors. Keep it evaluation-only until an untouched timestamped
  holdout validates merging it into the product algorithm.
- The remaining whole-track ambiguity with equally plausible metrical levels
  and edge spans where the model emits no candidate peak cannot be solved by
  safe core smoothing. Next evaluate an explicit DP/DBN-style path decoder over
  beat/downbeat logits, preserving ambiguity and avoiding timestamp invention,
  against tempo-change, rubato, and compound-meter slices. Compare an alternate
  backend if the decoder cannot recover evidence that Beat This never emitted.
- 2026-08-23: an edge-connected Viterbi decoder recovered long repeated weak
  Beat This peaks without changing any of the 15 timestamped ARTBeaT cases. In
  combination with `sequence-phase-v1`, it raised FSLD from 9 to 10 of 15 by
  reducing the 110 BPM clip's P95 tempo error from 33.48 to about 2.92 percent.
  A minimum six-candidate sequence plus local support rejected the regressive
  four-point 128 BPM edge run. The 130 BPM clip remained unchanged because the
  model did not emit a sufficiently supported edge sequence. The subsequent
  disjoint holdout rejected promotion, so keep the decoder evaluation-only and compare
  an alternate observation backend for the missing-evidence cases instead of
  weakening the no-invention rule.
- 2026-08-23: on the development VDI, optimized inference completed the 15-case
  ARTBeaT decoder matrix in under one minute, while an incremental release-LTO
  relink took about three minutes and an unoptimized first-case inference took
  more than one minute. The observed performance issue is build/profile
  configuration, not Viterbi decoding. `cargo xtask` now uses an optimized
  non-LTO `evaluation` profile while distribution release builds retain
  Thin-LTO.
- 2026-08-24: the registered edge-connected decoder failed its precommitted
  nine-case ARTBeaT holdout. Mean beat F1 fell from 0.67915 to 0.67771, one
  syncopated case regressed, and only four cases met the locked 0.80 gate. The
  11/8 half-time-risk case improved only from 0.76744 to 0.76836. This rejects
  promotion and confirms that the remaining failures are whole-track pulse or
  phase ambiguity, not merely weak edge peaks. Do not tune another decoder on
  the opened holdout; compare an alternate observation backend and expose
  competing phase hypotheses without inventing timestamps.
- 2026-08-24: product policy was narrowed to one zero-tuning shipping estimator
  across Rust, CLI, C ABI, and WASM. Named decoder and estimator candidates are
  evaluation-only experiments. Do not introduce an internal strategy selector
  merely because candidates trade wins and regressions: first prove an
  irreducible music-class split, a truth-free runtime classifier for that split,
  and a no-regression gain on a precommitted untouched holdout.
- 2026-08-24: do not integrate KRAISLER while its repository metadata combines
  a CC BY 4.0 label with an additional non-commercial restriction. It is not a
  runtime dependency and remains blocked even as an evaluation-only corpus
  until the rights holder provides a clear, commercially usable grant.
- 2026-08-24: added backend-neutral uncommitted beat candidates and calibration-
  only pulse/phase coverage. Beat This retains one real frame per radius-one
  local-maximum plateau without a confidence floor. A fixed truth-free
  hypothesis generator compares the selected sequence, both alternating
  half-time phases, and real-midpoint augmentation; no candidate changes the
  shipping result. Use candidate recall and best-top-K F1 to decide whether the
  next bottleneck is missing backend evidence, hypothesis construction, or
  truth-free ranking.
- 2026-08-24: the 15-case ARTBeaT calibration measured 0.9820 mean candidate
  recall, 0.8052 selected-sequence beat F1, and a 0.8370 best-top-K ceiling.
  The initial confidence/continuity ranker achieved only 0.5224 top-1 F1 and
  selected the truth-best hypothesis in 0/15 cases because deleting alternating
  events artificially improved regularity. Treat this as a rejected scoring
  baseline. The next candidate must explicitly account for retained and
  discarded backend evidence, remain truth-free at ranking time, and stay
  evaluation-only until a fresh precommitted holdout exists.
- 2026-08-24: miss-only attribution found real candidate peaks near 118 of 128
  truth beats omitted by the selected sequence (0.9219 micro recall), but the
  median of per-case median miss confidence was only 0.0323. The retention-aware
  ranker safely restored mean top-1 F1 to the selected baseline of 0.8052 and
  ranked the truth-best member first in 9/15 cases, yet chose `selected` in all
  15. Do not promote it as a selector. Next build one evaluation-only local
  sequence decoder that accumulates coherent midpoint evidence across runs and
  penalizes normal/double-time state transitions; do not lower a global
  threshold or expose a product Strategy list.
- 2026-08-24: a truth-free two-state midpoint gap path was rejected on ARTBeaT.
  It activated 11/15 cases, improved six, regressed five, and lowered mean
  top-1 beat F1 from 0.8052 to 0.7699. All activations formed one nearly
  whole-track run: stable subdivision peaks are phase-indistinguishable from
  missing double-time beats. Do not retain this as another policy or tune its
  thresholds on the opened calibration set. The next discriminator must add
  independent beat-level evidence; otherwise preserve explicit metrical
  ambiguity instead of selecting an alternative sequence.
- 2026-08-24: added a backend-neutral, deterministic spectral-flux onset
  envelope and measured it at real midpoint candidates. Cases where midpoint
  augmentation improved by more than 0.02 F1 averaged 0.55 onset strength,
  versus 0.45 for non-improving cases, but the distributions overlap: the
  regressive 85-to-127.5 case scored about 0.69 while useful ramp cases scored
  about 0.41 and 0.28. Keep onset strength as auditable observation metadata,
  not a pulse selector or another strategy. A safe selector still needs meter-
  or structure-level evidence and a fresh precommitted holdout.
