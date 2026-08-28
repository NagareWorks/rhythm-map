# Roadmap

## Product horizon: audio metadata engine

Timing analysis is the first capability pack, not the permanent limit of the
product. Rhythm Map will grow into an offline, embeddable metadata engine whose
features are selected at build/package time while each included feature remains
one-call and zero-tuning in normal use. The durable composition and uncertainty
rules are recorded in [`docs/METADATA-PACKS.md`](docs/METADATA-PACKS.md).

Planned capability order after the timing foundation is shippable:

1. section-aware style metadata, reusing shared audio decoding and interval
   references but owning its taxonomy, model, and evaluation;
2. MIDI-assisted audio alignment and key-sound cut suggestions for rhythm-game
   authoring, with MIDI treated as optional evidence;
3. AI-origin suspiciousness as an explicitly probabilistic research pack only
   after unseen-generator and production-chain evaluation exists.

These are not reasons to merge every analysis into `rhythm-map-core` or one
mandatory model. A future pack gets a crate only when it has meaningful behavior
and artifacts to own, and every product surface includes only the packs selected
for that distribution.

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
- A pinned CC BY 4.0 Vienna 4x22 annotation source and Rust match importer for
  corpus-disjoint expressive-piano beat/downbeat and beat-local tempo truth.
- A locked 25-track RUBATO v0.3 real-performance calibration suite spanning 12
  works and multiple instruments, with ZIP-range acquisition plus official
  beat, measure/downbeat, structure, and beat-local tempo truth import.

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

## Phase 3: broaden metadata capabilities

- Define the aggregate metadata document after a second pack has real output;
  preserve the existing timing schema as an independently versioned payload.
- Implement and evaluate a section/style pack with a versioned taxonomy and
  out-of-domain uncertainty.
- Implement MIDI/audio alignment and key-sound cut candidates before attempting
  general source separation.
- Research AI-origin signals separately; require calibrated suspiciousness,
  generator-disjoint evaluation, codec/mastering robustness, and prominent
  non-verdict semantics before any product distribution.

## Phase 4: evidence-driven timing model work

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
- 2026-08-24: split the existing spectral flux into low, mid, and high
  contributions without another FFT or model pass. Useful midpoint
  augmentations averaged 0.35 high-frequency contribution versus 0.21 for
  non-improving cases, but the regressive 60-to-80 case also scored 0.33.
  Frequency balance is useful metadata but not a selector.
- 2026-08-24: whole-track downbeat periodicity was rejected because useful and
  regressive cases shared the same 4-to-8 and 8-to-8 event-density shifts. A
  stricter local rule activated 10/15 ARTBeaT cases, improved five, regressed
  five, and moved activated-case mean beat F1 only from 0.7809 to 0.7910. The
  post-hoc combination with mean onset strength above 0.6 selected four gains
  and no regressions, but that threshold was observed on opened calibration
  data and must not be promoted. Remove both bar algorithms rather than retain
  another strategy. Further selection work requires a genuinely untouched
  timestamped corpus or independent meter evidence from another backend.
- 2026-08-24: a pure-Rust BeatNet technical spike passed RTen ONNX execution
  and reproduced the published 272-feature madmom-compatible frontend. On the
  15-case ARTBeaT calibration suite its raw local maxima covered 459/460 truth
  beats (99.78% micro recall), versus 450/460 for Beat This, so the alternative
  backend supplies some genuinely missing evidence. A single evidence-snapped
  variable-tempo Viterbi path reached mean beat F1 0.8080 versus Beat This
  0.8052, but improved seven cases and regressed seven. Do not open Vienna or
  add a runtime backend selector: first improve the one BeatNet sequence
  decoder on calibration, review the published model's training-corpus
  provenance, and preselect one complete candidate for the untouched holdout.
- 2026-08-24: reproduced BeatNet's published offline joint tempo-meter boundary
  with 2/3/4-beat bar states. A hard DBN observation model reached 0.8071 mean
  beat F1 and over-constrained meter changes, piano rubato, and decelerating
  ramps; a softened variant reached 0.8085. Reject both rather than retain
  another selectable strategy.
- 2026-08-24: replaced the virtual-grid-only BeatNet selector with one guarded
  candidate graph. It scores only real pulse maxima, carries interval and bar
  phase, treats the old grid as a soft prior, allows penalized half/double-time
  transitions, rejects paths that materially destroy interval continuity, and
  restores a track-edge event only from a real grid-supported maximum. On the
  same 15-case ARTBeaT calibration, mean beat F1 rose from 0.8080 to 0.8536,
  precision from 0.8418 to 0.8513, recall from 0.8264 to 0.8894, and complete
  passes from 1 to 3. Four cases improved, ten were unchanged, and piano
  rubato regressed by about 0.04; 90-to-120 and 240-to-96 reached beat F1 1.0,
  while the two tempo ramps rose to about 0.91 and 0.82. Keep this as the one
  experimental BeatNet decoder and do not open Vienna yet. The remaining
  piano error is a genuine model-supported half/double-time ambiguity: the
  wrong dense path wins the internal evidence score by a wide margin, so a BPM
  band or another calibration-derived threshold would hide rather than solve
  it. Next obtain independent meter/accent evidence or retain the ambiguity in
  the product result before selecting the one holdout candidate.
- 2026-08-25: tested two truth-free attempts to force a unique piano-rubato
  selection and rejected both. A PCM-only adaptive half-time path reached beat
  F1 0.45 versus the selected 0.59 and found only a 1.09 retained/discarded
  accent ratio; ordinary spectral flux and RMS do not separate the dense model
  peaks. Removing the candidate graph's per-event bias left piano unchanged,
  reduced mean ARTBeaT F1 from 0.8536 to 0.8364, and regressed three real
  variable-tempo cases. Retain neither experiment and do not expose their
  constants as strategies.
- 2026-08-25: promoted metrical uncertainty into analysis schema v2 as
  `beat_hypotheses`. The product now returns selected, alternating half-time,
  and real-candidate-supported double-time sequences with normalized
  truth-free evidence scores. Discarded observed events remain candidates,
  unsupported timestamps are never invented, and alternatives outside
  40--320 BPM are omitted. ARTBeaT primary metrics and all 15 selected paths
  are unchanged; piano explicitly reports competitive half-time scores instead
  of presenting the dense path as unqualified certainty. Next investigate a
  locally varying metrical path backed by genuinely independent long-range
  meter/harmonic evidence before opening Vienna.
- 2026-08-25: added the frozen evaluation candidate
  `local-metrical-path-v1` and analysis schema v3. It computes chroma cosine
  distance around model-supported events and uses that evidence in a candidate-
  only dynamic program with penalized local half/double-time transitions. The
  default selected paths and mean beat F1 remain exactly unchanged across all
  15 ARTBeaT calibration cases. Best-top-K mean beat F1 rises from 0.8620 to
  0.9180: seven cases improve, none regress, and piano rubato rises from 0.5873
  to 0.8041. Ordinary zero-mean onset autocorrelation was rejected because it
  still favored the dense piano pulse. The local path is necessary as an
  additional hypothesis but is not yet a unique selector: it is emitted on 13
  cases and can itself be worse than the primary path. Keep one shipping
  algorithm and no public strategy selector. Next freeze this complete candidate
  for a one-shot Vienna holdout, then design a truth-free ambiguity selector
  only if the holdout confirms that locally varying coverage generalizes.
- 2026-08-25: opened the precommitted Vienna 4x22 holdout exactly once for the
  `local-metrical-path-v1` definition frozen at commit `03fc058`. The local path
  was emitted on 6/12 performances and improved beat F1 in all six, raising the
  truth-assisted primary/local coverage ceiling from 0.4018 to 0.4772. This
  confirms that the additional hypothesis generalizes beyond ARTBeaT, especially
  on 6/8 and Schubert 3/4 material. It does not solve product selection: the
  existing truth-free relative score chose the primary path in all 12 cases, so
  deployable F1 remained 0.4018, and no case cleared the locked 0.8 gate even at
  the coverage ceiling. Vienna is now permanently closed to tuning. Preserve the
  local path as an internal parallel hypothesis, keep the shipping path
  unchanged, and develop any selector or missing 2/4 coverage only on new
  calibration data before obtaining another independently precommitted holdout.
- 2026-08-26: the new 25-case RUBATO core-oracle calibration passed beat and
  downbeat F1 exactly on all cases but passed the tempo budgets on only 20/25.
  Both Mozart KV618 performances and three of four Verdi performances failed
  tempo P95 despite exact input timestamps; mean tempo median/P95 error was
  2.64/16.18 percent. This is new multi-instrument evidence that the remaining
  tempo-level and local-curve failure is deterministic, not a reason to retrain
  Beat This. Calibrate meter-aware tempo interpretation on RUBATO without
  forcing a preferred BPM band, changing official timestamps, or reopening the
  Vienna holdout. The full Beat This run is a scheduled/offline job because the
  VDI projected multi-hour inference for 1.8 hours of audio.
- 2026-08-26: timestamp-level RUBATO diagnostics isolated the 20/25 result to a
  single common defect rather than competing musical strategies: the primary
  tempo curve clamped every cadence to 40--320 BPM. Removing that mutation while
  retaining the same bounds for published metrical alternatives raises RUBATO
  to 25/25, lowers mean tempo median/P95 error from 2.6407/16.1781 percent to
  0.1548/4.5714 percent, and improves or preserves every case. Keep this in the
  unified shipping algorithm; do not expose a slow-tempo strategy switch.
- 2026-08-26: raw-interval diagnostics then showed that all 25 remaining RUBATO
  points above 25 percent error were isolated unconditional metrical repairs,
  not unresolved whole-track tempo levels. Local repair now requires real
  observation support: slower intervals need backend candidates at every
  implied missing pulse, while faster-interval repair is limited to declared
  fixed-frame model output. RUBATO stays 25/25, mean tempo median/P95 error
  improves again to 0.1544/4.4053 percent, and high-error points fall from 25
  of 6,694 to zero. ARTBeaT ideal metrics remain byte-for-byte unchanged and
  the end-to-end beat F1 remains 0.80516. Keep one shipping algorithm.
- 2026-08-26: added an opt-in content-addressed observation cache keyed by audio
  SHA-256, model-manifest SHA-256, backend/decode contract, and complete decoder
  policy. It persists raw beats, candidates, confidence, source identity, and
  decoded-audio shape only after successful estimation; PCM enrichment and the
  selected estimator still rerun. A real generated-v1 Beat This cold/hot check
  preserved every observation and metric exactly while reducing summed per-case
  analysis time from 322,967 ms to 826 ms (391x). This remains evaluation
  infrastructure, not a product runtime strategy.
- 2026-08-26: promoting the adjacent short/long fixed-frame jitter repair from
  `sequence-phase-v1` changed one ARTBeaT warning but changed no beat or tempo
  metric. Keep it evaluation-only: a shipping behavior without measured gain
  would add another hidden branch rather than improve the unified algorithm.
- 2026-08-26: populated the content-addressed observation cache for all 15
  ARTBeaT and 15 FSLD calibration cases, then added report-schema-v9 attribution
  that separates global fixed-tempo hypothesis coverage from leading, interior,
  and trailing timestamp errors. FSLD cold/hot replay preserved every result
  exactly while reducing summed case runtime from 528,919 ms to 399 ms (1325x).
  The shipping primary passes 6/15 FSLD cases, but the existing product-visible
  half/selected/double alternatives contain a global BPM within 5 percent of
  truth on 14/15; the 140 BPM case remains 12.82 percent away even at that
  coverage ceiling. On ARTBeaT, selected-sequence misses split into 2 leading,
  121 interior, and 5 trailing beats, with backend candidates supporting
  2/112/4 respectively. This rules out track-edge repair as the main ARTBeaT
  bottleneck and shows that whole-track hypothesis construction is mostly
  adequate; truth-free selection and locally varying internal paths remain the
  unresolved work. Replaying the current evidence-required `sequence-phase-v1`
  yields 8/15 FSLD passes, not the historical 9/15: it still safely fixes the
  60 BPM edge-extra and 200 BPM false-half-time cases, while the stricter support
  rule no longer clears the 150 BPM P95 gate. Do not weaken observation support
  or promote the candidate without a fresh precommitted timestamped holdout.
- 2026-08-26: added a calibration-only cross-backend consensus diagnosis and
  compared Beat This hypotheses against BeatNet's independently inferred
  primary beat sequence. Naive global timestamp agreement improved three
  ARTBeaT cases but regressed `60-to-80` by 0.26 beat F1, reducing mean F1 from
  0.80516 to 0.80416. Quarter-track margins show the backend relationship flips
  inside that clip: BeatNet tracks roughly double-time early and the same level
  late. Therefore do not add a second-model selector or a user-visible strategy.
  The useful role of independent backend evidence is to raise confidence when
  whole-track level and phase agree, and to preserve or localize ambiguity when
  they do not. The next selector experiment needs an absolute meter/downbeat
  anchor or a locally varying path; global beat-sequence agreement cannot decide
  which octave-related pulse is musically canonical.
- 2026-08-26: extended the consensus diagnosis with a weight-free meter gate.
  BeatNet downbeat confidence alone is not a safe selector: maximizing its
  2/3/4-pulse periodic likelihood drops ARTBeaT mean beat F1 to 0.71893 and
  regresses seven cases. Requiring an alternative to strictly improve both
  cross-backend beat agreement and class-balanced downbeat periodic likelihood
  is materially safer on calibration: it changes only `75-to-150` and
  `ramp-80-to-200`, raises mean F1 from 0.80516 to 0.82097, and has zero case
  regressions. It also vetoes the prior `60-to-80` failure because its meter
  margin is slightly negative. Keep this as one frozen evaluation candidate,
  not a product strategy; the fixed 2/3/4 meter assumption and use of decoded
  rather than dense downbeat activations require a fresh, meter-diverse
  timestamped holdout before promotion.
- 2026-08-27: removed a decoder-selection bias from the cross-backend meter
  experiment by retaining BeatNet's complete 50 Hz pulse/downbeat activation
  series and sampling it at every primary-hypothesis timestamp. The prior
  decoded-event gate's apparent +0.01581 mean beat F1 does not survive: all
  four agreement-driven alternatives have negative dense meter margins, so the
  dense gate changes zero cases and remains at 0.80516 mean F1. Reject the
  decoded-event candidate and do not spend a fresh holdout on it. Keep dense
  activations as backend-neutral diagnostic evidence, not product output or a
  selectable strategy. The next selector experiment must address locally
  changing metrical level with independent evidence rather than tune another
  whole-track 2/3/4 periodic threshold.
- 2026-08-27: implemented that local experiment as
  `anchored-pareto-decoded-event-dense-pulse-v1`. It compares only maximal
  selected/local disagreement spans between timestamps shared by both paths;
  unanchored track-edge spans remain unchanged, and a bounded local span must
  strictly improve both BeatNet decoded-event decisions and dense pulse
  Bernoulli likelihood. The rule has no fitted weight or BPM band, selects nine
  ARTBeaT regions, and improves three cases. It still fails calibration because
  both BeatNet representations endorse four wrong sparse-pulse substitutions in
  the early 240 BPM part of `240-to-96`; that case loses about 0.11 beat F1 and
  aggregate mean F1 falls from 0.80516 to 0.80308. Keep the implementation as a
  reproducible diagnostic and do not promote it. The remaining ambiguity is not
  localization but canonical beat-level semantics shared by both beat models.
  A next selector needs genuinely different evidence trained or annotated for
  perceived beat/meter level; until then, publish competing supported paths and
  confidence instead of silently choosing a BPM convention.
- 2026-08-27: promoted the frozen harmonic-aware locally varying path into the
  single shipping estimator as alternative metadata only. Cross-corpus evidence
  showed additional coverage without a safe truth-free selector, so the engine
  now emits the path automatically when its real-timestamp and harmonic-evidence
  gates pass, adds an availability warning, and leaves selected beats, BPM
  curves, sections, and changes untouched. Cached Beat This evaluation emitted
  it on 10/15 ARTBeaT cases with exact primary-metric parity; deterministic
  post-processing averaged 42.51 ms per track versus 24.31 ms previously. The
  old `local-metrical-path-v1` name is a report-compatibility alias, not a second
  product strategy. Canonical selection remains blocked on genuinely different
  perceived-beat or meter evidence.
- 2026-08-27: added analysis schema v4 `metrical_ambiguity_regions` instead of
  another track-edge repair heuristic. Shared real timestamps between selected
  and locally varying hypotheses define anchors; the product reports leading,
  bounded, trailing, and fully unanchored disagreement spans with event counts
  and the alternative score, but never inserts, deletes, or selects a beat.
  Cached ARTBeaT replay exposed 40 regions in 10/15 cases: 34 bounded interior,
  three leading, three trailing, and zero fully unanchored. Primary metrics,
  hypotheses, and warnings exactly matched schema v3. This confirms edge
  ambiguity should be visible to consumers but is not the dominant repair
  target. A canonical selector still requires genuinely different perceived-
  beat or meter evidence and a fresh holdout.
- 2026-08-27: froze `rubato-holdout-v1` before model inference as the acceptance
  set for a future canonical selector. Its four commercially usable CC BY-SA
  recordings come from Beethoven Op. 47 and Handel HWV 56, the only two works
  absent from the 12-work RUBATO calibration slice, so the split is work-
  disjoint rather than merely recording-disjoint. The selection excludes NC,
  ND, ambiguous-license, synthetic, reproduction-piano, and structurally
  deviating material. Complete the byte-addressed lock, truth, suite, selector
  identity, and thresholds before opening it; never use these recordings for
  selector diagnostics or tuning.
- 2026-08-28: the completed 25-case RUBATO Beat This calibration attributes the
  remaining failure to the observation path. Official observations still pass
  25/25 with exact beat/downbeat F1 and 0.1544/4.4053 percent mean tempo
  median/P95 error, while the audio path passes 1/25 with 0.5213 mean beat F1.
  Real local maxima cover 82.11 percent of truth beats but only 52.07 percent
  of beats missed by the selected sequence. The current truth-free hypothesis
  rank improves one case and regresses none on RUBATO, yet its known ARTBeaT
  regressions prevent promotion; even the truth-assisted top-K ceiling reaches
  only 0.5232 mean F1. Keep the holdout sealed, retain explicit ambiguity, and
  compare one existing alternate observation backend on the same calibration
  suite before considering a single internal selector or any model work.
- 2026-08-28: BeatNet is complementary evidence but not a replacement or safe
  selector. Its real peaks cover 95.30 percent of RUBATO truth and uniquely
  support 1,040 of Beat This's 2,510 missed beats, raising two-backend candidate
  coverage to 93.51 percent. Its decoded path nevertheless emits 17,051 events
  for 6,726 truth beats, reaches only 0.3865 mean F1, and regresses 22/25 cases
  relative to Beat This. The frozen local decoded-event/dense-pulse Pareto rule
  changes 51 regions, improves two cases, regresses one, and lowers mean F1 by
  0.0010; the existing dense-downbeat meter gate improves none and regresses
  three. Reject a mandatory second backend and keep the holdout sealed. Search
  for commercially usable pretrained perceived-beat or meter evidence; absent
  that, preserve ambiguity rather than training on or tuning against holdout.
- 2026-08-28: audited the available pretrained perceived-beat candidates before
  adding another backend. Madmom's model assets remain non-commercial, while
  BeatNet+ and BEAST do not publish sufficiently explicit source/weight terms.
  Beat Transformer is MIT-licensed, but its released checkpoints all target a
  five-stem Spleeter input and total roughly 284 MB across eight
  cross-validation folds; its lighter non-demixed architecture has no released
  matching weights. Its example DBN also forces a 55--215 BPM band and 3/4
  meter, which cannot define Rhythm Map's semantics. Do not integrate it, pick
  an arbitrary fold, add a Spleeter runtime, or open the holdout. Under the
  current no-training constraint there is no audited complete selector;
  preserve the single estimator and explicit metrical ambiguity until a
  licensed matching checkpoint supplies genuinely independent meter evidence.
