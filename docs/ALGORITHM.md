# Timing Analysis Algorithm

Rhythm Map is currently a hybrid system: a neural observation backend locates
beats and downbeats, while a deterministic Rust estimator derives the BPM
curve, tempo segments, change points, and rhythm-homogeneous sections. The
neural backend does not directly classify tempo changes or emit a tempo map.

## Observation boundary

The default backend runs Beat This! and converts its output into
backend-neutral `RhythmObservations`:

- beat timestamps;
- downbeat timestamps;
- per-event confidence derived from 50 Hz beat and downbeat logits;
- a deterministic short-time PCM activity envelope;
- a deterministic short-time spectral-flux onset envelope; and
- model identity and audio duration.

No model tensor or Beat This-specific type crosses into `rhythm-map-core`.
Alternative trackers and caller-supplied observations therefore use the same
tempo-map estimator and produce the same `Analysis` schema.

The experimental BeatNet adapter exercises this boundary without Python. A
native Rust frontend resamples mono PCM to 22,050 Hz, uses centered 1,411-sample
Hann frames every 441 samples, applies 24 logarithmic triangular filters per
octave, takes `log10(1 + magnitude)`, and appends the positive one-frame
spectral difference. FFT resolution collapses duplicate low-frequency filters
to 136 bands, yielding the model's 272-value frame input. RTen produces
beat/downbeat/non-beat probabilities at 50 Hz.

For calibration, BeatNet retains every radius-one pulse maximum as uncommitted
candidate evidence. The current selected sequence is one internal guarded
candidate-graph decoder covering 40--320 BPM:

1. the earlier variable-tempo frame grid is retained only as a soft prior and
   is snapped within three frames to real pulse maxima;
2. Viterbi states then traverse only real candidate maxima and carry the most
   recent interval plus phase inside a 2-, 3-, or 4-beat bar;
3. event evidence is the beat-plus-downbeat versus non-beat log odds, with a
   tempered beat/downbeat class term for meter phase;
4. ordinary tempo motion pays a squared log-interval-ratio penalty, while a
   half/double-time transition pays an additional fixed cost so sustained
   evidence can change level but isolated events cannot toggle it cheaply;
5. a candidate path is rejected if it loses over 30 percent of the grid
   prior's interval continuity, or if multiple added events also reduce
   continuity; and
6. one missing track-edge event may be restored only when it is both a real
   grid-supported maximum and within 30 percent of the selected median
   interval.

No state can emit a timestamp absent from the neural candidate set. The grid
prior and graph are fused inside one decoder; they are not caller-selectable
strategies. The original calibration is recorded in
`evaluation/baselines/artbeat-beatnet-viterbi-v1.md`, and the guarded graph in
`evaluation/baselines/artbeat-beatnet-guarded-graph-v2.md`. The implementation
uses the published BeatNet/madmom joint tempo-meter boundary as an algorithmic
reference but embeds neither madmom's Python runtime nor its model weights.

The default Beat This decoder follows the upstream peak picker: a frame logit
must be strictly above zero (probability above 0.5), must be maximal within
three 50 Hz frames on each side, and adjacent peak frames are averaged before
conversion to timestamps. Downbeat peaks are decoded the same way and snapped
to the nearest decoded beat. The Beat This adapter can retain frame logits and
replay an explicit peak policy for evaluation, but these backend-specific
diagnostics remain outside `rhythm-map-core` and do not change the default
algorithm.

The adapter also exposes an experimental supported-midpoint decoder for
evaluation. It begins with unchanged upstream peaks, searches only radius-three
local maxima above logit -3, and considers a candidate only when it lies within
15 percent of the midpoint between two upstream beats. A candidate is inserted
only when at least three such gaps occur inside a five-gap neighborhood. This
uses real model peaks plus local phase continuity; it never inserts an
extrapolated grid timestamp. It is not the default because the first public
evaluation improved several half-time and rubato cases but regressed one
metrically ambiguous case.

The evaluation contract prevents that experimental result from silently
becoming a default: candidate sweeps and truth-assisted logit inspection run
only on calibration suites, while a holdout suite accepts one preselected named
candidate and compares it only with the immutable upstream baseline, reporting
overall, per-case, and capability-tag deltas.

### Candidate evidence and pulse coverage

The Beat This adapter also retains one real frame from every radius-one local
maximum plateau in the beat logits. No confidence floor is applied at this
stage. These points are serialized only in backend-neutral observations as
uncommitted `beat_candidates`; they do not change the selected beats or any
shipping analysis result.

Candidate coverage is reported twice: over all annotated beats, and over only
the annotated beats missed by the selected backend sequence. The second slice
prevents already-selected high-confidence events from hiding weak evidence at
the exact timestamps that a recovery algorithm would need to promote.

Calibration reports use that evidence to construct at most four whole-track
pulse hypotheses: the selected sequence, its two alternating half-time phases,
and a midpoint-augmented sequence when at least three real candidate peaks lie
near selected-beat midpoints. No regular grid timestamp is generated. The
hypotheses are ranked without truth using an auditable evidence score: 45%
mean event evidence, 30% log-interval continuity, and 25% preservation of the
selected sequence's total evidence. Event evidence combines backend beat
confidence, local PCM activity, and downbeat confidence. This prevents a
half-time subset from winning merely because deleting every other strong event
makes its intervals more regular, while still allowing a genuinely weak
alternating phase to be removed. Independent beat truth is then used to report
top-1 and best-top-K F1. This separates three cases: no candidate evidence,
evidence present but no coherent hypothesis, and a coherent alternative that
the ranking placed below the primary result.

The first ARTBeaT run and its rejected naive ranking are recorded in
`evaluation/baselines/artbeat-candidate-coverage-v1.md`. Ranking weights remain
an evaluation candidate, not a product option.

The PCM onset envelope is computed at an approximately 10 ms hop with a
centered approximately 40 ms Hann window, rounded up to a power-of-two FFT
size. For every non-DC frequency bin, only positive magnitude growth from the
previous frame contributes:

```text
flux[t]     = sum_k max(0, magnitude[t, k] - magnitude[t - 1, k])
onset[t]    = log(1 + flux[t])
strength[t] = onset[t] / max_t(onset[t])
band[t, b]  = strength[t] * flux[t, b] / flux[t]
```

The first frame establishes the magnitude baseline and has zero flux. Strength
is therefore finite and track-normalized to `[0, 1]`; it is salience within one
recording, not a calibrated probability that a frame is a beat. Each frame also
splits that normalized strength into low (below 250 Hz), mid (250 Hz through
2 kHz), and high (above 2 kHz) contributions whose sum equals `strength[t]`.
The split reuses the same FFT and does not run another model.

Calibration reports include the mean full-band and band-split onset strength at
real backend candidates added by the midpoint hypothesis. Full-band strength,
frequency-band balance, whole-track downbeat periodicity, and a local repeated-
downbeat rule were each insufficient to select a pulse without regressions.
They remain diagnostics and do not change hypothesis ranking. Results are
recorded in `evaluation/baselines/artbeat-spectral-flux-v5.md` and
`evaluation/baselines/artbeat-band-bar-evidence-v6-v8.md`.

The estimator normally preserves backend timestamps. It can reject events
inside sustained low-activity spans, select one phase of a strong/weak
alternating sequence, or reconstruct a short corrupted transition from stable
beat grids on both sides. Paired evaluation reports retain the raw backend
events and confidence values before these decisions. They also expose raw
median BPM plus alternating-phase PCM salience and backend confidence so a
metrical decision can be audited independently of its final tempo curve.

## Audio activity and silence

The end-to-end engine computes RMS over centered 100 ms windows at a 50 ms hop
and converts each value to decibels relative to the loudest window. A span at or
below -40 dB for at least 0.8 seconds is treated as low activity. Backend beats
inside that span are rejected before tempo estimation, and the span midpoint is
reported as `rhythm_discontinuity`.

This is deterministic PCM evidence, not another neural model. Observation-only
callers may omit the envelope; in that case no activity-based filtering occurs.

## Local tempo observations

For consecutive beat timestamps `t[i]` and `t[i + 1]`, the raw local tempo is:

```text
interval[i] = t[i + 1] - t[i]
raw_bpm[i]  = 60 / interval[i]
```

Each observation is placed at the midpoint of its two beats. This produces a
local time series instead of assigning one BPM to the whole recording.

## Metrical-level selection and robust smoothing

Beat trackers may report a musically plausible half- or double-time level. A
120 BPM pulse can therefore appear as 60 or 240 BPM without the event times
being random. But the same power-of-two relationship can be a real tempo
change: 75 to 150 BPM must not be flattened merely because both values share a
metrical octave.

The estimator therefore preserves the sustained cadence implied by consecutive
beat timestamps and clamps only to the accepted 40--320 BPM range. Within the
configured seven-interval window, it compares the median context on the left
and right of each interval. A center value is repaired only when both contexts
agree within the 12 percent jump threshold and the center is approximately an
integer metrical octave away. Each three-point neighborhood is then averaged in
log-tempo space only when its range also stays below that threshold; otherwise
the center value is retained so smoothing does not blur a real step. This
repairs an isolated half- or double-length interval from a missed event while
retaining a sustained step, ramp, or rubato gesture. It does not fold the whole
recording into a preferred BPM band.

The evaluation-only `metrical-consistency-v1` candidate extends that repair to
runs of at most three consecutive intervals. A run is rewritten only when it
is bounded on both sides, the two surrounding tempo medians agree within the
jump threshold, and every interval in the run is approximately an integer
metrical octave from that shared context. Replacement values interpolate in
log-tempo space between the two boundaries. An edge run, a longer run, or a
sustained 75 to 150 BPM transition is therefore preserved. Applied repairs are
reported as `short_metrical_outlier_run_repaired`. The shipping default remains
the one-interval rule until this candidate passes an independent timestamped
holdout.

The evaluation-only `sequence-phase-v1` candidate includes that bounded-run
repair and adds
three sequence-level checks for cases that lack symmetric context:

1. A proposed whole-track half-time selection transfers downbeat evidence from
   discarded subdivisions to the nearest retained beat, then rejects the fold
   if the resulting downbeats occur less than two retained beats apart. This
   prevents a strong alternating PCM accent from turning a coherent fast beat
   sequence into a bar-phase-inconsistent half-time sequence. The rejection is
   reported as `inconsistent_half_time_selection_rejected`.
2. A one-sided edge run is treated as spurious double-time only after at least
   six stable anchor intervals establish a grid and every later observation can
   be partitioned into retained grid events and midpoint extras through the
   beginning or end of the track. At least four extras and six retained events
   are required, and the retained events must carry at least 1.15 times the
   confidence-weighted PCM evidence of the extras. The estimator only removes
   observed extras; it never extrapolates new timestamps. The decision is
   reported as `edge_double_time_events_rejected`.
3. For a backend with a declared fixed frame rate, an adjacent short/long
   interval pair may be replaced by its mean when the pair straddles the stable
   surrounding period and its mean agrees with that context. This corrects
   opposite frame-quantization errors without changing caller-supplied or
   oracle timestamps, whose source has no frame rate. The repair is reported as
   `quantized_interval_jitter_repaired`.

These checks preserve equally supported real tempo doublings, sustained tempo
changes, rubato, and ambiguous whole-track pulses. `sequence-phase-v1` remains
a named evaluation candidate rather than a product strategy until independent
timestamped evidence supports merging it into the single shipping algorithm.

Half- and double-time alternatives are preserved at two levels. The compact
`tempo_hypotheses` field reports octave-related global BPM summaries. Analysis
schema v2 also returns `beat_hypotheses`, containing the selected sequence, both
alternating half-time phases when their implied tempo remains in range, and a
double-time sequence when real backend candidates support enough interval
midpoints. Every listed time must come from an accepted beat or backend
candidate; hypothesis construction never interpolates a timestamp.

Sequence scores are truth-free and relative, not calibrated probabilities. For
each sequence the estimator combines mean event evidence (45 percent),
log-interval continuity (30 percent), and retained selected-sequence evidence
(25 percent), then normalizes the strongest returned sequence to 1.0. A
half-time or edge-cleanup decision moves discarded real events back into the
candidate pool so the result remains auditable. Hypotheses outside the same
40--320 BPM analysis range are omitted. They are result metadata rather than
caller-selectable strategies and do not silently replace the primary tempo map.

Before interval smoothing, an evidence-based rule handles inserted
subdivisions. When the raw median is at least 150 BPM and its half lies in the
preferred band, the estimator compares the mean audio salience of the two
alternating event phases. It keeps the stronger phase only when its
confidence-weighted activity is at least 1.35 times the discarded phase. Equal
salience therefore preserves a genuine fast pulse instead of blindly dividing
every high tempo by two. The decision is recorded as
`metrical_level_selected_half_time`.

Weak double-time events require model evidence rather than estimator-generated
timestamps. The evaluation build of the Beat This adapter therefore exposes the
`supported-midpoints-logit-minus-3.0` policy through the same end-to-end
evaluation path as the upstream decoder. Each inserted event must still be a
real local maximum in the model logits, near an interval midpoint, and part of
a supported run. Evaluation reports record both decoder and estimator policy
IDs so their effects cannot be confused with the default product path.

## Edge-connected sequence-path recovery

The evaluation-only `viterbi-edge-logit-minus-3.0-bias-2.0` decoder operates on
the
retained Beat This beat logits before the deterministic estimator. Its dynamic
programming state is an integer beat period and phase at the backend's 50 Hz
frame rate. Period states cover 40--320 BPM. Beat and non-beat emissions use
the Bernoulli log likelihood implied by each logit, with a beat-state bias of
2.0. A transition between beat periods pays 100 times the squared natural-log
period ratio, so the path may follow a real tempo change without changing
period freely at every event.

The best-scoring path is evidence, not an event generator. Every recovered
timestamp must satisfy all of these checks:

1. it snaps within three frames to a radius-one local maximum above logit -3;
2. its connected weak-peak sequence contains at least six candidates, with no
   adjacent candidates more than three path beats apart;
3. at least three weak candidates lie within three path beats of the emitted
   event; and
4. the connected sequence reaches the first or last model-supported path
   event within two path beats.

All upstream peaks remain unchanged. A path state without a qualifying model
peak emits nothing, so the decoder cannot fill a silent grid or extrapolate
through a region where Beat This produced no event evidence. Live backend
observation and single-inference evaluation share the same policy dispatcher,
preventing an evaluation-only policy from silently decoding differently in a
deployed adapter.

On the 15 timestamped ARTBeaT calibration cases, the conservative policy is
identical to the upstream decoder in every event count and beat metric. When
combined with `sequence-phase-v1`, it raises the tempo-only FSLD slice from 9
to 10 passing cases by recovering a long repeated weak-peak sequence in the
110 BPM clip. Short four-point sequences remain rejected. The 130 BPM clip is
unchanged because it lacks enough qualifying edge peaks; solving that case in
this backend would require timestamp invention, a less conservative policy, or
new observation evidence. On the precommitted nine-case ARTBeaT disjoint
holdout, the candidate lowered mean beat F1 from 0.67915 to 0.67771 and
regressed the syncopated case by adding false events. It therefore remains a
research experiment and must not be promoted. The holdout also shows that edge-connected
recovery does not solve whole-track pulse-level or phase ambiguity.

## Short transition beat-grid recovery

A model can smear an abrupt tempo jump across several seconds and emit a mix of
late, duplicate, or missing events. The estimator only reconstructs this region
when all of the following evidence is present:

1. the tempo curve contains a ramp block no longer than four seconds;
2. stable constant-tempo segments bracket the block and differ by at least 12
   percent;
3. at least six observed beats on each side fit a linear beat grid with no more
   than eight percent of one period in maximum timing error (boundary outliers
   may be trimmed while fitting, then are included in the repaired span); and
4. the transition contains an interval shorter than 65 percent of both stable
   periods, or an interval that fits a two- or three-beat multiple of one grid
   while fitting neither grid's ordinary period.

The right-hand stable grid is fitted by least squares and extrapolated backward
only across the transition block. Each reconstructed timestamp inherits the
confidence and downbeat confidence of a nearby raw event when one exists.
Everything outside the block is preserved. A clean tempo jump, a gradual ramp,
or an isolated extra event without a tempo change therefore does not trigger
the repair. The decision is recorded as
`short_transition_beat_grid_recovered`.

## Bar-level downbeat selection

Beat This can identify the correct two-beat metrical pulse while assigning
downbeat confidence to both the first and third beat of a four-beat bar. Rhythm
Map treats those events as half-bar candidates rather than assuming every
second candidate must be removed.

After transition-grid recovery, the estimator finds continuous runs where
downbeat candidates are exactly two analyzed beats apart. Runs shorter than six
candidates are left unchanged. Within each longer run it compares the mean PCM
activity amplitude of the two alternating candidate phases. The weaker phase
is rejected only when the stronger phase is at least 1.2 times as salient.
Equal or weakly differentiated phases remain untouched, preserving legitimate
2/4 meter and acoustically ambiguous material.

An irregular candidate interval splits the run, so a tempo change may select a
different phase on each side. If a selected four-beat grid predicts one
downbeat immediately outside a run, and the model placed a candidate one beat
away, the label moves to the predicted beat only when its PCM activity satisfies
the same 1.2 salience ratio. This repairs a displaced label at a reconstructed
tempo boundary without extrapolating a bar grid through a long unknown region.
The decision is recorded as `bar_level_downbeats_selected`.

This stage changes only downbeat labels and confidence. It neither inserts nor
removes beat timestamps and therefore cannot improve ordinary beat F1 or tempo
accuracy by itself. Observation-only callers that omit PCM activity retain the
backend's downbeat decisions.

## Robust BPM curve

The normalized observations pass through two deterministic filters:

1. an odd-window median filter, seven intervals by default, removes isolated
   spikes caused by a missed or extra beat;
2. a three-point local mean in `log2(BPM)` space performs symmetric
   ratio-domain smoothing, equivalent to a geometric mean in BPM space.

For each curve point, confidence starts with the lower confidence of its two
source beats and is reduced exponentially when the unsmoothed observation
disagrees with the regularized curve:

```text
deviation  = abs(log2(normalized_bpm / smoothed_bpm))
confidence = min(beat_confidence) * exp(-8 * deviation)
```

`global_bpm` is the median of the smoothed curve and is only a summary. The
complete `tempo_curve` is the source of variable-tempo information.

## Constant and ramp segments

The curve is simplified recursively in time--`log2(BPM)` space. A point becomes
a retained knot when its error from the line connecting the current endpoints
exceeds 0.04 octaves, approximately 2.8 percent. This is analogous to
Ramer--Douglas--Peucker polyline simplification and preserves sustained shape
while discarding insignificant jitter.

Adjacent knots form `TempoSegment` values. A segment is classified as:

- `constant` when its endpoint BPM values differ by at most 3 percent;
- `ramp` when the endpoint difference exceeds 3 percent.

Every segment retains start and end BPM values. Rhythm sections additionally
carry a representative median BPM; no separate model inference is run for an
individual segment.

## Change-point detection

Three independent deterministic detectors contribute change points.

### Tempo jump

At each eligible curve index, the estimator compares the median of the two
preceding BPM points with the median of the current and next BPM points. A
sustained relative difference of at least 12 percent produces `tempo_jump`.
The simplified segments provide a second path: a consecutive ramp block lasting
at most four seconds with stable plateaus on both sides and at least the same
12-percent endpoint difference is treated as a model-smeared jump, timestamped
at the start of the transition. Same-kind detections within 0.75 seconds are
merged, retaining the strongest.

### Ramp boundary

Consecutive ramp segments are considered as one block. A significant block
longer than the four-second jump limit produces `ramp_boundary` at its available
constant/ramp edges.

### Rhythm discontinuity

An inter-beat gap produces `rhythm_discontinuity` when it is both longer than
one second and more than 3.5 times the median beat interval. A sustained PCM
low-activity span produces the same kind even if the neural backend hallucinates
a regular pulse through silence. Both use the span midpoint.

## Rhythm-homogeneous sections

The beginning, detected change points, and audio end form section boundaries.
Each resulting section reports:

- start and end time;
- median local BPM when defined;
- beat count; and
- stability derived from mean absolute log-tempo deviation.

These sections describe timing homogeneity. They are not semantic labels such
as verse, chorus, build, or drop.

## Current interpretation

The estimator is deterministic and training-free, but its input is still
model-derived. Missed beats, inserted subdivisions, and incorrect metrical
levels can propagate into the tempo curve. The recovery rules correct bounded
cases with stable evidence on both sides; they cannot reconstruct a long span
of missing evidence, determine bar phase without an acoustic accent, or infer
musical intent from an ambiguous pulse alone.

The defaults are internal product policy rather than parameters users must
supply. They will be tightened or replaced only against the checked-in
evaluation protocol. See [`../ROADMAP.md`](../ROADMAP.md) for the planned
bottleneck-attribution experiments.
