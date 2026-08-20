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
- a deterministic short-time PCM activity envelope; and
- model identity and audio duration.

No model tensor or Beat This-specific type crosses into `rhythm-map-core`.
Alternative trackers and caller-supplied observations therefore use the same
tempo-map estimator and produce the same `Analysis` schema.

The estimator normally preserves backend timestamps. It can reject events
inside sustained low-activity spans, select one phase of a strong/weak
alternating sequence, or reconstruct a short corrupted transition from stable
beat grids on both sides. Paired evaluation reports retain the raw backend
events and confidence values before these decisions.

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

## Metrical-level normalization

Beat trackers may report a musically plausible half- or double-time level. A
120 BPM pulse can therefore appear as 60 or 240 BPM without the event times
being random.

The estimator first folds every raw observation by powers of two into the
preferred 70--180 BPM band and takes the median as a robust reference. For each
raw observation it then considers `raw_bpm * 2^level`, for levels from -3 to 3,
within the accepted 40--320 BPM range. It chooses the candidate minimizing:

```text
abs(log2(candidate / reference)) + 0.02 * abs(level)
```

The small level penalty avoids unnecessary octave changes when two candidates
are otherwise similarly plausible. Half- and double-time alternatives are
also preserved in `tempo_hypotheses`; the public result does not pretend that
metrical ambiguity has disappeared.

Before interval normalization, a second evidence-based rule handles inserted
subdivisions. When the raw median is at least 150 BPM and its half lies in the
preferred band, the estimator compares the mean audio salience of the two
alternating event phases. It keeps the stronger phase only when its
confidence-weighted activity is at least 1.35 times the discarded phase. Equal
salience therefore preserves a genuine fast pulse instead of blindly dividing
every high tempo by two. The decision is recorded as
`metrical_level_selected_half_time`.

## Short transition beat-grid recovery

A model can smear an abrupt tempo jump across several seconds and emit a mix of
late, duplicate, or missing events. The estimator only reconstructs this region
when all of the following evidence is present:

1. the tempo curve contains a ramp block no longer than four seconds;
2. stable constant-tempo segments bracket the block and differ by at least 12
   percent;
3. at least six observed beats on each side fit a linear beat grid with no more
   than eight percent of one period in maximum timing error; and
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
of missing evidence or infer musical intent from an ambiguous pulse alone.

The defaults are internal product policy rather than parameters users must
supply. They will be tightened or replaced only against the checked-in
evaluation protocol. See [`../ROADMAP.md`](../ROADMAP.md) for the planned
bottleneck-attribution experiments.
