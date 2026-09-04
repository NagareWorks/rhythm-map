# When would timing-model training be justified?

The product contract remains one-call, zero-tuning audio metadata extraction.
Timing is the first capability pack. Experiments do not become user-selectable
policies, and a rejected heuristic is not proof that training is necessary.

## Separate the musical clock from observations

A detected event is evidence about a beat, not the definition of a beat. Two
events one second apart can mean 60 BPM, or 120 BPM with one undetected beat.
Weak attacks, rests and syncopation do not by themselves establish a tempo
change. Conversely, continuing a previous tempo is a prior, not new evidence.

Keep three concepts separate:

- the latent musical tempo/phase and possible beat-count advancement;
- acoustic/model observations and their availability;
- returned events, supported alternatives and uncertainty.

An inferred clock must not silently manufacture observed beat timestamps.
Numerical BPM coverage is not evidence coverage: a segment can extend beyond
the final detected beat. Unknown regions must not disappear from scoring.

## Controlled observation-loss audit

`rhythm-map-eval`'s `observation_dropout` example keeps calibration truth intact
and deletes observations from its ideal beat sequence. Five fixed patterns
cover intact input, one deletion every eight beats, alternating deletions in
the middle third, four central deletions, and eight trailing deletions. Inputs
shorter than eight events are left intact; the first two events are preserved.
Each pattern is run with oracle downbeat labels and with that channel zeroed.
Zero is a controlled stand-in, not an explicit unknown-channel representation.
There is no audio transformation, model inference or holdout access.

All variants are queried at the same truth beat-interval midpoints. These are
beat-interval-weighted, not uniformly time-weighted; do not compare these
numbers directly with the main evaluator's tempo metrics. Report missing
output separately, as well as queries outside the returned beat span. Empty
error samples mean null, not zero error. Jump counts on expressive recordings
are descriptive, not false-positive counts without matching change truth.

The authored controls include constant 120 BPM, 120/60/120 and 120/90/120 BPM,
with each section lasting eight seconds. Constant 120 with alternating central
observations removed and zeroed downbeats is **exactly the same input** as
intact 120/60/120 with zeroed downbeats. This proves non-identifiability at that
observation boundary, not identical audio or the necessity of training.

## Frozen missing-step clock experiment v1

Before running the candidate, freeze the following deliberately simple
training-free state model. It belongs only to evaluation, not production:

1. For each observed interval `d[i]`, a latent advancement `k[i]` is one of
   1 through 8 musical beats. Its period is `p[i] = d[i] / k[i]`.
2. Find the globally minimum-cost path by dynamic programming:
   `sum(0.05 * (k[i] - 1)) + sum(log2(p[i] / p[i-1])^2)`.
   The first interval pays only the missing-observation cost. Exact ties keep
   the smaller state/earlier predecessor. There is no preferred BPM band.
3. Output piecewise interval tempo `60 / p[i]` for diagnosis, never generated
   beat events and never extrapolation beyond the observation span. The bound
   of eight and cost 0.05 are fixed reference-model assumptions, not fitted
   probabilities or calibrated confidence. No parameter sweep is authorized
   by this experiment.
4. The decoder receives only observations, never truth, mask identity or
   fixture identity. Downbeat factors must yield identical candidate paths
   because this reference model does not use that evidence yet.

This differs from repairing individual outliers: the unobserved beat count is
an explicit state and the whole sequence is optimized. It is still a limited
model: selected-event phase anchors, no false-positive deletion, no onset or
meter likelihood, no explicit gradual-tempo dynamics, and no uncertainty
calibration. Its smoothing prior cannot resolve identical-input truths.

Precommitted decision: improved dropout tempo alone is not promotion. Report
intact-case regressions, real octave and non-octave changes, coverage losses,
and the equivalence witness. A candidate that erases a true tempo change or
loses coverage must not replace the default on aggregate improvement alone.
Keep primary beats/API/packaging unchanged. If useful, the next evidence must
come from richer observations and real cached model output, not another sweep
of the missing-step penalty on these same labels.

## Practical decision gate, not an impossibility claim

Before proposing training, maintain a case-level evidence ledger covering:

1. **Pipeline validity:** audio/frontend/model parity, timestamp mapping,
   trustworthy labels and commercially usable data/model rights.
2. **Representation limits:** clean oracle, controlled missed/extra events,
   weak evidence, track edges, true jumps and gradual tempo. Oracle success
   alone does not establish robustness to imperfect observations.
3. **State inference:** a clock/availability model, not only local repair
   rules. Preserve the distinction between inferred phase and observed events.
4. **Available evidence:** dense activations, independent acoustic features and
   compatible licensed pretrained alternatives. Reusing another model counts
   as training-free; an untested viable source remains an open alternative.
5. **Product acceptance:** fixed per-slice accuracy, timestamp identity,
   coverage, uncertainty and runtime gates, then a frozen candidate on sealed,
   work-disjoint holdout. Do not tune on holdout or weaken gates after failure.

Training becomes a defensible next investment when representative failures
persist across these checks, the missing discriminative evidence is identified,
and available training-free approaches cannot meet the scoped product
requirements within declared runtime/license constraints. That is practical
evidence, not a theorem ruling out every future algorithm.

The resulting proposal must say **what to learn** (observation recovery,
perceived beat level, transition/availability evidence or confidence), why
existing evidence is insufficient, which independent labels and rights exist,
and how success will be measured. Prefer the smallest justified learned
component; do not assume a whole beat tracker must be retrained. Present that
proposal before starting training or opening a holdout for a new candidate.
