# Normalized missing-observation model: a bounded negative result

This implements the next reference observation model after the
[complete-frame scoring checkpoint](frame-likelihood-v1.md). It is a
single-head, authored likelihood diagnostic, not a new clock decoder. No
training, real-music evaluation, holdout access or production change occurred.
All previous frozen scorers and reports remain unchanged.

## Explicit generative reference, not calibrated model confidence

For a real-valued detector logit z, let q = sigmoid(z) and h = q(1-q).
Use densities with respect to **the real logit axis**:

- observable event: `f_visible(z) = 2 q h`;
- no event/background: `f_background(z) = 2 (1-q) h`;
- event present but possibly unobserved:
  `f_present(z) = (1-r) f_visible(z) + r f_background(z)`.

Both base densities integrate to one: substituting `dq = h dz` gives the
integrals of `2q` and `2(1-q)` from zero to one. The mixture weights sum to one.
Tests independently check numerical normalization. This is not a claim that
Beat This logits actually follow these densities, nor that nearby frames are
independent. The construction preserves the earlier visible/background odds
`exp(z)` while making the measure and assumptions explicit.

The authored run fixes r=0.1, without fitting or sweeping it. Tests at other
rates check algebraic boundaries only and never select a candidate. The scorer
uses stable log-sum-exp, scores the same complete frame domain for every given
state mask, and skips explicitly unavailable observations without treating
them as negative evidence. This does not implement gap-safe clock decoding.

## Why this does not solve weak evidence

The event-to-background likelihood ratio is exactly:

`f_present / f_background = r + (1-r) exp(z)`.

For every negative z and `0 <= r < 1`, this is below one. At r=1 it becomes
one for every input: all event/absence distinctions disappear. Increasing
dropout can reduce the penalty on a hypothesized weak beat, but cannot make
its observation support that beat over absence.

More generally, mixing any visible density with the same background density
maps its ratio L to `r + (1-r)L`. It moves evidence toward neutrality without
changing which side of one it falls on. This limitation is not specific to
the chosen logistic reference densities. It does not rule out temporal priors,
other missing-state emissions, contextual features or other pretrained models.

Also, `P(missing | event-present, z)` is conditional on an assumed event.
It exceeds 0.99 at z=-8 with r=0.1, while the same observation favors **no
event**. Calling that quantity beat confidence would turn background frames
into apparently confident inferred events. The implementation and tests keep
these meanings separate; missing-frame counts are not recovered beats.

## Fixed-path tempo counterexamples

The [authored report](../parity/dropout-likelihood-v1.json) uses 1,152 frames
at 50 Hz in three 384-frame sections. Given masks use periods 24/24/24,
24/48/24, 24/12/24 and 24/32/24 frames, plus an all-absent alternative.
These represent constant 125 BPM, middle half-speed, double-speed and
non-octave change. Each section starts at offset four and uses radius-one
rectangular pulses: strong +8, weak -2, background/erased -8. These are
detector-output controls, not original audio or learned likelihood parameters.

Four intact controls rank their correct given path first. But:

- constant tempo with alternating weak or erased central pulses favors the
  half-speed path;
- a genuine doubled-tempo section with alternating weak pulses favors the
  unchanged-tempo path;
- an all-weak constant grid favors all-absent observations.

Constant tempo with erased alternate central pulses and an intact real
half-speed section yield identical input hashes and identical scores for
**every** candidate. No input-only decoder can recover both incompatible
truths from that representation. This is not an identical-audio claim.

The scorer receives only logits, a proposed state mask and availability, not
truth or case identity. These masks are supplied hypotheses; ranking them is
not a search over unknown clock positions, tempo changes or meter. The
single-head isolation does not establish failure with an informative downbeat
head or other acoustic evidence. It prevents a promising intact-case result
from concealing the weak/erased-observation failure.

Run `cargo run --locked -p rhythm-map-eval --example dropout_likelihood` to
reproduce the authored report. Six Rust tests cover normalization, the ratio
property, conditional posterior meaning, unavailable evidence, invalid inputs
and the fixed-path outcomes. No parameter was revised to improve these results.

## Decision

Do not wire this background-copy dropout mixture into the full decoder or run
another real-cohort comparison expecting it to recover weak negative logits.
Do not tune the missing rate or reinterpret conditional missing probability
as event confidence. This eliminates one proposed formulation, not training-free
timing analysis generally, and does not satisfy the training decision gate.

The next observation formulation must distinguish a **weak structured pulse**
from background using pulse shape, local off-phase evidence or cross-beat
context, with an explicit null/unsupported alternative. Normalization must
include that alternative; conditioning on an event being present is not enough.
First test it against the same weak/erased, extra-event and genuine-change
controls, retaining the identical-input ambiguity. Only then freeze a full
joint sequence candidate for the retained dense calibration captures.
