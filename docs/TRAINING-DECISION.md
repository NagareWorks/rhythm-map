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

The subsequent [frozen proposal-miss diagnosis](../evaluation/baselines/clock-proposal-diagnosis-v1.md)
leaves every old candidate and score unchanged. Only 38/2,690 misses have a
crop-boundary-only witness. Seventeen of nineteen step misses lack any
count-matched candidate; fourteen have existing local-max peaks near every
missing raw reference tick, while extra events coexist. This is evidence to
specify one peak-supported, locally variable-advance clock construction, not
proof that selecting those peaks works or that a scorer should now be trained.
Expressive missing-peak and nonuniform-interpolation errors remain; reference-
assisted offset/anchor geometry is not independent coverage or a product fix.
Freeze joint assignment, support and finite candidate/search cost before replay.

The [observation-only proposal coverage baseline](../evaluation/baselines/clock-proposal-v1.md)
now measures 3,348 fixed-grid queries from all forty prior inputs before any
annotation is loaded. Its native-grid proxy covers 484, misses 2,690 and leaves
174 reference-unavailable; only 5/24 step queries are covered. This is not
independent listening or product accuracy, but it exposes a candidate-generation
prerequisite before training a scorer. The [two-pass packet contract](CLOCK-ANNOTATION-PACKETS.md)
preserves uncertainty, source/context identity and generator misses; no actual
listener records or cleared groups are added. Next inspect retained miss geometry,
not train the readout or broaden the candidate/threshold search on these labels.

The [clock-compatibility proposal](CLOCK-READOUT-PROPOSAL.md) now separates
observation-only proposal coverage, conditional evidence learning and automatic
product admission. It specifies multi-supported/unknown annotation semantics,
a costed 25,249-parameter frozen-encoder readout, a bounded label-feasibility
pilot and explicit fitting/acceptance prerequisites. Only model-free budget and
vote-contract checks exist: no annotations are newly collected, no independent
groups admitted and no training authorized. The later packet/proposal audit
above implements the generation prerequisite but does not pass its coverage
gate; do not present global BPM summaries or supplied oracle clocks as that
automatic generator.

The [conditional MusicFM replay](../evaluation/baselines/musicfm-paired-v1.md)
now measures the original seven exposed music pairs, separately from admission
of the previously rejected no-rhythm composition. Valid extraction and geometry
leave 3/7 pairs with both correct native shapes and 0/7 resolving all density
rivals; false changes and erased real changes remain. Context/frontend differ
from Beat This, and the rule uses oracle clocks, so this neither ranks encoders
in isolation nor measures automatic accuracy. It does show that a presence
rejector alone would not repair this fixed clock readout. Keep the old negative
failure and close this recurrence comparison without a same-label metric sweep.

The next justified **proposal**, not an approved training run, is a small learned
timing readout for clock continuation through weak/missed observations versus
true transition and perceived beat level. Do not substitute a presence-only
classifier for that target. The prior bounded failures support examining this
investment; they do not prove every training-free method impossible or that a
particular hidden representation contains enough information. Independent
paired labels, provenance/rights, group-disjoint acceptance, resource limits and
stopping criteria remain prerequisites. No optimizer or new holdout is opened.

The preceding [bounded metadata admission review](../evaluation/baselines/rhythm-support-admission-review-v1.md)
finds two known project-exposure matches and records eight source-review cases,
not a training set. Standard CC terms do not require an extra AI-specific grant
solely for being a new technology; individual provenance/other rights and
independent region labels remain unresolved. This closes the metadata-only
side pass. No training decision is approved, and no amount of presence-only
metadata resolves weak/omitted beats versus true tempo changes. Return the next
technical decision to that competing-clock target and existing evidence.

The subsequent [null-assumption checks and initial worksheet](../evaluation/baselines/rhythm-support-readiness-v1.md)
close limited second-order-preserving and naive-permutation shortcuts, not all
training-free methods. The subsequent [verified metadata inventory](../evaluation/baselines/rhythm-support-inventory-v1.md)
recovers the official development metadata and screens 34,976 CC0/CC BY clip
records. Per-clip rights clearance, provenance/encoder-overlap review and region
labels remain incomplete; uploader strings are not independent groups. No
neural admission gate ran. Neither unavailable data nor recovered metadata is
evidence that model training is necessary.

The [minimal rhythm-support proposal](RHYTHM-SUPPORT-PROPOSAL.md) now specifies
the region target, analytic-null assumptions, a costed frozen-encoder readout,
rights/label gaps and independent acceptance requirements. It recommends no
training yet. Presence alone cannot resolve missed beats versus true tempo
changes in rhythmic music. The next bounded work is metadata-only inventory
and model-free assumption checks, with no new neural execution or holdout use.

The later [rhythm-support source screen](../evaluation/baselines/rhythm-presence-review-v1.md)
records the MusicFM rejection failure and separates source-semantic limits from
unresolved license/artifact prerequisites. It freezes seven positive and four
negative authored inputs for a no-oracle admission gate, but that gate remains
unrun without a justified producer. This is not evidence of eleven new failures
and does not prove that training is necessary. The next small-component proposal
must identify the missing target, data rights and null/reference assumptions;
it must not silently equate frame non-beat, silence, unavailable input and a
beatless musical region. A frozen-encoder fitted readout still counts as training.

The [pretrained-representation review](../evaluation/baselines/pretrained-representation-review-v1.md)
records which evidence routes have already failed and which genuinely richer
feature remains untested. Reading Beat This's existing pre-head sequence is
training-free; fitting even a small supervised readout would be training and
needs a separate proposal. Neither an affine information bottleneck nor a
successful feature-export parity check establishes musical discriminability.

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
