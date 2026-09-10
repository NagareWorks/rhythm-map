# Small learned clock-compatibility readout: proposal v1

Status: **design and model-free contract checks only**. No training, feature
extraction, new annotation collection, holdout access, model export or production
change is approved by this document. The pilot sizes and resource caps below
are proposed limits, not allocated resources or a claim that usable data exists.

Follow-through: the [blinded packet contract and fixed observation-only proposal
baseline](CLOCK-ANNOTATION-PACKETS.md) now exist. The original proposal below
remains a design, not an implemented readout. Measured native-grid coverage of
the new candidate set is insufficient to clear the generation prerequisite;
independent labels and fitting remain unstarted.

The [subsequent miss diagnosis](../evaluation/baselines/clock-proposal-diagnosis-v1.md)
preserves those scores and separates crop-boundary witnesses, offset-compatible
residuals, count mismatch and anchor geometry. Step misses primarily lack a
count-matched candidate, often despite nearby existing peaks. Joint observation
assignment and locally varying clock advancement need a bounded construction
contract before a learned scorer; the diagnostic does not supply a repaired path.

## Decision: learn evidence for a clock, not just presence or one BPM

The [latest conditional MusicFM experiment](../evaluation/baselines/musicfm-paired-v1.md)
retains 3/7 correct native-shape pairs and 0/7 full density resolutions despite
valid capture and supplied clocks. False changes and erased real changes remain.
The [Beat This pre-head recurrence test](../evaluation/baselines/prehead-recurrence-v1.md)
also has mixed cases; do not combine per-track winners. These reject particular
readouts, not all information in an encoder or all training-free methods.

Propose one small **clock-conditioned compatibility readout**, reusing the frozen
Beat This encoder and unchanged beat/downbeat heads. For an audio query and a
candidate continuous clock, ask whether the declared audible context supports
that interpretation. A candidate can continue through weak attacks or rests,
make a real step/ramp, or use a different beat level. It is not a list of public
strategies. The [audio metadata engine contract](METADATA-PACKS.md) remains one
call with no per-song tuning; timing is first, future packs remain optional.

This is distinct from the earlier [rhythm-support proposal](RHYTHM-SUPPORT-PROPOSAL.md):
a region can be rhythmic while a particular clock is contradicted. Conversely,
rejecting every proposed clock can mean the generator missed the correct clock,
not that the audio lacks rhythm. Do not turn either result into an AI-origin,
genre, meter or source-separation claim.

## Inputs, outputs and the missing generator boundary

The intended research chain is:

```text
audio -> frozen encoder/observations -> observation-only clock proposals
           |                                      |
           +-> shared temporal features ----------+-> compatibility scores
                                                        |
                                              joint timing decision (later)
```

Only the compatibility component is specified here. The existing public
`tempo_hypotheses` are octave-related **global BPM summaries**, not local
phase-continuous paths. They cannot be relabeled as a ready-made local proposal
generator. The earlier eight oracle clocks likewise do not discover boundaries.

- A proposal must carry a continuous phase function `phi(t)` and positive period
  `p(t)`, with explicit physical support. Steps change its phase slope, not reset
  phase. Ramps and expressive paths cannot be silently forced into step labels.
- The future generator receives only audio-derived observations and their valid
  time coverage. Annotation BPM, phase, change locations, listener votes, file
  roles and source IDs are excluded. Deduplicate equivalent physical clocks
  before fitting/scoring; aliases are not extra votes or loss examples.
- Start with a research cap of eight proposals per query. This is a cost bound,
  not eight required semantic classes or a user parameter. Empty proposals and
  cap overflow are explicit generator failures, never dropped queries.
- Measure **proposal coverage separately from conditional scoring**: can an
  observation-only generator offer at least one independently supported clock
  on a query where listeners can follow timing? Record ambiguous, unsupported,
  invalid and unassessed queries separately. A supplied-oracle experiment is a
  labeled diagnostic upper bound, never this automatic coverage result.
- Proposed output is an uncalibrated compatibility logit per candidate plus
  coverage/provenance. It is not an absence likelihood, a beat timestamp or
  evidence that frame observations are independent. Do not multiply overlapping
  query scores into a joint probability. A separate calibration split and
  separately verified sequence integration are required before confident use.
- Multiple candidates may remain supported; none may qualify. Preserve
  ambiguity/unknown rather than forcing a winner with a softmax over the list.
  A missing context is unavailable evidence, not an incompatible-clock label.

The generator is **not implemented or frozen by this proposal**. Its no-label
interface, phase geometry, bounded search and per-slice coverage must be reviewed
before a learned-fit pilot is authorized. A ranker cannot recover a missing
proposal. Failure there redirects work to generation, not a larger readout.

## Annotation task and supervision

Use two independent listeners with the same complete declared analysis input
and the same query bounds. A short preview may navigate the input, but it must
not provide different context to the two listeners or to the encoder. Preserve
the clip's relation to the original work/recording; a crop edge is not a natural
intro/outro. Missing sides of a query remain missing, not zero-valued truth.

1. Before any detector scores or candidate overlays, each listener marks a
   followable clock/accepted beat levels, timing uncertainty, possible transition
   intervals and whether the available context is sufficient. Do not show tags,
   the source's expected tempo, experiment outcomes or the other listener's vote.
2. Present candidate-clock overlays/metronomes with opaque IDs in a fixed,
   randomized presentation order recorded in the view identity. Judge each
   candidate separately: `supported`, `contradicted` or `uncertain`. Do not say
   that exactly one must win. Both listeners evaluate the same clocks/context.
3. Keep both first-pass annotations and candidate votes. Equal known votes
   produce a binary compatibility target; any uncertainty or disagreement is
   retained without a binary target. Subsequent adjudication is a separate
   versioned record, never an overwrite of the original votes.

| Situation | Supervision consequence |
| --- | --- |
| Constant clock remains perceptible through weak/omitted attacks | The continuing clock can be supported; an observed long interval is not itself a negative label. |
| Independently audible real change | The changed clock can be supported; old-tempo continuation may be contradicted. The change interval can remain uncertain. |
| Both native and half/double beat levels sound defensible | Multiple positive targets; do not manufacture negatives from the old native-only evaluation convention. |
| Every offered clock is wrong but timing is audible | Known negatives for these candidates plus a generator miss, not a no-rhythm label. |
| Listener disagreement, free-time uncertainty or insufficient context | No binary target; retain reason and denominator. Never silently convert to zero. |

Beat/source MIDI or an authored tempo map is construction evidence, not by itself
a guarantee of a uniquely perceived beat level. Detector pseudo-labels cannot
independently assess that same detector. A source's musical structure and its
recorded execution can differ; keep both when available.

The [model-free contract helper](../evaluation/parity/clock_readout_proposal.py)
checks matching view identities, complete candidate votes, distinct listener
identifiers and conservative consensus. It permits multiple positives and
all-negative proposal sets. It does **not** certify listener independence,
recording rights, a truthful first pass, actual listening or absence of leakage.
Its authored test objects are not collected annotations or accepted data.

Future loss: independent binary cross-entropy on known candidate targets, averaged
first within a query and then within a provenance group. Groups with at least
one trainable query receive equal weight. Unknown-only queries/groups have no
gradient but stay in the coverage ledger. Duplicating candidates or crops must
not increase a group's weight. Do not fit the original seven exposed pairs,
assume every alternative is wrong, or add a fitted pairwise penalty after failure.

## One costed architecture, not a trained or exported model

Use the existing normalized 512-channel, 50 Hz `final0` pre-head sequence under
the [validated capture contract](../evaluation/baselines/prehead-capture-v1.md).
Freeze encoder parameters and the existing two task heads. No MusicFM runtime
or new mandatory encoder is proposed. A failed readout does not authorize
unfreezing the encoder, layer search or a whole-tracker retraining job.

The candidate's six per-frame fields are `sin(phi)`, `cos(phi)`, `sin(2*phi)`,
`cos(2*phi)`, `log(p / 1 second)` and the one-frame difference of that log period;
`phi` is in radians for these fields. A continuous cycle-coordinate clock is
converted with `phi = 2*pi*cycles`. For unavailable preceding period support,
use zero difference with incomplete-context status, not a claimed zero change.
Two additional flags describe a valid center and complete required readout
context. Silence/rests with valid features do not clear these flags. Invalid
features cannot enter the head; physical-edge padding is explicitly marked.

| Operation | Parameters including biases |
| --- | ---: |
| Shared frame projection 512 -> 32 | 16,416 |
| Six shared residual temporal blocks: depthwise kernel 5, pointwise 32 -> 32 | 7,488 |
| Per-candidate fusion: 32 audio + 6 clock + 2 coverage fields -> 32 | 1,312 |
| Masked query mean after fusion, then 32 -> 1 compatibility logit | 33 |
| Total, parameter-free ReLU and no additional normalization | **25,249** |

Keep temporal order through the shared temporal blocks, with dilations
`1, 2, 4, 8, 16, 32`. Pointwise nonlinear fusion precedes query pooling; this
is not a mean of raw encoder frames. Pool only valid query centers; zero valid
centers are unavailable, not zero logit. The architecture is a proposed baseline,
not evidence that these frozen features contain the necessary information.

The readout radius is 126 frames and receptive field 253 frames: 5.04 seconds
between outer centers. This is **not** the encoder's audio context or end-to-end
latency. Gather left/right readout halos from the original owned feature timeline,
using one fixed padding/mask rule only at physical input ends. Six encoder-border
frames cannot stand in for a 126-frame readout halo. Never change encoder ownership
or infer selected query crops to make head seams look better.

FP32 weights alone are 100,996 bytes. For an illustrative 15,000 owned frames,
four-second queries with two-second hops give 149 full queries. At eight
proposals/query, shared projection/temporal processing plus fusion/output take
657,510,144 multiply-accumulates; this excludes the frozen encoder, activation,
pooling, masks, trigonometry, allocation and data movement. It is a cost estimate,
not a latency measurement. It assumes shared frames are processed once; a
stateless chunk implementation that recomputes halos has additional work.
Retain the final partial query when needed for full
coverage; do not hide an uncovered tail to fit this illustrative count.

One 1,500-frame FP32 encoder-feature chunk is 3,072,000 bytes; allowing both
126-frame readout halos makes raw input-buffer capacity 3,588,096 bytes before
other activations/copies. A five-minute full-feature export instead costs about
30.72 MB. Production would require bounded buffers, same-pass versioned tensor
export, unchanged old head/event identity, complete-vs-chunk head equivalence,
and measured Rust/FFI/WASM operator, memory and warmed-CPU behavior. These are
not implemented or established by a small parameter count. No public switch,
empty crate, new package or model download is added now.

## Data: a finite label-feasibility pilot, not another source-pool expansion

Keep the existing forty calibration identities, seven exposed pairs, untyped
RUBATO, authored controls and sealed holdouts in their existing roles. None is
new independent acceptance; do not derive fitting labels by reading their old
scores. The FSD50K metadata inventory supplies no clock-compatibility labels.

Propose at most **24 new, independently reviewed provenance groups**, four with
each primary stratum below and two scheduled four-second queries per group
(48 query records, not 48 independent recordings). Secondary tags can overlap.
Use complete declared inputs and record why each query was selected without
model outputs. This is for rights/label/generator feasibility, not fitting or an
accuracy claim; no such groups are admitted by this proposal.

1. Constant timing with weak/drumless/syncopated pulse or contextual rests.
2. Genuine octave-related steps, with a matched constant context where available.
3. Genuine non-octave steps, retaining both acceleration and slowdown.
4. Gradual/expressive timing, not relabeled into hard step ground truth.
5. Natural intro/outro or asymmetric context; mark crop edges separately.
6. Perceptually ambiguous beat level, free-time/no reliable clock and periodic
   non-music counterexamples. Sound-source category does not determine truth.

Prioritize material with documented work/recording/asset provenance and timing
construction evidence, followed by independent listening. Candidate routes are
new rights-cleared recordings and controlled multi-generator renders; neither
is declared collected or cleared. Keep genuinely recorded musical material in
each applicable stratum so synthetic success alone cannot pass. Stop this pilot
at 24 groups or six listener-hours, whichever is first; report missing coverage
instead of silently extending the budget or replacing difficult examples.

Record exact PCM/source identity, work, recording, performance, creator/session,
shared samples, transformations/generator families, license/evidence URLs,
attribution and distinct training/redistribution review statuses. Form connected
provenance groups before splitting; uploader names and different crop IDs are
not independent groups. Preserve unknown encoder-training overlap. Existing
license metadata is not individual clearance, and no third-party contact or
audio acquisition is authorized here. See the retained
[admission limits](../evaluation/baselines/rhythm-support-admission-review-v1.md).

Pilot data stays development-exposed. Any later fit needs separate training,
development/checkpoint-selection, score-calibration and new acceptance groups.
No connected component crosses those boundaries; derived audio and repeated
performances remain together. Balance conditions within source/generator
families so a model cannot pass by recognizing a corpus or rendering chain.
The split manifest must state its counts and missing strata before extraction.
No training-set size is asserted sufficient without those observed labels.

## What must pass, and when to stop

There are three distinct decisions; passing one cannot substitute for another:

| Stage | Required evidence | Stop condition |
| --- | --- | --- |
| Feasibility, before fitting | Rights/provenance review, reliable independent labels, observation-only proposal generation and coverage, unmodified encoder/export fidelity | Missing groups/rights, pervasive disagreement, or supported clock omitted by the generator. Fix/narrow that prerequisite, not network size. |
| Conditional learning pilot, only after separate approval | One architecture beats frozen recurrence and a fitted clock-only control on new development groups; all known slices and unknowns retained | Improvement comes only from a BPM/clock prior, one source family or one favorable seed; missed/weak and true-change cases still trade regressions. Close this readout, no automatic sweep/fine-tune. |
| Automatic product admission, separately authorized | Annotation-free generator + sequence integration, calibrated decisions, per-slice beat/tempo/change/coverage and cross-platform/runtime checks on locked new acceptance | Oracle-only success, collapsed coverage, new false changes/erased changes, forced beat-level certainty, or failed runtime/identity checks. Keep default unchanged. |

For label feasibility, report independent agreement by stratum **before**
adjudication, known/unknown counts, multiple-supported levels and generator misses.
A proposed pilot continuation floor is 80% exact three-way agreement and at
least 75% known candidate targets in each stratum. These are feasibility gates,
not musical accuracy; interpret small denominators explicitly. Legitimate
multi-level ambiguity is not disagreement when both listeners support both
clocks. If a naturally uncertain stratum fails, narrow the learning scope and
retain it as an abstention obligation, rather than forcing labels or dropping it.

Before fitting, preregister actual independent acceptance sizes and numerical
error/coverage budgets; the current 24-group pilot cannot certify high accuracy.
Separate false changes on constant inputs, missed/erased real changes, gradual
curve error, false beat-level certainty, unsupported-clock acceptance, proposal
coverage, decision coverage and uncertainty. Report primary beat F1/identity
separately from latent clock estimates. Use fixed queries/full durations and
group-level denominators, not overlapping windows as independent trials.

An illustrative zero-error binomial bound at one-sided 95% needs 299 independent
units to put a single error rate below 1%, assuming representative independent
Bernoulli units. That is not a required training count, a multi-slice guarantee
or evidence about these exposed examples. The earlier
[planning caveats](RHYTHM-SUPPORT-PROPOSAL.md#acceptance-and-stopping-rules) still
apply. Freeze risk and coverage floors together: always-unknown must not pass.
Do not decide numerical adoption margins after looking at new acceptance scores.

Proposed later fitting envelope: one architecture; three fixed main seeds
`142, 143, 144`, plus one clock-only control at seed `142` (zero audio features,
same clock/coverage inputs). At most 20 epochs or 30 minutes per fit, at most
four fits/two hours total. Seed 142 is the primary candidate; the other two
measure stability, not a menu from which to pick a winner. Select its checkpoint
by the preregistered development loss, earliest epoch on a tie. A primary failure
cannot be repaired by silently promoting another seed.

Count encoder capture separately (proposed cap: 120 CPU-core minutes, one feature
capture per admitted complete input plus small fixed fidelity controls). Record
both wall and device time. Any temperature/intercept calibration is a separate
fitted artifact on the score-calibration split, capped at one fixed-form fit and
ten minutes; it is not training-free postprocessing. Architecture/threshold
sweeps, pseudo-label expansion, extra seeds, additional encoder passes or renewed
data acquisition require a new proposal after a recorded stop.

First experimental integration is shadow-only. Do not let inferred clock ticks
become observed beat events, veto correct existing events, or hide unavailable
regions. Any later observation selection or sequence-model change needs its own
joint gate. Preserve alternative clocks/uncertainty in the timing pack; no public
per-song policies or automatic fallback to a larger mandatory model.

## Next deliverable, with training still closed

Implement and review a **blinded annotation packet contract and observation-only
proposal-coverage audit** before collecting new votes or fitting a head. The
packet must bind input/context, candidate geometry, provenance and separate
listener records, without leaking detector verdicts. The generator must account
for empty/overflow/ambiguous cases on the existing development evidence and
cannot read annotations. This proposal's budget/consensus checks are a small
prerequisite, not that completed tool or a training-readiness certificate.

Present the resulting coverage, feasible label/rights inventory, concrete split
sizes and acceptance/resource budget for separate approval. No access to a
sealed holdout, listener coordination, optimizer execution or deployment follows
automatically from accepting this design direction.
