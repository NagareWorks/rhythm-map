# Minimal rhythm-support component: feasibility proposal v1

Status: **design only; no training or producer execution authorized by this
document**. The source screen is complete, but the eleven-input admission gate
is still `not_run`, not eleven measured failures. This proposal changes no
model, decoder, default, API, calibration identity or holdout policy.

## Decision and product boundary

Do not start training yet. The missing prerequisites are representative,
rights-cleared region labels and a justified comparison against an analytic
null, not a larger list of model names or thresholds. There is practical
evidence that the tested compositions are insufficient, but not proof that
every training-free approach is exhausted. Retain the
[source screen and frozen gate](../evaluation/baselines/rhythm-presence-review-v1.md)
and the [training decision ledger](TRAINING-DECISION.md).

A support component would address **spurious clock evidence in unsupported
audio**. It would not distinguish a missed beat from a tempo change when both
regions remain rhythmic; nor would it select half/double tempo or a meter.
Those require evidence about competing clocks. Passing a presence gate must
not be reported as solving the original variable-tempo accuracy problem.

The product remains the [audio metadata engine](METADATA-PACKS.md), with timing
first and later optional packs. This work adds no public strategy selector,
threshold or mandatory model pack. First integration, if justified later, is
diagnostic-only: determine whether new timing evidence is admissible. Do not
hard-veto existing correct beats, invent timestamps or hide low-confidence
regions to improve an average. Production composition needs its own joint gate.

## Target and annotation contract

Ask: **does the query region, heard with its declared surrounding context,
support a perceptible sequence of beat-level timing relationships?** This is
not music/non-music classification, energy detection or a single-BPM test.

| Region label | Meaning and important examples |
| --- | --- |
| `supported` | A listener can follow a pulse or changing pulse through the region; includes drumless music, syncopation, weak/omitted attacks, contextual short rests, true tempo changes and followable rubato. Exact beat level or meter need not be unique. |
| `not_supported` | Adequately heard context supplies no reliable beat-level timing relationship; isolated steady tone, DC and noise are controls, not an exhaustive definition. |
| `unknown` | Missing/invalid context, insufficient evidence or unresolved listener disagreement. Retain a reason; never relabel as negative or remove from the denominator. |

Periodic machinery, metronomes and rhythmic speech are not automatically
negative because they are not music. Conversely, a music tag does not make a
free-time introduction supported. A short rest within clear surrounding music
is not equivalent to an entirely silent input. Do not impose constant speed on
expressive recordings or derive this label from an existing tempo estimator.

Use fixed physical query windows and context policy, not beat-count windows
chosen by the model. The existing authored gate remains the exact full PCM at
24 kHz, 288,240 samples, queried at `[4, 8)` seconds. For a later real-data pilot,
freeze window/context lengths and edge ownership before feature inspection;
do not silently pad missing context into negative ground truth.

Two independent listeners label each pilot window with the same context, before
seeing model scores, file tags or the other vote. Keep both votes, uncertainty
reasons and adjudication history. Measure agreement before trusting the target;
persistent disagreement means revise or narrow the target on pilot data, not
force consensus and advertise high accuracy. Human labeling is a real resource
requirement. Another model's pseudo-labels cannot independently validate its
own representation. No region-support labels have been collected in this step.

The future producer receives only PCM, sample rate and physical window. Source
IDs, annotations, expected statuses, BPM, phase and change locations stay with
the evaluator. An offline schema checker alone cannot prove absence of leakage:
the runner and producer implementation require independent review.

## One bounded analytic alternative, with an assumption gate

Candidate: compare temporal structure with **input-level surrogates** under a
declared no-support reference, rather than interpreting the largest recurrence
score as confidence. This is a research alternative, not a selected producer.
The general method requires a specific null; rejection has only that null's
meaning. See Schreiber and Schmitz's
[surrogate-method review](https://arxiv.org/abs/chao-dyn/9909037v1).

Before spending any new encoder forward, reject these invalid shortcuts:

1. Fourier phase randomization preserves `|FFT(x)|^2`, hence circular
   autocorrelation. A statistic that uses only that spectrum/autocorrelation
   cannot distinguish the original from those surrogates, except numerical
   effects. Edge artifacts are not evidence of rhythm.
2. Circularly shifting a fixed feature sequence preserves its circular lag
   relations. Arbitrarily permuting overlapping/contextual encoder tokens does
   not establish exchangeability under a no-rhythm null.
3. A stationary linear/Gaussian null is not the class of unsupported audio:
   non-stationary noise and isolated transients can violate it without musical
   timing. Rejection is not `P(rhythm | audio)`. The original authors explicitly
   discuss this [non-stationarity problem](https://arxiv.org/abs/chao-dyn/9904023v1).

A viable version must specify nuisance properties to preserve (including
colored/noisy envelopes and context), why the observed input and surrogates
are exchangeable under its null, and a temporal statistic that actually changes
under the proposed transformation. Apply the *same full extraction and entire
clock/phase search* to every draw; comparing a maximized original to one fixed
surrogate clock is selection bias. Freeze seeds and draw count, never retry
until significant. If a valid rank test exists, use
`p = (1 + count(T_surrogate >= T_original)) / (B + 1)`; even then this is a null
tail probability, not rhythm confidence. With 99 surrogates the smallest such
value is 0.01 and full re-extraction costs 100 forward sets, not one. Dependent
windows and multiple reported tests need a separately declared error policy.

**Current disposition:** no no-support reference satisfying these requirements
has been established. The next analytic check is model-free: demonstrate the
invariances above and document which assumptions a proposed replacement meets.
Do not run the neural gate under a mismatched null. If this bounded candidate
has no defensible reference, close it as unsuitable; that is not a theorem
against all algorithms, nor permission to sweep the old recurrence thresholds.

## Smallest learned candidate if a later decision justifies fitting

Reuse the already captured Beat This `final0` normalized 512-channel, 50 Hz
pre-head sequence, freeze its encoder and existing heads, and fit one temporal
region-support readout. See the [capture contract](../evaluation/baselines/prehead-capture-v1.md).
Temporal order must survive until the readout; a mean of raw frame features
discards the very relationships under investigation. An illustrative budget:

| Operation | Parameters including biases |
| --- | ---: |
| Per-frame linear projection, 512 to 32 | 16,416 |
| Six residual temporal blocks; each has depthwise kernel 5 then pointwise 32 to 32 | 6 x 1,248 = 7,488 |
| Linear 32 to 1 support logit | 33 |
| Total, with parameter-free activations and no additional normalization | **23,937** |

Dilations `1, 2, 4, 8, 16, 32`, stride one and centered padding give a 253-frame
readout receptive field (5.04 seconds between outer frame centers). Aggregate
only the query region after temporal processing to train against its region
label; frame logits are not supervised beat/non-beat labels. These dimensions
are a costed design, not implemented or empirically selected architecture.
Unknown labels are retained for abstention evaluation, not forced into binary
training targets. A sigmoid alone is not calibrated confidence; later score
cutoffs/calibration require a separate calibration split and count as fitting.

Float32 head weights would be 95,748 bytes. A 1,500-frame feature chunk is
3,072,000 bytes before readout activations/copies; exporting a five-minute
50 Hz sequence instead would cost 30,720,000 bytes. Reuse chunk ownership and
bounded context buffers, not whole-track private trace export. The encoder has
its own longer context: 5.04 seconds is **not** raw-audio latency. New readout
halos/padding need a separate seam/fidelity contract; six existing encoder border
frames do not cover a 126-frame readout radius. Existing shipping exports do
not expose this tensor, so versioned export, Rust/WASM operator parity, memory
and warmed CPU latency must be measured before integration. Small weight size
does not establish those results. Do not add the separate MusicFM stack as a
mandatory runtime just to obtain a support score.

Encoder/weight grants, readout training provenance and runtime dependency
distribution obligations need separate records. Freezing the encoder is still
supervised training when its readout is fitted. No optimizer, new model export,
weight acquisition or new neural execution is part of this proposal.

## Data rights, independence and the actual gap

Existing [dataset locks](../evaluation/datasets/README.md) preserve attribution
and exact bytes; they do not supply region-support labels. Keep all forty
calibration identities, seven exposed ARTBeaT pairs, untyped RUBATO and sealed
holdouts unchanged. The eleven authored inputs remain exposed regression
controls, never the training set or a new independent acceptance set.

| Source route | Permitted role at this stage and missing evidence |
| --- | --- |
| Existing ARTBeaT/FSLD/RUBATO calibration | Regression context only. Beat/tempo/loop annotations do not establish unsupported-region truth; do not rebrand exposed cases as new evaluation. |
| Newly authored recordings and synthesis | Candidate pilot/training material only after explicit recording, composition and sample-asset grants. Include new generators and real recordings; seed variations alone do not establish independence or real-music accuracy. |
| FSD50K metadata-first inventory | Candidate sound-source pool requiring new support labels and provenance review, not a pre-labeled rhythm-negative set. Do not import audio or approve use from its dataset-level license alone. |

The [official FSD50K record](https://zenodo.org/records/4060432) lists mixed
CC0, CC BY, CC BY-NC and Sampling+ audio and separate dataset-level CC BY
terms. Its metadata files map clip IDs to licenses/uploaders; the record also
asks commercial users to contact the authors. Inventory only individually
documented CC0/CC BY candidates under the project's initial rights policy;
exclude NC, Sampling+ and ambiguous grants from that inventory. Filtering is
not a completed commercial-use clearance, and no contact is authorized here.
Sound-event labels do not establish absent rhythm. This step inspected the
publication, not clip metadata/payloads, and selected zero new recordings.

Each future inventory entry must keep source/version, work/recording/creator
groups, exact PCM hash, clip and annotation license URLs, attribution, declared
training/evaluation/redistribution permissions with evidence, and unknowns.
Do not assume absence from Beat This's corpus list proves recording-level
non-overlap; check source IDs, shared recordings and derived assets against
the frozen encoder's published training provenance, marking unverifiable
overlap as unknown. Such items cannot support an independent-test claim.

Keep crops, augmentations, alternate performances of a work, shared samples
and synthesis-generator families in connected provenance groups. Assign groups
before extraction to disjoint training, development, score-calibration and
new acceptance pools. Avoid source-only class shortcuts: both positive and
negative conditions need diverse recording chains. Keep some source/creator
families wholly reserved. Do not train on one corpus's positives and another
corpus's negatives, then mistake corpus recognition for support recognition.

Minimum pilot inventory must cover percussive pulse, drumless/weak/syncopated
pulse, contextual rests, true jumps, gradual/expressive timing, uncertain
free-time material, and stationary/non-stationary non-support controls,
including periodic non-music counterexamples to the label shortcut. A pilot
checks rights, label agreement and sampling feasibility; its size is not an
accuracy claim or automatic training authorization.

## Acceptance and stopping rules

First retain all eleven exact authored identities and all prior positive
shape/density obligations. This necessary gate still cannot establish musical
generalization. For later independent acceptance, preregister error budgets
and slice sizes **before** fitting or inspecting acceptance outputs. Report:

- false support on labeled unsupported regions;
- false rejection and `unknown` separately on supported regions;
- complete coverage, label disagreements and unknown reasons on every slice;
- original beat identity, true-change recovery, false changes, half/double
  ambiguity and edges under any proposed production composition;
- artifact size, peak native memory and incremental warmed runtime.

An always-unknown component must not pass by claiming zero false support.
No aggregate gain offsets regression on weak/rest/drumless or true-change
slices. Predeclare a coverage floor as well as error budgets; do not derive
either from favorable pilot scores. Production remains unchanged until these
joint requirements and platform parity pass.

For planning, with zero errors in `n` independent, representative Bernoulli
units the one-sided 95% exact upper error bound is `1 - 0.05^(1/n)`; certifying
at most 1% this way needs at least 299 units, 2% needs 149 and 5% needs 59.
These are illustrative budgets, not adopted product requirements or training
sample counts. The [exact binomial method](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm)
does not make overlapping windows independent. Use a prespecified recording/
provenance-group sampling unit or an appropriate clustered analysis; declare
what the error rate then measures. Simultaneous per-slice guarantees require
multiplicity control and more data. None of these statistical claims applies
to the deliberately selected, already exposed eleven synthetic controls.

## Finite next step and training decision

Next deliver **a metadata-only rights/label worksheet and model-free null
assumption checks**. List eligible independent source groups and missing label
coverage; preserve unresolved rights and overlap explicitly. Do not acquire
audio, ask third parties, open sealed holdouts or run features as a side effect.

Only propose a bounded frozen-encoder training pilot when: representative
failures are retained; this target can be labeled consistently; usable data
and independent acceptance groups exist; reusable/analytic evidence cannot
meet the scoped rights/runtime/accuracy requirements; and a fixed run budget,
baseline, metric/coverage policy and stop rule are ready. Present that evidence
and obtain separate authorization before fitting. A failed support readout is
not permission for encoder fine-tuning or training an entire tracker. If the
remaining failure is clock selection rather than unsupported audio, return to
that target explicitly instead of enlarging this component by stealth.
