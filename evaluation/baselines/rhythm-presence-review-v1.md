# Rhythm support: reusable-source screen and admission contract

Date: 2026-09-09. **No new detector was accepted or evaluated on audio.** This
bounded review distinguishes existing measured failures, source-level semantic
limits, and unresolved artifact/rights prerequisites. They are not interchangeable
reasons to say a model failed. It does not prove that all training-free methods
are exhausted or that a presence head must be trained.

The [source record](../parity/rhythm-presence-source-v1.json) pins inspected small
source-file bytes and previous reports. The [reproducible audit](../parity/rhythm-presence-audit-v1.json)
retains three model-free semantic witnesses and an **unrun** 11-input admission
gate. Its evaluator is not a rhythm detector or a new production strategy.

## What could actually be reused?

| Source | What it supplies | Disposition in this review |
| --- | --- | --- |
| Existing Beat This | Weighted, shift-tolerant beat/downbeat heads | Preserve the [source/checkpoint proof](beat-this-semantics-v1.md): direct raw/sigmoid/weight-offset conversion does not supply an absence likelihood. Wider/context/pre-head experiments remain recorded separately, not reopened. |
| Existing MusicFM | Contextual representation and relative recurrence | Preserve the [measured composition failure](musicfm-negative-v1.md), including four successful gain/rest pairs and six failed negative ledgers. No null class is added by normalization. |
| Essentia maximum agreement | Agreement among candidate tick sequences | Useful agreement metadata, not independent acoustic evidence of rhythm presence. No implementation or weights imported. |
| BeatNet+ | Three frame classes, including non-beat | Off-tick is not beatless-region supervision. Published small weights exist, but no explicit grant was found in the inspected tree/README/setup. No acquisition under an assumed license. |
| BeatFCOS | Beat/downbeat interval scores, regression and leftness | No acquirable task checkpoint identified in the inspected tree, README or release listing. A paper and training code are not pretrained inference artifacts. |

### Essentia: agreement and licensing are separate questions

The [official maximum-agreement documentation](https://essentia.upf.edu/reference/std_TempoTapMaxAgreement.html)
defines its input as alternative tick lists and its score as mutual agreement.
The [multi-feature tracker](https://essentia.upf.edu/reference/std_BeatTrackerMultiFeature.html)
combines several onset functions and documents a confidence range of roughly
0 to 5.32, with dataset-derived quality guidance. Dividing that value by its
maximum does not turn it into a probability that arbitrary input has a beat.

Identical tick lists do not reveal whether their source was music, a shared
tracking prior, or another cause. This is an input-information limit, **not** an
experiment showing that Essentia emits beats on our noise fixture. The audit's
generic 40-bin Shannon concentration reaches `log2(40) = 5.321928...` for one
occupied bin and remains unchanged when counts are multiplied. It is not a
ported or parity-tested implementation of TempoTapMaxAgreement. It establishes
only why a concentration scale or duplicated agreement cannot certify acoustic
cause or independent evidence.

[Publisher licensing guidance](https://essentia.upf.edu/licensing_information.html)
separates the library, pretrained models and transitive dependencies. Preserve
that separation: an AGPL library is not categorically prohibited from commercial
use by AGPL itself, while the listed model grant is CC-BY-NC-ND and a separate
commercial option exists. No integration/license decision or legal clearance
is made here; source-level semantics already rule out a direct presence-score
substitution. Generic algebra below copies no Essentia implementation.

### BeatNet+: a normalized third class still has the wrong target

At revision `bb90eb0a9065b101a4b4c4cb2b2061950266cb4b`, the
[README](https://github.com/mjhydri/BeatNet-Plus/blob/bb90eb0a9065b101a4b4c4cb2b2061950266cb4b/README.md)
describes beat/downbeat/non-beat targets and class weights `[60, 200, 1]`.
The [model](https://github.com/mjhydri/BeatNet-Plus/blob/bb90eb0a9065b101a4b4c4cb2b2061950266cb4b/src/BeatNetPlus/model.py)
uses weighted cross entropy; inference applies a frame softmax. The inspected
training code also uses auxiliary/teacher feature matching. Consequently even
the simple population weighted-CE correction is not a calibration certificate
for these actual finite, regularized models.

An ideal 100-frame label sequence with four ticks can contain 96 non-beat frames
while representing a perfectly regular clock. Rearranging those same four ticks
changes the timing but not the three class counts. Our exact authored witness
retains both sequences. It does not label the irregular sequence as beatless;
it proves that pooled class counts discard temporal organization. Nor does this
prove the **complete** frame trajectory is useless.

For an ideal pointwise weighted-CE population optimum only,
`q_c = w_c p_c / sum_j(w_j p_j)`; division by weights and renormalization recovers
the frame posterior `p`. The audit checks this identity, zeros, invalid values
and finite large weights. The recovered target is still a metrical frame class,
not region-level rhythmic support or a normalized observation density.

Three approximately 3 MB weight objects are listed in the source record by Git
blob ID and size, **not downloaded or hash-verified payloads**. GitHub license
metadata was null; no license file or grant was found in the reviewed surfaces.
This is an unresolved prerequisite, not a categorical legal judgment. Do not
inherit an original BeatNet license onto this separate repository by assumption.

### BeatFCOS: promising output shape, incomplete reuse artifact

The [paper](https://arxiv.org/html/2510.14391v1) describes temporal beat objects,
classification/regression/leftness heads and Soft-NMS. A detection score is not
automatically a calibrated beatless-region probability. Its task could still be
useful under a separately justified aggregation and empirical gate.

The inspected [repository revision](https://github.com/zaiisao/BeatFCOS/tree/82f095c6bc78605abedfe4ec2e473f79bfd97818)
has Apache-2.0 repository metadata, an empty pretrained-model README section,
no task checkpoint in the reviewed tree, and an empty release list on this date.
The [model source](https://github.com/zaiisao/BeatFCOS/blob/82f095c6bc78605abedfe4ec2e473f79bfd97818/beatfcos/model_module.py)
refers to local WaveBeat backbone paths. Those paths are not downloadable trained
BeatFCOS weights. Paper CC-BY-4.0 licensing does not grant rights to absent weight
artifacts or clear transitive backbone provenance.

There is also no basis for silently merging paper and code settings: the source
forward default has score threshold `0.05`, while the paper's final Soft-NMS
discussion uses `0.2`. These identify different described configurations, not a
measured reproduction bug. No model construction, dependency install, threshold
choice or download follows from this screen. This is a finding about inspected
surfaces, not a claim that no checkpoint exists anywhere.

## Preserve the old statistical limits

The audit also keeps a two-family example with equal signal likelihoods and a
99.5% family prior. Its posterior remains 99.5% without new acoustic information.
This restates the existing prior-only counterexample; it is not a new successful
confidence model. The [background-copy/dropout formulation](dropout-likelihood-v1.md),
[rank/family preference](rank-clock-v1.md) and [presence-density reference](presence-likelihood-v1.md)
retain their original failures. None is restarted under a new name.

## Frozen necessary admission gate: seven positives, four negatives

Retain **all** six previous temporal PCM identities and **all** five later
negative/robustness identities: constant-clean, constant-omitted, constant-weak,
step-fast, step-slow, silence, quiet-gain, short-rest, steady-tone, DC and noise.
Their authored rhythmic/nonrhythmic roles are known; this is not a new blind
holdout. No new seed, gain, waveform, case substitution or model feature is used.

A future producer receives complete 12.01-second, 24 kHz PCM and the same physical
`[4, 8)` second query window. It receives no fixture ID, expected verdict, beat
annotation, oracle BPM/phase/change point or supplied clock family. In contrast
to the old relative-score test, rejection cannot depend on choosing the fast or
slow oracle family. Processing must preserve complete-input context rather than
infer a crop as if it were the original recording.

The three outcomes are:

- `supported`: evidence supports rhythmic timing in this query window;
- `not_supported`: the fixed rule does not support assigning rhythmic timing;
- `unknown`: evidence is missing, invalid or insufficient to decide.

None means a calibrated probability or a philosophical assertion of musical
beatlessness. Missing/invalid pipeline output should remain unknown or an error,
not be converted into a convenient negative. A 0.5-second rest in a larger
rhythmic window must not be treated as a tempo change.

All seven rhythmic controls must be supported; all four nonrhythmic controls must
be not-supported. Unknown remains in the denominator but does not pass. The
evaluator rejects missing, duplicate, extra or changed PCM identities and invalid
status fields. All-support, all-reject and all-unknown predictors fail. These are
tests of the evaluator, **not** outcomes of a real producer. Schema validation
cannot prove a producer did not access labels: a separately reviewed execution
adapter and fixed producer artifact/rule are required before any empirical run.

This necessary gate cannot select BPM, resolve density, validate abrupt/gradual
change or establish music accuracy. Preserve every old shape/density verdict,
the seven exposed music pairs, all forty calibration identities, untyped RUBATO
and sealed holdout. Passing rejection alone cannot erase positive regressions.

## Next decision, with a finite boundary

No direct pretrained presence producer qualifies from this **bounded** inventory.
The empirical admission gate is explicitly `not_run`, with no selected producer
and no pass percentage. Do not confuse that with an 0/11 measured model failure.

The next deliverable should be a **small presence/abstention-component feasibility
proposal**: define the actual target and observation domain, needed independent
positive/negative labels and rights, explicit null/background reference, temporal
dependence assumptions, runtime and acceptance tests. Distinguish a fixed analytic
rule from fitting a readout on pretrained features; the latter is training even
when the encoder is frozen. Retain a genuinely specified, feasible training-free
alternative if one exists; do not launch another open-ended checkpoint, threshold,
layer or context search merely because this screen found no drop-in head.

That proposal is a practical investment decision under
[`TRAINING-DECISION.md`](../../docs/TRAINING-DECISION.md), not permission to train.
No training, new model, holdout access, default/API change, user option or release
is authorized by this report. The product remains one-call, zero-tuning timing
analysis within the optional audio-metadata capability-pack architecture.

## Reproduction

```text
python -m unittest discover -s evaluation/parity -p test_rhythm_presence_audit.py -v
python evaluation/parity/rhythm_presence_audit.py --check evaluation/parity/rhythm-presence-audit-v1.json
```

The audit is standard-library-only and offline. New reports require a fresh
`--output` path; existing files are never overwritten. Ordinary CI includes the
contract/semantic checks and exact retained-report reconstruction. No heavy
inference or source download belongs in this test suite.
Illustrative logarithmic values are serialized to twelve decimal places to
avoid platform-libm last-bit churn; this formatting is not a confidence threshold
and does not participate in any categorical gate decision.
