# Frozen clock proposal misses: geometry diagnosis v1

The old **484 covered / 2,690 missed / 174 reference-unavailable** results are
unchanged. This is a reference-assisted decomposition, **not an accuracy gain,
new generator, trained readout or proof that training is necessary**.

## Fixed protocol and identities

After 22 authored geometry controls, the [diagnostic lock](../parity/clock-proposal-diagnosis-lock-v1.json)
pinned the protocol, code and private archive bytes before diagnostic results
were inspected. This follows an already-observed development baseline; it is
not an independent experiment. All forty inputs and 3,348 four-second queries
from [proposal v1](clock-proposal-v1.md) were retained. Capture, annotation,
projection, query, candidate and old result identities are checked during replay.
No new clock is generated, shifted or selected. No new audio, listening, neural
inference, training, holdout access or production/API changes occurred.

The [retained report](../parity/clock-proposal-diagnosis-v1.json) has SHA256
`0806f2725d842bd9ee421dc1bb4c8cc9ffaf1565de44b2ef20ef73363355d5fa`.
Detailed coordinates remain private. Replay took 16.18 seconds including input
reads, validation and diagnosis, not audio/model processing or a runtime promise.

## Partition of the 2,690 original misses

| First applicable diagnostic | Queries | Meaning |
| --- | ---: | --- |
| No candidates | 46 | Reference is available but the old generator produced no clock. |
| Query-boundary-only witness | 38 | Unshifted ticks match across a crop boundary with no unmatched central ticks under the fixed halo alignment. |
| Equal-count offset-compatible | 427 | At least one ordered equal-count candidate has residual span at most 120 ms. |
| Equal-count nonuniform error | 1,179 | Some candidate has the right tick count, but none has either preceding witness. |
| No count-matched candidate | 1,000 | Every existing candidate has the wrong tick count. |

These partitions use the stated precedence; they are not disjoint causal
mechanisms. A query in an earlier row may also have a nonuniform candidate.
No original or halo tick-budget failures occurred. Another 135 of the old
181 empty-proposal queries have unavailable reference and stay in that category,
not the 46 assessed misses. Original hits never become misses in this diagnosis.

The halo is exactly the existing 60 ms tolerance, clipped to each clock's
physical support, with half-open ownership by the original query. Chronological
one-to-one matching gives a constructive witness, not a least-error fit or proof
that another alignment cannot exist. A crop artifact is not necessarily an
audio-track edge error. There is no edge extrapolation or change to old scoring.

Offset-compatible means the ordered residual intervals share a possible
translation tolerance interval. No offset is chosen or applied; translation can
change which ticks belong to the crop, so these 427 are **not recovered coverage**.
A nonuniform residual is not necessarily a tempo drift or true change: it can
also reflect incorrect event correspondence, local timing noise or beat level.

## What the step slice reveals

The 24 step queries still contain five original hits and nineteen misses:
seventeen misses have no count-matched candidate, one is offset-compatible and
one has nonuniform equal-count error. None has a boundary-only witness. Boundary
handling alone therefore has no demonstrated rescue in this slice.

Of the nineteen misses, eighteen have at least one unmatched reference tick in
the raw-event halo alignment; twelve also have unmatched raw events. Fourteen
queries have existing model local-max peaks near every missing reference tick.
Across overlapping queries, 50 of 54 missing-reference occurrences have such
peaks and four do not. Seven queries have different reference-index advances
between matched raw endpoints. These flags overlap and are **not independent
trials, unique-event counts or newly established perceptual beat labels**.

This points to a concrete hypothesis: the raw-event-only constructions can
discard useful existing peak evidence and cannot express changing advancement
between retained anchors by globally halving/doubling a whole path. It does not
show that selecting all peaks is correct: extra raw events coexist, and the
missing-peak inventory is separate from retained-event assignments. Joint
selection, phase continuity and false-change control remain untested here.

The expressive/untyped slice prevents overgeneralizing the step result. Among
its 2,654 misses, 1,836 queries have a missing raw reference anchor with no nearby
candidate peak. Across all failed slices, uniform subdivision between matched
raw endpoints places 189 of 365 interior reference occurrences outside 60 ms,
in 108 queries (all expressive/untyped). Reference-assisted subdivision is not
an implemented generator; even that geometry does not justify filling all gaps
uniformly. Full-trace endpoint alignment and local halo inventory use different
domains, so their counts are not interchangeable recovery bounds.

## Next bounded work and stop conditions

Do not train a scorer over this insufficient candidate set or retune the old
phase/threshold/likelihood rules. First specify **one peak-supported clock
proposal construction with locally variable advancement**, including false
event omission, missing evidence and support boundaries. It must differ
materially from the previously rejected top-K event paths: produce continuous
clocks, not another ranking of the same event sequences or global BPM summaries.

Before any measured replay, fix its observation-only input allowlist, joint
assignment semantics, positive/continuous phase geometry, deterministic budget
and deduplication. Retain the existing eight-candidate ceiling and all empty/
overflow queries; no hidden favorable-prefix pruning. Cost construction and
state/transition storage explicitly before implementing or running it. Reuse
the old full-duration development grid, never query labels or model confidence
as calibrated probabilities. Label-assisted diagnostics above are not its inputs.

The contract must explain how it can include a supported path through weak or
omitted attacks without erasing true changes, and how it leaves genuinely
missing acoustic evidence unresolved. If this cannot be done within the fixed
budget, record that construction limit instead of enlarging the search or
starting training implicitly. Even successful development coverage would still
need separate independent listening and product admission, not default promotion.

## Reproduction

```sh
python evaluation/parity/clock_proposal_diagnosis_audit.py \
  --prior-private <corrected-private-proposal-v1> \
  --artbeat-captures <private-artbeat-captures> \
  --rubato-captures <private-rubato-captures> \
  --output <new-summary-report> --private-output <new-private-diagnostic-report>
```

The driver refuses existing outputs, validates all original code/data/result
identities and asserts that every old candidate and score remains unchanged.
Compare replay fields except measured `elapsed_s`; private content identity
and all counts must agree. CI uses authored controls, byte-locked source/report
identities and per-input/per-slice conservation tests, not private audio.
