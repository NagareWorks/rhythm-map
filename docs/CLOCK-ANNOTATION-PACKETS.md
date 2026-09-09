# Blinded clock packets and proposal coverage v1

Status: implemented **research contracts and one development coverage baseline**.
No new audio, listener records, rights clearance, learned weights, production
selection or public API is added. The [readout proposal](CLOCK-READOUT-PROPOSAL.md)
is not training-ready. The [measured baseline](../evaluation/baselines/clock-proposal-v1.md)
shows a proposal-generation prerequisite still needs work.

## Fixed observation-only proposal construction

`clock_proposals.generate(observations, query)` accepts exactly duration, ordered
raw beat times and disjoint valid time spans, plus a query interval. Extra source,
confidence, label or BPM fields are rejected. The capture adapter projects this
allowlist from byte-identified prior observations. It does not use dense scores,
truth tempos, change locations, source IDs or the original oracle-pair windows.

Every complete input uses four-second queries, two-second hops, and a retained
partial tail. Query count/position depends only on duration. For a query contained
in one valid span, require raw events on both sides; no extrapolation across a
gap or from a missing intro/outro. Use intervals intersecting the query plus a
126-frame/50 Hz context halo, capped at 128 intervals without decimation.

One fixed set contains two constructions:

1. A constant continuation using the median interval in that context, anchored
   to the raw event nearest the query midpoint (earlier event breaks a tie).
2. A phase-continuous path advancing one cycle per raw event and linearly
   interpolating cycles between adjacent events. Its tempo may step at events;
   it does not reset phase at a tempo change.

Each supplies native, two half-level phases, and double-level clocks. This gives
at most eight proposals before physical-clock deduplication. Whole-cycle phase
aliases and collinear redundant knots are equivalent; opposite half-level phases
are distinct. Compare piecewise-linear phase at all knots with numerical `1e-9`
cycle tolerance. The two constructions are necessary rival hypotheses in this
baseline, not selectable public strategies and not evidence either is correct.

Clocks carry absolute input seconds and unwrapped cycles at ordered knots. All
slopes must be positive and finite. `phi_radians = 2*pi*cycles`; interval period
is `delta_seconds / delta_cycles`. The representation does not establish accurate
ramp curvature or a musical interpretation between its knots.

Return `ready`, `empty`, `overflow`, `invalid_query_support`, `unbracketed_query`,
or `observation_budget_exceeded`. Overflow returns no favored prefix. Missing
candidates do not mean no rhythm. Audio/readout context availability is separate
from the availability of a clock's context; valid silence is not missing audio.
The original production beat timestamps are never changed by these latent paths.

## Two views, a private ledger, two independent records

`clock_annotation_packet.build(binding, provenance, observations, query, nonce)`
calls that generator internally, not a caller-supplied candidate list. It returns:

| Object | Contents | Recipient |
| --- | --- | --- |
| `first_view` | Exact PCM hash/sample rate/count and query bounds | Each listener before candidate overlays |
| `candidate_view` | Same complete input/query, first-view hash, opaque IDs and clock geometry in fixed pseudorandom order | Each listener after saving the first pass |
| `private_ledger` | Provenance/review records, input relation, nonce, generation/observation hashes and both view hashes | Auditor only; never distribute to listeners |

PCM identity denotes the entire declared analysis input, not just the query.
Listeners may navigate a preview but must have the same context as the encoder.
The private input relation distinguishes complete recordings, crops with source
intervals, and unknown relationships. A complete input starts at source zero;
known source intervals must match input duration. Crops are not natural edges.
No paths, corpus tags, scores, source identities, reference tempo or verdicts enter
the listener views. Auditors retain provenance instead of deleting it for blinding.

The private record requires work, recording, performance and creator/session
identities; shared assets and transform-family links; evidence hash; separate
training and redistribution review states; and encoder-overlap status. Unknown
identities need explicit unresolved catalog records, not invented independent
groups. These fields do not compute connected components or certify clearance.
`unreviewed` is valid for preparation, never authorization to train or distribute.

Use a fixed externally generated randomization nonce, independent of observations
or scores. Hash ordering gives the same opaque permutation for both listeners.
Content hashes bind PCM, query, geometry and presentation order. A changed view
invalidates its votes. `validate_bundle` checks the internal contract;
`verify_replay(bundle, observations)` additionally reconstructs it from the
supplied observations. Auditors must verify those observation bytes separately.

The first-pass record stores listener ID, first-view hash, context sufficiency,
`followable`/`not_followable`/`uncertain`, independently drawn accepted clocks and
transition uncertainty intervals. A followable judgement requires its own clock,
not a candidate ID copied from the second view. Second-pass records bind the
original first-pass content hash and candidate-view hash and judge every candidate
`supported`/`contradicted`/`uncertain`. Preserve originals; adjudication is new data.

`reconcile` reuses the conservative consensus helper: multiple positives are
allowed; disagreement/shared uncertainty has no binary target. Insufficient or
unknown context clears all binary targets even if overlay votes are known.
When both context-sufficient listeners independently follow timing but reject
all candidates (or none were generated), retain a `generator_miss`. Otherwise an
unresolved coverage judgement remains `unassessed`, not a negative rhythm label.

These checks **cannot prove** that people listened independently, did the passes
in order, avoided known source tags, provided honest provenance, or had legal
rights. A future annotation UI/coordinator must enforce view release and append-only
records, and the rights/split review must still happen. No such UI, coordinator,
listener recruitment or actual annotation packet delivery is implemented here.
The authored tests are contract controls, not independent listener data.

## What the current coverage audit means

The audit byte-verifies all forty old calibration captures, generates all queries
before loading annotations, freezes a candidate hash, then attaches labels and
checks the hash is unchanged. No neural inference, audio decoding or feature
extraction runs. Capture arrays/candidate geometry remain private; only identities
and count summaries are committed.

For each query with complete native annotation support and at least two ticks,
ask whether one proposal has **the same tick count with every ordered tick within
60 ms** of the native annotation grid. No phase shift is fitted during scoring.
This two-direction grid check is a mechanical development proxy, not independently
supported-clock coverage: alternate defensible beat levels may fail it. Missing
reference support is unassessed; absent proposals remain misses when references
are available. Candidate tick-budget failures are counted, not dropped queries.

The fixed grid has overlapping queries and long expressive inputs dominate its
pooled counts. Report per-input and per-slice denominators. These forty input
identities are not forty proven independent provenance groups; no confidence
interval or population accuracy is inferred from their query count.

Next: inspect the retained misses to separate missing anchors, local drift and
unrepresented change/phase paths before proposing a different bounded generator.
No larger candidate sweep or readout training follows automatically. Independent
labels, rights/split review and numerical acceptance budgets remain separate gates.
