# Full-span clock/response ledger v1

## Decision

A bounded exact ledger now accounts for complete supplied clocks and all
supplied responses on one time span. Unlike five-point templates, a half-time
subset of an ideal constant response train leaves responses unexplained. A
single response cannot support two ticks. All equally optimal assignments are
retained through exact counts; one selected traceback is not treated as unique.

This is **accounting, not a likelihood adapter or tempo selector**. It preserves
constant-with-omissions / slowdown ambiguity and exposes a weak-subdivision
failure: counting all responses equally can make a double clock cover more
than the intended constant clock. No default algorithm, detected Beat output,
user strategy, model, dependency, holdout or training changed. No real music or
frozen audio capture was read by this audit.

## Contract and boundaries

The [locked contract](../parity/clock-response-ledger-lock-v1.json) and
[reference](../parity/clock_response_ledger_audit.py) use integer frames at 50 Hz
and one shared half-open domain `[0, frame_count)`.

The input consists of an explicit per-frame availability mask, ordered unique
response centers with paired finite beat/downbeat logit marks, and a supplied
clock. The renderer includes every clock tick in the domain, including edge
ticks. Period changes must fall on scheduled ticks; the new period determines
the interval following the change tick. The five clock specifications in the
report are authored candidates, not discovered tempos or user parameters.

Responses are supplied packets, not raw frames and not verified acoustic
events. This tool does not threshold logits, cluster peaks, infer downbeats,
fit reliability, or assert that its packets reproduce a backend's outputs.
Marks are validated but do not enter the matching objective. Missing metrical
annotations are not negative labels and no truth label enters the solver.

For each clock, match ticks and responses chronologically, one-to-one, within
three frames. The **inventory objective** first maximizes the number of matched
responses, then minimizes their total absolute frame offset. It is not a
calibrated observation cost, and its optimum is not a musical judgment.
Unmatched clock ticks receive no invented omission penalty. Neither unmatched
responses nor low logits are cast as acoustic absence likelihoods.

Every result includes:

- matched and unmatched tick/response totals and total timing residual;
- exact counts of distinct optimal assignments and of each pair's occurrence;
- whether each tick/response is matched in every, some, or no optimal assignment;
- one deterministic witness and its explicit unmatched indices;
- each tick's available query-frame count and full-window observability.

Outside-domain frames are unavailable, not padded silence. A partially
unavailable query may still match a known response, but remains marked partial.
An unmatched fully observed query means no assigned response, **not a proven
silent or absent beat**. Zero responses yields null complete-cover status,
rather than declaring every clock a successful tempo explanation.

The `complete_response_cover_clocks` list says only that an assignment covers
every supplied response. A clock with extra omitted ticks may also be on this
list. The list is neither a posterior-equivalence class nor a set of accepted
tempos: all results keep `selected_tempo=null` and emit no detected beats.

## Exactness and resource bounds

State `(i, j)` means the next tick index and first response still eligible for
matching. Its alternatives are: leave this tick unmatched, or match it to one
compatible response `k >= j` and continue at `(i+1, k+1)`. Skipped responses have
no separate transition. Thus each distinct chronological matching is counted
once; a grid DAG with independent tick/response skip paths would overcount it.

The recurrence retains all optimum transitions, not only one predecessor.
Forward prefix counts multiplied by suffix counts give exact tick, response
and pair occurrence counts. These are integer combinatorial counts, not
calibrated probabilities. A deterministic witness prefers matching, then the
nearest response, then the earlier response; the other optima remain visible.

Limits are 4096 frames, 128 ticks, 128 responses and 16,641 states. Exceeding
any limit fails without returning partial evidence. Unique integer response
centers bound each tick's compatibility list to seven entries, so the dynamic
program does not enumerate exponentially many tracebacks. The unit test with
64 independent two-way ambiguities counts all `2^64` optimal matchings exactly.

An independent combinations-based enumerator checks the objective, witness,
all pair counts and all tick/response counts on 80 fixed random small inputs.
Other controls verify unavailable padding/translation, unchanged accounting
under logit-mark changes, edge ticks, change semantics, invalid input rejection,
and a unique empty matching rather than duplicate skip paths.

## Fourteen authored controls; all failures retained

Each main control spans 148 frames. The family has a constant period 12 clock,
constant period 24 and 6 clocks, and period 12 changing at frame 64 to 24 or 6.
Every clock starts at phase 4. These integer fixtures are not real recordings.
All 70 case/clock ledgers, input marks, availability, optima and witnesses are
frozen in the [report](../parity/clock-response-ledger-v1.json).

| Authored responses | Constant clock | Other full-span evidence | Retained limitation |
| --- | --- | --- | --- |
| 12 constant responses | 12 matched, 0 unmatched | Half clock matches 6, leaves 6 responses | Double clock also covers all, with 12 unmatched ticks |
| Two missing middle responses | 10 matched, 2 unmatched ticks | Double also covers all | Omission count alone must not choose tempo |
| 9 slowdown responses | 9 matched, 3 unmatched ticks | Slowdown covers all with 0 unmatched ticks | Same packets also represent constant tail omissions |
| 19 speedup responses | 12 matched, 7 unmatched responses | Speedup and globally double clocks cover all | Same packets can mean prefix omissions on a fast clock |
| 12 strong plus 12 weak subdivision responses | 12 unmatched responses | Only double covers all 24 | Completeness would select subdivisions if used as a tempo rule |
| The same 24 responses, now all strong | Same matching inventory | Exactly unchanged from weak subdivision case | Marks still need a justified reliability interpretation |

The report also retains shared shifts, alternating jitter, all-weak constant
responses, explicit missing observation regions, an observed span with no
response, and a fully unobserved span. The two omission/tempo interpretation
pairs produce identical ledgers. Renaming fixtures or clocks cannot alter
matching. Complete-cover growth is never reported as beat recovery or accuracy.

Compared with the [shared-phase audit](shared-phase-context-v1.md), the genuine
advance is complete response ownership and visible residual density, not a
new winning strategy. Compared with the
[presence-likelihood reference](presence-likelihood-v1.md), this ledger supplies
no new normalized observation law. The
[Beat This loss-semantic limitations](beat-this-semantics-v1.md) still apply.

## Next gate

First connect the existing backend's **response identity and coverage** to this
ledger without changing extraction thresholds: distinguish the shipped raw
events from its existing candidate pool as diagnostic sources, not runtime
strategies. Establish how duplicate/displaced peaks map to unique packets and
preserve paired marks, source provenance and unobserved regions. Validate that
adapter before another real-cohort replay. Then classify residuals as missing
packets, extra packets, or assignment ambiguity; do not turn this count-based
inventory into a confidence model or fit an omission/noise penalty to these
fixtures. This work still does not establish that neural training is necessary.

## Reproduce

```sh
python evaluation/parity/clock_response_ledger_audit.py --output /data/reports/clock-response-ledger-new.json
python -m unittest discover -s evaluation/parity -p test_clock_response_ledger.py -v
```

The generator refuses overwrite. Twelve tests include all 14 report inputs and
70 complete reproductions. Earlier report bytes remain unchanged.

- Contract SHA-256: `559f0a6023abf8f7514663ed42c396dad16ec44d73873851ef9826198255d9c1`.
- Script SHA-256: `856e24197e5a52480ae4b1233c9b7b452a67be29bc01ebe7e6f6654f8d3ab141`.
- Report SHA-256: `c082d0f4b14b397866659492d2a96b7aed2317ec13ab7ef9652152aa8948e9c2`.
