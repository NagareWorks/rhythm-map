# Observation-only clock proposals: development coverage v1

The [packet and generator contract](../../docs/CLOCK-ANNOTATION-PACKETS.md) is now
executable. This single frozen baseline **does not clear the training gate**.
It is neither a learned model nor a selected tempo output.

## Fixed execution

- All forty original calibration inputs: fifteen ARTBeaT, twenty-five RUBATO.
- Four-second queries, two-second hops, partial tails retained: **3,348 queries**.
- Raw-event-only constant continuation and continuous event-interpolation paths,
  each with native/two half-phase/double levels; at most eight unique proposals.
- All generation completed before annotation loading; content hash unchanged
  after comparison. No annotation-selected windows, phase/threshold search,
  selected winner, new data, neural inference, training or holdout access.
- The generation-stage loop took **5.98 seconds**, including capture reads and
  proposal construction, excluding later annotation loading/comparison. This is
  not audio-to-result runtime, a benchmark guarantee or a model measurement.

The [report](../parity/clock-proposal-v1.json) retains identities and per-input
counts. Its exact SHA256 is
`d17f36dd8f3ea3d4c9ccc46eb78c41c3ab9ee11d97f22e2869e5d714582411cf`.
The [lock](../parity/clock-proposal-lock-v1.json) pins implementations and numerical
rules. Private candidate coordinates are not redistributed.

## Results: native-grid proxy, not musical accuracy

| Post-hoc slice | All queries | Native grid covered | Native grid missed | Reference unavailable |
| --- | ---: | ---: | ---: | ---: |
| Constant | 51 | 31 | 10 | 10 |
| Step | 24 | 5 | 19 | 0 |
| Ramp | 16 | 7 | 7 | 2 |
| Expressive/untyped | 3,245 | 441 | 2,654 | 150 |
| Other/boundary | 12 | 0 | 0 | 12 |
| Total | **3,348** | **484** | **2,690** | **174** |

There are 3,167 ready queries and 181 without raw-event bracketing. Twenty-one
queries deduplicate to four candidates; 3,146 retain eight; 181 retain zero.
No candidate overflow, observation-budget or scoring-tick-budget failures occurred.
Audio/readout context is complete on 3,177 queries; complete clock context on
3,070. Absence of a proposal and absence of audio evidence are different states.

Covered means exact ordered tick-count agreement and at most 60 ms error for
every tick. It does not mean the selected tempo is right (there is no selection),
that all intermediate BPM values match, or that other metrical levels are wrong.
Long expressive recordings dominate the pooled count; overlapping queries are
not independent trials. No independent listener-supported coverage is measured.

## Interpretation and next boundary

Even supplied clocks were hard for the previous frozen readouts. This earlier
stage now also has a concrete limitation: this raw-event-derived candidate set
frequently fails to contain the reference timing path, including 19/24 step
queries. A candidate scorer cannot recover a missing path. The reference grid
can itself be only one defensible interpretation, so these are not 2,690 proven
perceptual errors and not evidence that every training-free generator must fail.

Do not tune a score or train the readout on these misses yet. First inspect the
retained failure geometry to distinguish missing anchors, drift and missing
change/phase constructions, and cost one explicit next generator proposal.
The blinded two-pass packet contract is ready for review, not a cleared data
inventory or collected listener labels. Keep defaults and user parameters unchanged.

Before publication a bookkeeping bug was corrected: unavailable candidates had
incorrectly suppressed the separate audio-context flag. The complete replay
changed 68 such flags but **no candidates, queries or match decisions**. Both
private runs are retained; the committed report is the corrected one. No algorithm
parameter was selected from these outcomes.

Reproduction, using previously byte-identified private captures:

```sh
python evaluation/parity/clock_proposal_audit.py \
  --artbeat-captures <private-artbeat-captures> \
  --rubato-captures <private-rubato-captures> \
  --private-output <new-private-candidate-report> --output <new-summary-report>
```

Reproduction never overwrites an output. Compare all fields except measured
`generation_elapsed_s`; code identities, generation hash and mechanical counts
must match. CI tests use authored controls and the retained public summary,
not private music or network downloads.
