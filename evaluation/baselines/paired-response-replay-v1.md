# Frozen paired constant/step response replay v1

All **seven frozen pairs / fourteen four-second windows** replay successfully,
without replacement or a rejected pair. They own 91 default events and 340
unthresholded candidates. This is a truth-assisted residual inventory, **not
automatic change-detection accuracy** and not a new production decoder.

## Locked comparison

The [annotation-only selection](annotation-exposure-v1.md) is regenerated and its
complete report and selected-coordinate hash verified before any private input
read. All 15 ARTBeaT and 25 RUBATO identities remain visible; only the seven
selected ARTBeaT whole-recording captures are opened. Suite/evidence, capture,
truth, source implementation and original packet inventories are identity-checked.
RUBATO is still untyped for this discrete constant/step comparison.

The [replay contract](../parity/paired-response-replay-lock-v1.json) and script
were frozen after 14 authored/public-identity controls passed and before the first
private-capture replay. That run completed without changing the protocol.

Unchanged helpers retain original whole-capture packet lineage and paired marks,
half-open window ownership, the previous four-beat prefix clocks, both half
phases and double, exact rational one-to-one assignment, all optimal assignments,
and the three-frame query radius. Cropping never re-extracts detections. Any
member/source/clock budget or support rejection would suppress **both** windows'
ledger comparison, keeping the pair and owned-response denominators. Successful
orphan sides would not enter aggregates. The 128-tick/response and 16,641-state
limits are unchanged; real maximum state count is 1,089 and all 140 real ledgers
have a unique optimum. Authored ambiguity controls still preserve multiple optima.

Metrics keep matched responses, unmatched ticks and unmatched responses separate.
For each source, annotated/half/double clocks are contrasted with continuation by
two deltas: matched responses and unmatched ticks. `left_dominates` means no fewer
matches and no more unmatched ticks, with at least one strict; the converse is
`right_dominates`. Other relations are `equal` or `tradeoff`. These are descriptive
count relations, not a musical winner or calibrated likelihood. Per-track
constant/change relation cross-tabs are retained instead of hiding disagreement
inside one pooled score.

## Matched residuals and competing candidates

Each context contains seven windows and 28 seconds of selected audio exposure.
The pair windows do not overlap, but their annotation prefixes can; no independence
of the two sides is claimed. Counts below concern the supplied annotated clock.

| Context | Annotated ticks | Raw matched | Boundary-censored raw misses | Interior raw misses | Candidate-supported interior misses | Interior misses without a candidate | Off-annotation interior nonpositive candidates, never matched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant | 54 | 42 | 2 | 10 | 10 | 0 | 117 |
| Step | 49 | 36 | 1 | 12 | 10 | 2 | 102 |

All 20 candidate-supported interior misses have nonpositive candidate marks.
Candidate ledgers match 52/54 constant and 47/49 step ticks, but the same exposure
contains 219 off-annotation, interior, nonpositive candidates that never match
the annotated clock. These are competing background observations, not permission
to lower the default threshold or automatically insert those 20 missing beats.
Boundary censoring remains separate from interior miss causes. Full packet and
tick strata are retained in the [aggregate report](../parity/paired-response-replay-v1.json).

## Why counts do not yet discriminate changes

The raw annotated clock dominates prefix continuation in **3/7 constant windows**
as well as 4/7 step windows. With candidates, the corresponding counts are 2/7
and 4/7. Thus annotation-clock advantage is not change-specific. Only one of the
seven constant windows has exactly identical annotated and continuation grids;
constant segment labels do not guarantee identical phase/period extrapolation.
This observation identifies a confound to audit, not its cause or a fitted fix.

Below, A = annotated clock dominates continuation, C = continuation dominates,
E = equal counts, T = tradeoff. Neither A nor C is an automatic BPM decision.

| ARTBeaT case | Raw constant / step | Candidate constant / step |
| --- | --- | --- |
| 06: 150 to 75 | A / A | A / C |
| 08: 112.5 to 75 | A / C | E / A |
| 09: 90 to 80 | E / A | E / A |
| 10: 90 to 120 | E / A | E / A |
| 12: 80 to 150 | E / T | E / A |
| 14: 240 to 96 | A / A | A / T |
| 15: 85 to 127.5 | E / C | E / T |

Candidate expansion is not uniformly helpful: case 08 improves this descriptive
step relation, while case 06 reverses from A to C and case 14 loses dominance.
Do not convert those mixed results into user-facing strategies or tune a selector
on these seven pairs.

Density ambiguity also remains. On candidates, the doubled clock dominates
continuation in **4/7 constant windows** and 2/7 step windows. Pooled constant
candidate matches rise 47 to 92, while unmatched ticks rise 8 to 18; on step
windows these are 45 to 86 matches and 9 to 25 unmatched ticks. A pooled tradeoff
can coexist with per-track dominance. Both levels must remain visible. The two
half phases likewise remain separate, not selected by the better result.

## Next gate and product boundary

Before judging joint-head/context discrimination, audit the **prefix-clock
phase/period extrapolation error inside the already selected constant windows**
against their annotations. Distinguish annotation precision, prefix instability
and boundary effects from actual tempo change; do not simply widen matching
radius, refit phase against response quality, or select another window. Freeze
that diagnostic before inspecting further evidence and preserve this result.

Then compare candidate-supported omissions with their competing background using
the existing paired-head/context evidence under a justified clock contract. This
seven-track calibration sample cannot establish broad generalization or prove
that model training is necessary. Authored controls explicitly show identical
observations admitting both omission and true-change interpretations when the
supplied truth clock changes. Raw logits are still not absence likelihoods.

No model inference, training, production decoder replay, holdout access, fitted
mapping, default-output change, new public parameter or accuracy gain is claimed.
The production contract remains one-call and zero-tuning; optional future audio
metadata packs are unaffected. Temporary diagnostic Python does not add a
shipping dependency.

## Reproduction

```bash
python evaluation/parity/paired_response_replay_audit.py \
  --artbeat-evidence /data/candidate-evidence-v1.private.json \
  --artbeat-captures /data/dense-artbeat-v1 \
  --output /data/paired-response-replay-v1.json
python -m unittest discover -s evaluation/parity -p test_paired_response_replay.py -v
```

Output refuses overwrite. Public tests use authored packets and checked-in
aggregates only; full replay requires the existing private immutable captures.
No raw events, selected coordinates, frame arrays, logits or private paths are
published. The first replay took about three seconds locally, not a model run.

SHA-256:

- Script: `e750e5df4f2ff1769419875269e086e632e43c8cf1b87118e66c9d4562509617`
- Lock: `d40ba206367c0b4cf970e15f0f3790a063a704e71d59b6da59a44df776fc7aeb`
- Report: `57d3005924fd6cf64b8b65632e77e1b2baff9e162dc6e4f9471fee14d60312f3`
- Selected plan: `5c15e677d0277dcac111aaebfb2be0b788996cbd44019a315c171fcf39bfe4f2`
