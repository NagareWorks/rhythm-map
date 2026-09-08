# Oracle-period paired count/sign gate v1

**Do not promote or further tune this count/sign selector.** With supplied
nominal periods and all four shared prefix phases, all seven pairs pass the
registered clock-geometry guard. Nevertheless, only **2/7 raw** and **1/7 candidate**
pairs support the correct constant/step shape in both windows at every phase.
Neither source resolves all registered half/double competitors in any pair.
These are diagnostic gate counts, **not accuracy rates** or a proof that every
feature of the existing model is insufficient.

## What was fixed before replay

The [contract](../parity/oracle-period-paired-lock-v1.json) and script were frozen
after 15 authored controls passed and before the first new real geometry/capture
replay. No rule changed after that run. The original seven four-second pairs,
forty input identities, source extraction, half-open windows, three-frame radius,
rational one-to-one assignment and resource limits remain. All prior frozen
reports and helpers are unchanged.

This is explicitly **oracle-assisted**: the nominal BPM before and after the
selected change is supplied to both members of the pair. It does not estimate
period from four noisy prefix beats or claim to locate an unknown change.
The annotated change's elapsed position in its window is transferred unchanged
to the constant window as a counterfactual change location.

For each window's four prior annotations `t_j`, oldest to newest, let
`a_j = t_j + (3-j)P0`, where `P0` is the nominal pre-period. Each `a_j` represents
the same last-prefix beat ordinal zero. The constant and step hypotheses share
that anchor, pre-period and boundary. The step's phase accumulates continuously:

```text
u(t) = (t-a_j)/P0                                      when t <= c
u(t) = (c-a_j)/P0 + (t-c)/P1                            when t > c
```

Clock ticks are the inverse image of integers (native), even/odd integers (both
half phases), or half-integers (double). A step is not forced to restart its
phase at the boundary. All four anchors and all eight shape/density clocks are
retained. There is no best-phase search, independent fitting of competing clocks,
phase averaging or new user parameter. These four conditions are a finite
sensitivity check, not a calibrated interval covering all possible phases.

Before interpreting a pair, its correct native hypothesis must stay within the
unchanged inclusive 60ms radius of every full-query annotation ordinal, and vice
versa, in both windows at all four phases. Same-ordinal checks do not wrap a cycle
error onto a nearby beat. Geometry failures would remain in the report without
entering discrimination conclusions; a resource/support failure would suppress
both members' ledgers instead of retaining an orphan comparison.

## Count/sign evidence and its limits

The existing rational ledger still maximizes cardinality, then minimizes exact
distance. It retains all optimal assignments and accounts for every owned packet.
For each clock we retain matches, unmatched ticks and unmatched responses, plus
conservative bounds on the number of matched positive packets for each head.
Always-matched positives give the lower bound; possibly-matched positives give
the upper bound. These bounds need not be jointly attainable extrema and are
not likelihoods. All 896 real ledgers happen to have unique optima; authored
ambiguity tests still exercise non-degenerate bounds.

A clock dominates another only if it guarantees no fewer matches, no more
unmatched ticks, and no fewer matched positive beat-head **and** downbeat-head
packets, with at least one strict axis. Incomparable or assignment-uncertain
results remain unresolved. No weights are fitted. Negative logits are not
calibrated absence evidence, and the ARTBeaT downbeat placeholders are not
downbeat-negative labels.

For native constant versus step, a packet-level cross-tab also retains pre/post
boundary location, full-query/boundary status, membership in each clock's optimal
assignments, and both head signs together. It shows competing evidence without
multiplying the heads as independent probabilities. This is coarse temporal
context, **not** the complete continuous frame sequence or the encoder's learned
representation. Boundary strata remain visible; whole-window dominance is not
by itself an interior-only causal result.

## Frozen result

All seven pairs are eligible and geometry-compatible, with no replacement.
The same 91 raw events and 340 candidates are reused across phase/clock
conditions, not counted as hundreds of independent examples. Correct-clock
maximum annotation error is 34.166ms in constant windows and 42.08ms in change
windows. Both are below 60ms; this does not make remaining phase effects vanish.
Maximum ledger state count is 1,089, below the unchanged 16,641-state cap.

Each cell below counts phases (out of four) where **both** pair members support
the correct native shape against the opposite native shape.

| ARTBeaT case | Raw shape support | Candidate shape support | Raw / candidate density-resolved phases |
| --- | ---: | ---: | ---: |
| 06: 150 to 75 | 0/4 | 0/4 | 0/4 / 0/4 |
| 08: 112.5 to 75 | 0/4 | 0/4 | 0/4 / 0/4 |
| 09: 90 to 80 | 4/4 | 0/4 | 0/4 / 0/4 |
| 10: 90 to 120 | 4/4 | 2/4 | 0/4 / 0/4 |
| 12: 80 to 150 | 0/4 | 4/4 | 0/4 / 0/4 |
| 14: 240 to 96 | 0/4 | 0/4 | 0/4 / 0/4 |
| 15: 85 to 127.5 | 0/4 | 0/4 | 0/4 / 0/4 |

The [complete report](../parity/oracle-period-paired-v1.json) preserves every
phase and clock. Three particularly direct counterexamples are retained:

- Case 06 raw evidence favors the step in **both** the true step and constant
  windows at every phase: it does not distinguish the two contexts.
- Case 08 raw and candidate evidence favors constant in **both** windows at
  every phase: adding candidates does not recover the actual step here.
- Case 10 candidates pass only two phase conditions. Choosing those conditions
  after looking at their outcomes would hide the registered sensitivity failure.

Candidate expansion helps case 12 but loses the all-phase support seen in raw
cases 09 and 10. This is not justification for exposing source strategies or
tuning a selector on these seven tracks. Density-resolved means dominating all
seven competing shape/density clocks in both members at all four phases; zero
such pairs means that strong criterion was not met, not that all clocks are
identical or all evidence is absent.

## Decision after the two gates

Gate 1 found a misleading prefix period reference. This oracle-period experiment
passes its geometry guard yet still fails the registered count/sign test.
Therefore **close the count/sign tuning branch**: do not add weights, thresholds,
per-track source switches, a favorable phase, or another local smoothing rule to
claim this comparison has been solved. Production defaults remain unchanged.

The result does **not** establish that every training-free algorithm must fail:
oracle inputs, seven calibration tracks, finite phase conditions and lossy sign
summaries cannot prove that. The next distinct evidence question is whether
existing continuous head trajectories or a reusable pretrained representation
retain information discarded by these summaries. That work needs a new explicit
representation contract and evidence of useful separation; it is not permission
to keep tuning this rejected selector. If new evidence is proposed, first show
what it adds and validate any positive rule on independent labeled examples.
Training/model adaptation remains an option to justify, not an automatic verdict.

The one-call, zero-tuning product contract and optional future audio-metadata
capability packs are unchanged. No release, production decoder change, inference,
fitted mapping, holdout audio/annotation/capture evaluation or training occurred.

## Reproduction

```bash
python evaluation/parity/oracle_period_paired_audit.py \
  --artbeat-evidence /data/candidate-evidence-v1.private.json \
  --artbeat-captures /data/dense-artbeat-v1 \
  --output /data/oracle-period-paired-v1.json
python -m unittest discover -s evaluation/parity -p test_oracle_period_paired.py -v
```

Output refuses overwrite. Public CI tests use authored packets, immutable public
artifacts and the annotation plan; full replay requires the existing private
captures and verifies their identities before use. No raw packet arrays, absolute
event coordinates, logits or private paths are published. The first replay
completed locally in under five seconds without running a model.

SHA-256:

- Script: `2086a72728279375158cad5f73178f9e093afd4541d7ce2c36034d6b3ff22732`
- Lock: `c92cbb9fdc2659e00dfa4c0e460d934a1e9967336ea16615cd509d4e7173c635`
- Report: `f2666efba8c9c048143d13a4dd47a4cecbdcf64fc55b13e12dbbd0e058469e49`
- Plan: `5c15e677d0277dcac111aaebfb2be0b788996cbd44019a315c171fcf39bfe4f2`
