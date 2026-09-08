# Frozen prefix-clock geometry audit v1

**Gate 1 result: the four-beat continuation is not an equivalent constant-clock
control.** Six of seven constant windows have same-ordinal annotation errors
beyond microsecond normalization; three exceed the unchanged 60ms radius even
where both clocks have complete interior queries. This is a concrete timing
confound, not evidence that model training is necessary or accuracy improved.

## Frozen scope and arithmetic

The [contract](../parity/prefix-clock-drift-lock-v1.json) and script were frozen
after 15 authored controls passed, before the first real annotation-geometry run.
The run used that unchanged contract. All seven original pairs, fourteen
four-second windows, and forty input identities remain. The complete earlier
annotation-selection report is reproduced and hash-checked; selected coordinates
must reproduce the original plan hash. No detector outcome selects or excludes
a window. Step windows are descriptive pre/post contrasts, not the gate's
decision population. Previous exposure and packet-replay artifacts are unchanged.

For prefix anchor `a`, its fourth-last beat `b`, and pre-segment nominal period
`P0 = 60,000,000 / BPM` in microseconds, the prefix period is `P = (a-b)/3`.
For annotation ordinal `n` after the anchor, predicted time is `a+nP`.
Correspondence follows beat ordinal, never nearest-neighbor matching or wrapping
by one period. Predictions outside the window are retained for error accounting.

With the first selected annotation `t1` at ordinal `n1`, the exact decomposition is:

```text
t_n - (a+nP) = E1 + (n-n1)(P0-P) + [t_n-t1-(n-n1)P0]
E1 = t1 - (a+n1*P)
```

The terms are first-beat alignment offset, accumulated relative prefix-period
drift, and relative annotation departure from the nominal period. `E1` already
includes pre-window accumulated drift: it is **not** a separately identified
physical phase. Nominal BPM is a supplied reference, not a fitted true clock.
Individual prefix interval deviations are also reported; they can cancel at the
endpoints without affecting extrapolation. Component extrema occur on potentially
different beats and must not be added as though they were one decomposition.

For independent nearest-microsecond normalization of each timestamp, the error
perturbation is bounded by `1+n/3` microseconds, from coefficients
`1, -(1+n/3), n/3`. This bounds only our numeric normalization. Original annotation
precision, performance jitter and musical timing truth are **not** established
by decimal digits or this bound. Exceeding it rules out normalization alone;
staying within it does not prove normalization caused the discrepancy.

## Constant-window result

Values below are milliseconds, rounded for display; the
[report](../parity/prefix-clock-drift-v1.json) retains exact rational quantities.
Positive period bias means the prefix period is longer than the nominal one.

| ARTBeaT case | Prefix period minus nominal | First-beat error | Max absolute extrapolation error | Max relative period drift | Max relative annotation departure |
| --- | ---: | ---: | ---: | ---: | ---: |
| 06: 150 to 75 | -9.444 | 3.611 | 95.832 | 75.555 | 16.666 |
| 08: 112.5 to 75 | 11.575 | -21.296 | 84.629 | 69.448 | 12.499 |
| 09: 90 to 80 | 1.852 | -7.407 | 8.332 | 7.408 | 20.140 |
| 10: 90 to 120 | 1.476 | -6.337 | 9.896 | 7.380 | 23.613 |
| 12: 80 to 150 | -2.344 | 6.251 | 14.845 | 9.376 | 4.688 |
| 14: 240 to 96 | -6.163 | 6.944 | 99.997 | 92.445 | 13.021 |
| 15: 85 to 127.5 | -0.000353 | 0 | 0 | 0.001765 | 0.001765 |

Only case 15 has identical grids. Across 54 constant annotations, 48 errors exceed
the normalization bound and 16 exceed 60ms. Fifteen of those 16 have full queries
on **both** the annotation and corresponding continuation tick: 4 in case 06,
4 in case 08, and 7 in case 14. This cannot be explained solely by query censoring
at the window edges. Those three cases are also the three raw annotation-clock
dominance cases in the [earlier replay](paired-response-replay-v1.md). That
cross-reference is an interpretation after the geometry run, not a selection
input or a causal detector ablation.

Half-open ownership differs in only one constant window: case 06 owns nine
annotation ordinals versus ten continuation ordinals. The extra continuation
ordinal's annotation is after the window. Cases 08 and 14 show large interior
errors without changing the total number of owned ticks. Authored controls show
the converse as well: a one-microsecond perturbation can change ownership while
remaining within the normalization envelope. Neither count difference nor count
equality is sufficient evidence of tempo change or clock equivalence.

The registered change contrasts contain 49 annotations; 21 same-ordinal errors
exceed 60ms, 16 with full queries on both clocks. Their pre/post strata are kept,
but ordinal divergence after a true step is expected and is not an event match,
change-detection score or independent validation sample. No step result drives
the constant-control gate.

## Decision and bounded next experiment

The gate rejects the interpretation **"annotation beats continuation, therefore
the model detected a tempo change"** under the current fixed four-beat reference.
It does not reject the entire training-free route. The magnitude and interior
location of the clock errors establish a nuisance that a valid discrimination
test must control; they do not identify which upstream source caused that timing.

Do not optimize the old count-dominance criterion, widen the 60ms radius, refit
phase against packet quality, replace the seven pairs or turn mixed outcomes into
public strategies. Keep this result as the first decision gate, not another
claimed production improvement.

The second gate must pre-register a shared phase/period nuisance treatment for
constant and step hypotheses, retain complete-clock density/omission accounting,
and test existing joint-head/context evidence. The same seven pairs can diagnose
the contract but cannot certify generalization: any positive rule then needs
new independent labeled examples. If meaningful separation still needs per-track
exceptions, conflicting thresholds, or trades recovered beats for fake changes,
stop extending this postprocessing-only route and evaluate additional pretrained
evidence or model adaptation. These observations do not yet prove training is
the only option. Normal use remains one-call and zero-tuning; optional future
audio-metadata capability packs are unaffected.

## Reproduction

```bash
python evaluation/parity/prefix_clock_drift_audit.py --output /data/prefix-clock-drift-v1.json
python -m unittest discover -s evaluation/parity -p test_prefix_clock_drift.py -v
```

Only checked-in calibration annotations, allowlisted source identity metadata,
and pinned public helpers are read. The new audit itself is Python-standard-library
only; it imports no model/capture adapter. Output refuses overwrite. No absolute
event coordinates, packet/logit arrays or private paths are published. There is
no inference, production decoder replay, fitting, holdout access, training, new
user parameter or automatic tempo selection.

SHA-256:

- Script: `9a1eb612285cf5fd3449d2fa989a4acb4a0833e4b31d45733ebf25893cefd331`
- Lock: `3261542bcd0add7f0b4bd91d506b4f1007fe5cc85b81aba787d969fde1d1f3cc`
- Report: `5c9445e4e00801a11fc92301b77fa38a29f0c604a96268a5c5c0e1963599761a`
- Plan: `5c15e677d0277dcac111aaebfb2be0b788996cbd44019a315c171fcf39bfe4f2`
