# Jump occurrence, destination and evidence limits, v1

## Decision

The short failures cannot be attributed only to an excessive jump penalty.
The frozen count-conditioned feature reference also has a finite evidence
ceiling: strengthening an already best-ranked assignment does not give it
unbounded support. Even a path inserting a pulse on a flat frame can be a
best-ranked assignment **within its own larger count class**.

This audit diagnoses that limitation; it does not replace the shipping estimator
or claim improved music accuracy. No prior, stationary boundary, omission law,
feature pipeline, public API or user parameter is changed. All ten old controls
are retained. Two fixed feature gains and four additional context controls are
development stress tests, not selected calibration, holdout data or neural-model
outputs. No training, audio replay, package dependency or release is involved.

## Separate the two jump terms

For the frozen period atoms, the evaluation prior is

```
a[p,q] = exp(-ln(100) * abs(log2(p/q))), q != p
A[p]   = sum_{q != p} a[p,q]
lambda = mean_p log(1 + A[p]) / p
T[p,p] = exp(-lambda*p)
T[p,q] = (1 - exp(-lambda*p)) * a[p,q]/A[p], q != p.
```

The first off-diagonal factor is the chance of changing on that transition;
the second is the conditional destination probability **given a change**.
An octave jump's unnormalized affinity is 0.01, not its final transition
probability. On period atoms 3,4,5,6, the 3-to-6 occurrence chance is 0.196597
and destination probability is 0.052229. The two factors are distinct in the
score, but the construction couples them through A when deriving lambda.

The following uses abstract time units, not an asserted musical BPM:

| Period atoms | Frame duration | Exposure-rate parameter per time unit |
| --- | ---: | ---: |
| 3,4,5,6 | 1 | 0.0729663 |
| 6,8,10,12: same physical hypotheses | 0.5 | 0.0729663 |
| 6,7,8,9,10,11,12: added hypotheses | 0.5 | 0.1868519 |
| 3,4,5,6,7: wider support | 1 | 0.0812640 |

Changing units on the **same atoms** preserves every transition probability.
Adding intermediate hypotheses changes the physical exposure-rate parameter by
about 2.56 times; the first row's jump chance becomes 0.429108. This is a
domain-dependent prior, not a units bug or failed row normalization. It is not
automatically appropriate as a discretization of a fixed continuous-time law.
Destination probabilities are masses on atoms, not densities. Restricting and
renormalizing the refined destination distribution back to the old alternatives
recovers it exactly; the occurrence term does not thereby revert. The audit
does not change any inference domain or choose a preferred prior.

## Why greater feature strength saturates

Fix one path with a plain labels and d accent labels on n available frames.
Its paired reference averages over
`K = choose(n,a) * choose(n-a,d)` disjoint assignments with those same counts.
Let S be its uncentered feature sum and S_j each reference assignment's sum.
Multiplying every feature by a positive gain g gives the feature ratio

```
R(g) = K * exp(g*S) / sum_j exp(g*S_j).
```

Centering both feature heads cancels out of this expression. Let S_max be the
maximum reference score and M the number of assignments attaining it. Dividing
by `exp(g*S_max)` proves

- if `S < S_max`, the ratio vanishes as g tends to infinity;
- if `S = S_max`, the ratio tends to the finite ceiling `K/M`.

The generator computes S_max and M with an integer max-plus coefficient DP,
including exact integer tie counts. It does not extrapolate the limit from
finite gains. An independent exhaustive assignment enumeration verifies the
coefficient counts, maxima, multiplicities, and finite-gain reference ratios.
Constant feature offsets cancel; fully flat available features have ratio one,
as do unavailable inputs without emission trials. Flat and unavailable still
have different retention semantics in the clock model.

Both the authored and frozen stationary-MAP paths in each short failure attain
S_max, but in **different count classes**:

| Control / fixed path | Plain / accent count | K | M | Limiting log feature ratio |
| --- | ---: | ---: | ---: | ---: |
| Half / authored | 2 / 2 | 18,360 | 1 | 9.817930 |
| Half / frozen MAP | 2 / 1 | 2,448 | 6 | 6.011267 |
| Double / authored | 3 / 2 | 85,680 | 1 | 11.358375 |
| Double / frozen MAP | 3 / 3 | 371,280 | 52 | 8.873468 |

The double MAP's six pulses include the flat frame 4; its three accents include
an unaccented feature at frame 7. Because that count class requires six pulses
and three accents, even its best assignments must use such low-feature frames.
Strengthening the existing contrast cannot make this assignment suboptimal
within that class. This is a limitation of conditioning away the count, not
evidence that the network supplied a real event at frame 4.

Adding the unchanged clock, meter and retention terms, the authored-minus-frozen-
MAP limiting log weights are **-1.958590** (half) and **-0.808713** (double).
The authored paths lose at gains 1,2,4 and in this limit. These are two
**fixed-path** comparisons, not posterior odds between all tempo families, a
proof for every intermediate gain, or a full infinite-gain MAP search.

## Stronger contrast versus more observed context

All ten original 18-frame cases are inferred at gains 1,2,4 with the unchanged
period 3..6, meter 2..3 domain and stationary boundary. The half MAP stays at
constant period 6; the double MAP stays at constant period 3. At gain 4 the
double path changes its meter/omission labeling, so unchanged tempo failure
does not mean every label is invariant. Identical half and erased-constant
inputs remain identical under inference; gain supplies no missing information.

Four 27-frame controls add **observed** context at the original feature strength.
All have 125,844 exact states under the unchanged 250,000-state cap:

| Control | Authored tick frames | Result |
| --- | --- | --- |
| Constant | 1,4,7,10,13,16,19,22,25 | Authored path is MAP |
| Half | 1,4,7,10,16,22 | Authored 3-to-6 path is MAP |
| Early double | 1,7,13,16,19,22,25 | Constant period 3 wins by 0.214444 log weight |
| Late double | 1,7,13,19,22,25 | Constant period 6 wins by 1.912057 log weight |

The changed period is the outgoing interval at a tick, not a retroactive interval
label. Feature arrays are authored from these clocks; only the arrays and fixed
domain reach inference. Path comparisons happen afterwards. These controls are
not matched audio crops, nor statistically representative evidence of a general
slowdown/acceleration asymmetry. The positive half result establishes that this
exact search can select a change with more context; both acceleration failures
remain explicit. None of these inferred labels are accepted detected events.

## Reproduction and next gate

```
python evaluation/parity/jump_evidence_audit.py --output /data/new-jump-evidence.json
python -m unittest discover -s evaluation/parity -p test_jump_evidence.py -v
```

The generator refuses overwrite and hashes itself, the frozen boundary/factor
sources and report, and the Rust prior source. Nine tests reproduce the added
inferences, preserve all old baseline results, independently verify assignment
limits and unit/domain accounting, reconstruct fixed-path scores, and retain
the state-budget failure. No old artifact is rewritten.

Do not reduce a jump weight or increase a feature gain to fix these controls.
Next define one shared-frame observation law that preserves evidence about
absence and event counts, with explicit distinction between weak, flat and
unavailable input. It must pass matched constant/omission/change controls before
any product integration, scalable approximation or real-cohort replay. Simply
removing the reference normalizer is not a normalized replacement. The present
evidence identifies a scoring limitation, not a need to train a new model.
