# Pulse/accent omissions and emission equivalence, v1

## Decision and bounded scope

Explicit omission states now distinguish **latent clock ticks, model-inferred
pulse labels and unavailable observations**. A truth-free post-inference check
also finds different supplied clocks that explain exactly the same inferred
full-frame label assignment. It does not need the authored case's identity.

This is a semantics and inference prerequisite, not production beat recovery.
The seven unchanged main controls and three fixed background-shift controls
retain their intended family winners. However, the MAP assignment still inserts
eight modeled pulses in an observed flat middle, so these inferred labels must
**not** become detected Beat events. The default estimator, public API and user
parameters remain unchanged. No model weights, training, real-music replay,
holdout access or release are involved.

The four clocks, fixture-derived beat phase/tempo boundaries, twenty inputs,
rank features and clock-duration prior come unchanged from [rank-clock v1](rank-clock-v1.md).
This new diagnostic marginalizes a **single constant meter and initial phase**
per clock (meters 2..7, 27 possibilities). It does not retain the previous model's
changing-meter graph. A matched constant-meter **no-omission** baseline is
reported to isolate this boundary; both models pass the seven main controls.
Neither is an unrestricted tempo, phase or change-point decoder. These are
development fixtures, not a blind validation set or twenty recordings.

## Observation process

Each available latent tick has a Bernoulli pulse-retention variable with one
run-wide rate `q ~ Beta(1,1)`. A retained latent bar-start tick additionally has
an accent-retention variable with a separate `r ~ Beta(1,1)`. Both rates are
integrated exactly, not supplied by the user or fitted by a rate sweep. These
exchangeable, run-wide priors are **new modeling assumptions**, not learned
detector calibration or a structured missing-beat model.

| Inferred label | Meaning in this model | Count contribution |
| --- | --- | --- |
| null | Audio frame unavailable | No retention trial or emission |
| 0 | Available tick, pulse omitted | One pulse-retention failure |
| 1 | Retained pulse without emitted accent | B; also Z if at a latent bar start |
| 2 | Retained accented pulse, only at a latent bar start | B, Z and D |

An unaccented label at a latent bar start therefore does **not** require dropping
the beat. This is necessary to express simultaneous beat and accent omissions
in the constant/half-speed witness. Missing audio is not an observed omission;
the supplied latent clock still advances phase there, without a q/r trial.
All labels are inferred latent states, not directly measured detector outputs.

Let N be the number of available ticks in one supplied clock, B its retained
pulses, Z retained pulses landing on latent bar starts, and D emitted accents.
For one assignment its integrated omission prior is

```
Beta(B+1, N-B+1) * Beta(D+1, Z-D+1).
```

There is no extra binomial coefficient here: this is the probability of one
particular assignment. The polynomial coefficients below sum assignments.
The accent prior only sees retained latent bar starts, not ordinary ticks or
bar ticks whose whole pulse was omitted. Initial meter/phase mass is `1/(6*m)`.
Covered-time clock priors are unchanged; retention trials count ticks, a separate
modeling assumption that should not be confused with a time-exposure hazard.

The shared full-frame, paired-head reference is still charged once using counts
`A = B-D, D`. Its emission log ratio is

```
sum(b[t] at retained pulses) + sum(d[t] at emitted accents) - log Z[A,D].
```

Head pairs remain coupled in Z. Do not normalize the beat and accent heads
independently, add the old meter score, or treat unobserved frames as zeros.

## Exact factorization and output

Conditional on a constant meter/phase, available ticks split into ordinary and
latent-bar sets. Factor their assignment sums as

```
F[k]   = [u^k]     product_ordinary (1 + exp(b[t])*u)
G[z,d] = [v^z w^d] product_bar      (1 + exp(b[t])*v + exp(b[t]+d[t])*v*w).
```

For every k,z,d, combine `F[k]*G[z,d]` with the two Beta factors and the **joint**
normalizer `Z[k+z-d,d]`. Thus the factors are coupled at the terminal count;
they are not independent normalized likelihoods. Forward/backward polynomial
messages yield every tick's three label marginals. Separate max-product factors
and traceback yield the joint MAP assignment. Summing the 27 meter/phase
components gives count, retention-rate and label marginals per supplied clock.

For O ordinary and S bar ticks, this costs O(O^2 + S^3 + O*S^2) per component,
plus the inherited full-frame normalization table. It avoids enumerating all
omission masks, but does not establish an unrestricted whole-song budget.

## Automatic same-emission explanations

After selecting the joint MAP clock/meter/assignment, retain just its nonzero
frame labels. For every supplied clock, enumerate constant meters/phases whose
ticks contain all emitted pulses and whose bar starts contain all emitted
accents. Extra ticks may be omitted and extra accents may be suppressed.

Every compatible explanation has the **same full-frame labels, feature numerator
and paired normalizer**. Their relative weights inside this class consequently
contain only meter, clock and omission priors. No extra musical evidence is
created by preferring one of them. The report exports:

- the shared inferred frame-label assignment and its common feature score;
- which clocks are compatible and their prior-only conditional weights;
- whether more than one latent clock is compatible, without a tuned threshold;
- this assignment class's probability mass inside the full model.

This is an automatic **selected-assignment** diagnostic, not uncertainty over
every possible assignment. Structural compatibility is not equal plausibility,
and it does not force equal probabilities or global abstention. The assignment
may cover only a small fraction of the posterior. No authored label enters this
check, and no private music or case-specific rule is consulted.

## Results and retained limitations

On the true-half/both-erased identical inputs, the full model prefers half tempo
with probability **0.946643** for both. Their selected assignment has 40 pulses
and is compatible with **constant, half and double** clocks. It accounts for
**0.852094** of the model's mass. Conditional on that identical assignment, the
half-clock weight is **0.999999676**, entirely from priors. That is not 99.99997%
audio evidence that the musical clock slowed. All outputs of the two contradictory
authored cases remain exactly equal.

Weak true doubling retains all 64 modeled pulses and wins its supplied family
with probability **0.999997890**. Its selected assignment is compatible only with
the double template in this limited family, with class mass **0.802862**. This
is not a claim that alternate phases, subdivisions, changing meters or unlisted
clocks are excluded in music.

The constant erased-beat case still chooses 48 modeled pulses despite eight
deleted beat-head peaks. The flat-middle control likewise assigns **eight modeled
pulses inside frames 480..672**, all without local pulses in either input head;
its selected assignment has only **0.089330** posterior mass. Global context,
the observation score and omission priors can favor inferred continuation.
Neither MAP labels nor high retention marginals are an event-acceptance rule.
Keep inferred clock advancement separate from actually supported observations.

Flat and all-unavailable inputs have zero full-family log ratio and >99% prior-only
constant preference. Flat available ticks have 0.5 posterior omission probability;
unavailable ticks have null label probabilities and no trials. The fixed noise
draw selects an empty MAP assignment and has family log ratio **-3.031537**;
one draw is not a false-positive calibration. The empty assignment's mass differs
between flat (0.020425), noise (0.423396) and unavailable (1.0) evidence.

All three 1/4096 contrast-shrink controls remain exactly equal to their original
outputs. Rank order still does not measure audible strength or detector confidence.
This conditional feature-pair reference is not a generative audio likelihood.
No omission-rate, clock-prior, feature-window or amplitude-threshold fitting is
authorized by these controls, and the result is not a training verdict.

## Verification and next gate

`cargo run --locked --profile evaluation -p rhythm-map-eval --example omission_clock`
reproduces `evaluation/parity/omission-clock-v1.json`.
`python -m unittest discover -s evaluation/parity -p test_omission_clock.py -v`
independently reconstructs all 80 clock inferences, **2,160** meter/phase components,
label/count marginals, both integrated rate moments, MAP scores, matched baselines
and all same-assignment explanations. Python uses probability-space polynomial
messages with scaled terminals; Rust uses log space. Small Rust exhaustive paths
independently verify both Beta integrals, MAP, counts and every label marginal.
Invalid/duplicate clocks and labels, unavailable trials, flat inputs, unchanged
input identities and the inferred-flat-tick counterexample are regression tests.

Next combine these explicit omission semantics with temporal meter/clock search,
while preserving a separate supported-event acceptance/provenance boundary and
testing structured missing spans. Do not publish modeled pulses as detected beats
or mistake selected-class compatibility for full-posterior uncertainty. The
exchangeable omission assumption and inherited strength blindness remain open
gates; do not hide them with another user strategy, Beta-rate sweep or long cohort
replay before the bounded semantics and search integration are checked.
