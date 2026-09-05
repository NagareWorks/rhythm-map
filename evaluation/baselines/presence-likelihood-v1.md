# Common-frame presence likelihood, v1

## Decision

The evaluation reference now accepts a normalized observation law that retains
evidence about both event presence and absence. All clocks explain the same
available frames, including frames without a tick. This removes the previous
count-conditioned evidence ceiling **without merely dropping a normalizer**.

The likelihood interface and probability accounting pass independent tests.
The illustrative Gaussian sensor does **not** pass the musical-selection gate:
on the original features it regresses the flat-middle omissions and the longer
half-time case. It is not a calibrated interpretation of Beat This outputs.
Neither the reference nor this sensor is promoted. Shipping Rust behavior,
detected events, APIs, user settings and package dependencies are unchanged.
No audio replay, training, holdout access or release occurs.

## Input contract: likelihoods, not class probabilities

For an available frame t the input has three entries:

```
[log f_absent(x_t), log f_plain(x_t), log f_accent(x_t)]
```

Each f is a class-conditional density (or probability mass) on the **same
observation space**, normalized over x. The entries do not need to sum to one
over classes; a softmax over these entries is not the observation model.
Arbitrary feature scores or detector class posteriors are not automatically
likelihoods. This short numerical reference can validate shape/finiteness,
but it cannot verify a producer's calibration from three numbers at one frame.

`None` means no observation: its likelihood integrates to one and it consumes
no retention trial. It is neither a zero feature vector nor evidence of absence.
All labels remain latent model states, not accepted Beat timestamps.

The clock root, tempo transitions, meter-wrap transitions, and integrated
`Beta(1,1)` pulse/accent retention rates are frozen from the stationary reference.
The same 32-frame and 250,000-state limits remain; exhaustion returns an error,
not a beam approximation or partial result. The new calculation uses log-space
sum-product, a separate max-product traceback, and reverse messages.

## One common observation domain

Let z_t be absent at every non-tick frame. At an available tick it is absent,
plain, or (only at a modeled bar start) accent, according to the unchanged
omission/mark prior. For a clock/label path H,

```
p(x_available, H) = p(H) * product_{available t} f_{z_t}(x_t)
B(x)            = product_{available t} f_absent(x_t)
relative(H)     = p(H) * product_{available tick t} f_{z_t}(x_t)/f_absent(x_t).
```

The full-frame background B is common to every hypothesis. Factoring it out
speeds up the graph but does not discard non-tick observations. A missed
positive observation loses its event likelihood ratio; an extra event on an
absence-supporting observation pays its negative log ratio. No count-specific
reference can cancel either effect.

At termination only the three original beta integrals remain. There is no
count-conditioned coefficient table and no extra survival charge. Since every
observation law is normalized and the clock/label prior sums to one, summing
over paths and integrating all observations gives one. Summing a missing
frame's possible observations also recovers the missing-frame calculation:
the marginalized emission choices integrate away before the beta priors.

`log_ratio` is log evidence relative to the all-absent observation model.
`log_evidence = log B + log_ratio` includes every available frame; for continuous
observations it is a density, not a probability or confidence percentage.
`joint_map_log_weight` excludes the common log B. Its normalized joint-path
probability is not evidence that a modeled tick is a detected musical event.

The new `emission_positions` rows contain absent/plain/accent marginals over
**all frames**. In particular absent includes both non-ticks and omitted ticks.
Unavailable frames have three null entries, not three zeros. Existing seven
clock/label/change marginals remain available separately.

## A fixed analytic sensor, not audio calibration

For the old authored pair `(b,d)`, the illustrative sensor uses independent
unit-variance Gaussian coordinates with class means `(0,0)`, `(1,0)`, `(1,1)`.
This is one fixed observation law with no fitted mean, variance, gain, or
threshold. It gives

```
log(f_plain/f_absent)  = b - 1/2
log(f_accent/f_absent) = b + d - 1.
```

These log ratios can grow positively or negatively with evidence. There is no
combinatorial ceiling tied to event counts. But the coordinate origin now has
meaning: `(0,0)` weakly favors absence, `(1/2,1/2)` is exactly neutral, and
`(-4,-4)` strongly favors absence. Treating every old zero as calibrated silence
would be unjustified. Constant offsets to feature heads are no longer free;
a common log-density scale applied to all three classes changes only absolute
evidence, as expected for a common coordinate Jacobian.

On the 18-frame sensor controls:

| Input | Relative log evidence | Posterior expected emitted pulses |
| --- | ---: | ---: |
| Neutral `(1/2,1/2)` | 0 | 1.986872, entirely prior-driven |
| Weak absence `(0,0)` | -0.826056 | 0.942778 |
| Strong absence `(-4,-4)` | -1.565253 | 0.008989 |
| Unavailable | 0 | Unreported, **not zero** |

Neutral and unavailable inputs both retain stationary latent tick marginals.
Strong absence still permits a latent clock but its MAP labels are all omitted.
No event-acceptance threshold is introduced.

## Retained selection failures and an explicit absence contrast

All ten old 18-frame arrays and four 27-frame context arrays are retained.
Original constants and phase shifts remain correct. The original half/double
controls both choose constant period 3. Flat-middle keeps period 3 but now emits
at both flat middle positions: an omission regression. The longer half case,
previously correct, now also chooses constant period 3, beating its authored
path by 1.091096 log weight. Both longer acceleration cases still fail.

The half authored path now beats the **old** stationary MAP by 0.234747 log
weight, yet a third, denser path beats both. A favorable fixed two-path
comparison therefore would have hidden the failed full search.

A separate, fully recorded input contrast maps **only paired zeros** to the
already-defined strong-absence coordinate `(-4,-4)`. Positive pulses with zero
accent coordinates remain unchanged; unavailable frames stay unavailable.
This is changed evidence, not a second sensor, decoder strategy, or calibrated
interpretation of the original features. The transform is applied to every
original/context case, not selected winners.

With this contrast, flat-middle omits frames 7 and 10 while retaining period 3,
and the longer half case recovers its authored 3-to-6 path. The short half still
selects a constant slow clock and misses a positive pulse; double still selects
constant period 3, although its unsupported frame-4 pulse becomes an omission.
Both longer acceleration failures remain, and the unaccented positive pulses
can still receive incorrect accent labels. Identical half/erased-constant
inputs remain identical. Explicit absence helps omissions but does not establish
a reliable tempo/meter selector.

## Verification and next gate

```
python evaluation/parity/presence_likelihood_audit.py --output /data/new-presence.json
python -m unittest discover -s evaluation/parity -p test_presence_likelihood.py -v
```

The generator refuses overwrite and hashes its frozen dependencies. Ten tests
cover all report runs, independent full-path enumeration, Gaussian integration,
all 81 observation sequences of a finite-channel witness, missing-observation
marginalization, unavailable padding, common-density-scale invariance, large
log ratios, score reconstruction, and failure-preserving input contrasts.
All earlier reports remain byte-identical. This is a bounded evaluation tool;
its exhaustive verification cost is not production audio-analysis latency.

Keep the normalized likelihood contract, not the illustrative sensor as an
audio adapter. Next inspect the existing backend's score/training-loss semantics
and already-frozen calibration evidence to determine whether presence/absence
likelihood ratios can be justified, including accent-versus-plain semantics and
class priors. That inspection must not silently cast logits/posteriors to
densities, fit this Gaussian to the known failures, or replay this decoder on a
real cohort before its observation adapter passes matched controls. The holdout
stays sealed. Current results do not establish a need for neural retraining.
