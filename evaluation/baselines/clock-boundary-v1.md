# Matched clock-boundary audit, v1

## Decision

An arbitrary analysis window should not silently mean a newly started clock.
On the frozen [unknown-clock reference](search-omission-v1.md), replacing only
its initial law with a stationary-window law removes a measurable dependence on
unavailable prefix length. The feature pairs, full-frame reference, interior
tempo/meter transitions, omission priors, and terminal marginalization are all
unchanged. This is a boundary-consistency result, **not a tempo-accuracy fix**.

Half and double controls still fail. The half control's new MAP path additionally
misses the accent at frame 13. The change is therefore not promoted. The default
Rust estimator, APIs, user parameters and package dependencies are untouched.
This Python-only evaluation audit reuses the previous independent Python factors;
it introduces no Python dependency into the product. No audio, model training,
real-cohort replay, holdout access or release is involved.

## Matched intervention

The ten 18-frame feature arrays and the period 3..6 / meter 2..3 domains are
identical to the previous gate. Both boundary laws search all permitted paths
under the same 250,000-state limit. Neither receives authored clocks, labels or
change points. Fixed authored paths are scored separately **after** inference to
diagnose the two change failures. The frozen Rust sources and old report are
not rewritten; the fresh-start reconstruction matches their partitions, maxima
and every reported marginal.

The old initial law is `1/(P*p*M*m)` for a first tick at offset `0 <= r < p`,
period p, meter m and a specified bar phase. This is a valid normalized
fresh-start assumption. It is not stationary under moving the window origin.
Nor is a stationary-window assumption automatically appropriate for a known
musical onset: this audit explicitly concerns an arbitrary cut in an ongoing
latent process, not a claim that every song starts in statistical equilibrium.

## Stationary first-tick law

Let T be the unchanged row-stochastic tempo transition matrix and pi its
stationary distribution **at ticks**, satisfying `pi*T = pi`. Define
`mu = sum_p pi[p]*p`. At an arbitrary frame, a containing interval q is sampled
with length bias, but the first tick at or after the window origin may choose a
different **outgoing** period p. These are not interchangeable durations.

Marginalizing the unobserved containing interval gives

```
W(r,p) = sum_{q > r} pi[q] * T[q,p] / mu,  0 <= r < max_period.
```

Consequently `sum_{r,p} W(r,p) = 1`, and offsets `r >= p` can be valid: the window
can cut a long old interval that ends at a tick choosing a shorter new interval.
Simply multiplying the old initial period prior by p misses those states.
For the fixed domain, `mu = 4.529733402819359` frames. The stationary marginal
tick probability on no-data inputs is constant at `1/mu` throughout the window.

At bar wraps the symmetric meter-transition law has equal stationary mass for
each beat-in-bar state, for every nonzero change rate h. Its stationary initial
weight is `1/S`, where `S = sum_{m=min_meter}^{max_meter} m`, not `1/(M*m)`.
This law is independent of h, so the same run-wide `Beta(1,1)` integration remains
valid. A singleton meter domain retains its uniform phase law. The combined root
mass is `W(r,p)/S`. No extra terminal weight is added: unobserved future decisions
still marginalize to one.

The distinction between an ordinary interval and one sampled at an arbitrary
time is standard renewal background; see
[Yibi Huang's renewal-process lecture](https://galton.uchicago.edu/~yibi/teaching/stat317/2021/Lectures/Lecture14.pdf).
The Markov-period formula and changing-meter integration above are derived and
tested for this repository's model, not asserted by that lecture.

## Unavailable-padding test

Prepending three **unavailable** frames retains all feature pairs and their
paired normalizer. It adds no observation or pulse/accent-retention trial.
It is not adding measured silence. The following are absolute differences
between padded inference and the original inference on the same observed region:

| Control | Fresh log-ratio difference | Fresh maximum tick/label marginal difference | Stationary differences |
| --- | ---: | ---: | ---: |
| Constant | 0.035752 | 0.040330 | below 2e-14 |
| Half | 0.057578 | 0.046491 | below 2e-14 |
| Double | 0.124441 | 0.030795 | below 2e-14 |

Appending three unavailable frames preserves the original-region marginals and
log ratio under **both** initial laws, confirming that the existing terminal
marginalization is not missing a survival penalty. A further two-left/one-right
padding test on the independent small control also passes for the stationary law.

These are marginalization identities, not MAP-path invariance. A full joint MAP
path can change when unobserved prefix/suffix states are included in the joint
maximization. The first original tick also has no in-window predecessor, whereas
a padded path can have one: tempo/meter-change marginals are compared only after
`max_period` frames, when a predecessor must be inside both windows. Tick and
label marginals are compared over the entire shared observed region.

## What still fails, and where the cost comes from

The stationary MAP retains constant period 3 for the constant, phase-shift and
two missing-middle controls. It still picks a global slow path at frames 1,7,13
for both identical half/erased-constant inputs. It now selects meter 3 there,
emitting an ordinary pulse rather than an accent at frame 13. The double control
still gets a constant period-3 path and a modeled pulse at the flat pair at frame
4. Neither MAP labels nor these probabilities are detected Beat events.

Hold the authored path and the **old frozen MAP path** fixed to isolate scoring
changes. The table gives authored minus old-MAP log-score components; it is not
posterior odds between tempo families or a comparison to every new MAP path:

| Component | Half | Double |
| --- | ---: | ---: |
| Jump occurrence | -1.626599 | -1.036924 |
| Jump destination | -2.952117 | -3.625472 |
| Unchanged-period transitions | +0.218899 | +0.437798 |
| Fresh clock + meter initial term | +0.287682 | -1.098612 |
| Stationary clock + meter initial term | -0.083680 | +0.083680 |
| Feature numerator + paired reference | +2.373423 | +2.040818 |
| Fresh total | -1.921856 | -2.435095 |
| Stationary total | -2.293218 | -1.252803 |

Omission and meter-transition terms are also exported separately. For fixed
paths, **only** the two initial terms change; all other terms match exactly.
The dominant negative terms come from actually taking a jump and choosing its
destination, not from an omitted terminal charge. This does not prove those
priors are wrong: these are short, arbitrarily scaled feature controls, not
calibrated detector likelihoods. Boundary consistency alone neither fixes the
selection nor supplies evidence that neural training is necessary.

## Verification and next gate

```
python evaluation/parity/clock_boundary_audit.py --output /data/new-boundary-report.json
python -m unittest discover -s evaluation/parity -p test_clock_boundary.py -v
```

The generator refuses to overwrite an existing report. New report provenance
hashes the generator, frozen independent factors, and old Rust-generated report.
Seven tests cover matched reconstruction of all ten old runs, reproduction of
all new runs, traceback score decomposition, every marginal of **5,012**
exhaustively enumerated paths, and unavailable-padding identities. A separate
closed-form reversible-chain calculation checks pi against the numerical solve;
a frame-expanded chain checks stationarity at three fixed meter rates and across
singleton/full meter domains. These are algebraic tests, not rate fitting.

Keep the stationary-window boundary as a verified research prerequisite, not a
user strategy or product promotion. Next isolate jump occurrence versus jump
destination semantics under fixed boundary and feature assumptions, including
time-unit/domain accounting and stronger context controls. Do not tune the
distance scale or declare a training requirement from two short failures.
Scalable search and supported-event acceptance remain separate open gates.
