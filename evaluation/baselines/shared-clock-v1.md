# Shared-frame joint clock-family gate, v1

This is the next scoring prerequisite after [common-meter v1](common-meter-v1.md),
not an unrestricted clock decoder. Four **supplied** clock templates share the
same 1152-frame input: periods 24 throughout, or 48/12/32 in the middle section.
Their phase and section boundaries are fixture-derived. Meter paths (2..7),
initial phase, changes and visible downbeat count are marginalized without
meter truth. These truth-assisted clock candidates cannot establish recovered
beat F1 or automatic tempo-change discovery.

## A common frame domain and one paired reference

Previously each clock sampled a different bag of beat-cell scores. Such
conditional meter scores are not comparable between clocks. Here every
candidate uses the same observed frame pairs `(b[t], d[t])`, with no per-clock
windows or extra samples. Every available frame belongs to exactly one label:
neither, ordinary beat, or downbeat (which is also a beat).

For A ordinary beats and D downbeats the score is:

```
numerator = sum of b at all beat frames + sum of d at downbeat frames
coefficient[A,D] = [u^A v^D] product_t (1 + exp(b[t])*u + exp(b[t]+d[t])*v)
Z[A,D] = coefficient[A,D] / (N! / (A! D! (N-A-D)!))
log R = numerator - log Z[A,D]
```

The normalizer averages over all three-label assignments with those counts.
It permutes feature **pairs together**, not the heads independently. Each fixed
label path therefore has unit mean ratio under the conditional, exchangeable
feature-pair reference. A downbeat is one joint category; multiplying separately
normalized beat and bar ratios would ignore their dependence and is not used.
An independent offset of either head cancels at the corresponding count.

For each supplied clock, B visible beats is fixed. Counted meter inference uses
the terminal factor `1/Z[B-D,D]`, sums D and meter paths, then adds the common
beat numerator once. The meter prior remains uniform initial meter/phase and
run-wide `rho ~ Beta(1,1)`, integrated by exact-degree quadrature. This replaces
the former conditional-meter emission; the old score is not added on top.

The unchanged time-exposure duration prior supplies relative clock weights,
normalized over these four templates. The final duration is censored at frame
1152, rather than charging time outside the recording. All candidates start at
frame 4 and cover the same time interval for this prior. Posterior weights are
conditional on this small, supplied family and its assumptions, not calibrated
music confidence or a detection probability.

## Feature intervention, missing data and limitations

The report retains a raw-smoothed diagnostic and one fixed-context intervention.
These are evaluation ablations, not two production strategies. Both start with
candidate-independent `[1,2,1]/4` smoothing using actual adjacent frames, not a
candidate's cyclic beat cell. The contextual variant subtracts the log mean
exponential of the smoothed scores in a nine-frame neighborhood: the largest
odd window shorter than the fixed minimum allowed period of ten frames.
This geometric rule is not chosen by a window-size sweep. It remains a model
assumption; local processing and the paired normalization are distinct stages.

Kernels clip and renormalize within observed contiguous runs. Missing frames
are omitted from the comparison bag, numerator and visible counts; they are
never replaced by silence. The supplied clock may nevertheless have ticks in
a missing span. Those ticks advance the latent meter phase without contributing
observed marks. This is a **given-clock bridging assumption**, not beat recovery
from missing audio. Partial bars at either edge are allowed. No full-audio
generative marginalization of unseen neural values is claimed.

The permutation reference is defined on the resulting features. Correlated
local windows, neural-head backgrounds and musical sections need not obey it.
In particular, permutation after feature extraction is not a simulation of
neural audio under a null model. The report is a controlled comparison, not a
false-positive calibration.

## Result: weak doubling improves, true halving regresses

All seven main head pairs and the overlapping boundary/noise controls match
the frozen time-clock inputs by hash. Fourteen controls are frozen in total.

| Control | Raw-smoothed family winner | Fixed-context family winner |
| --- | --- | --- |
| Constant intact / all weak | Constant | Constant |
| Constant with weak alternating beats | Half | Constant |
| Intact true half speed | Half | **Constant: regression** |
| Intact true double speed | Double | Double |
| True double speed with weak alternating beats | Constant | **Double: improvement** |
| Non-octave change | Non-octave | Non-octave |
| Constant with erased beats, retained bar cues | Half | Constant |
| Constant with erased beats and bar cues | Half | Constant; identical input to true halving |
| Flat middle | Half | Constant |
| Unavailable middle | Constant | Constant, with missing ticks explicitly unobserved |
| Flat / entirely unavailable | Prior-only preference | Prior-only preference |

For weak true doubling, the contextual double-minus-constant joint evidence is
about **+39.00**; the inherited duration-prior disadvantage is **-10.57**,
leaving **+28.43**. The raw joint evidence instead favors constant by about
67.88. The meter search inside the constant-clock competitor remains free to
change meter, so this improvement is not produced by forcing four beats there.

For intact halving, contextual half-minus-constant evidence is already about
**-1.21**. The duration prior adds **-9.34**, giving **-10.55** total. The raw
evidence favored halving by about 108.45. Thus this is not merely the inherited
jump penalty: the localized feature reference has weakened absence evidence.
Do not tune the jump cost to conceal this regression or promote the joint score.

Erasing alternate beats and alternate bar cues from a constant clock produces
exactly the same two head arrays as the authored half-speed case. Hashes, all
scores and all probabilities are identical. Neither algorithm changes nor
training on these same inputs can uniquely recover both contradictory labels.
Real metadata must expose this ambiguity rather than equating a preferred
template with certainty.

Flat and entirely unavailable controls have joint family log ratio zero. Yet
their constant-clock probability inside the supplied family exceeds 99%, solely
from its prior. This is a direct regression guard against calling conditional
family probability a signal-confidence score. One fixed noise draw gives
negative family evidence in both variants, not an estimated rejection rate.

## Verification and next step

Run `cargo run --locked --profile evaluation -p rhythm-map-eval --example
shared_clock` for `evaluation/parity/shared-clock-v1.json`.
`python -m unittest discover -s evaluation/parity -p test_shared_clock.py -v`
independently reconstructs every input, shared feature, paired normalizer,
meter/count marginal, duration prior and family probability. Rust checks
enumerate all three-label assignments and small meter paths with missing marks
and Beta-integrated change rates; correlated-head and offset checks guard the
joint reference boundary.

Retain the common domain, paired normalization, counted meter and missing-data
accounting. The next prerequisite is an observation/missing-pulse treatment that
retains weak repeated evidence **and** supports genuine slowing, with explicit
ambiguity for observationally identical cases. Evaluate that before expanding
to unknown clock search or replaying a long music cohort. The current full-frame
table costs O(N*Amax*Dmax); meter search is still repeated only for four supplied
clocks. This is not a demonstrated whole-song, unrestricted-search budget.
No default estimator, public API, user knob, model weight, private music,
holdout, training or release is changed.
