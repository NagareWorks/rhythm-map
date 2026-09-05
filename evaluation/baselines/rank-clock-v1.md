# Background-centered rank clock-family gate, v1

## Decision and scope

The seven unchanged main synthetic controls now select their authored clock
templates, including **both weak doubling and genuine halving**. This repairs
the previous [shared-frame gate](shared-clock-v1.md)'s halving regression without
changing the duration prior. Three fixed section-background perturbations also
select their authored templates. These are prerequisite scoring results, **not
beat recovery, automatic boundary discovery or measured real-music accuracy**.

The same four supplied clocks, fixture-derived phase/section boundaries, paired
full-frame normalization and exact meter marginalization remain. No unrestricted
clock search or missing-pulse state marginalization is implemented here. The
default estimator, public API, model weights, user options and releases are
unchanged. Holdout remains unopened. No training is performed or justified by
this result.

## One fixed observation transform, with its failed ablation retained

The earlier local log-mean subtraction rescued weak pulses but weakened absence
evidence enough that genuine halving lost even before its duration prior. The
intervention here changes the observation feature only:

1. Apply the unchanged candidate-independent `[1,2,1]/4` head smoothing.
2. Subtract each head's local median in a fixed nine-frame window: the largest
   odd window strictly shorter than the fixed minimum period of ten frames.
3. Compute empirical midranks independently per head over **all available
   frames**, and use their log odds as features.
4. Permute the resulting head **pairs together** using the unchanged three-label
   count normalizer. Ordinary beats, downbeats and neither remain joint labels;
   do not multiply two independent head likelihoods or add the old emission.

If a feature has L observed values below it, E equal and G above it, its score is

```
logit(mid-CDF) = log(L + E/2) - log(G + E/2)
```

Ties share scores, both terms are positive, and no clipping epsilon, amplitude
cutoff, rank temperature or per-song parameter is fitted. Local median subtraction
retains relative peaks despite a section-level background shift; global ranking
lets a weak peak exceed background without exponentially privileging the loudest
peak. Unlike purely local normalization, an observed flat location retains its
position in the whole recording's feature distribution.

Both local kernels clip and renormalize within available runs. A clipped even-size
median is the midpoint of its two central values. Missing samples never become
observed zeros and never affect neighboring available runs. Their supplied latent
ticks still advance meter phase without observations, as in the prior audit.

The report also keeps **raw rank without median subtraction**, because it initially
passed the main controls but failed the section-background challenge. This is an
evaluation ablation, not an alternate production strategy. The median intervention
was investigated after that failure; this is development evidence, not a blind
validation set. There was no window-size, offset-size or prior-weight sweep.

## Frozen results

All fourteen previous head pairs match their frozen hashes. Six additional
authored interventions make twenty inputs total; they are not twenty songs.
Every input is run through both rank variants on the same four templates.

| Control | Raw rank winner | Background-centered rank winner |
| --- | --- | --- |
| Constant / weak alternating / all weak | Constant | Constant |
| Genuine halving | Half | Half |
| Genuine doubling / weak alternating doubling | Double | Double |
| Non-octave change | Non-octave | Non-octave |
| Constant with erased beats, retained bar cues | Constant | Constant |
| Constant with erased beats and bar cues | Half | Half; ambiguous with true halving |
| Flat middle / unavailable gap | Constant | Constant |
| Constant, middle background shifted | **Half: failure** | Constant |
| True half, middle background shifted | Half | Half |
| Weak double, middle background shifted | **Half: failure** | Double |
| Flat / all unavailable | Prior-only constant preference | Prior-only constant preference |

The shift is exactly -16 on **both heads throughout frames 384..768**, with
unchanged pulse positions and within-section contrast. Boundary smoothing can
change a few ranks, so this is not an exact invariance claim. The centered model
passes all three fixed shifts; it is not proven robust to arbitrary drift, dense
subdivisions, pulse shapes, head dependence or full-song nonstationarity.

In the final centered model, relative to the constant template:

- weak true doubling: joint evidence **+22.01**, duration prior **-10.57**,
  total **+11.44**;
- genuine halving: joint evidence **+14.62**, duration prior **-9.34**,
  total **+5.28**;
- constant with erased beats but retained bars: half-minus-constant evidence
  still **+4.44**, but its prior **-9.34** gives **-4.90** total. Thus the correct
  constant preference is **not observation evidence alone** resolving omission.

The prior is imported unchanged, including common-endpoint censoring, from the
frozen shared-clock report. There is no weakening of its jump cost to get passes.

## Explicit ambiguity and strength limits

The constant both-erased input remains exactly identical to the true-half input,
including availability and every inference output, despite conflicting authored
clock labels. The scorer receives arrays and supplied templates, never case names
or labels. The report's paired witness is a **post-scoring diagnostic**, not an
automatic ambiguity detector or an omitted-tick hypothesis model.

The conditional half-template probability exceeds 99% on *both* identical inputs.
It says nothing about the omitted-pulse explanation absent from this family.
An explicit latent omission model must not equate its preferred observed pulse
density with the underlying musical clock or report an inferred tick as observed.

Three additional controls shrink all deviations from -8 by **1/4096** for constant,
half and weak-double cases. Each variant's complete outputs are exactly equal to
its unshrunk counterpart. This preserves order, but discards absolute contrast:
tiny coherent ripples can score like strong authored pulses. Rank-based evidence
therefore cannot itself be detector confidence or proof of audible rhythm. It
still needs a tested observation-strength/noise interpretation, not a fitted
amplitude cutoff disguised as confidence.

Flat and all-unavailable controls retain zero family log ratio while the constant
template receives over 99% prior-only probability. The fixed noise draw has
negative family evidence in both variants; one draw does not estimate a false
positive rate. The normalized reference is conditional on processed feature
pairs, not a calibrated generative audio null; local windows remain correlated.

## Verification and next gate

Run `cargo run --locked --profile evaluation -p rhythm-map-eval --example
rank_clock` to reproduce `evaluation/parity/rank-clock-v1.json`.
Run `python -m unittest discover -s evaluation/parity -p test_rank_clock.py -v`
for independent reconstruction. Python uses pairwise comparison counts rather
than Rust's sorted tie groups, NumPy medians and a probability-space meter DP.
It reconstructs all inputs, normalizers, observed/missing marks, count/downbeat
marginals and family weights for all **160 template/variant combinations**.
Rust tests cover ties, monotone order invariance, missing runs, odd/even clipped
windows, invalid input, the paired reference and exact enumerated meter paths.
Source/report identities bind the inherited family and unchanged components.

Next freeze an explicit latent-clock versus observed-pulse omission formulation,
including the observationally identical witness, rather than another feature or
duration-weight sweep. Retain this single centered-rank scoring prerequisite,
but do not promote it until ambiguity, observation strength and unknown-clock
search have their own evidence. No long cohort replay is warranted by these
truth-assisted controls alone. The inherited full-frame table is still
O(N*Amax*Dmax); an unrestricted whole-song cost has not been demonstrated.
