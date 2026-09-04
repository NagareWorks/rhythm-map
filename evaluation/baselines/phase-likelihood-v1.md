# Contextual phase likelihood: authored prerequisite, not a shipped decoder

This follows the [negative dropout result](dropout-likelihood-v1.md). Instead
of interpreting a negative logit as absence, it compares pulse shape at a
proposed phase with competing phases on the same observed frame window.
Production analysis and all previous frozen experiments remain unchanged.

## Normalization and null reference

For a complete cell of n frames, compute cyclic triangular pulse statistics
`s[u] = (x[u-1] + 2*x[u] + x[u+1])/4`, with indices modulo n. The proposed
phase p receives the log likelihood ratio

`log R(p,x) = s[p] - log(mean_u exp(s[u]))`.

This is not a local maximum detector. Every possible cyclic phase enters the
normalizer. `mean_u R(u,x) = 1` exactly; with a rotation-invariant reference
distribution f0 on the cell, `f1(x|p) = R(p,x) f0(x)` therefore integrates to
one. The null/unsupported alternative is f0, with log ratio zero. Independent
cell products remain normalized under a product reference. Comparing different
tilings additionally requires a common frame-domain reference invariant under
each tiling (an i.i.d. background suffices). These are declared reference-model
assumptions, not calibrated claims about correlated neural logits.

The cyclic convention is a periodic cell hypothesis, not padding or reading
frames from another component. The pulse weights are fixed, with no temperature
or weight search. Adding a uniform offset to all logits leaves the evidence
unchanged. A structured peak at -2 on background -8 can now support its phase;
the absolute-logit dropout model could not. The score is neither event
confidence nor evidence that every noise peak is a musical beat.

Exactly flat cells give ratio one at all phases: they remain neutral, not
observed events. Cell lists must tile the full input without overlap or omitted
tails. If even one frame is unavailable, the entire cell normalizer is unknown;
the scorer reports both missing frames and available frames in unscored cells.
Do not compare paths with unequal evidence coverage as if they used the same
data. This code does not implement gap-safe decoding or endpoint extrapolation.

## Same-input comparison

The [authored report](../parity/phase-likelihood-v1.json) reuses all eight
logit arrays from the prior dropout audit, verified by their canonical f64
little-endian hashes. The frame domain, three sections, support width and
proposed 125 BPM / half / double / non-octave paths are unchanged. The candidates
are GIVEN paths, including their origins and change locations, not a search
over unknown musical timing. The null reference remains an explicit candidate.

Seven of eight cases rank their authored path uniquely first, versus four of
eight for the preceding reference. Specifically, constant weak alternating,
all-weak constant, and weak alternating inside a genuine doubled-tempo section
now rank correctly. This is an observation-scoring improvement on constructed
inputs, **not real-music F1, beat recovery or production accuracy**.

The remaining constant-tempo erasure case still chooses half speed. Its input
is exactly identical to the genuine intact half-speed case, and every proposed
path gets identical scores for both. The incompatible truths cannot both be
recovered from this single-head representation; that ambiguity is retained.

Separate downbeat-head controls compare two-, four- and eight-beat cells at a
given beat period. Both strong and weak authored four-beat pulses favor four
over two and eight. This avoids the old extra-bar tie on these controls, but
does not establish general meter estimation or arbitrary starting-phase search.

## Do not maximize every window independently

For fixed period 24, sum each proposed phase's log ratios across 48 windows,
then marginalize ONE shared phase with a uniform `1/24` prior:

`log R_shared = logsumexp_p(sum_windows log R(p,x)) - log(24)`.

This accounts for unknown shared phase under an independently rotatable cell
reference; it is not a maximum phase selected without a search cost. A separate
diagnostic deliberately sums per-window maxima to expose that invalid shortcut.

| Authored input | Per-window maxima (invalid evidence) | Shared-phase marginal log ratio |
| --- | ---: | ---: |
| Fixed-seed nonperiodic noise | 63.141 | -16.156 |
| Coherent weak pulses | 132.588 | 129.410 |
| Weak pulses drifting against the fixed period | 132.588 | -119.412 |

The noise fixture uses LCG seed `0x13572468`, multiplier 1664525, increment
1013904223 and upper-byte values mapped to [-8,-2]. No seed or weight sweep
was performed. These three controls diagnose phase coherence, not a measured
false-positive rate. Rejecting a fixed clock for phase drift is not rejecting
music or claiming the drifting sequence has no valid varying-tempo clock.

This shared-phase diagnostic was added during authored development without
changing the cell statistic or weights. The report is not a preregistered
real-music validation. A whole decoder must account for searches over tempo,
meter, change locations and cell boundaries too; phase marginalization alone
does not solve their multiplicity or musical priors.

## Next gate

The weak-evidence prerequisite now has positive results worth taking forward.
Freeze a joint sequence formulation that owns unknown boundaries/tempo/meter,
explicit unavailable spans and supported-versus-inferred state. Its tests must
include all meter/phase combinations, real changes, edges, extras and noise;
do not replace unknown timing with truth-derived cells in that decoder.
Only then compare a fixed candidate on all retained dense calibration captures,
with the established per-track regression gates. No training or holdout access
is justified by these authored results alone. No user strategy is added.

Reproduce with `cargo run --locked -p rhythm-map-eval --example phase_likelihood`.
Seven Rust tests cover normalization, offset/rotation behavior, weak contrast,
missing coverage, invalid partitions, fixed-path/meter outcomes and coherence.
