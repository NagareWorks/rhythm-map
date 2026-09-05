# Censored conditional meter audit, v1

This evaluation separates bar-boundary uncertainty from beat-clock search. It
does **not** implement a new production decoder. It reuses the exact inferred
ticks, periods, gaps and unused edge frames in [time-clock v1](time-clock-v1.md).
No missing beat is added, no BPM is corrected, and no truth clock is substituted.
The fifteen original input head pairs are hash-checked against that frozen run.

## Model and changed observation boundary

For each frozen beat cell, apply the existing rotation-normalized `[1,2,1]/4`
kernel to the **downbeat** head at phase zero. Call the resulting log ratio
`l[t]`. A state is `(meter m, beat-in-bar j)`, with `m=2..7` and `j=0..m-1`.
It emits `exp(l[t])` when `j=0`, and ratio one otherwise. This replaces the
previous whole-bar cyclic observation model; it is not a prior-only intervention.
The frozen beat head is checked for identity but is not rescored here.

The initial prior is `1/(6*m)`: uniform meter, then uniform initial phase.
Within a bar, the phase advances deterministically. At a bar boundary:

```
P(next meter = current meter | rho) = 1-rho
P(next meter = any particular other meter | rho) = rho/5
rho ~ Beta(1,1), shared across this available run
```

Every terminal state has weight one. A recording may therefore start or finish
partway through a bar; it does not have to invent a shorter final complete bar.
This is censoring at **beat-grid** boundaries, not integration of truncated
neural beat cells or recovery of the excluded audio edge frames. Unavailable
gaps still split independent runs. Their meter priors are not bridged.

The run-wide unknown `rho` lets repeated structure inform meter persistence
without a user-selected change penalty. It is integrated, not fitted to truth
or selected from a menu. The initial uniform meter/phase prior, uniform
hyperprior and uniform alternate-meter choice remain consequential assumptions;
zero user knobs does not mean assumption-free inference. The uniform hyperprior
was introduced during development after an independent-per-bar reset discarded
accumulated meter evidence. This is an exploratory gate, not a preregistered
one-factor result. No coefficient or threshold sweep is performed.

For `N` beat cells there are at most `floor(N/2)` visible bar boundaries. Path
probabilities are polynomials in `rho`; Gauss-Legendre quadrature with
`Q=floor((floor(N/2)+3)/2)` integrates the partition, state occupancies and first
moment of `rho` exactly up to floating-point error. Forward/backward runs at each
node. The present short-control implementation uses O(N^2 * S) time and storage
for fixed meter domain `S=27`, not a production whole-song performance claim.

Output contains marginal meter and downbeat probabilities per frozen tick.
Independent marginal maxima need not form a legal sequence and are **not a MAP
path**. The internal fixed-rate MAP is used only by mathematical unit checks.

The partition equals one without mark evidence. Its reference assumes
independently rotation-invariant fixed cells. However, the clock was already
selected using these same heads: the conditional scores are post-selection
diagnostics, not calibrated confidence, a fresh clock-detection Bayes factor,
or additive evidence for comparing different clocks/coverages. No support
threshold, production label or automatic clock selection uses them.

## Reproducible checks

Run `cargo run --locked --profile evaluation -p rhythm-map-eval --example
censored_meter` to reproduce `evaluation/parity/censored-meter-v1.json`.
The report hashes the runner, decoder, frozen cell kernel and source clock
report. `python -m unittest discover -s evaluation/parity -p test_censored_meter.py -v`
reconstructs marks and independently checks every saved marginal using dense
probability-space matrices and NumPy quadrature. Rust checks also enumerate
small paths directly with Beta-integrated change/stay counts, independent of
both the graph recurrence and quadrature. Neutral future cells leave earlier
marginals and evidence unchanged; all neutral crop lengths have unit mass.

In addition to the fifteen frozen head pairs:

- 139 supplied-grid controls cover meters 2..7, every initial phase, and removal
  of 0..m-1 final beats from 48-beat inputs. Weak authored downbeat pulses are
  unchanged across crops. This tests bar censoring, not end-to-end beat finding.
- Four supplied-grid true-meter-change controls contain 24 beats in each meter:
  4->3, 3->4, 4->2 and 2->4. The truth arrays are reported for scoring only;
  inference receives just the observed mark array and the common meter domain.

## Frozen result: edge uncertainty is explicit, promotion still fails

| Check | Result |
| --- | --- |
| Six main controls other than weak true doubling | Final marginal preference becomes 4 instead of the old complete-bar MAP's 3; P(4) only 0.4704..0.4899 |
| Weak true doubling | Frozen incorrect clock stays unchanged; final meter preference is 2, P(2)=0.8713 |
| Three-beat control | Final preference 3, P(3)=0.9223 |
| Flat heads | Conditional log ratio 0, posterior mean rho 0.5; no evidence acquired |
| One fixed-seed noise draw | Conditional log ratio +0.5532, largest final meter probability 0.2248; not a detection result |
| 139 crop controls, correct marginal meter at every position | 87/139: 2-beat 4/4, 3-beat 9/9, 4-beat 0/16, 5-beat 25/25, 6-beat 0/36, 7-beat 49/49 |
| True 4->3 change | Correct marginal meter at 40/48 positions |
| True 3->4 change | 48/48 positions |
| True 4->2 and 2->4 changes | Each only 24/48 positions |

These counts are authored structural diagnostics, not real-music accuracy.
In particular, the six improved tail preferences are not reliable metadata:
each remains below 50% and much of the ambiguity lies between two and four.
The short first/last bars are expressible without forced closure, but the
observation model still rewards insufficiently constrained explanations.

There is an exact counterexample behind the composite-meter failures. On a
four-beat supplied clock, the mark pattern repeats `[positive,0,0,0]`. A two-beat
meter scores every positive mark too; its extra predicted bar starts occur in
flat cells with log ratio zero. Thus the two constant paths have **identical
emission sums**. Priors/path multiplicity decide between them, not evidence
against the unsupported extra bar starts. Adding a persistence hyperprior does
not repair this information discarded by the per-cell normalization. The
earlier whole-bar scoring had a larger comparison context; this experiment
deliberately changed that context and exposes a regression.

Keep censored boundaries, marginal outputs and normalization tests as design
requirements, **not this complete emission/transition model as another product
strategy**. Next construct a common-context observation model that distinguishes
supported and unsupported bar starts while retaining weak repeated structure;
it must define how unobserved edges are marginalized rather than treating
missing frames as observed silence. Gate it on the composite-meter and true
change counterexamples above before integrating with joint tempo search.
Do not tune the meter prior to force four, narrow the meter domain, compare
post-selected conditional scores as joint clock evidence, or replay a long
music cohort to rescue a failed structural gate. Weak tempo doubling remains
unresolved, and this experiment provides no evidence that training is required.
Production defaults, external APIs, model weights and release state are unchanged.
