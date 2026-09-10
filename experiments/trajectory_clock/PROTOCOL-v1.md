# Native-trajectory objective v1: fixed loss-only experiment

Freeze this protocol, attribution report, code and plan before optimizing.
This is one paired exploratory experiment, not a retry of either closed run.
No automatic seed, epoch, coefficient, architecture or data sweep follows failure.

## Motivation and single changed factor

Post-fit truth-assisted attribution of coupled v1's retained outputs finds that
even the best whole-track constant phase shift leaves work-macro phase errors
0.2227 cycles on development and 0.2353 on ARTBeaT, versus raw 0.1427 / 0.1078.
Thus origin correction alone cannot recover the raw baseline. These are oracle
diagnostics, not improved predictions, independent labels or product selectors.
Removing a best constant rate bias also leaves nonlinear count residuals; no
global tempo correction is claimed sufficient.

Test whether directly supervising complete-span cumulative native beat error
helps sustained alignment. Keep the identical 23,970-parameter CNN, one physical
origin, positive integrated clock, frozen features, padding and numerical guards.
Only the supervised objective changes, including the development objective used
to select a checkpoint. Numerical training losses across experiments are not
comparable; compare the unchanged musical metrics instead.

For each contiguous reference-supported span, let e[j] = predicted q[j] - native
reference q[j]. Select ONE integer k = round(mean(e)), then penalize mean((e-k)^2).
This integer minimizes the squared error over arbitrary integer beat numbering.
It is not chosen per frame: a one-beat slip later in the crop remains an error.
The detached integer branch has the exact objective gradient away from ties;
round-to-even selects a branch at ties. Mean-square decomposes into centered
trajectory variance plus squared registered mean error. There is no guarantee
of convex model optimization or accurate convergence.

Weight valid frames equally within each recording. Separate annotation spans
have separate unknown integer count origins; never bridge a reference hole.
The model still integrates one complete physical crop with NO gap/window reset.
Neither truth nor integer registration appears in inference. This replaces,
rather than adds a tuned weight to, the old circular-phase/multi-lag loss.
Removing the old local loss can regress local tempo: the original metric gates
remain mandatory. Native supervision does not settle perceived beat convention.

## Unchanged population and execution

Use the exact coupled-v1 input manifest and all 40 hash-pinned packets:
20 RUBATO recordings / 9 works fit; 5 recordings / 3 works develop;
15 ARTBeaT recordings diagnose only. Same crops, features and raw/v1 comparators.
No new audio, encoder forward, pretrained weight, annotation or holdout access.
The fit set still lacks audited weak-attack, omission, hard-change and no-rhythm
region labels. It cannot certify those capabilities. All cohorts are previously
exposed development evidence; the attribution additionally informed this design.
ARTBeaT does not train or choose an epoch and is not independent acceptance.

Two fresh fits, audio and matched zero features. Seed 142, identical initial
state to coupled v1, NumPy shuffle seed 142 + epoch index, AdamW lr 0.001 / weight
decay 0.0001, gradient norm clip 1, four-recording accumulation, 20 epochs /
100 updates maximum, 1,800 seconds per fit. Identical complete-crop training and
inference, no mixed precision, deterministic algorithms, TF32 disabled.
Keep the old fit/development loop literally unchanged except the loss call;
tests check its source equivalence. Reuse fixed loading, weighting, journaling
and musical measurement helpers without monkey-patching the frozen old module.

Choose the lowest complete-epoch development trajectory loss, mean recording
within work then across works; ties keep the earlier epoch. A budget exhaustion
or numerical failure fails the gate, retains partial logs, and does not trigger
a retry. Save both selected checkpoints and every prediction privately.

## Fixed musical gate and stopping rule

Retain ALL coupled-v1 conditions and measurement semantics: native level (no
octave minimum), exact cell/reference/raw support, v1 trapezoidal rate projection,
phase error, median/P95 BPM error, 4-second count drift, work-macro means,
complete coverage and every-recording regressions. No denominator change.

- Both 20-epoch fits complete with finite required outputs in budget.
- Reference-only development tempo median and phase error each improve at least
  10% over the freshly fitted zero-feature control.
- On development and ARTBeaT common support, median/P95 tempo, phase and count
  drift are no worse than both raw events and the closed independent v1 head.
- No more than half of recordings regress versus raw for either median tempo
  or phase in either cohort.
- Additionally, each of those four common-support work-macro metrics must be no
  worse than the previous coupled head, in both development and ARTBeaT. Retain
  previous-coupled per-recording regressions too, not only favorable means.

A pass only supports proposing independent admission and representative-label
work. A failure closes this exact experiment; inspect the result before proposing
something materially different. Neither outcome changes Rust defaults, emits
inferred beat crossings, adds public strategies, publishes weights or releases.
