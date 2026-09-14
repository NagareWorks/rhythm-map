# Relative-tempo continuation v1

Freeze this contract and its source hashes before neural capture or optimizer
updates. This is a new sparse-point mechanism experiment, not admission of the
failed 15-pair alignment population. Leave every old report, nominal-only flag,
gate, packet and held-out asset unchanged. No downloads or new audio rendering.

## Population and measurement semantics

Use the 3 retained source prefixes and all 15 Rubber Band PCM previews, verified
against the old renderer report. Recompute log-mel spectra from actual PCM using
the pinned upstream Beat This frontend, then capture the frozen final0 encoder's
512-channel, 50 Hz pre-head features using the existing split/stitch ownership.
Never stretch a cached feature axis. Pin the checkpoint and every imported
upstream/dependency Python source; verify task-head reconstruction and unchanged
encoder state. This capture does not establish new Rust/PyTorch frontend parity.

Use only explicitly control-checked **grid** points from alignment v1. Require
interior status and source AND output centers at least 2.56 seconds from their
respective segment boundaries. This excludes the CNN's local receptive field
from render seams/edges; the upstream encoder still has wider context. Keep all
915 grid points in an exclusion ledger; the 1,300 beat-query points are not used.
No dense validity mask or interpolation of support. Unsupported orchestral
points remain excluded, and supported points from that source remain present.
The local 2-second alignment witness does not certify every CNN context sample.

The existing tempo branch predicts outgoing-cell log2(period/second), with cell
centers (i + .5)/50. Linearly interpolate the **predicted scalar field**, not the
features or labels, at each admitted output t and its source g(t). The auxiliary
target is z_output(t) - z_source(g(t)) = -log2(requested rate). This tests local
speed equivariance of the interpolated predictor; it is not a new exact finite-
cell annotation or proof of perceptual beat level. Use nominal correspondence
centers, not best-lag correction. Both query branches receive gradients.

Train pair terms on identity_seams, slow and fast (9 source/profile groups).
Reserve local_slow_fast and local_fast_slow (6 groups) for final evaluation only.
They share source music, transform algorithm and rate values with training:
this is unseen-schedule transfer, NOT independent song/domain generalization or
proof against renderer-artifact shortcuts. No new source-disjoint claim.

## Models, optimization and controls

Continue ONLY the tempo trunk/head from the pinned separated-natural export of
separated_evidence_fit v1. Two arms start byte-identically: native-only and
native-plus-relative. Phase and encoder weights are not optimized. This is the
same 32-channel, kernel-5, dilations 1/2/4/8/16/32 CNN; no architecture search.

Retain all 20 original RUBATO fit recordings / 9 works for native-cell log-period
MSE, with the unchanged valid cells and equal expected work weight. Original
5 development recordings / 3 works and 15 ART diagnostics never contribute
gradients or select checkpoints. Use the already pinned natural feature packets.

Each arm: 20 epochs, 4 natural recordings per update, 5 updates/epoch, exactly
100 AdamW steps, learning rate .0003, weight decay .0001, gradient norm cap 1.
Seed 142; identical natural-record permutation per epoch in both arms. Pair arm
additionally draws 4 admitted points with replacement from each of the 9 fit
groups per update using a separate seed-8142 RNG. Group mean squared pair errors
are averaged equally across all 9 groups with coefficient 1. Native loss weights
are unchanged. The native-only arm performs no paired training forward.

Select the final 100th update in both arms, not the best observed result. No
retries, extra epochs, threshold changes or tuning after opening outcomes.
Save initial/final states, predictions, actual returned update receipts and loss
history. Reject nonfinite tensors/gradients and incomplete execution. Run FP32,
deterministic Torch with TF32 disabled, at most 40% GPU allocation, 1800 seconds
per arm and a 5400-second outer watchdog including capture/evaluation. Keep
artifacts outside Git on the data drive; private network-disabled container.

## Fixed readout and decision

Evaluate initial, native-only and paired final models on exactly the same sparse
points and all 40 unchanged natural records. Also query paired weights with zero
features. Publish per-source/profile counts and errors, not just pooled scores.
Relative rate prediction is -(z_output-z_source); target is log2(rate).
Report RMSE, signed response slope through the origin on changed points, and
sign correctness; zero-change RMSE uses the same targets and denominator.
Do not count identity points toward change-response improvement.

Mechanism gate: for EVERY source on reserved local changed points, paired RMSE
<= .75 times zero-change RMSE and <= .8 times native-only RMSE, slope in [.5,1.5],
and sign correctness >= .8. On EVERY processed identity group require mean
absolute difference <= .05 log2 units. These are preregistered pilot thresholds,
not a product accuracy contract or calibrated confidence.

Natural retention gate: work-macro native-cell MSE and native BPM median error
on BOTH development and ART diagnostics must be <=1.05 times BOTH initial and
native-only values. Report per-record regressions too. This does not certify
native rubato response, which remains separately reported through existing
4-second change measurements. Zero-input scores are diagnostics, not training.
Overall exploratory gate requires complete execution, mechanism and retention.
Even a pass warrants broader, independently sourced/renderer-controlled testing,
not default promotion, packaging, a release or a claim of general accuracy.
