# Frozen local-tempo conflict audit v1

Freeze this contract and executable source before musical gradient evaluation.
This is attribution of the closed local_tempo v1 result, not another fit or an
acceptance experiment. No optimizer, proposed parameter updates, modified model
weights, encoder calls, new PCM, hyperparameter search, checkpoint selection or
shipping change. Original holdout remains sealed. Development is already
selection-exposed and now additionally gradient-diagnostic, never a fit input.

## Population and complete state coverage

Pin the local_tempo v1 report/plan, predecessor code closure, all 42 saved epoch
exports, old 18 frozen-feature captures, original 40 natural-packet manifest and
starting full separated-natural export. Check both epoch-zero state identities
match the starting head, compute their identical state once, and inspect every
epoch 1--20 for both arms: 41 distinct checkpoint slots, no favorable subset.
Retain all 40 packets in the verified loader, but forward/differentiate only the
20 fit records / 9 works and 5 development records / 3 works. No ART forward or
gradient, fresh-schedule forward, holdout read or diagnostic-to-fit role change.
All 559 old admitted points / 15 groups supply the expected pair objective;
the full 915-point exclusion ledger remains unchanged. New schedules are not
used by this audit even though their outcomes are now exposed.

## Quantities measured at each immutable checkpoint

Compute parameter gradients without filling Parameter.grad:

- Native-fit: expectation of the old work-weighted natural minibatch objective,
  equivalently equal work means over the 20 fit recordings.
- Pair: equal means over all 15 source/profile groups, not pooled point means.
  Preserve identity (3 groups), global (6), local (6) contributions separately,
  each still divided by 15; their sum is the same full pair objective.
  Query the old exact-context scalar prediction field. Split each group's points
  into <=16-point batches and weight each batch by its fraction of that group.
- Retention-fit: coefficient-1 detached initial-output MSE on exactly the same
  valid natural fit cells and with the same native work weights. Compute this
  diagnostic even on the unretained arm, but do not claim it was in that arm's
  historical optimizer. Initial teacher remains frozen.
- Development: native-cell MSE for each of five recordings; aggregate within
  work, then equally across three works. MSE derivatives are NOT derivatives of
  median BPM error. Also recompute the actual MSE/BPM metrics and compare with
  every saved epoch's development record and training-pair MSE.

Save named parameter geometry, theta and all primitive gradient vectors privately.
Model/loss math is deterministic FP32 (native-label MSE retains its existing
float64 target arithmetic); detached vector statistics use float64. Report norms,
gradient dot products/cosines and unit-negative-gradient directional derivatives
for native, each pair part, pair total, retention, native+pair, and
native+pair+retention against every development record/work/macro gradient.
Zero vectors have undefined cosine/unit direction, not a protective score.
No diagnostic gradient is projected into, combined into or passed to an optimizer.

These full-population expectation gradients are NOT the historical stochastic
batch gradients. Separately reconstruct the very first recorded minibatch's
native/pair/retention gradients at the initial state, using its exact four natural
IDs and 15 groups x four sampled indices with multiplicity. Verify both arms'
first draw is identical and recomputed unclipped combined norm matches the saved
first-update receipt within relative 1e-5 / absolute 1e-6. Retention is exactly
zero at the teacher's own state. This is a gradient reconstruction only: no first
AdamW displacement is synthesized and no step is performed.

## Actual movement, and the attribution limit

For each arm's 20 consecutive saved epoch intervals compute actual delta theta.
For every development record/work/macro retain observed endpoint MSE change,
g_start dot delta, g_end dot delta, their trapezoidal estimate, and its residual
against the observed change. Retain all 40 intervals, including disagreements.
These use actual saved endpoints, but a straight endpoint approximation does not
reconstruct the intervening five AdamW steps or integrate the true training path.

Positive gradient dot product means the negative training-gradient direction is
locally MSE-decreasing; negative dot means locally opposing. AdamW momentum,
coordinate scaling, clipping, weight decay and finite-step curvature prevent
equating that sign with actual loss change or assigning exact causal percentages.
The old run did not save intermediate updates 1--4 or optimizer moments. Do not
pretend this audit recovers them. Authored tests must explicitly exhibit Adam-like
diagonal scaling reversing raw-gradient alignment, and curved loss disagreeing
with an endpoint linear prediction, without installing an optimizer.

No all-pass musical accuracy gate or data-dependent rescue follows. The outcome
reports which directions conflict, where the first-batch reconstruction agrees,
whether retention has a nonzero opposing/agreeing signal, and how well endpoint
derivatives describe observed movement. Any next training mechanism needs a new
explicit contract; do not choose a teacher coefficient using these gradients.

## Execution and replay

Single T4 diagnostic, deterministic math, TF32 off, two CPU threads and 40% GPU
allocation cap. Outer watchdog 1800 seconds, internal 1700 seconds; fail closed,
preserve partial packets, no automatic retries or sample shrinking. Save plan,
source/runtime/input pins, per-state packets and receipts before the next state.
Require unchanged input/state/teacher hashes and empty Parameter.grad buffers.
Recompute all public vector statistics from private packets without a model;
test analytic gradients, weighting, sparse batching and no-write boundaries on
authored inputs. A final exact-commit GPU replay covers the first batch, shared
initial state, both epoch-one states and both terminal states without training.
