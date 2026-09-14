# Protect the optimizer's actual displacement

Research boundary only. The CNN, clock, original supervision, old experiments,
shipping estimator, holdout and release state are unchanged. This is not a new
user-selectable strategy or permission to repeat a musical fit.

The [previous experiment](../gradient_routing/README.md) established a specific
problem: count-descending **gradients** can become count-ascending **AdamW
updates** because of diagonal preconditioning or retained momentum. Decoupled
weight decay can do the same. Protecting the gradient alone is insufficient.

## What changes

`ProtectedFitRecorder` reuses the existing recorder's update lifecycle and two
observable transports. It aggregates all weighted complete records separately
for phase and count, applies the same three-group routing, and installs one
global clipped gradient. It does not project individual songs and then sum.

`ProtectedAdamW` gives native PyTorch AdamW a shadow copy of the complete current
parameters, gradients and optimizer state. It does not reimplement AdamW. Given
old weights theta and its proposal theta_adam, the actual displacement is
`d = theta_adam - theta`. For each parameter group with count gradient c:

```
c dot d <= 0: keep the actual AdamW proposal
c dot d > 0:  project d onto the halfspace c dot d <= 0
              round theta + projected_d to the model's native dtype
              recheck; if still adverse, restore that group's old weights
c == 0:       protection undefined; retain the AdamW proposal and say so
```

The final sign check uses exact integer accumulation of stored binary floats,
including the two endpoints of each displacement. Tiny underflowed products or
cancellation cannot accidentally be accepted as zero. The binary gradient's
common exponent cancels from the sign; enormous gradients need not be expanded.
The gradient itself still has finite precision. Projection is an approximate
float64 computation; the native endpoint certificate is checked separately.

All proposed AdamW moments/time counters advance on a committed proposal,
including vetoed groups. This defines a projected optimizer, **not** unchanged
AdamW training. Receipts distinguish returned optimizer calls, committed
proposals, moved proposals, group projections and rounded-projection vetoes.
There is no public tuning parameter, line search, automatic retry or scheduler.

## Failure ownership

`before-update.pt` precedes zeroing gradients or any forward. `proposal.pt`
contains the old live model/state, whole weighted batch, exact count packets,
proposed weights/moments, rounded candidate and group decisions before live
writes. Normal completion additionally persists `after-update.pt`.
Every completed record retains an owned CPU copy of its full input/metadata
and pre-forward RNG state, not just its gradient or the last record's payload.

Proposal/save errors leave live weights and moments untouched. A caught partial
commit attempts rollback and verifies native values; the failure snapshot says
`restored` or `failed_unknown`, then the recorder closes. A journal failure after
commit can have one committed proposal but zero returned calls; this is explicit.
Hard process/device death is not crash-atomic across multiple tensors. The last
durable before/proposal snapshots are diagnostic evidence, not automatic resume.

Supported ownership is intentionally narrow: the unchanged deterministic,
buffer-free PhaseSyncReadout; one complete ordered AdamW parameter group;
disjoint native float32/float64 storage on one CPU/CUDA device; one exclusive
caller. Unsupported hooks, aliases, parameter replacements, sparse/missing
gradients, captured/differentiable/fused/AMP/distributed execution fail closed.

## Evidence and limits

The [predeclared authored protocol](PROTOCOL-v1.md) tests real AdamW
preconditioner/momentum/weight-decay failures, safe native AdamW state parity,
rounding/zero movement, exact signs, complete-batch CNN/VJP integration, and
injected failures before/during/after commit. Captured signs replay without a
model forward or optimizer call. CPU tests are in CI; CUDA tests require a GPU.

The fixed 40-step teacher task uses the old AdamW settings (lr 0.001, decay
0.0001, clip 1), with no sweep or best-checkpoint selection. Its result is
recorded in `authored-windows-v1.json`. All source identities and all 40 step
receipts are retained, not just the final losses. This is a short, synthetic
learnability observation, not an audio benchmark or a general performance bound.
On the recorded Windows CPU run, phase/count losses go from 0.108576/0.007531
to 0.00005496/0.00003417. All 40 proposals move weights; 25 of 120 group decisions
are projected and 18 vetoed after rounding. Nevertheless, 15 of the 40 steps
increase finite count loss, directly illustrating the first-order limit.
Elapsed time including durable snapshots is 52.30 s on the shared VDI; no peak
RSS was measured there. The initial instrumentation attempt stopped before any
update on non-scalar loss metadata; its fix does not change the loss or optimizer.

First-order nonincrease does **not** imply finite-step loss reduction: even a
perfectly orthogonal displacement can increase a curved loss. Neither does it
prove the count objective is the correct perceived BPM on every recording.
Finite count-loss increases and group vetoes are reported, not hidden. This
boundary addresses an identified optimizer failure; it does not solve musical
generalization, integer-beat ambiguity or the failed old accuracy gates.

Next: register one bounded musical comparison using this explicit optimizer
identity, matched zero-feature control and unchanged accuracy gates. Include
all veto/stall and finite-loss regression counts. Do not silently restart the
old fit or expose an additional product policy.

Run `python -m unittest discover -s experiments/protected_update -v`. To run the
fixed authored observation, use `python -m experiments.protected_update.authored
--output <fresh-private-directory> [--device cuda]`. The output must be outside
the repository; only load your own trusted snapshots with `weights_only=True`.
