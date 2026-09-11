# Observable training updates, before another musical experiment

Status, 2026-09-11: implemented diagnostic infrastructure, NOT a stability fix or
a new musical fit. The [failed phase-sync fit](../phase_sync_fit/README.md) stays
source-pinned and closed. Its unsaved state cannot be recovered retroactively.
There is no dataset loader, public strategy, automatic resume or retry here.

## What the recorder owns

`FitRecorder` surrounds an explicitly bounded series of optimizer updates. Each
update preserves full-record loss accumulation, supplied work/batch scaling,
ordinary gradient clipping and the caller's optimizer. It does not change the
loss, model, precision, crop, update budget or development-score aggregation.
Before the next fit, its caller must register source/input/initial-state hashes,
the model/loss, population, seed, budget and controls in the execution identity.
This helper cannot establish those research permissions from a dictionary.

The private output directory must be new and outside the repository. It keeps:

- `initial.pt`: initial model, optimizer and random-number-generator states;
- `before-update.pt` and `after-update.pt`: last durable update boundaries;
- `best.pt`: the lowest **complete-epoch** development loss supplied by the
  caller, with the earlier checkpoint retained on ties;
- `failure.pt`: current AND best weights, optimizer state, existing/partial
  parameter gradients, finite/nonfinite summaries, epoch/batch/recording index,
  exact current inputs and loss scaling, pre-record RNG, failure-time RNG and
  optional operation trace;
- `journal.jsonl`: fsynced stage transitions and successful update receipts with
  snapshot hashes, rather than only an epoch counter.

The caller still owns complete-epoch validation and work-macro aggregation; it
must not submit partial development scores to `checkpoint`. Initial/per-update
weights are not automatically eligible for accuracy measurement or release.

Model/optimizer tensor copies and serialization preserve their dtype and do not
consume RNG or alter gradients. The recorder captures Python/global NumPy/Torch
RNG and already-used CUDA devices, not arbitrary external RNG objects. Random
samplers and custom generators must additionally be pinned by the fit protocol.
The extra I/O/copies are **not free**: a new complete-crop cost and wall-budget
check is required before music training. Old cost measurements do not cover this
instrumentation. Diagnostic weights/inputs stay private, not Git artifacts.

## Failure semantics, including what cannot be promised

A backward/clip failure before `optimizer.step()` retains the exact number of
returned optimizer calls and the failed recording or batch stage. If `step()`
itself raises, some parameters may already have changed: the recorder marks
`optimizer_step_status = entered` and `update_count_exact = false`, retains the
pre-step snapshot, and never calls that partial update successful.
The durable `optimizer_step_prepared` event is only intent. A caught error writing
that intent is still a known pre-step failure; a process kill after writing it
can leave optimizer entry unknown. Only the returned-call snapshot/receipt proves
a completed update.

Snapshots use same-directory temporary writes, file flush/fsync and atomic
replacement. Failed writes do not replace the previous complete file. Disk/device
failures during capture are reported separately without hiding the original
exception. A process kill or host failure can bypass the handler entirely; the
journal and last committed snapshots are evidence, not an exactly-once guarantee.
The interrupted update remains unknown until reconciled. We do not claim
power-loss durability of every filesystem's rename/directory metadata.

After any caught failure the recorder closes. A saved checkpoint is permission
to inspect evidence, **not** permission to restart a closed experiment. Load only
your own trusted snapshots with `torch.load(path, weights_only=True)`; this is not
an interchange format for arbitrary user uploads or the shipping Rust library.

## Capturing and replaying the actual clock backward

`TracedPhaseSyncReadout` is an observation-only adapter over the frozen model. It
uses the same fields, synchronizer and first derivative. Inside that one known
scan, it copies the saved local Jacobians and the **actual accumulated upstream
gradient** reaching the clock output. The gradient hook returns nothing; it does
not replace, clip or detach the derivative. An altered scan needs a new explicit
trace contract, not silent tensor-shape guessing across a whole network.

Pass `trace=lambda: model.last_trace` to the recorder. `replay_backward` explicitly
replays only the saved field-level scan with that upstream gradient and no
optimizer. It is not CNN-backward replay, full-epoch resume, or a promise of
cross-device bit equality. On a later clip/optimizer failure, this trace describes
the last recording's scan, not proof that the scan caused the failure; the stage
and current-record identity must be read together.

Authored tests compare ordinary versus observed optimizer updates, model forward
values and all parameter gradients; inject backward, clip, partial-step and disk
failures; preserve earlier best states; and save/replay a **new authored** opposed-
phase failure. This does not reconstruct the old musical failure from epoch 17.
CUDA coverage checks device identity, RNG capture and same-device backward replay.
Run `python -m unittest discover -s experiments/training_observer -p 'test_*.py' -v`.

## Why merely reducing phase feedback is not a stability proof

For fixed period/evidence, the old scan has a differentiable state map
`F(q) = q + d(q)`, with one-cycle periodic advancement: `d(q + 1) = d(q)`.
Therefore `F(q + 1) = F(q) + 1` and the integral of `F'(q)` across one cycle is 1.
If `abs(F'(q)) <= 1` everywhere, continuity and that integral force `F'(q) = 1`
everywhere, hence `d` is constant in phase. A nontrivial differentiable periodic
correction cannot be globally nonexpansive in this scalar lifted-clock metric.

This elementary property explains why smaller nonzero gain alone cannot certify
all-length/all-evidence stability. It does NOT prove typical trajectories always
explode, exclude bounded-horizon guarantees, or rule out other states/objectives.
Clamping the custom backward Jacobian would no longer be the derivative of the
unchanged forward operation. Any such approximation must be explicit and checked,
not called an observation-only repair.

Next establish a justified long-sequence formulation and its derivative/cost
contract before a separately registered fit. No new music inference, old model
loading, holdout access, encoder update, product default change or release occurred
in this observability step; tiny authored unit-test optimizer steps are not a
musical training run.
