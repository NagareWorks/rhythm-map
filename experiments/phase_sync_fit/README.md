# Distributed phase clock: musical validation stopped on gradient failure

Status, 2026-09-11: **closed, incomplete, no musical accuracy conclusion**.
The main fit completed 16 epochs, then failed in epoch 17 with a nonfinite clock
gradient. The fitted-zero control and final 200 sequence evaluations never began.
No default, model pack, holdout, encoder, dataset or public strategy changed.

## What actually ran

The [frozen protocol](PROTOCOL-v1.md) and [target-runtime plan](plan-v1.json) retain
the same 20/5 RUBATO and 15 diagnostic-only ARTBeaT roles, seed 142, native
circular-phase/multilag-count loss, AdamW settings, 20-epoch/100-update budget,
work balancing and all old nonregression gates. Only the previously validated
24,037-parameter synchronizer replaces the old model construction. The optimizer
loop's source-equivalence test remains in CI; no failed old model is refitted.

A first pre-optimizer guard rejected the VDI-generated initial-state hash on the
actual Torch 2.10 T4 runtime. Every other plan key matched. The target runtime
re-registered the SAME source/protocol/data/seed before any musical forward or
optimizer step; the strict plan equality guard was not relaxed. This is not an
extra fitted seed or a training retry. The earlier private plan hash is retained
in the [failure outcome](results-v1.json). A seed alone is not a byte-identity
guarantee across runtime/host builds; reproducible execution requires the planned
initial-state hash as well.

The real main run then passed 33 component checks and 9 pre-fit protocol checks,
loaded the pinned packets, and performed learning. Its [append-only journal](journal-v1.jsonl)
records 16 complete epochs / 80 confirmed updates. The failure occurs in epoch
17 before another complete epoch can be journaled: the exact update count is
unknown, bounded by 80--84, not asserted to be exactly 80. The container exited
1 after 28.113 s including setup/tests/loading; it was not OOM-killed. That time
is neither training-only throughput nor end-to-end audio inference latency.

Development total loss reached its lowest logged value, 1.20325, at epoch 11,
then rose to 12.32362 at epoch 16. These are optimization losses, NOT BPM/phase
accuracy measurements. Do not select the attractive epoch from the log as a
successful model. The old loop kept its best state in RAM and wrote weights only
after a completed fit; this failure left NO saved checkpoint, failing-model
snapshot, current-recording ID or exact within-epoch update record. We cannot
recover the lost state or precisely attribute this failure to a musical case.

[Execution identity](execution-v1.json), [failure type](failure-v1.json) and
[outcome/limitations](results-v1.json) preserve this distinction. No control
training, musical metric comparison, weight promotion or automatic retry followed.

## What the optimizer-free stability check proves

In the synchronizer, the derivative from one state to the next is
`A = 1 + d*k*h`, where `h` is the signed derivative of the phase residual.
Positive advancement `d > 0` does not imply `abs(A) <= 1`. Reverse propagation
multiplies these state derivatives across time; clipping the final parameter
gradient occurs AFTER backward, too late if backward already overflows.

The [authored post-failure witness](stability-v1.json) uses finite float32 inputs,
the existing initial gain, 3,000 cells at 50 Hz and a 120 BPM prior. Fixed
adversarial observations nearly oppose each incoming predicted phase. Its forward
clock remains strictly increasing at 119.999999--120.000001 BPM. Yet each local
state derivative is about 1.0615914; their product is about `10^77.87`, beyond
float32's `10^38.53` maximum. A float64-input control retains that finite large
derivative, while the float32 case triggers the same nonfinite-gradient guard.
A short version agrees with independent plain-PyTorch autograd.

This proves a structural stability gap compatible with the observed failure,
NOT that the actual unsaved learned model encountered this exact phase pattern,
field, dtype conversion or recording. The witness deliberately constructs
adversarial evidence; it is not music, a fitted model, representative data or
a proposed float64 training fix. It uses zero optimizer steps and changes no
weights. Forward coherence and the previous synthetic cost gate remain true,
but did not establish stable optimization over learned full-crop trajectories.

The diagnostic lives under `diagnostics/`, separate from the 24 pre-fit files
pinned in the original plan, and records its own source hashes. Run only the
optimizer-free probe with `python -m experiments.phase_sync_fit.diagnostics.stability
--output NEW_PRIVATE_JSON`. CI exercises its counterexample and derivative check.

## Next boundary

Before proposing another musical fit:

1. make numerical failures observable: persist the current and best model,
   exact update/recording identity and useful gradient/Jacobian diagnostics;
2. establish long-sequence gradient stability, not merely positive forward
   advancement, for the next synchronization formulation;
3. pass the existing and this new finite-input counterexample contract, then
   freeze a separately justified budget/control plan. Do not silently detach
   recurrence, shorten crops, lower learning rate or add epochs to this closed run.

This is not evidence that a larger encoder or Transformer is required, nor
permission to train against sealed holdouts. Half/double-time semantics,
representative weak-attack/change/no-rhythm labels and independent acceptance
remain unresolved. There is no successful model to export to Rust/DLL/WASM yet.
