# Separately learned tempo and phase evidence

This is an executable component comparison, not a trained music model, decoder,
new public strategy or release. It follows the failed
[protected-optimizer musical run](../protected_phase_fit/README.md) by changing
task ownership rather than tuning that optimizer. The [protocol](PROTOCOL-v1.md)
states exactly what is tested and what is still required before a musical fit.

## What is genuinely different from the first direct head

The old direct phase/period head already supervised both quantities. Merely
adding those losses again is not a new mechanism. Here the tempo and phase tasks
have separately owned projections, residual CNN blocks, heads, clipping and
AdamW states. A phase update cannot change the tempo trunk through a shared
parameter or scale its step through a combined gradient norm. The shared arm
still has that learning path and serves as an explicit control.

Both arms start with identical per-task functions, using fresh weights and the
same full temporal context. The price is explicit: 24,003 trainable parameters
for shared versus 47,907 for separated, with two trunk evaluations instead of
one. Per-task capacity is held constant; total capacity is not. A favorable
result would not prove that interference alone caused all previous failures.

`model.py` owns the actual neural branches and chunk geometry. `supervision.py`
owns supported native-cell tempo targets and frame-phase targets. `training.py`
owns a task's complete-record accumulation, native optimizer and failure state.
`authored.py` records a fixed synthetic learning comparison with before/after/
failure snapshots and all paired receipts. It is not a hidden musical fitter.

## Important output boundary

The model returns `Evidence(log2_cell_period, phase_vector)`, NOT `Clock`.
These are T-1 cell intervals and T frame observations; there is no fabricated
last tempo cell, beat timestamp list, calibrated confidence or no-rhythm label.
We must not ship two independent fields that contradict one another. A coherent
time-axis layer is a separate required stage, to be designed after musical
evidence shows the separate tasks are useful.

Phase wraps around every beat and cannot identify integer counts. Tempo targets
follow the existing native annotation convention, not a universal perceptual
beat level. Missing reference masks never become silence labels or allow a
tempo cell to bridge missing endpoints. No smoothing penalty conceals a step.

## Verification and the next decision

Authored tests cover full-trunk/storage isolation, exact initial-function parity,
complete-context chunk output/gradient parity, native half/double targets,
reference gaps, standalone-equivalent momentum/decay updates, large cross-task
scale changes and failed/returned optimizer boundaries. The fixed forty-step
fixture deliberately provides teacher channels and contains a rate ramp, a step
and a phase-observation gap. Learning it is not evidence of real-audio accuracy,
generalization or weak-beat recognition. All shared and separated scores remain
in the report even when one arm is worse.

The next musical comparison must include shared/separated arms AND independently
fitted zero-feature controls, unchanged fit/development/diagnostic roles and
explicit per-task checkpoint selection. No fresh data or holdout is opened in
this component stage. Its result cannot select another epoch or reopen an old
fit. First ask whether the isolated tasks learn; only then test coherent fusion.
If isolated tasks fail, investigate supervision/features/transfer rather than
adding a fusion policy. If only fusion fails, investigate the clock. If gains
stay on training works, investigate generalization/data coverage. These are
directions for the next controlled experiment, not unique causal conclusions.

```sh
python -m unittest discover -s experiments/separated_evidence -p 'test_*.py' -v
python -m experiments.separated_evidence.authored --device cpu --output NEW_PRIVATE_OUTPUT
```

CUDA uses the same fixed fixture and budget, with `CUBLAS_WORKSPACE_CONFIG=:4096:8`
before process startup. Use a 300-second external watchdog for an authored run.
No command in this component loads music packets or fits a musical checkpoint.
