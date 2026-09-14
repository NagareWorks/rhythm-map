# Protected-optimizer musical comparison

Status: registered-source preparation for one user-authorized fit pair. This
is a separate experiment, not a resume, release, new product strategy or sweep.

The [protocol](PROTOCOL-v1.md) keeps the last observed fit's model, seed,
population, work weighting, optimizer hyperparameters, budget, development
selection and accuracy gates. It changes the update mechanism to the validated
[count-priority native displacement boundary](../protected_update/README.md).
No old learned weights or new encoder inference are used.

Main and fitted-zero controls each have 20 epochs / 100 returned updates and
a 1,800-second limit. They use the same target-runtime initialization and the
same 20 fit / 5 development / 15 diagnostic-only recordings. Both must finish
before the unchanged 200-sequence evaluation. Holdout remains sealed.

In addition to all term-gradient and update receipts, record group projections,
native-rounding vetoes, fully stalled steps and original weighted count/phase
losses before and after every update. These finite-loss audits do not change
the optimizer, selection score or step size. Native first-order nonincrease
does not imply finite loss descent or correct musical beat level.

`MusicalRecorder` owns fit deadlines and these receipts; the frozen protected
recorder owns batch gradients and commit/rollback. The new fit driver owns this
experiment's fixed lifecycle and exports, importing the original evaluation
and comparison gate unchanged. It never monkey-patches an older fit runner.
Snapshots include every completed record's owned input/RNG, not just the last.

Commands require fresh private output paths and the same target runtime:

```sh
python -m experiments.protected_phase_fit.register --device cuda --output PLAN.json
python -m experiments.protected_phase_fit.run --device cuda --plan PLAN.json \
  --expected-plan-sha256 SHA --inputs PRIVATE_INPUTS --output NEW_OUTPUT
python -m unittest discover -s experiments/protected_phase_fit -p 'test_*.py' -v
```

Use deterministic `CUBLAS_WORKSPACE_CONFIG=:4096:8` and an outer 3,900-second
watchdog. Any caught failure closes this run; if main fitting fails, do not run
fitted zero or diagnostic music. A failed musical gate must not be replaced
by a favorable synthetic result or a relaxed comparison. Full public-safe
results and private evidence identities will be retained after execution.
