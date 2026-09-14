# Protected-optimizer musical comparison

Status, 2026-09-14: **completed, musical gate failed, closed without retry**.
Both fits finished 20 epochs / 100 returned updates and all 200 sequence
evaluations. This is a separate experiment, not a resume, release or sweep.

## Actual musical result

The frozen [plan](plan-v1.json), [execution](execution-v1.json),
[epoch/evaluation journal](journal-v1.jsonl), [update receipts](receipts-v1.jsonl)
and [every-case result](results-v1.json) retain the complete comparison.
Main and fitted-zero initial weights matched exactly on Torch 2.10.0 / T4.
The original complete-epoch development-loss rule selected epochs 15 and 7,
not the final epoch or the lowest training loss. Fit times including snapshots,
finite-loss audits, development and export were 325.02 s and 319.84 s.

Same-support work-macro comparison against the previous unprotected fit:

| Cohort | Previous BPM error | Protected BPM error | Previous phase error | Protected phase error |
| --- | ---: | ---: | ---: | ---: |
| Fit | 41.70% | 17.47% | 0.1867 | 0.2080 |
| Development | 62.73% | 42.19% | 0.1837 | 0.2054 |
| ARTBeaT diagnostic | 24.58% | 29.12% | 0.2171 | 0.2309 |

BPM columns are work-macros of recording median percentage errors, not accuracy
or a pooled median. Phase is circular mean absolute error in cycles; lower is
better in every column. Four-second count error improves on development from
1.4552 to 0.8525 cycles but worsens on ARTBeaT from 1.7490 to 2.2160. The report
retains P95 tempo, all earlier baselines, support counts and every intervention.
The protected mechanism improves fit/development tempo but does not jointly
improve tempo and phase or transfer that tempo gain to ARTBeaT.

The fixed joint development gate uses reference-only support: phase improves
15.84% versus fitted zero (0.2436 to 0.2050 cycles), but BPM error worsens from
41.17% to 42.00%, instead of improving by at least 10%. On common support, raw
development BPM/phase errors are 41.59% / 0.1427 cycles, and ARTBeaT raw errors
are 14.45% / 0.1078 cycles. Against raw, median BPM regresses on 4/5 development
and 11/15 ARTBeaT recordings; phase regresses on 3/5 and 15/15 respectively.
All required accuracy gates remain unchanged and the overall gate fails.

Both cohorts' natural-input phase still beats the same-checkpoint zero, mean
and half-roll interventions at work-macro level. This is operational timing
dependence, not proof of correct speed, integer beat count or usable timestamps.
Numerical completion and this intervention gate cannot override regressions.

## What the protected steps establish

All 1,600 term-gradient diagnostics, 800 completed records, 200 returned updates
and 200 protected-step receipts are retained. All musical gradients used one
native VJP band and zero alignment loss; this is not a multi-band stress test.

| Fit | Unchanged groups | Projected groups | Rounded vetoes | Moved / committed steps | Finite count increases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Audio | 176 | 65 | 59 | 94 / 100 | 32 / 100 |
| Fitted zero | 199 | 48 | 53 | 98 / 100 | 11 / 100 |

All 600 group certificates have a defined, nonpositive stored-gradient dot
product with actual native displacement. Six audio and two zero steps stall
entirely while still advancing AdamW state and consuming the fixed budget.
The 32/11 finite count increases explicitly show why first-order protection is
not finite-loss monotonicity. They are observations, not grounds for retrying
steps, tuning a step size or selecting another checkpoint.

A separate read-only audit recomputes 320 metric blocks from all 40 prediction
archives, verifies 10 snapshot hashes and both selected exports, checks the
eight retained complete-record inputs of the last two batches, and independently
recomputes 12 exact Fraction signs for the last two native proposals. It checks
those candidates against actually committed weights and proposed optimizer
states against committed states. No model forward or optimizer update is used.
The first auditor revision compared missing-reference NaNs as unequal; only
that read-only equality check was corrected. Training and predictions were not
rerun. Public outcome checks reproduce all macros, gates, shuffles, work weights,
receipts and finite after-loss sums without private inputs or learned weights.

Whole-process host peak RSS was about 1,752 MiB. The private run archive is
identified by SHA-256
`72af4b2f77d41a87645b815724d8110db27ccb6eb575e6c72f341db26d80c785`;
audio, feature packets, learned weights and snapshots are not distributed.

Close this fixed experiment. Do not tune projection tolerances, blend public
strategies or add epochs to rescue it. The measured count-direction interference
was real, but protecting that direction is insufficient for joint musical
accuracy. The next proposal must address the speed-prior / phase-alignment and
representative-supervision problem, using these paired fixed outputs to make
a distinct, falsifiable prediction before any new fit. This run alone does not
identify which architectural or data change will work. No checkpoint promotion,
holdout access, extra fit, production change or release is authorized by failure.

## Frozen execution design

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
by a favorable synthetic result or a relaxed comparison. The artifacts above
retain the original failed outcome rather than rewriting an earlier experiment.
