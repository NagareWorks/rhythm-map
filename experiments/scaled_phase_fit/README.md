# Observed range-safe phase-clock musical validation

Status, 2026-09-11: **completed, research gate failed, closed without retry**.
Both fits completed 20 epochs / 100 updates and all 200 sequence evaluations.
This is a new experiment, not a resume of the closed `phase_sync_fit` or a release.

## Actual outcome

The [target-runtime plan](plan-v1.json), [execution](execution-v1.json),
[epoch/evaluation journal](journal-v1.jsonl), [all gradient/update receipts](receipts-v1.jsonl)
and [every-case result](results-v1.json) retain the complete comparison. Main and
fitted-zero initialization matched exactly on Torch 2.10.0/T4. Both selected
epoch 20 by the fixed development-loss rule. Complete fit times, including
snapshot collection and evidence-report I/O, were 146.89 s and 144.30 s.
Private best checkpoints, exported weights and every prediction archive were
independently hash/selection-audited without a model forward or optimizer step.

Reference-only development phase error improves 22.60% over fitted zero, but
BPM error worsens from 42.46% to 62.60%, so the joint 10% improvement gate fails.
The common-support work-macro comparison is:

| Cohort | Raw BPM error | Audio BPM error | Fitted-zero BPM error | Raw phase error | Audio phase error | Fitted-zero phase error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 41.59% | 62.73% | 42.36% | 0.1427 | 0.1837 | 0.2369 |
| ARTBeaT diagnostic | 14.45% | 24.58% | 42.55% | 0.1078 | 0.2171 | 0.2429 |

BPM columns are work-macros of recording median percentage errors, not accuracy
or one pooled median. Phase is circular mean absolute error in cycles. On common
support, 4/5 development recordings regress versus raw in median BPM error and
2/5 in phase; ARTBeaT regresses on 11/15 BPM and 15/15 phase. The report also
retains P95 tempo, four-second count drift, v1/coupled/trajectory comparisons
and every intervention. ARTBeaT tempo improves versus the previous learned
clocks but not raw; this cannot hide the development regressions.

Both cohorts' natural-input phase beats all three SAME-checkpoint zero/mean/
half-roll controls. Thus numerical completion and operational timing dependence
pass, but neither implies correct beat counts, beat level or usable BPM. The
overall musical gate fails. No checkpoint is promoted, no holdout is opened,
and no extra fit/epoch/seed/strategy is selected after observing this outcome.

## What the gradient evidence says (and does not say)

All 800 completed training-record gradients and 200 successful update receipts
are retained. Every record used exactly ONE native CNN VJP band; alignment-loss
counts stayed zero. Main parameter-gradient norm log2 ranged from -0.0270 to
10.6230, fitted zero from -2.9128 to 9.1119. All 100 main batches and 96/100 zero
batches had pre-clip norm above 1. These are diagnostic distributions, not new
acceptance thresholds. Whole-process host peak RSS was about 1,683 MiB.

This musical run did NOT exercise multi-band range extension or reproduce the
old unsaved overflow. The new transport/weighting/accumulation rounding leads
to a different optimization trajectory despite identical initial weights. The
synthetic multi-band witness remains separate evidence; do not claim this run
proved arbitrary-length stability or repaired the exact old failure. It does
remove missing crash evidence as an excuse for an unmeasured outcome.

Next: read-only attribution from these fixed predictions/fields and existing
labels, separating circular phase alignment from native tempo/count errors
(including octave ambiguity). Do not infer a specific cause from the aggregates,
alternate losses or add epochs without a new falsifiable mechanism. Independent
acceptance, representative region labels and rhythm availability remain open.

## Execution design and reproduction boundary

The [protocol](PROTOCOL-v1.md) changes only the numerical gradient transport
and diagnostic observation around the same CNN, forward clock, loss, data,
work weights, optimizer and fixed accuracy gates. Its strict target-runtime
plan pins the prior failure outcome, numerical prerequisites, executable source
closure, population and initial weights before any musical forward or update.
The main and zero-feature fits must start from that exact same initialization.

`MusicalRecorder` delegates full-record gradient math and the optimizer lifecycle
to the already validated observed transport. It adds fit-wall deadline checks,
per-record gradient receipts and clean development-record boundaries. Training
failure snapshots retain current/best state; development has no backward trace.
Selection/export failures are marked separately. Evaluation reuses the frozen
metric/control implementation after both fits, not a new timing estimator;
an evaluation failure retains the selected checkpoints and partial prediction
artifacts but does not claim a training-backward replay for that inference.

The two-record synthetic cost gate is prerequisite evidence, not a four-record
musical throughput measurement. Actual fit timing, memory, update durations,
gradient norm ranges, VJP counts and snapshot identities belong in the outcome.
All-zero gradients remain counted with null norms. No gradient-size threshold
chooses a favorable checkpoint or replaces the unchanged musical accuracy gates.

Commands (fresh private output paths; never publish weights or cached inputs):

```sh
python -m experiments.scaled_phase_fit.register --device cuda --output PLAN.json
python -m experiments.scaled_phase_fit.run --device cuda --plan PLAN.json \
  --expected-plan-sha256 SHA --inputs PRIVATE_INPUTS --output NEW_OUTPUT
python -m unittest discover -s experiments/scaled_phase_fit -p 'test_*.py' -v
```

CUDA requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process startup. Register
on the actual execution runtime, not the VDI and then another Torch build.
Use a host-disk output bind mount and a separate 3,900-second process watchdog.
There is no resume/retry command or user-facing musical strategy switch.
