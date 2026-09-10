# Sustained native-clock alignment: loss-only experiment

Status, 2026-09-10: both fixed 20-epoch fits completed; the musical gate failed.
See the [complete outcome](../../evaluation/baselines/trajectory-clock-learning-v1.md),
[pre-fit protocol](PROTOCOL-v1.md), [plan](plan-v1.json), [execution](execution-v1.json)
and [every-case metrics](results-v1.json). No model/default/pack/release change.

## Why this experiment exists

The retained coupled-v1 predictions remain badly aligned even after a
truth-assisted best constant phase shift. [Attribution](attribution-v1.json)
separates a single origin correction from within-span constant-rate drift and
nonlinear residuals. This is post-fit oracle analysis, never a repaired output
or an independent benchmark. No annotation hole is used as a count bridge.

The experiment changes only the training/development loss. It reuses the exact
23,970-parameter CNN, positive integration, frozen features, population,
initialization, optimizer settings, complete-crop loops and musical metrics.
The optimization-loop snapshots are source-tested against coupled v1 with only
the objective call renamed; old code and reports are not monkey-patched or edited.

## The new training target, explicitly

For one contiguous reference span:

```text
e[t] = predicted cumulative beats[t] - native reference cumulative beats[t]
k = nearest integer to mean(e)            # one beat-number origin for the span
loss = mean((e[t] - k)^2)
```

Integer beat numbering is arbitrary, so a uniform shift of exactly three beats
does not change the loss. But an error that grows from zero to three beats cannot
be erased separately at each frame. The loss decomposes into trajectory-error
variance plus squared registered mean error. It is measured in beats squared,
not relative tempo percent, and can sacrifice local tempo to reduce global error.

The integer registration is detached; away from branch ties the derivative with
respect to each predicted count is `2 * (e[t] - k) / number_of_valid_frames`.
At ties, round-to-even selects one minimizing branch. The neural optimization is
not thereby convex. This replaces the old circular-phase/multi-lag loss; no
coefficient sweep or new architecture is hidden behind the change.

Registration uses reference labels **only inside the loss**. Inference still
returns the model's original positive clock with its predicted origin; no oracle
shift, per-window reset or label-informed correction is applied to reported
musical metrics. Separate annotation spans have separate unknown integer gauges,
while the predicted clock itself never resets.

## Reusable pieces and retained boundaries

- `attribution.py`: exact circular L1 origin bound and within-span OLS error
  decomposition, using the old private predictions without fitting.
- `supervision.py`: native trajectory objective and gradient contract.
- `register.py`: exclusive pre-fit source/parent-data plan.
- `run.py`: the fixed pair of fits, checkpoint choice and unchanged musical gates,
  plus a nonregression gate against the previous coupled head.
- `test_supervision.py`: authored loss/gradient/source-equivalence tests.
- Public JSON files contain hashes and diagnostics, not audio/features/weights.

The 20/5 RUBATO fit/development split and 15 ARTBeaT diagnostics stay unchanged.
There are still no audited omission/change/no-rhythm training labels, independent
acceptance, performer-independence proof or new holdout access. A good aggregate
on ARTBeaT cannot promote this model over its development/phase regressions.
Close this fixed run. A next proposal must combine local acoustic phase evidence
with native count/tempo consistency; do not alternate scalar-loss variants or
expose failed experiments as user-selectable product strategies.
