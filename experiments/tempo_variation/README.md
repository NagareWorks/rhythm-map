# Native-offset ablation: less regression, still no retained learned checkpoint

The [fixed comparison](PROTOCOL-v1.md) completed both 100-update fits in 45.72 s
including evaluation. Removing the recording-wide native residual offset term
improves several outcomes relative to the matched control, but **both arms still
select epoch 0**. Every post-update epoch fails the unchanged development rule.
This closes the hypothesis as insufficient, not as a new shipping estimator.
The [result](results-v1.json) and [point ledger](plan-v1.json) preserve all cases.

## What changed, and the exact control

Keep the same 23,937-parameter CNN, frozen features, starting weights, teacher
penalty, paired-speed objective, optimizer, batches and selection policy.
Only the active native loss changes:

`mean(residual^2) = mean(residual)^2 + mean((residual - mean(residual))^2)`.

The control retains both terms; the primary trains only the second. Centering is
inside each FIT recording's loss, never at inference/evaluation. Reference masks
and absolute BPM metrics stay unchanged. The two arms share all 100 draws.
All 21 control epoch states and their development/pair scores exactly reproduce
the prior local-retained fit. This isolates the loss ablation in this fixed run;
it does not establish independent generalization or a universal causal result.

## Outcomes, including regressions

Work-macro native BPM median error; lower is better:

| Cohort | Initial / both selected | Full-native control terminal | Variation-only terminal |
| --- | ---: | ---: | ---: |
| Fit: 20 recordings / 9 works | 16.27% | 13.01% | 16.28% |
| Development: 5 recordings / 3 works | 35.22% | 42.37% | 37.80% |
| Exposed ART: 15 variants / 1 source | 27.98% | 27.41% | 29.73% |

Training-pair MSE falls from .08440 to .02417 (control) / .01524 (variation).
Development MSE is .32077 initial, .41567 control terminal and .34417 variation
terminal. Thus removing offset reduces the control's development regression,
but does not retain the starting model's absolute timing performance. It also
loses the control's ART benefit. Both components of native MSE remain reported
per record: lower within-record variation error can coexist with a larger
recording-level offset, so it cannot substitute for absolute BPM accuracy.

In epoch 1, variation fails Boccherini MSE/BPM and Mussorgsky BPM constraints.
At its terminal epoch Boccherini and Mozart fail both metrics. No learned epoch
is admissible, not merely the final epoch. Selected-epoch-zero retention passes
only because those selected weights are unchanged; it is not learning success.

Changed-only log2-rate RMSE on the six **already exposed** schedule diagnostics:

| Source / schedule | Control terminal | Variation terminal |
| --- | ---: | ---: |
| Vivaldi / slow-fast-normal | .2680 | .2056 |
| Vivaldi / fast-slow-normal | .2209 | .1902 |
| Bach / slow-fast-normal | .1927 | .1720 |
| Bach / fast-slow-normal | .2395 | .2146 |
| Berlioz / slow-fast-normal | .1410 | .1279 |
| Berlioz / fast-slow-normal | .1830 | .1898 |

Five improve relative to control, one regresses. These are the same three source
works, six cached transformations and renderer already inspected in local-tempo
v1. Their 366 grid points / 224 admitted / 140 changed are relabeled exposed,
never called fresh validation. No new encoder calls, music, labels or support.
They neither train nor select this run. The primary selected model passes only
one schedule's complete old numerical checks and fails the learned-checkpoint
requirement. The overall exposed exploratory gate remains **failed**.

## What is closed; what comes next

Offset-only removal is insufficient, although this comparison supports it as
one contributor to the fixed run's regression. Do not sweep its weight, lengthen
training, relax a work-level gate, oracle-center outputs, or retain rejected
heads as product strategies. Original holdout/defaults remain untouched.

The next useful workstream is training-source coverage: inventory legally usable
native and paired-speed training works disjoint from development and the sealed
holdout, before registering a broader-source comparison. The current 9 native
fit works and 3 paired sources do not represent broad musical variation. This
is a next hypothesis, not proof that more data solves the gap, and not evidence
that the frozen encoder or CNN must be replaced. Avoid another scalar-loss cycle
on these same exposed works without new mechanistic evidence.

## Reproduction and retained failures

All 200 updates retain before/after weights, optimizer moments, RNG, exact draws,
unclipped/clipped gradients and entered/returned receipts. All 42 epoch exports,
selected/terminal outputs and per-frame predictions remain private. The public
plan maps the frozen evaluator's legacy NPZ aliases to current arm names; those
aliases do not change any model or numeric metric. See the
[closed-direction index](../RESEARCH-DECISIONS.md) before proposing another fit.

Read-only replay passed all 42 epoch scores and 408 prediction comparisons
bit-for-bit. An independent NumPy AdamW formula checked all 200 saved updates;
maximum absolute weight discrepancy was 3.72e-8 (below the fixed 2e-7 numerical
tolerance). Replay took 14.37 s, with zero optimizer steps or encoder calls.
The 28 authored/frozen-result tests include exact control parity, decomposition,
role boundaries, all selection decisions and favorable/unfavorable outcomes.

```sh
python -m unittest discover -s experiments/tempo_variation -p 'test_*.py' -v
timeout --signal=TERM --kill-after=10s 5400s python -m experiments.tempo_variation.run \
  --local-run PINNED_LOCAL_TEMPO_RUN --old-run PINNED_RELATIVE_RUN \
  --inputs PRIVATE_NATURAL_PACKETS --weights PINNED_SEPARATED_NATURAL_EXPORT \
  --output FRESH_PRIVATE_OUTPUT --device cuda
```
