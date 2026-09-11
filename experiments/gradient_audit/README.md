# Fixed-checkpoint gradient interference

Status, 2026-09-11: **audit complete; no model update or product admission**.
The [registered protocol](PROTOCOL-v1.md), [actual T4 plan](plan-v1.json),
[all 80 cases and summaries](results-v1.json), and [completion journal](journal-v1.jsonl)
retain the outcome. The same two selected epoch-20 checkpoints were inspected on
the same 40 recordings, without changing fit/development/diagnostic roles.

## Main finding: the count signal arrives, but the joint direction can oppose it

This is now measured at the actual CNN parameters, not inferred from scalar
loss magnitudes. On the fixed natural-input checkpoint, sum all 20 FIT recordings
with weights `1/(9 * recordings_in_work)`. The resulting full-fit gradients are:

| Parameter group | Cosine(count, direct prior) | Cosine(phase, count) | Phase/count norm ratio | Cosine(total, direct prior) |
| --- | ---: | ---: | ---: | ---: |
| Period output row | +0.9973 | -0.9558 | 4.52 | -0.9239 |
| Shared CNN | +0.9944 | -0.9701 | 4.59 | -0.9437 |
| Complete model | +0.9950 | -0.9645 | 4.57 | -0.9361 |

Cosines are dimensionless direction comparisons: +1 means the same direction,
-1 the opposite, not an accuracy percentage. The direct-prior diagnostic is
mean squared error between the pre-feedback log2 period and the native reference
log2 period. It is an inspection target, NOT a newly fitted auxiliary loss.

For example, a positive dot product between count and direct-prior gradients
means an infinitesimal step in the negative count-gradient direction decreases
the direct-prior diagnostic. Here the count signal is strongly aligned, but
the phase gradient is larger and almost opposite. Their sum points against
direct-prior calibration in both the period head AND the shared CNN. This
rejects a blanket explanation that the count supervision simply never reaches
the rate-learning parameters at this checkpoint.

The separately fitted zero-feature checkpoint is a useful counterexample, not
a replacement model: its full-fit period-head count/prior cosine is +1.0000,
phase/count cosine -1.0000, but phase/count norm ratio is only 0.88. The total
still aligns with the prior (+1.0000 in that head; +0.9430 over all parameters).
There is therefore no universal rule that adding phase always reverses speed
learning. Neither checkpoint meets the earlier musical acceptance gate.

## Per-record evidence remains mixed

| Natural-input cohort | Count/prior head cosine < 0 | Total/prior head cosine < 0 |
| --- | ---: | ---: |
| Fit | 5/20 | 12/20 |
| Development | 1/5 | 1/5 |
| ARTBeaT diagnostic | 1/15 | 5/15 |

These are recording counts, not work-macro percentages. Work-macro averages of
per-record head cosines are different from the cosine of the summed gradient:
for natural-input development, count/prior is +0.4818 and total/prior +0.5008.
That does not contradict the negative full-FIT aggregate above. Do not average
unit-normalized gradients or combine development/diagnostic with fit gradients.

One concrete development contrast is retained: Mussorgsky/Staab's head count
gradient aligns +0.9989 with direct-prior calibration, phase/count is -0.9976,
and total/prior becomes -0.9987. Boccherini/Krux and three other development
recordings do not show that total-head reversal. Even count-only gradients are
not uniformly aligned per recording, so removing phase is not a demonstrated
musical solution either.

## What was actually executed and checked

- 80 recording/checkpoint pairs: 320 CNN field forwards and 240 clock forwards.
  Autograd is enabled; there is no optimizer construction, clipping/installing
  of parameter gradients, parameter update, new encoder inference or holdout.
- Original phase/count/total losses use the unchanged native support and frozen
  clock_loss. Every field/clock forward exactly reproduces the saved evaluation
  arrays. All cohort original native loss components reproduce the earlier
  fixed-prediction report.
- All model-state hashes and CPU/CUDA RNG states match before/after. Parameter
  `.grad` buffers remain None. No AdamW state or failed-fit resume state is loaded.
- Independently differentiated phase + count agrees with total: maximum field
  residual is 8.19e-15 and parameter residual 3.76e-7, relative to the sum of the
  component norms. No alignment loss or numerical audit failure occurred.
- Every complete case is persisted as metrics AND gradient packets before
  proceeding. A separate CPU-only, no-forward packet audit reloaded all 80
  packets/metric rows and recomputed every gradient comparison, all cohort
  macros, and both complete-fit aggregates successfully.
- Actual audit time, including result persistence, was 55.38 s, versus the
  predeclared 1,200 s budget. Peak CUDA tensor allocation was 19,574,784 bytes;
  that is NOT total driver/context memory or an end-to-end audio benchmark.

Before musical gradients, review found that an interrupted run could lose
completed metrics held only in memory. Per-case persistence and a regression
test fixed that. Term-entry deadlines also clear previous traces, and loading/
summary failures are labeled separately. The executed source was frozen only
after those corrections and all 17 authored CPU/T4 checks; outcome checks live
outside that source closure. The two earlier preflight sources did not execute
any musical gradients. There was one completed musical audit, not a retry sweep.

Private full-gradient packets and the packet-audit helper/log are retained, not
published as model artifacts. The full private report's SHA is recorded in the
public result and journal. The packet-audit helper SHA256 is
`b7affe5a9ae5060c4ea90bdbb9c172ccc1f7c1f45f9c0c2c0b1fb72c77d9d3c6`.

## Limits and next falsifiable mechanism

This establishes gradient interference at the SELECTED checkpoints. It is not
a reconstruction of the preceding 100 AdamW updates: per-batch shuffles,
momentum, preconditioning and weight decay can change update directions.
Nor does an infinitesimal Euclidean direction prove a finite-step BPM change.
The correct pre-feedback prior need not equal the optimum jointly corrected
clock, so this diagnostic alone does not prove the entire musical failure's
cause or justify increasing/decreasing a scalar loss weight.

The next proposal should test a controlled gradient-routing mechanism: protect
native-speed learning while retaining acoustic phase correction. A test must
cover the shared CNN as well as the final period output row; blocking only the
last row does not eliminate the measured shared-parameter interaction. First
verify the learning paths and representable authored timing cases, then define
one falsifiable musical comparison before any new fit. This is not approval to
detach the old recurrence, alter this experiment, sweep loss weights, add a
public policy switch, promote a checkpoint, or release a package.

Data representativeness, native beat-level ambiguity, rhythm availability and
independent acceptance remain open. Shipping APIs/defaults and the one-call,
zero-per-song-tuning contract are unchanged.

```sh
python -m experiments.gradient_audit.register --output NEW_PRIVATE_PLAN.json
python -m experiments.gradient_audit.run --inputs PRIVATE_INPUTS \
  --predictions PRIVATE_SELECTED_PREDICTIONS --plan NEW_PRIVATE_PLAN.json \
  --expected-plan-sha256 SHA --output NEW_PRIVATE_OUTPUT
python -m unittest discover -s experiments/gradient_audit -p 'test_*.py' -v
```

Registration/execution requires the pinned T4 runtime and deterministic CUDA
workspace. The commands document the frozen execution; a fresh path does not
authorize another musical attempt or changing the completed result.
