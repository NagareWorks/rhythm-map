# Frozen-gradient audit: the native objective also conflicts with retention

The [fixed audit](PROTOCOL-v1.md) completed in 31.61 s on the private T4. It
inspected all 41 distinct checkpoint slots from the closed local-tempo fit
(42 exports, with the common epoch-zero state computed once), all 40 consecutive
epoch movements, and the exact first recorded minibatch. **Zero optimizer steps,
encoder calls, ART/fresh-schedule forwards or model changes.** The
[full result](results-v1.json) and [plan](plan-v1.json) retain every inspected state.

## Strong initial conflict is not limited to the added pair term

At each frozen state, compute the expected work-balanced native-fit gradient,
all 559 admitted points' equal-group pair gradient, the fit-only teacher penalty,
and separate development-record gradients. Development gradients are diagnostic
only. No gradient routing, optimizer proposal or weight update is installed.

Cosines below compare each training gradient with the development work-macro
**MSE gradient**. Positive means its negative-gradient direction locally reduces
development MSE; negative means local opposition. They are neither BPM accuracy
percentages nor predictions of the actual AdamW update.

| Initial state | Gradient norm | Cosine with development MSE |
| --- | ---: | ---: |
| Native-fit expectation | 2.55344 | -.93803 |
| All pair groups | .45152 | -.13549 |
| Global pair contribution | .23828 | -.08654 |
| Local pair contribution | .24902 | -.16289 |
| Native + pair | 2.61357 | -.93986 |
| Teacher retention | 0 | undefined |

The stronger initial raw-gradient opposition comes from native fitting, not
solely from the new paired relation. This does not mean native annotations are
wrong or pair supervision is harmless. Work-level directions differ sharply:

| Initial development work | Native-fit cosine | Pair cosine |
| --- | ---: | ---: |
| Boccherini | +.92378 | -.20406 |
| Mozart | -.98167 | -.07861 |
| Mussorgsky | +.97959 | +.08241 |

The native direction conflicts with Mozart while locally agreeing with the other
two works. The old [tempo attribution](../tempo_attribution/README.md) already
found large Mozart recording-level bias and weak local response; this new audit
does not repeat that scalar accounting or prove bias alone causes the gradient
conflict. The datasets remain tiny and exposed, not new generalization evidence.

## Actual first minibatch confirms the initial direction

Reconstruct its exact four natural records and 15 groups x four sampled points,
preserving repeated indices. The combined unclipped norm is 3.31820407 versus
the recorded 3.31820393, within the frozen float32-accumulation tolerance.
Both arms have the identical first draw and recorded losses. The native-gradient
cosine is -.92129; all-pair -.13158; combined -.92745. This is not an artifact of
using a full-population expectation instead of that particular first batch.

The initial-output MSE and its gradient are exactly zero at the teacher's own
weights. Thus this soft penalty supplies no opposing gradient at the very first
update. It may respond after movement, unlike a hard constraint. Zero initial
gradient does not imply the loss has zero curvature or is useless afterward.

## All states, including exceptions

Counts below use all 41 inspected slots, not independent samples. A contribution
is called opposing only when its gradient dot development-macro gradient is
negative; the direction is that term alone, not the optimizer displacement.

| Term | Opposing slots | Aligned slots | Zero-gradient slots |
| --- | ---: | ---: | ---: |
| Native-fit | 34 | 7 | 0 |
| Global pair contribution | 2 | 39 | 0 |
| Local pair contribution | 37 | 4 | 0 |
| All pair groups | 17 | 24 | 0 |
| Teacher retention | 1 | 39 | 1 |
| Native + pair | 34 | 7 | 0 |
| Native + pair + retention | 10 | 31 | 0 |

Global/local/identity gradients keep their original contribution weights (group
means divided by 15); they are not individually reweighted to unit strength.
Retention on unretained-arm states is a diagnostic of the penalty, not a term
that participated in that arm's historical training. Its one opposing nonzero
slot is retained epoch 1; it is usually locally helpful, but is neither always
helpful nor sufficient for the old acceptance gate. Local-pair conflict remains
visible, so removing only native opposition is not a promised complete solution.

## Observed movement is a separate check

For consecutive saved epochs, delta is the actual weight difference. Compare
observed development MSE change with the start-gradient dot delta and the mean
of the two endpoint-gradient dot deltas. No synthetic checkpoint is evaluated.

| First epoch / development macro | Actual MSE change | Start linear | Endpoint trapezoid |
| --- | ---: | ---: | ---: |
| Local-only | +.058634 | +.048178 | +.057810 |
| Retained | +.010258 | +.008503 | +.010183 |

For every one of the 20 intervals in each arm, both estimates match the sign of
the observed macro MSE change. Maximum absolute trapezoid residual is .001335
for local-only and .000524 for retained. Every per-record/work residual remains
in the report; the table does not imply exact reconstruction or exact additivity
of objective responsibility.

Five AdamW steps lie between epoch exports. Their intermediate weights and
optimizer moments were not saved. Adaptive coordinate scaling, momentum,
clipping, weight decay and curvature prevent treating raw-gradient alignment as
the actual update or assigning causal percentages to native versus pair losses.
Authored tests show both diagonal-scaling sign reversal and nonzero endpoint
approximation residuals. No existing failed model is selected or promoted here.

## What this changes next

Do not merely increase epochs or tune the teacher coefficient. A concrete next
controlled training hypothesis is to separate the native loss's recording-level
log-period offset term from its within-recording variation term, while keeping
the paired-speed target, architecture, data roles and retention policy fixed.
Compare the original native objective with variation-only native supervision
under a new fixed contract. Any reference centering belongs only inside FIT
loss computation, never an oracle output correction at evaluation/inference.
Keep absolute-BPM metrics and the original failure gates unchanged: avoiding a
damaging absolute calibration direction is not itself learning correct BPM.

This is a proposed test, not an implemented loss change or a claim that it will
fix the problem. It follows the strong native/development conflict and the prior
bias evidence; the current packets do not decompose the native gradient into
offset and variation components. Original holdout remains sealed, no new product
strategy or user parameter, no encoder-size conclusion or release.

```sh
python -m unittest discover -s experiments/tempo_conflict -p 'test_*.py' -v
timeout --signal=TERM --kill-after=10s 1800s python -m experiments.tempo_conflict.run \
  --local-run PINNED_LOCAL_TEMPO_RUN --old-run PINNED_RELATIVE_RUN \
  --inputs PRIVATE_NATURAL_PACKETS --weights PINNED_SEPARATED_NATURAL_EXPORT \
  --output FRESH_PRIVATE_OUTPUT --device cuda
```

Private artifacts retain primitive float64 gradient vectors, actual saved theta,
per-state metrics and entered/returned diagnostic receipts. Their statistics are
independently recomputed without neural forwards. Old experiments stay byte-pinned.
