# Paired-speed continuation: learns training pairs, not reliable local transfer

The [fixed protocol](PROTOCOL-v1.md) executed on the private T4. Both arms made
100 AdamW updates from the identical existing tempo-branch weights. No encoder
fine-tuning, phase update, new rendering, holdout access or shipping change.
The [complete result](results-v1.json) retains all 40 natural recordings and all
15 source/profile groups. The [plan](plan-v1.json) retains all 915 grid points,
including exclusions. This is an exploratory failure, not a model release.

## What actually trained

The existing 23,937-parameter tempo CNN was continued, not replaced. The
native-only control used the 20 original fit recordings / 9 works. The paired
arm used that same objective and schedule, plus sparse relative-speed MSE on
the 3 source recordings' processed-identity, global .8x and global 1.25x audio.
Both query sides receive gradients; no rate/profile ID enters the model input.
The final 100th update was used, with no result-dependent checkpoint selection.

The frozen encoder processed **18 actual PCM files**: 3 unchanged prefixes and
15 retained Rubber Band previews. Log-mel spectra were recomputed through the
upstream frontend; hidden features were not time-warped. Head reconstruction and
unchanged encoder-state checks passed. This is not a new Rust frontend parity
certification. Feature capture took 9.79 s, native-only continuation 3.91 s,
paired continuation 10.91 s, and the complete experiment 27.63 s, excluding input
transfer and subsequent CI. The two arms need not have equal compute cost.

Of 915 grid points, 559 qualify for this sparse experiment: **334 training and
225 reserved-schedule points**. Excluded: 142 unsupported by alignment, 79
noninterior, and 135 too close to CNN-context seams/edges. No support was filled
between points. The 1,300 beat-query points did not enter training or evaluation.
All three sources remain; this does not overturn the old full-population failure.

## Relative speed results

For actual .8x/1.25x changes, the zero-change predictor has log2-rate RMSE 0.32193.
Below, RMSE uses **changed points only**, never padding the score with identity
points. Lower is better. Global rows were used in training; local rows were not.

| Source / profile | Native-only RMSE | Paired RMSE | Paired slope | Correct sign |
| --- | ---: | ---: | ---: | ---: |
| Vivaldi / global slow | .1953 | .1447 | .677 | 100% |
| Vivaldi / global fast | .6205 | .2016 | .448 | 90% |
| Bach / global slow | .3433 | .1367 | .745 | 100% |
| Bach / global fast | .1393 | .0901 | .882 | 100% |
| Berlioz / global slow | .1708 | .1075 | .877 | 100% |
| Berlioz / global fast | .2955 | .0954 | .813 | 100% |
| Vivaldi / local slow-fast | .3958 | .2724 | .494 | 72.4% |
| Vivaldi / local fast-slow | .4904 | .3433 | .108 | 69.0% |
| Bach / local slow-fast | .4656 | .3630 | -.054 | 53.6% |
| Bach / local fast-slow | .3725 | .2559 | .375 | 80.0% |
| Berlioz / local slow-fast | .2392 | .2520 | .270 | 86.7% |
| Berlioz / local fast-slow | .2883 | .1645 | .675 | 93.3% |

All six trained global-change groups improve against native-only. Among six
reserved local-change groups, five improve against native-only, four beat the
zero-change RMSE, but only **one** (Berlioz local fast-slow) satisfies every fixed
transfer condition. The other Berlioz schedule regresses on changed-only RMSE;
including unchanged points would misleadingly make that row look improved.
All three processed-identity groups pass the .05-log2-unit preservation gate.
Zero-feature paired-model queries produce zero relative response, as expected.

The model can learn the supplied global speed relation. This does **not** show
reliable detection of local changes, their exact onset, perceptual beat level,
or independence from renderer artifacts. Reserved schedules share source music,
rates and rendering algorithm with training. The sparse points are correlated,
and the orchestral coverage is selective. Do not call 1/6 a general accuracy rate.

## Natural music retention fails

Work-macro native BPM median error, using the same outgoing-cell denominator
for all three models in this experiment (not an imported old headline metric):

| Exposed cohort | Initial | Native-only final | Paired final |
| --- | ---: | ---: | ---: |
| Fit, 20 recordings / 9 works | 16.27% | 11.49% | 13.09% |
| Development, 5 recordings / 3 works | 35.22% | 47.18% | 45.57% |
| ART diagnostic, 15 variants / 1 source | 27.98% | 26.98% | 28.62% |

Development MSE likewise increases .32077 -> .49163 / .45566. Because **both**
continuations worsen development while improving fit, the regression cannot be
attributed solely to the pair term. Paired development is better than native-only
but still materially worse than the initial model. ART paired MSE .44715 exceeds
1.05 times native-only .41944. The report includes each natural recording's
4-second large-change response, MSE, median and P95 error, preserving failures.

## Decision and next boundary

Execution completed; identity passed; schedule transfer and natural retention
failed. Keep the old default and the holdout sealed. No automatic retry, parameter
sweep, extra epochs or fusion. These results identify a useful but insufficient
learning signal: global pair learning does not reliably transfer to local speed
schedules, and continued natural fitting shows a fit/development generalization
gap. They do not establish that the frozen encoder lacks the information or that
a larger CNN/Transformer is necessary.

A follow-on experiment should explicitly test local-schedule supervision while
retaining the existing work-disjoint natural validation and genuinely withheld
schedule controls, and address retention/checkpoint policy under a new fixed
contract. Reusing these exposed local scores would no longer be a fresh check.
Do not spend another run merely increasing this same continuation's step count.

```sh
python -m unittest discover -s experiments/relative_tempo -p 'test_*.py' -v
timeout --signal=TERM --kill-after=10s 5400s python -m experiments.relative_tempo.run \
  --pairs PRIVATE_RETAINED_PCM --assets PINNED_ENCODER_ASSETS --deps PINNED_DEPS \
  --inputs PRIVATE_NATURAL_PACKETS --weights PINNED_SEPARATED_NATURAL_EXPORT \
  --output FRESH_PRIVATE_OUTPUT --device cuda
```

The private artifacts retain features, final weights, per-frame predictions and
entered/returned optimizer receipts. Audio/checkpoints are not bundled into Git.
