# Local-tempo continuation: learning without a retained checkpoint

The [fixed protocol](PROTOCOL-v1.md) completed two matched 100-update fits on the
private T4, but **neither arm produced an admissible learned epoch checkpoint**.
Both selected epoch 0, the unchanged starting tempo head. That fallback is not
training success. No shipping model, public option, encoder or phase update.
The [result](results-v1.json), [full point ledger](plan-v1.json) and
[audio preparation report](preparation-v1.json) preserve the rejected outcomes.

## Mechanism and ownership

Keep the existing 23,937-parameter CNN and frozen Beat This features. Continue
the same pre-relative separated-natural weights in both arms:

- `local-only`: existing natural-label loss plus sparse relative-speed loss.
- `local-retained`: those same losses plus coefficient-1 MSE against the initial
  teacher's detached log2-period on the same valid natural FIT cells.

The teacher is an output-movement penalty, not extra truth or a guarantee about
unseen works. Both arms receive identical natural batches and pair draws. All
559 previously admitted pair points now train, including the 225 exposed local
points. They are explicitly relabeled `fit_exposed`; the old experiment is not
rewritten. No interpolated feature warping, profile ID input or dense support.

At epoch 0 and every full epoch, check the five work-disjoint natural development
recordings. Each work must retain MSE and BPM median error within 1.05 times its
initial value; at most two recordings may exceed that BPM tolerance. Rank the
admissible checkpoints by equal-group **training** pair MSE, with earlier ties.
Both arms share this rule. ART and the new schedules never select checkpoints.
Every epoch and both selected/terminal exports remain in the private artifacts.

## Fresh schedule preparation

Two schedules, `.8 -> 1.25 -> 1` and `1.25 -> .8 -> 1`, produce six actual
61-second WAV files from the same three source prefixes. This adds different
schedule placement, **not independent songs, rates, domains or renderer families**.
Some source pieces/rate combinations were already exposed. The six files were
encoded only after all four selected/terminal exports were frozen.

All six authored renderer witnesses and six musical wrong-map controls passed.
Of 366 fixed grid points, 224 qualify; 140 are in changed-rate pieces and 84 in
unchanged pieces. Changed-point counts are 29/29 for Vivaldi, 29/29 for Bach,
13/11 for Berlioz. The ledger retains all 142 excluded points. Sparse alignment
does not certify exact change onsets, perceptual beat level or dense labels.

Preparation took 16.91 s; fits 19.69 / 20.42 s; six fresh frozen-encoder captures
3.73 s; complete fit/capture/evaluation 48.66 s. Transfer, audit and CI are separate.

## What learned, and what did not

Equal-group training-pair MSE falls from .08440 to .02042 without retention and
.02417 with retention. However, every post-update **epoch** checkpoint fails the
development rule in both arms. The first unretained epoch fails Mozart's work
MSE/BPM tests; the first retained epoch instead fails Boccherini's. We did not
evaluate intermediate updates 1--4 as selection candidates.

Work-macro native BPM median error; lower is better:

| Cohort | Initial / both selected | Local-only terminal | Retained terminal |
| --- | ---: | ---: | ---: |
| Fit, 20 recordings / 9 works | 16.27% | 12.67% | 13.01% |
| Development, 5 recordings / 3 works | 35.22% | 44.86% | 42.37% |
| Exposed ART, 15 variants / 1 source | 27.98% | 28.00% | 27.41% |

Retention reduces terminal development error relative to the matched local-only
arm, but remains worse than the initial model. Development MSE is .32077 initial,
.44898 local-only terminal and .41567 retained terminal. Thus this is not only a
strict per-work gate hiding an improved development macro score.

Changed-only log2-rate RMSE on all six new schedules (zero-change RMSE .32193):

| Source / new order | Previous global-only final | Local-only terminal | Retained terminal |
| --- | ---: | ---: | ---: |
| Vivaldi / slow-fast-normal | .3087 | .2510 | .2680 |
| Vivaldi / fast-slow-normal | .1933 | .2046 | .2209 |
| Bach / slow-fast-normal | .1851 | .1938 | .1927 |
| Bach / fast-slow-normal | .2863 | .2129 | .2395 |
| Berlioz / slow-fast-normal | .1454 | .1397 | .1410 |
| Berlioz / fast-slow-normal | .2192 | .2121 | .1830 |

Both terminal models improve RMSE against the previous global-only final on
four pairs and regress on two. Neither terminal model passes all the fixed
conditions on any new pair; this descriptive application does not select them.
The actual primary, selected epoch 0, passes one new pair and preserves identity
and natural scores **because it is the unchanged initial model**. Its overall
gate explicitly fails the learned-checkpoint and fresh-transfer requirements.
Do not compare this 1/6 with relative v1's 1/6 as the same population or progress.

## Closed outcome and next question

This tested mechanism improves fit but does not solve cross-work retention or
reliable schedule transfer. A penalty only on the original fit recordings does
not establish protection outside those recordings. Do not relax the selection
gate, enlarge the network, mix rejected models into a product strategy, or sweep
the retention coefficient after seeing these results.

Before another fit, use the retained per-epoch weights to attribute the conflict:
does natural supervision or pair supervision drive the earliest development
regression, and is the fit-only preservation signal blind to that direction?
Any gradient diagnostics on development must be labeled selection/exploration
and must never become optimizer inputs. Use the evidence to choose a new
mechanism or broader training-source coverage; it is not yet proof that the
frozen encoder is inadequate. The six new schedules are now exposed too.

```sh
python -m unittest discover -s experiments/local_tempo -p 'test_*.py' -v
timeout --signal=TERM --kill-after=10s 900s python -m experiments.local_tempo.prepare \
  --pairs PRIVATE_RETAINED_PCM --ffmpeg PINNED_FFMPEG --output FRESH_PREPARATION
timeout --signal=TERM --kill-after=10s 5400s python -m experiments.local_tempo.run \
  --prepared FRESH_PREPARATION --preparation-sha256 VERIFIED_PREPARATION_SHA \
  --old-run PINNED_RELATIVE_RUN --assets PINNED_ENCODER_ASSETS --deps PINNED_DEPS \
  --inputs PRIVATE_NATURAL_PACKETS --weights PINNED_SEPARATED_NATURAL_EXPORT \
  --output FRESH_PRIVATE_OUTPUT --device cuda
```

Audio, features and checkpoints stay private. Original holdout stays sealed.
No model release or independent-acceptance claim follows this experiment.
