# Direct clock learning v1: fitted, failed, closed

On 2026-09-10, two real optimizer runs fitted a **24,003-parameter direct
phase/period head** and its zero-audio control. The encoder stayed frozen.
The fixed development gate failed. This is a completed learning experiment,
not another candidate audit, a production accuracy gain or a released model.

## What changed from the earlier proposal

The earlier candidate-compatibility scorer was not fitted. A scorer cannot
recover an absent path. This experiment instead directly predicts three
50 Hz fields from Beat This's normalized 512-channel pre-head sequence:
`log2(period_seconds)`, phase cosine and phase sine. No candidate clocks,
annotation, song identity, split role, absolute time or BPM prior enter the
network. The target follows the source's native beat annotations; other
perceived beat levels are not thereby proven wrong.

The [pre-fit protocol](../../experiments/clock_readout/README.md), source,
[execution identity](../../experiments/clock_readout/execution-v1.json),
[input manifest](../../experiments/clock_readout/inputs-v1.json) and
[complete unedited report](../../experiments/clock_readout/results-v1.json)
are retained. The report SHA256 is
`990f7a2c71b2d3d61ab77c81daa69c3c5add3a78ead43d776fbf03d25ed07736`.
The manifest binds original audio/truth hashes, source files, crop relations,
frame counts and private tensor hashes. Features, PCM, predictions and fitted
weights remain private; this commit redistributes none of those artifacts.

## Data and fit fixed before execution

- RUBATO: 20 recordings from nine works fit the head; five recordings from
  three other works select its checkpoint. Versions of a work stay together.
  All 25 recordings were already exposed calibration, now explicitly reused
  for this exploratory fit. This is **not independent acceptance**; performer
  independence and encoder-training recording overlap remain uncertified.
- All 15 ARTBeaT inputs are diagnostic only, scored after both fits. They do
  not fit parameters or select checkpoints. Their shared construction source
  is not fifteen independent provenance groups. No sealed holdout was opened.
- Every RUBATO input is its fixed first 60 seconds; all ARTBeaT inputs fit
  within that limit and retain their complete duration. The new frame metrics
  cannot be compared directly to prior full-track/query-level reports. Frames
  outside bracketing annotation support are unavailable, not negative labels.
  No hard-change or no-rhythm labels are invented for RUBATO.
- A 512-to-32 projection, six residual depthwise/pointwise temporal blocks
  with dilations 1/2/4/8/16/32, then 32-to-3 output. Readout radius is 126
  frames. Central windows own 200 frames without overlap and retain tails;
  raw-feature padding at physical ends is identical for fit and prediction.
- Seed 142 for each of two fits, 20 epochs maximum, 30 minutes per fit maximum.
  AdamW learning rate 0.001, weight decay 0.0001, gradient norm cap 1, batch 4.
  Loss is squared native log-period error plus mean squared cosine/sine error.
  Fit contributions are work-balanced. Lowest work-macro development loss
  selects the checkpoint; exact ties keep the earlier epoch. No sweep ran.

Both runs completed all 20 epochs. The audio checkpoint is epoch **1** and
the control checkpoint epoch **2**. Training loss continued falling without
improving the audio head's selected development loss: evidence of poor transfer
on this split, not grounds for extending epochs or choosing another seed.

## Measured results

All values below average recordings within each work, then average works.
Tempo error is native relative BPM error; phase error is absolute wrapped
distance in cycles. Neither is beat F1, change-point accuracy or calibrated
confidence. Missing outputs must remain in denominators: headline metrics are
null unless all assessed reference frames have the relevant finite output.

On all supported development reference frames, the audio head's tempo-median
error is **37.2279%**, versus **40.4426%** for the fitted zero-audio control:
only **7.95% relative improvement**, below the precommitted 10% floor. Its phase
error improves from **0.25044 to 0.13922 cycles** (44.41%). Finite output coverage
is complete, but the conjunctive gate still fails. No floor is relaxed.

The following comparison uses only the **same reference frames bracketed by
raw detected events**, and all rows use the same crop/context. The baseline is
raw-event interval interpolation, **not the shipping core's smoothed estimator**.

| Slice / output | Work-macro tempo median error (%) | Tempo P95 error (%) | Phase error (cycles) |
| --- | ---: | ---: | ---: |
| RUBATO development / raw | 41.5893 | 104.0834 | 0.14274 |
| RUBATO development / learned | 37.3199 | 67.3552 | 0.13929 |
| ARTBeaT diagnostic / raw | 14.4585 | 49.4982 | 0.10780 |
| ARTBeaT diagnostic / learned | 31.1228 | 51.2062 | 0.14318 |

Do not promote the favorable development average: **four of five development
recordings regress in tempo-median error**. The large improvement of one Mozart
recording conceals those regressions, and that recording is still very wrong.

| Development recording | Raw tempo median error (%) | Learned tempo median error (%) |
| --- | ---: | ---: |
| Boccherini / Krux 2022 | 11.6447 | 15.7538 |
| Mozart / Papalin 2012 | 113.5476 | 64.0091 |
| Mozart / Gli Armonici 2015 | 101.9969 | 103.4379 |
| Mussorgsky / Bertoglio 2012 | 8.2593 | 11.2058 |
| Mussorgsky / Staab 2016 | 2.4427 | 13.7591 |

ARTBeaT has **12/15 tempo-median regressions and 13/15 phase regressions**
against raw interpolation. The complete report includes every input's median,
P95, phase, coverage, coherence and regression flags, including the fit set;
none of the poor recordings was removed from a headline.

## Phase is not yet a coherent clock

The independent outputs do not enforce that phase advances at the predicted
tempo. Across development recordings, positive phase advancement covers only
83.96% to 99.35% of valid adjacent frame pairs. ARTBeaT ranges from 62.20% to
86.10%. This rules out treating the predicted phase crossings as trustworthy
beat timestamps. The phase vector's magnitude is not calibrated confidence.

The network learns some timing signal relative to the no-audio control. That
does not establish robust tempo, canonical beat level, transition localization
or useful generalization. The tiny work split and RUBATO-to-ARTBeaT domain shift
also prevent attributing failure solely to model size, encoder information,
data quantity or the unconstrained output form. A coherent-clock architecture
is a next design question, not a claimed fix inferred from these scores.

## Execution and reproducibility

The unchanged final0 encoder and its original task heads were checked against
the Rust RTEN captures on all forty identical inputs. Maximum absolute logit
disagreement was **0.000164748**, below the frozen 0.002 + 0.0001-relative budget.
Head reconstruction checks passed; encoder state hashes match before/after.
No checkpoint parameter was unfrozen. Capture used the original chunk ownership.

On one T4, Python 3.12 / Torch 2.10.0+cu129 / NumPy 2.2.6, capture took **22.11 s**;
the audio fit **19.73 s**, and the control **13.97 s**. These are workload wall
times, not GPU kernel time or end-to-end CPU/product benchmarks. Local release
input preparation separately took **952.78 s**, with two RTEN threads and seven
identical cached ARTBeaT traces reused. Transport and setup are excluded.

Before either fit, a debug-exporter attempt timed out on its first input; it
produced no fitted model. The unchanged preparation ran successfully using the
existing release exporter in a new output directory. Container startup required
explicit cache directories for an unlisted numeric user. Failed setup logs were
retained; no scientific parameter, timeout or threshold was relaxed after a fit.

All forty stored prediction metrics were recomputed privately from the hashed
input/prediction arrays. Windows-vs-Linux float32 math differed by at most
0.000061036 percentage points for tempo P95 and 0.000000060 cycles for phase;
counts and advancement fractions match. The original Linux report and its
scientific gate are unchanged. Unit tests additionally pin source/result/input
identities, recompute all group summaries and regression flags, and check the
checkpoint decision, masks, tails, gradients and whole-vs-chunk output.

To reproduce, prepare private inputs with `experiments/clock_readout/prepare.py`
using the existing `beat_this_trace` release executable, final0 model pack and
the declared calibration audio roots. Then run `experiments/clock_readout/run.py`
with `--inputs`, `--upstream`, `--checkpoint`, `--output` and `--device cuda`.
Use exclusive output directories outside Git, the source identities above and
the pinned upstream source from the existing capture contract. For deterministic
CUDA execution, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch.
This is a research runner, not a new user-facing analysis mode.

## Disposition

**Close this fixed direct readout.** No extra seed, threshold adjustment,
encoder fine-tuning, larger-head sweep, holdout opening or shipping integration
follows from this result. Keep the current default and API unchanged.

The next bounded design should couple phase progression and tempo as one clock,
address native/half/double beat-level semantics, and declare representative
continuation/change supervision. It must state what changes relative to this
failed baseline before another fit. This result justifies that concrete design
review, not another open-ended source search or claim that training is solved.
