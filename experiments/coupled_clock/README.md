# Coupled clock: executable architecture and supervision contract

Status, 2026-09-10: the [fixed complete-crop experiment](PROTOCOL-v1.md) has now
run both 20-epoch fits. Its [research gate failed](../../evaluation/baselines/coupled-clock-learning-v1.md):
coherent advancement does not establish reliable musical alignment. This is a
separate experiment, not a retry of the closed [direct readout](../../evaluation/baselines/direct-clock-learning-v1.md).
No new encoder capture, audio, holdout access, distributed weight or production
integration. The synthetic unit tests remain separate from the actual music fit.

## What is implemented, and what is reused

Reuse the unchanged v1 convolution block: a 512-to-32 projection and six
residual depthwise/pointwise blocks, dilations 1/2/4/8/16/32, radius 126 frames.
Replace the three independent outputs with two fields: log-period and an
anchor field, of which only the first physical frame supplies the clock origin.
The new head has **23,970 parameters**. It accepts only `[batch,time,512]`
frozen encoder features; truth, detector events, file identity and candidate
clocks are not inputs. V1 fitted weights/predictions are not inputs or initialization;
the old selected weights are replayed only as an evaluation comparator.

- `model.py`: CNN fields with canonical raw-feature padding and ownership.
- `clock.py`: positive advancement, one anchor and differentiable integration.
- `supervision.py`: native cumulative-count targets, reference masks and loss.
- `test_coupled_clock.py`: authored representation, numerical and gradient tests.
- `prepare.py`: pinned feature/trace reuse and v1 comparator replay, no encoder call.
- `run.py`: full-crop gradient accumulation, two fixed fits and development-only selection.
- `measurement.py`: cell/phase/count scoring on fixed reference and common support.
- `test_experiment.py`: synthetic measurement, work-weight and replay contracts.
- `inputs-v1.json`, `execution-v1.json`, `results-v1.json`: public hashes and metrics,
  never audio/features/prediction arrays or trainable weight files.

The old experiment's source/report hashes and closed decision remain unchanged.
This module is not imported by the Rust core, CLI, bindings or model packs.

## One clock, not separately predicted tempo and phase

For T frame points at 50 Hz, there are T-1 complete cells. Let `p[j]` be the
predicted cell period in seconds, and `a` one predicted origin in beat cycles:

```text
positive cell rate[j] = 2 ** (-log2_period[j])       # beats / second
q[0] = a
q[j+1] = q[j] + rate[j] / 50                       # cumulative beats
phase[j] = (cos(2*pi*q[j]), sin(2*pi*q[j]))
cell_bpm[j] = 60 * rate[j]
```

Phase progression and cell tempo are thus two views of the same clock.
There are no per-window phase resets, backward phase corrections, integration
of the failed v1 predictions, or extra continuity/smoothing penalties. Positive
rate can jump immediately at a cell boundary. Whole-input and chunked CNN work
use the same halo/padding; fields are joined BEFORE the one global integration.
The `chunk_frames` argument is an internal computation/test detail, not a
user-facing musical strategy. This is bidirectional offline inference.

Integration uses float64, preserves gradients back to float32 CNN parameters,
and rejects nonfinite, nonpositive or numerically unrepresentable advancement.
The accumulation agreement check uses rtol 1e-9 / atol 1e-12 cycles per cell;
this is a numerical check, not a tuned musical cutoff. No tempo bounds silently
clamp invalid output. Phase trigonometry uses fractional cycles to avoid feeding
large whole-cycle counts directly to sin/cos. The final point has phase but no
invented outgoing tempo cell, and partial physical tail cells are not inferred.

**Cell-average tempo is not instantaneous left-frame tempo.** At a change
inside a 20 ms cell, the cell advance includes both sides. Reference point
counts can be reproduced exactly by such cell averages, but subcell shape and
exact subcell beat-crossing times are not thereby recovered. Do not compare
these cell values directly with v1's instantaneous inter-beat targets or
silently replace the old metrics/denominators.

## Labels: count native beats, including through weak or absent attacks

For reference beats `b[i] <= t < b[i+1]`, define:

```text
reference_q(t) = i + (t - b[i]) / (b[i+1] - b[i])
```

Only frame points bracketed by annotations and before the physical duration
are valid. Unbracketed edges and explicit annotation holes stay unknown/NaN.
Missing acoustic attacks or missing detector events do NOT remove reference
supervision. A reference clock can continue through a rest; an observed-event
interval alone cannot decide whether this is continuation or genuine slowing.

This preserves the source's native beat convention. It does not establish that
native is the sole perceived beat level, resolve disagreement between annotators,
or certify consistent conventions between datasets. Those remain data/admission
questions. No hard-change, no-rhythm, bar or downbeat labels are invented.

## Loss: circular phase plus unwrapped advancement

At valid frame points, phase loss is `1 - cos(2*pi*(q - reference_q))`.
For lags 1/25/100/200 frames (20 ms / 0.5 s / 2 s / 4 s), advancement loss is:

```text
(log2(q[t+lag] - q[t]) - log2(reference_q[t+lag] - reference_q[t])) ** 2
```

Each lag averages its fully supported pairs; available lags are then averaged
equally. Every intermediate reference frame must be valid, not only the two
endpoints. Short tails keep available lags; missing lags are not zero-error
terms. The total is phase loss plus advancement loss, returned separately with
frame/pair counts. A future fitter must balance recordings/works outside this
per-recording objective and reject a recording with no supported interval.

At coincident beat ticks, circular phase alone can give a doubled clock zero
phase error. The native advancement ratio is still two (or one half), giving
unit log-ratio squared error. This removes that objective loophole; it does
NOT demonstrate that a network can identify the correct level from audio.
No minimum over native/half/double targets is used to make ambiguity look solved.

## What the authored checks establish

Run from the repository root, in the existing NumPy/PyTorch test runtime:

```sh
python -m unittest discover -s experiments/coupled_clock -p 'test_*.py' -v
```

The witnesses cover constant clocks, octave/non-octave steps, an analytic
linear-rate ramp, an annotation change inside a frame cell, retained tails,
clock carry across chunks, reference holes, coincident octave phase aliases,
invalid numerics, finite differences, and gradient reachability into the
anchor/rate/convolution weights. The weights remain unchanged after backward.
A random model is structurally coherent; it is not musically accurate.

For the continuation/step witness, two reference timelines share a possible
sparse observation sequence but have different native count targets. This is a
supervision/identifiability check, not training data or an audio discrimination
result. Synthetic arrays cannot substitute for representative musical labels.

## Requirements registered for the second fit

1. Freeze a separate input manifest and scientific protocol. Reusing v1's
   exposed calibration and private features is exploratory development, not
   independent acceptance. Audit source-native count labels and declare which
   genuine continuation/step/ramp cases may fit, select checkpoints or diagnose.
   Do not silently repurpose ARTBeaT diagnostic cases as training examples.
2. Use the same complete crop and single-anchor semantics in fitting and
   prediction. Do not copy v1's independent 200-frame training windows: that
   would reset the clock and hide whole-crop accumulation drift. CNN chunking
   alone does not bound the backward graph or make this a streaming model.
3. Fix fit budget, seed, checkpoint criterion, matched fitted zero-feature
   control, and same-support comparisons against raw events and the closed v1
   output BEFORE running. Report tempo and phase separately, every recording's
   regressions, coverage, count drift and execution cost. Structural coherence
   is required but cannot replace those musical criteria or relax the old gate.
4. Retain phase-origin uncertainty at the track start, long-range gradient/
   accumulation difficulty, beat-level convention, negative/no-rhythm inputs,
   and independent data as unresolved. A positive clock always advances even
   on silence: its existence is NOT evidence that rhythm is present. Do not
   emit its crossings as observed beats or use unit phase-vector norm as
   confidence. Rhythm availability/admission remains separate.

The [protocol](PROTOCOL-v1.md) fixes these choices for the completed run. Its
fit population has expressive native beats, not audited continuation/change
region labels; ARTBeaT remains diagnostic-only. That explicit coverage limit is
not claimed resolved by fitting. Close this fixed run without an automatic
seed/epoch sweep; see the result report for the remaining phase-locking and
supervision questions. No public option or package release follows.
