# Coupled-clock complete-crop fit v1: pre-fit protocol

This is one bounded exploratory continuation, not independent acceptance or a
shipping model. Freeze this document, all experiment source files and the input
manifest before either optimizer runs. Retain failures; no automatic extra
seeds, epochs, loss weights, smoothing, encoder fine-tuning or data search.

## Question and population

Can the already implemented 23,970-parameter CNN learn a useful *coherent*
tempo/phase clock from the identical frozen Beat This final0 features?

Reuse exactly the previous direct-readout population, crop bytes and roles:
20 RUBATO recordings / 9 works for fitting; 5 recordings / 3 works for
development; 15 ARTBeaT examples for diagnostics only. RUBATO crops are the
first 60 seconds; ARTBeaT uses the existing complete recordings. No new encoder
forward, audio download, holdout access or annotation. Source and weight license
decisions are unchanged. Features, predictions and fitted weights stay private.

RUBATO supplies native beat times in expressive performances, **not** audited
hard-change, missing-attack, or no-rhythm labels. ARTBeaT includes the existing
step/ramp cases, but may not enter fitting or checkpoint selection. This is a
transfer diagnostic, not training on a representative transition corpus.
Both cohorts have been exposed previously. Work disjointness does not establish
performer independence or absence from encoder training. A pass is not a release
gate and a failure cannot by itself condemn CNNs or frozen encoders generally.

## Fixed optimization

- Two fresh fits: audio features and identically shaped all-zero features.
  Both use seed 142, identical initialization, ordering, optimizer and budget.
- 20 epochs maximum, 1,800 seconds per fit maximum, no mixed precision,
  deterministic PyTorch algorithms, TF32 disabled. Stop on numerical failure;
  retain the journal and do not silently restart.
- AdamW: learning rate 0.001, weight decay 0.0001; global gradient norm clip 1.
- Shuffle the 20 complete crops with NumPy seed `142 + epoch_index`. Accumulate
  gradients over four recordings, without padding recordings to each other.
  Each recording has one physical-start anchor and all available frame cells.
  No truncated backpropagation or per-window phase resets.
- For N fit recordings in G works and n recordings in the current work, weight
  its one-recording loss by `N / (G*n) / actual_batch_size`. This gives equal
  expected work weight and equal recording weight within a work. Five optimizer
  updates per epoch, at most 100 updates per fit.
- Keep the existing circular phase plus equal-mean multi-lag log-count loss
  (lags 1, 25, 100, 200 frames). No loss-coefficient search or octave minimum.
- After each complete epoch, score development only, taking the mean loss per
  recording within each work, then across works. Select its lowest total loss;
  exact ties keep the earlier epoch. No diagnostic metric chooses weights.
- Record every epoch, selected epoch, duration, updates and peak GPU allocation.
  Budget exhaustion or an incomplete fit fails the experiment gate even if a
  checkpoint exists. Do not claim this is an architecture-only ablation: loss,
  complete-crop optimization and update count differ from the old window fit.

## Common measurement contract

Native beat times define piecewise-linear unwrapped reference q at 50 Hz. Only
annotated, bracketed points strictly before crop duration are valid. Adjacent
valid points define complete cells; no tail extrapolation. Missing attacks do
not remove reference support. Any annotation hole invalidates an entire lag
interval crossing it, not just its endpoints.

Report reference-only coverage for audio and zero, plus an identical intersection
of native-reference and raw-event support for audio, zero, raw and v1. Raw means
interpolation of the original detected events, **not the shipping estimator**.
Reconstruct raw q from hash-pinned trace timestamps, not wrapped-phase unwrapping.
Each v1 comparison is a deterministic replay of its hash-pinned selected audio
weights on the same frozen features; do not train or repair the v1 clock.

- Cell BPM: `60 * 50 * (q[j+1]-q[j])`. Reference and raw use exact interpolation
  at both cell endpoints, including a sub-frame tempo transition. The v1 head
  predicts rates at frame points; project it to a cell using the explicitly
  declared trapezoidal average of its two endpoint rates. This is a new
  cell-measurement comparison, not a rewrite of the original v1 point metrics.
- Per recording: median and P95 relative cell-BPM error (percent), and mean
  shortest wrapped phase error (cycles) at valid points. V1 keeps its independent
  atan2 phase; zero-amplitude/nonfinite phase is unavailable, never zero error.
- Count drift: mean absolute error in native beat advance over fully supported
  200-cell / 4-second intervals. Integrate the projected v1 rate *for this
  diagnostic only*; do not replace its independent phase or produce new events.
- Aggregate each metric equally across recordings within a work, then works.
  Show per-recording regressions as well. Report point, cell and interval
  denominators; any missing/nonfinite required prediction makes that metric null
  and fails a required gate, rather than shrinking its denominator.

## Fixed research-continuation gate

All conditions are necessary; evaluate only after both checkpoints are fixed:

1. Both fits complete all 20 epochs in budget with complete finite predictions
   on every required development and diagnostic point/cell/4-second interval.
2. Development audio improves reference-only median tempo error and phase error
   by at least 10 percent relative to the *fitted* zero-feature control.
3. On common support, development and ARTBeaT work-macro median/P95 tempo, phase
   and count drift are each no worse than **both** raw and replayed v1.
4. On each of development and ARTBeaT, no more than half of recordings regress
   against raw for either median tempo or phase. Count strict increases without
   post-hoc tolerance tuning; list regressions against v1 too.

A pass authorizes only proposing fresh independent validation and resolving
rhythm availability / beat-level semantics. A failure closes this fixed run;
inspect its evidence before proposing a materially different experiment. Neither
outcome changes Rust defaults, adds a user-facing strategy, or publishes a pack.
