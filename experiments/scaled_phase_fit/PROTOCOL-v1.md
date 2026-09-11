# Observed range-safe phase-clock fit v1

This is a NEW bounded experiment, not a resume of the closed phase_sync_fit.
Register this protocol, complete executable source closure, prior outcomes,
population and target-runtime initial-state hash before any musical forward or
optimizer step. One natural-feature fit and one matched fitted-zero control.

## Single experimental change

Keep the frozen 24,037-parameter PhaseSyncReadout, seed 142, initial construction,
positive forward recurrence, native circular-phase/multilag log-count loss,
work balancing, full-crop shuffle/accumulation and complete-epoch selection.
Use the validated ObservedTransport and ScaledFitRecorder for backward and
durable diagnostics. Only the derivative's numerical representation/transport
and observation change. Do not load old learned weights, detach recurrence,
reset phase, shorten crops, adjust gain/learning rate, use AMP or tune seeds.

Range-safe transport does not bound the derivative or cure conditioning. It is
not bit-identical to the old native float32 parameter-gradient/AdamW path.
The synthetic range/parity/cost evidence permits this bounded attempt, not an
all-sequence convergence guarantee. Preserve and hash the old failed outcome.

## Fixed data, optimization and budgets

Reuse the exact coupled-v1 input manifest and all forty previously exposed
packets: 20 RUBATO fit recordings / 9 works, 5 development / 3 works, and 15
diagnostic-only ARTBeaT recordings. Same first-60-second RUBATO crops and complete
ARTBeaT captures. No new audio, encoder inference, labels, role changes or holdouts.
Native beats do not resolve perceptual octave ambiguity or provide audited
weak-attack/continuation/change/no-rhythm region labels.

Both fits start from the SAME target-runtime state hash, checked before their
first forward/step. AdamW lr 0.001, weight decay 0.0001, one global norm clip 1
after summing four complete recordings. Weight remains N/(G*n_work)/batch_size.
Shuffle with NumPy default_rng(142 + zero-based epoch); 20 epochs and at most
100 optimizer calls PER fit, with a 1,800-second PER-fit wall budget including
recorder creation, snapshot I/O, development, checkpoint selection and export.
Two CPU threads, one inter-op thread, deterministic PyTorch, TF32 off.

Check the wall deadline before/after each full record, before clipping, after
optimizer return, around development, and before selection/export. Never commit
a partial batch or select an incomplete epoch. Cooperative checks cannot preempt
a single slow kernel/I/O call; an outer 3,900-second watchdog bounds the whole
execution, including loading and evaluation. A killed process may leave the
last update unknown. A caught failure closes the experiment; no retry, resume,
seed/epoch/loss sweep or fallback transport. If the main fit fails, do not start
the fitted-zero control or evaluate diagnostic music.

The synthetic cost witness covers two records/update on four CPU threads, not
this full four-record, two-thread musical runner. Do not claim its subsecond
timing as this fit's throughput. Retain actual wall time, RSS, CUDA allocation,
per-update timings and snapshot bytes for this run; unchanged wall limits decide
budget admission. Diagnostic output goes to a private host disk, never Git.

## Durable evidence and selection

Retain the recorder's current/best model, optimizer, RNG, input identity,
complete/partial gradient packets, clock Jacobians/upstream, optimizer
entry/return state and returned-update count. Clear training traces before a
development record so stale evidence is not relabeled as development backward.
Development forwards have no adjoint; a later selection/export failure is not
attributed to the last clock. Journals retain every completed record's native
VJP count, field/parameter gradient norm in log2 units and binary exponent,
plus each complete batch's clipping report. Zero norms are null, not missing
records. Summaries are descriptive, not new thresholds for cherry-picking.
Alignment loss, band-budget failure and nonfinite state keep the existing
fail-closed behavior. Do not silently drop tiny gradient components.

Development loss averages within work, then across the three works. Select
only the lowest COMPLETE-epoch score, retaining the earlier checkpoint on ties.
Both fits must finish before using either selected checkpoint for the existing
five interventions per recording (200 sequence evaluations): natural, same
checkpoint zero/temporal mean/half-roll, and separately fitted zero. Reuse the
unchanged phase_sync_fit evaluation and comparison gate by import, not a
rewritten metric. Snapshot retention does not authorize resuming a failed run.

## Frozen accuracy gates, no product admission

All five requirements of phase_sync_fit/PROTOCOL-v1.md remain necessary, with
the exact same denominators, work macros, comparisons and strict regressions:
complete both 20-epoch/100-update fits within budget; at least 10% development
tempo AND phase improvement against fitted zero; no development OR diagnostic
regression in median/P95 tempo, phase and 4-second count against each raw/v1/
coupled/trajectory baseline; no majority of per-record raw regressions; and
natural phase strictly better than EVERY same-checkpoint timing intervention
on both cohorts. Missing metrics cannot reduce the denominator.

New execution admission additionally requires unchanged target-runtime plan,
exact initialization before each fit, durable 100 successful update receipts
per fit, all 400 record diagnostics per fit, completed finite conditioning
summaries and the registered source/prerequisite identities. Numerical
completion alone is not musical success. Report every failed gate and every
recording, including an unfavorable corpus. A failure closes this fixed run.
A pass only justifies proposing independent validation and representative-label
work; neither outcome changes shipping defaults, model packs or release status.
