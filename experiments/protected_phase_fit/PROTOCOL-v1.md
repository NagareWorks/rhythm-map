# Protected phase-clock musical comparison v1

This is a new, user-authorized bounded fit pair, not a resume of any closed run.
Freeze the executable source closure, this protocol, prior evidence, population,
runtime and initial weights before any musical forward or optimizer step.

## One mechanism, unchanged model and accuracy gates

Use the validated ProtectedFitRecorder/ProtectedAdamW exactly as implemented:
separate phase/count gradients, complete-batch weighting before three-group
routing, one global clip, native AdamW shadow proposal, groupwise displacement
projection, native rounding and veto, then an owned commit of parameters and
all proposed moments. No new scalar loss, direct-period auxiliary target,
architecture, decoder policy or numerical fallback. First-order protection is
not finite-loss monotonicity or a guarantee of correct perceived beat level.

Keep PhaseSyncReadout's 24,037 parameters, seed 142, initial construction,
positive recurrence and original circular-phase/native-multilag-count loss.
No old learned weight loading, encoder fitting/inference, shortened crops,
detached recurrence, AMP, changed gain initialization or per-song settings.

## Fixed population and budgets

Reuse the byte-locked coupled-v1 manifest and all 40 exposed feature packets:
20 RUBATO fit recordings / 9 works, 5 development / 3 works, 15 ARTBeaT
diagnostic-only recordings. Same first-60-second RUBATO crops and complete
ARTBeaT captures, masks, targets and measurement support. No holdout access,
new labels/data, role changes or choosing a favorable cohort.

Two fresh fits, natural features and separately fitted zero features, starting
from the SAME registered target-runtime weight hash. AdamW lr 0.001, decay
0.0001, clip 1; four complete recordings per update, weight
N/(G*n_work)/actual_batch_size. Same epoch shuffle default_rng(142+epoch).
20 epochs / 100 returned optimizer calls PER fit, 1,800 seconds PER fit
including all snapshots, post-step loss audits, development and export.
Two CPU threads, one inter-op thread, deterministic PyTorch, TF32 off.
One outer 3,900-second watchdog bounds loading, both fits and evaluation.

Check deadlines at record/term boundaries, before and after clipping, every
proposal/commit notification, each post-step/development record, checkpoint and
export. Checks are cooperative, not single-kernel/I/O preemption. A killed
process can leave an entered call unknown. A caught fault closes the run; no
retry, extra epochs, new seed, line search, checkpoint restart or fallback.
If the main fit fails, do not start fitted zero or diagnostic evaluation.

## Evidence, selection and unchanged evaluation

Keep the existing durable before/proposal/after/best/failure snapshots. Every
completed batch record retains its owned CPU input/metadata, starting RNG and
both weighted gradient packets. Journal all 800 term-gradient diagnostics per
fit, 100 returned update receipts, and actual group projection/veto decisions.
Retain returned calls, committed proposals and moved proposals separately;
an all-group native stall still consumes an update and advances AdamW time.

After each returned step, evaluate the ORIGINAL losses on the SAME complete
weighted training batch with no gradients. Record before/after phase and count,
finite count increases, per-record results and elapsed time. These audits do
not decide acceptance, change the step, tune its size or select a checkpoint.
Clear training traces before these inference-only audits and development so a
later failure cannot be mislabeled as the previous record's backward.

Development selection remains minimum ORIGINAL total loss averaged within
work then across the 3 works; only complete epochs, earliest ties. Both fits
must finish before 200 sequence evaluations over all 40 recordings: natural,
same-checkpoint zero/mean/half-roll, and separately fitted zero. Import the
unchanged phase_sync_fit evaluate/comparison_gate, not rewritten metrics.
All original tempo/phase/count support, work-macro denominators, 10% joint
development improvement versus fitted zero, raw/v1/coupled/trajectory
nonregression, per-record breadth and timing-intervention gates remain required.

Additionally require all 100 returned/committed receipts, 400 complete records
and 800 phase/count gradient diagnostics per fit, finite post-step audits,
zero alignment loss, and no positive defined native displacement certificate.
Do not require every step to move or every finite loss to decrease; count both
explicitly. Any stopped/incomplete fit fails execution admission.

Report the same-support per-case and work-macro differences from the last
closed scaled_phase_fit using its fixed report, without loading its weights.
This mechanism comparison is descriptive; it does not replace any old gate.
Record runtime, memory, artifacts and every adverse cohort. A pass only permits
proposing independent evaluation; a failure closes this fixed experiment.
Neither outcome releases a model, changes product defaults or opens holdout.
