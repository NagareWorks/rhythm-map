# Four-arm musical evidence comparison v1

Freeze this protocol, executable source closure, input manifest, target runtime
and fresh initialization before opening musical packets. This is a new bounded
comparison, not a restart or reinterpretation of a closed fit. No default,
public strategy, release, encoder computation, new data or holdout access.

## Hypothesis and actual cost

Compare shared-natural, shared-zero, separated-natural, separated-zero in that
fixed order. All four start from seed 142 and the same per-task functions of
the validated separated_evidence component. The shared model has 24,003 total
parameters; the independent branches have 47,907. Width, features, native
targets, physical boundaries and additional 126-frame context radius per task
stay identical. Total capacity and compute are NOT held equal. A gain cannot
be attributed exclusively to removal of gradient interference.

Use the existing frozen 512-channel / 50 Hz features, never original audio or
old learned weights. Preserve the byte-locked 20 RUBATO fit recordings / 9 works,
5 development / 3 works, and 15 diagnostic-only ARTBeaT recordings. Keep first
60-second RUBATO captures, complete ARTBeaT captures and all annotation masks.
Loading diagnostic packets for integrity checking is not fitting or selection;
their neural forwards may occur only after ALL four fits and selections finish.

## Fixed training and budgets

Twenty complete epochs, 100 paired batch updates per arm, four COMPLETE
recordings per batch, shuffle default_rng(142 + zero-based epoch). Weight each
record by N/(G*n_work)/actual_batch_size. Native AdamW lr .001, decay .0001,
norm-1 clipping. No AMP, truncation, recurrent scan, projection or retry.

Shared: accumulate the sum of native-cell log2-period MSE and frame cos/sin MSE,
then one global clip and AdamW call. Separated: reuse TaskTrainer unchanged,
tempo then phase, each with its own complete-record accumulation, clip and
optimizer. Each task takes 100 steps; shared takes 100 optimizer calls per arm,
separated 200. Across four arms: 400 paired updates, 600 optimizer calls, 1,600
record presentations (each task sees 400 per arm). Report both per-task and
total resources; do not conceal the two trunk evaluations in the candidate.

Use two CPU threads / one inter-op thread, deterministic PyTorch, TF32 off.
The wall budget is 1,800 seconds per arm including development, snapshots and
export. An outer 7,500-second watchdog bounds registration checks, input loading,
all four fits, 400 selected evidence-pair evaluations and report persistence.
Cooperative checks bracket each complete task/shared update, each development
record, checkpoint and export; they do not preempt a backward kernel or I/O.
The external watchdog is mandatory for music. A fault closes the invocation;
do not start later arms, evaluate a partial set, resume or tune the budget.

## Equal selection privileges, explicitly non-coherent exports

After each COMPLETE epoch, compute each task's original training loss on all
five development recordings, first within work then equally across three works.
For BOTH architectures select tempo by minimum tempo loss and phase by minimum
phase loss, earliest exact ties. Select only after BOTH finite development
aggregates are complete. Never select using ARTBeaT, final metrics, a favorable
record, or a partial epoch. Retain all 20 pairs of scores and selected epochs.

The shared arm consequently exports its tempo branch from the shared model at
the tempo-selected epoch and its phase branch from the shared model at the
phase-selected epoch. These can differ. It does not secretly train two shared
models; it has the same two-export selection privilege as separated. These
exports are an evidence pair, NOT a jointly valid checkpoint or unified Clock.
Even equal epochs do not imply agreement between independent output fields.
Both exported pairs contain 47,907 parameters and use two trunk evaluations.
The 24,003 versus 47,907 comparison above describes TRAINING parameter ownership;
do not advertise a one-trunk shared inference baseline for these selected exports.

## Measurement, controls and decisions

For each architecture evaluate natural selected fields, separately fitted-zero
fields, and the natural selected pair on same-zero, time-mean and half-roll
features. No checkpoint or parameter changes in evaluation. Ten evidence pairs
per recording, 400 total. Retain raw T-1 log periods and T cos/sin vectors in
private prediction packets for every variant, including failures of support.

Convert cell log period directly to rate exp2(-log_period); DO NOT average frame
periods or manufacture a final cell. Phase is atan2(sin,cos)/(2*pi); magnitude
at most 1e-6 has unavailable phase, not zero error or silence. Reuse the unchanged
coupled_clock.measurement.measure/macro and phase_sync_fit.comparison_gate.
Four-second count error integrates the tempo branch only. It is not agreement
between that integral and phase, an observed-beat count or a coherent clock.
Keep the full reference support and common valid & raw_valid support, no edge
trimming, half/double oracle choice or unavailable-prediction denominator drop.

Retain per-record and work-macro tempo median/P95, circular phase MAE and 4-second
count MAE for fit/development/diagnostic. Raw, v1, coupled and trajectory gates
remain unchanged, including 10% development natural-vs-fitted-zero gains,
recording breadth and temporal controls. The last protected fit is an additional
descriptive baseline, not a new threshold. Component task losses choose epochs;
they do not replace the existing evaluation metrics or admission gates.

Report each architecture's unchanged numerical gates independently. Justifying
separation additionally requires all development/diagnostic metrics no worse
than shared, a strict development tempo-median OR phase improvement, and no
majority of either cohort regressing in tempo-median or phase. Also expose each
task's 10% development natural-vs-fitted-zero signal check separately. Failures
are retained, not repaired by another seed, epoch, weight or data-role change.

Even success only supports proposing coherent-fusion validation. Independent
acceptance, coherent_clock and production_change are always false. If isolated
tasks fail, inspect targets/features/generalization before fusion; if separation
does not beat shared, its added capacity has not been justified by this test.

## Durable execution evidence

Before and after each shared/task call, atomically persist complete model and
optimizer states, gradients, counters/stages, owned weighted input records,
metadata and Python/NumPy/Torch/CUDA RNG. Keep the latest before/after boundaries,
each task's best snapshot, every update receipt and every epoch score. A caught
fault attempts a failure snapshot without masking the primary error. A killed
entered call without a returned receipt is UNKNOWN, never fabricated as zero.
The next task's before snapshot preserves a prior task's committed state; there
is no pairwise rollback. Native returned calls are counted before finite-state
checks. Failed updates cannot be retried. Snapshots are evidence, not resumable
checkpoints. Persist hash-pinned branch exports and the whole execution plan.

The authored tests use reduced budgets on generated tensors only. The music
entry point rejects any modified plan before reading inputs. Component tests,
review and private CPU/CUDA validation must pass on the exact candidate before
a separate target-runtime plan is materialized and an authorized music run starts.
