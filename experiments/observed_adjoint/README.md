# Observable full updates with exponent-transport gradients

Status, 2026-09-11: synthetic integration and cost validation, **not another
musical fit**. The prior musical fit remains closed; its unsaved failure cannot
be reconstructed. No old weights, holdout recordings or shipping defaults are
used or changed here.

## One update lifecycle, with explicit gradient ownership

`ScaledFitRecorder` extends the existing `FitRecorder` through five internal
boundaries: begin accumulation, begin a record, backward one record, clip the
accumulated batch, and check a returned optimizer state. The original recorder still owns snapshot
durability, record identity and RNG capture, epoch/update budgets, optimizer
entry/return accounting, best-checkpoint handling and failure closure. Its
default native backward/clip path keeps the same results and event sequence.
We do not copy the training loop or offer a song-dependent policy switch.

`ObservedTransport` exposes stages around the frozen mathematical primitives:
CNN fields, clock forward, loss, output adjoint, field adjoint, band plan,
native field-gradient cast, native CNN VJP, completed band and parameter sum.
The old `scaled_adjoint.record_gradient` remains an unchanged numerical/cost
reference. This new orchestration retains partial state explicitly, rather than
reading stack locals or globally replacing autograd. No recurrence, derivative,
band width, weighting or clipping rule is changed by observation. Tests compare
the packets and traces exactly on the same runtime, including the multi-band
full-CNN path; this is not a claim of cross-device bit identity.

Each complete record's parameter-gradient packet is multiplied by its registered
weight and retained until the complete batch is summed. There is still exactly
one global clip before the optimizer. Two observed updates, their AdamW state
and RNG are compared against manually running the frozen scaled transport.
The earlier native-backward/AdamW rounding limitations remain unchanged.

## What the durable snapshots now retain

The existing snapshot fields remain available. The `scaled_accumulation`
extension (`observed-scaled-update-v1`) adds:

- every completed record's ID, weight, weighted parameter-gradient packet,
  actual CNN VJP count and pre-weight gradient norm in log2 units;
- the complete accumulated batch packet and clipping summary, when available;
- the current transport stage and band index/plan;
- the saved clock inputs, actual local Jacobians and actual accumulated
  upstream gradient, as soon as those values exist;
- the field-gradient packet, every completed parameter-band packet, and the
  complete current parameter gradient only if that accumulation returned.

Packets use CPU float64 tensors plus an integer binary exponent and explicit
alignment-loss count (`binary-gradient-v1`). They can be loaded from our own
trusted files with `torch.load(..., weights_only=True)`; serializing them never
materializes the overflowing gradient or writes NaN/Infinity into JSON logs.
Unknown/zero norms use JSON null. This is not a public untrusted-file format.

A second-record failure keeps the first record's weighted gradient and the exact
current input. Record-local traces are cleared before capturing the next input,
so a record-start journal fault cannot label a previous trace as the new record.
The upstream stage is set before either backward-start journal write, so even
that write's failure retains the correct stage without claiming autograd ran.
A failure inside the third native VJP keeps the preceding two
completed pieces and a replayable clock trace, without pretending that the third
piece completed. A clip failure retains all completed records and the batch
packet. `replay_field_adjoint` replays **only the captured field adjoint**, without
an optimizer or CNN. It neither resumes a fit nor reconstructs earlier record
audio/features from their gradient packets. A later optimizer failure's clock
trace describes the last record, not proof that the clock caused that failure.

An exception inside `optimizer.step()` can leave partial state: it stays
`entered`, with an unknown successful-update count. Once the call returns,
the return count is exact. The scaled recorder then checks model parameters
and tensor-valued optimizer states for finiteness. A failed post-step check
retains that returned-call count but emits **no** successful `update_completed`
receipt and closes the run. A returned optimizer call is not musical admission
or proof of a numerically valid update. The same distinction applies to a disk
failure after return. Pre-entry journal failure remains a known pre-step failure.

Caught faults preserve current/best weights, optimizer, RNG and exact current
record through the existing atomic-save handler. Process kills/device/disk
failure can still prevent capture. Acknowledged fsync/rename is not an
exactly-once or arbitrary-host-power-loss guarantee. A before-update checkpoint
is diagnostic evidence, not permission to retry a closed musical experiment.

## The full-CNN multi-band pressure fixture

The `natural` fixture is the existing 24,037-parameter CNN on Gaussian features.
The `opposed` fixture uses that **same full CNN** plus a fixed, authored field
offset buffer. Offsets are constructed once to place phase observations opposite
the incoming clock near 120 BPM. Construction carries the actual rounded fields
so the witness does not assume float32 subtraction/addition cancel exactly.
Offsets are not trainable or recomputed after an update. Every native CNN
parameter still participates in differentiation; there is no straight-through
surrogate, detached recurrence or replacement Jacobian.

This is deliberately adversarial, not a proposed model, reference-driven musical
repair, or training dataset. It makes the old float32 backward fail while the
observed path exercises at least four exponent bands per record. Passing it
demonstrates numerical/state coverage, not realistic musical accuracy or the
maximum possible band count. Large derivatives and near-zero-gradient AdamW
sensitivity remain real conditioning concerns.

## Predeclared complete-update cost

`contract-v1.json` fixes two complete 60 s records per update, weights 0.3/0.7,
the existing native count/phase loss, one global clip, one AdamW step, seed 777,
four CPU threads, one warmup and three measured probes for each fixture. Each
probe starts from the same initial-state hash and fresh optimizer; these are
independent synthetic checks, not sequential training epochs or retries after a
failure. There are exactly eight synthetic optimizer calls per device when the
entire check succeeds. Exceptions terminate the run; there is no automatic retry.

Timing includes recorder construction, initial/before/after snapshot writes,
fsynced stage journal, both complete-record computations, all actual VJP bands,
batch weighting/sum/clip, optimizer step and finite-state checks. Fixture/source
construction and resetting the independent probe are outside timing. Snapshots
must use a private **host disk bind mount**, not tmpfs/ramfs. The fixed Linux CPU
gate is median **below 2 s for both cases**, process peak RSS **below 2 GiB**, and
at least four bands for each pressure record. This scales the prior one-record
1 s envelope to two records; thresholds were set before measurement. CUDA is
additional device evidence, not a CPU or macOS substitute. This is not an
all-band/all-length runtime bound or end-to-end decode/feature-extraction cost.

Both cases passed the fixed CPU gate. The device measurements are:

| Environment | Ordinary two-record update | Five-band two-record update |
| --- | ---: | ---: |
| Linux CPU | 0.761 s | 0.981 s |
| T4 | 0.796 s | 0.924 s |

The pressure case exercised **five bands on each of both records** in every
measured update. CPU process peak RSS was 441 MiB; T4 host process peak RSS
(including its CUDA runtime, not GPU tensor memory) was 1,505 MiB. Each probe
wrote roughly 8--9 MiB of snapshots/journal to the host filesystem. The
[CPU](cost-cpu-v1.json) and [CUDA](cost-cuda-v1.json) reports retain each timing,
source/input/initial-state hash, actual band count, gradient exponent and storage
type. Eight independently reset synthetic steps ran per device, including
warmups, in the reported measurement set. An earlier synthetic measurement set
was superseded after record-boundary journal fixes; its private evidence is
retained separately. No failed musical run was retried and no musical score was
measured.

Run `python -m experiments.observed_adjoint.benchmark --device cpu --output
<new-private-directory>` (or `cuda`). Private output contains diagnostic weights
and inputs; only sanitized summary reports belong in Git. Run tests with
`python -m unittest discover -s experiments.observed_adjoint -p 'test_*.py' -v`.

Next, if these numerical and cost gates pass: register one new, bounded musical
validation with this explicit transport/recorder identity, preserved complete
records/weights and matched zero-feature control, failure/conditioning summaries
and frozen accuracy gates. Keep prior failed outcomes and holdouts unchanged.
Do not silently restart the old runner, select the most favorable corpus, add a
public strategy or claim improved musical accuracy from this integration.
