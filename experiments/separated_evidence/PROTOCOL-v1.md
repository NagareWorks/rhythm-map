# Separately owned tempo / phase evidence: component contract v1

This is the next user-authorized design and controlled component verification,
not a restart of protected_phase_fit, a music fit, a fusion decoder or a release.
All earlier source identities, failed gates and input roles remain unchanged.

## Actual architecture and comparison limits

Both arms use the same frozen 512-channel / 50 Hz Beat This interface. Reuse
the fixed 32-channel, kernel-5 depthwise / pointwise residual CNN with dilations
1/2/4/8/16/32 and radius 126 frames. Physical padding and full-context chunk
ownership are identical. The upstream encoder may already carry wider context;
126 frames is the additional readout radius, not the audio receptive field.

Shared control: one trainable trunk and direct tempo/phase heads, 24,003 parameters.
Separated candidate: independent deep copies of that trunk, projection and each
task's own output head, 47,907 total parameters. Copies start with EXACTLY the
same per-task functions as the shared control without consuming another RNG
draw. They share no trainable storage, gradients, clipping norm or AdamW state.
Frozen input features alone are common. Each task retains its old trunk width.

This is a matched initial-function / per-task architecture comparison, NOT an
equal-total-parameter experiment. Any improvement could involve increased total
capacity as well as removal of shared updates. Do not attribute it entirely to
gradient interference or hide the parameter/computation increase. The candidate
is neither a Transformer nor an encoder fine-tune, scalar routing heuristic,
gradient projection, detached recurrent clock or hidden per-song policy.

## Targets and ownership

Tempo emits T-1 outgoing-cell log2 periods in seconds. Use squared error to
the native cell target -log2(50 * (reference[t+1]-reference[t])). It preserves
the same annotation beat convention, never minimizes over half/double choices,
and does not impose a smoothness penalty. Native multi-lag counts are a future
musical diagnostic, not another weighted loss in this component.

Phase emits T frame-point cos/sin components. Use mean squared error over both
components against the wrapped native reference angle. A unit target is NOT a
calibrated output confidence. Circular phase cannot by itself identify integer
beat counts or choose a perceived beat level. Reference gaps are missing labels,
not weak attacks/no-rhythm labels, and no tempo target crosses a missing endpoint.

Use separate native AdamW instances (lr .001, decay .0001) and separate global
norm-1 clipping for the separated branches. Complete record gradients are
weighted and accumulated before each branch clips. No cross-task normalization
or shared optimizer moments. Tests compare real momentum-bearing updates with
standalone task updates and verify the inactive task's parameters AND optimizer
remain bit-identical, including under a 1,000-fold other-task weight.

An exception closes that task. Optimizer-entered exceptions retain an unknown
mutation boundary; returned calls are counted before checking resulting state.
The two tasks are independent transactions: failure in the second task cannot
claim the first never committed. There is no automatic task or music retry.

## Fixed authored verification

Seed 142, two CPU threads, 40 paired updates on one 96-frame authored ramp/step/
phase-observation-gap packet. The packet explicitly encodes teacher period and
phase channels; it is a learnability fixture, not realistic encoder evidence.
One shared model and one separated model begin with matched functions; no zero
music control or frozen learned checkpoint is opened. Shared AdamW uses the sum
of the same two losses and one global clip, with the same hyperparameters.
Each separated task takes 40 steps; shared takes 40 steps. No checkpoint choice,
epoch/seed sweep, line search, new data or favorable-result-only report.

Retain initial/final losses for BOTH arms, every update receipt, source/runtime
identity, RNG, before/after snapshots and any failure state. A component pass
requires both separated task losses to decrease versus their OWN initialization,
finite state and complete budgets. It does NOT require outperforming shared on
this authored packet, nor admit music accuracy. Run the same fixed fixture on
CPU and CUDA; no runtime-specific retuning. Bound each invocation externally by
300 seconds. A caught failure closes the invocation and preserves its artifacts.

## Explicit next boundary: music, then coherent time

These outputs are Evidence, NOT Clock: there are no beat timestamps, tempo
segments, rhythm availability or confidence claims. Adding a shipping adapter
around these independent fields would recreate the rejected incoherent v1.

The next musical protocol must freeze all four arms (shared/separated times
natural/fitted-zero), matched initialization, the existing 20 fit / 9 works,
5 development / 3 works and 15 diagnostic-only ARTBeaT cases, budgets, branch
selection, support, every per-record metric and failure receipts BEFORE a fit.
No role migration, holdout access, new music or encoder computation belongs to
this component verification. Fitted-zero controls remain necessary; separation
alone does not establish acoustic learning. Report per-task and total resources.

Task-specific checkpoint selection and the shared control's export semantics
must be explicit in that future protocol; silently cherry-picking whichever
epoch wins each final metric is forbidden. No complete musical admission gate
can pass without a single coherent output clock and independent evaluation.

Decision order: if isolated speed/phase evidence still fails, inspect supervision,
features and generalization before building a fusion decoder. If both task
outputs pass but a coherent decoder regresses, investigate fusion. If gains
stay on fit works, inspect transfer/data coverage rather than extending epochs.
These observations prioritize hypotheses; they are not proofs of a unique cause.
