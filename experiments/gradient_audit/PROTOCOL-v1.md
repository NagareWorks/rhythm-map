# Fixed-checkpoint gradient-path audit v1

Freeze before musical gradients. Reuse the two selected scaled_phase_fit v1
checkpoints (natural and separately fitted zero), the exact 40 exposed packets
and their saved predictions/fields. Keep 20 fit / 5 development / 15 diagnostic
recordings and work identities. No new encoder/audio/labels, role changes,
holdout, checkpoint selection, training, parameter update or public strategy.

## Bounded execution and identity

Register executable sources, this protocol, old source closure, prior reports,
input/prediction/weight hashes and the target runtime before the audit. Use the
same Torch 2.10.0+cu129 / NumPy 2.2.6 / T4 runtime, deterministic mode, TF32 off,
two CPU threads and one inter-op thread. All natural/zero forwards must exactly
reproduce saved clock fields and cycles. Compare every model state before/after;
no optimizer or checkpoint-resume state is constructed or loaded, and parameter
`.grad` buffers must stay None. Models are in eval mode with autograd enabled.

One streaming pass over all 40 recordings, both selected checkpoints: 80 cases.
For each case, independently differentiate the original circular phase term,
native multi-lag count term, and their unchanged total, using ObservedTransport
and the frozen clock_loss. Each path recomputes the same CNN/clock; additionally
one CNN-only direct-prior comparator. This is 320 CNN field forwards / 240 clock
forwards, not 80 inference calls. Exponent-band VJPs retain range checks and
fail closed on any lost alignment or nonfinite value. This reuses the existing
adjoint, not a second manually written recurrence.

There is a 1,200 s cooperative audit budget and separate 1,500 s process watchdog.
Retain every completed case and private field/parameter gradient packets. On
failure retain the exact case/kind and transport snapshot where applicable,
close the audit without automatically changing limits or skipping the case.
A direct-prior failure must not be attributed to a stale clock trace.

## Comparators and measurements

Use the SAME native annotation support as the fitted loss, not raw/common
support. Each count interval must contain all its reference frames. Report
each original loss and support count. The diagnostic comparator is
`mean((ell - (-log2(native_cell_rate)))**2)` on supported cells, where ell is
the CNN's pre-feedback log2 period. Its field gradient is exactly
`2*(ell + log2(native_cell_rate))/supported_cell_count`, zero elsewhere.
It is NOT a proposed new training objective or an assertion that the correct
pre-feedback prior equals the best jointly corrected clock.

Retain gradients separately for phase, count, total and direct prior. At the
field level compare their period blocks on supported cells. At parameter level
compare the dedicated period output row (output.weight[0], output.bias[0]),
shared CNN (projection/blocks), and the complete model. The shared/full gradient
includes paths through phase vectors and origin, not only the period path.

For each group retain log2 L2 norms, cosine(phase,count), cosine(count,prior),
cosine(total,prior), cosine(phase,prior), and phase/count norm log2 ratio. Cosines
are null for zero vectors, never reported as positive agreement. Positive
cosine(total,prior) means an infinitesimal negative TOTAL gradient in that
parameter subspace decreases the direct-prior diagnostic, before AdamW moments
or weight decay; it does not prove a finite step or actual old update does so.
Also retain supported period-cell gradient sign agreement/disagreement/zero
fractions with the direct-prior gradient, with the informative-cell count and
unweighted and |prior-gradient|-weighted disagreement. No magnitude/sign
threshold is chosen after music inspection.

Audit linearity against the independently computed total at both field and
parameter level. Retain L2 residual / sum of phase and count L2 norms (zero
denominator => zero only when all zero), plus total/component norm ratio to
expose cancellation. Require residual <=1e-10 for field gradients and <=5e-5
for native float32 CNN gradients; these are numerical audit tolerances, not
musical acceptance gates. Never silently sum independently normalized vectors.

Summarize every role separately by mean within work then across works; any null
metric keeps an explicit undefined count, not a smaller favorable denominator.
Additionally sum the 20 FIT parameter gradients with exact work weights
1/(number_of_works * recordings_in_work), separately for both checkpoints.
This is the fixed-checkpoint full-fit objective gradient, NOT a replay of one
historical shuffled batch or of AdamW. Do not mix development/diagnostic into it.
Retain every recording, both checkpoints and all comparisons.

## Interpretation and stop boundary

This can reject a missing-gradient hypothesis, locate sign/direction conflict,
or show shared-parameter interference at the selected states. It cannot prove
the complete training history, optimal loss weighting, sufficient training
duration/features/data, or that another fit will improve accuracy. No gradient
audit result automatically authorizes another fit, direct-prior auxiliary loss,
feedback gain change, smoothing policy, model promotion or release.
