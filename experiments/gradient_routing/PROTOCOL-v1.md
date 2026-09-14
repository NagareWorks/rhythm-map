# Count-priority routing mechanism v1

Registered before reading any retained musical gradient through this mechanism.
This is NOT a new musical fit, loss-weight sweep or admitted product strategy.

## Single mechanism

Keep PhaseSyncReadout, its complete recurrence, native phase/count losses,
annotation support and all old source identities unchanged. Use the original
count gradient c as the protected direction, NOT the direct-prior diagnostic.
In each of three exhaustive, disjoint parameter groups (period output row,
shared CNN, all remaining phase/origin/gain parameters), replace phase gradient p
by p' = p - min(0, dot(p,c)/dot(c,c)) c. Return c + p'. If c is zero, retain p
and report undefined protection; do not fabricate a useful count direction.
Sum all weighted recordings in a batch BEFORE this nonlinear operation, then
apply any single global clipping operation. Do not route individual recordings
and call their sum the routed batch. There is no random task order or song knob.

The ideal Euclidean direction satisfies dot(c,c+p') >= ||c||^2 in EACH group.
This protects the local count objective, not necessarily direct prior, musical
BPM, phase, finite-step losses or AdamW displacement. Opposing collinear goals
can discard the complete phase gradient. No mechanism can promise simultaneous
strict descent for exactly opposite gradients. Zero directions stay explicit.

Use frozen binary-exponent gradient packets; projection is evaluated at the
phase mantissa scale, avoiding exponent-ratio materialization. Reject input or
output alignment losses, unknown model geometry and a routed direction whose
count projection is below (1-1e-10) times the original count gradient. This is a
floating-point check, not arbitrary precision or a blanket conditioning cure.

## Authored gate before any musical packet analysis

- Independent dense formula, aligned/orthogonal/opposed/zero cases, large binary
  exponents, rejected alignment loss, exhaustive parameter ownership, no mutation,
  aggregation before routing, positive global-clipping sign preservation.
- Full unchanged CNN + clock phase/count gradients with native CPU execution;
  T4 equivalent when available. Check all three groups, `.grad` buffer ownership,
  unchanged forward/model state before any synthetic update.
- Representable constant, abrupt step, gradual ramp and zero-observation-gap
  clocks retain the earlier model's unsmoothed coherent output. These are authored
  fields, not proof that an audio encoder can predict those fields.
- A fixed-seed short authored full-CNN optimization check with plain SGD and
  the original losses must reduce both losses on its fixed teacher-generated
  reference. Maximum 40 updates; no search or music/old weight loading. Report
  a failure rather than changing that gate after observing the result.
- Actual AdamW witnesses test whether first-step diagonal preconditioning and
  retained momentum can break count descent despite protected raw gradients.
  These are REQUIRED negative controls, not failing tests to hide or relax.
  A read-only actual-displacement check must detect the counterexamples.

## Fixed-packet diagnostic, once

After authored checks, reuse ONLY the 80 parameter-gradient packets from the
completed gradient_audit, identified by its private report SHA256
19a3b3d540fa968aa8ec37a63712d5953b9f53cbf1b14430d4050b9ac54d4a91 and per-case
packet hashes. No CNN/clock forward, model/optimizer state loading or update.
Verify the source report, exact population and every packet identity. Inspect
each pair separately, and independently sum the complete 20 FIT recordings with
weights 1/(9 * recordings_in_work) for each selected checkpoint before routing.
Never combine development or ARTBeaT diagnostic recordings into that aggregate.
Report every case, undefined denominator and negative direct-prior alignment;
the direct prior is a comparator only, never an input to the routing decision.
Persist the diagnostic only at a fresh output path, with all source hashes.
Do not select grouping, tolerance, model or treatment based on these results.

## Stop rule

Even successful routing and authored SGD learning do not authorize another
musical fit. If AdamW can reverse the protected direction, gradient-only routing
is NOT ready to insert into the old fitter. Next require an optimizer-boundary
design that checks the actual proposed, native-rounded parameter displacement
and preserves recorder/complete-batch semantics, then preregister a separately
falsifiable musical comparison. No architecture split, auxiliary prior target,
AdamW history reconstruction, holdout, default change or release in this step.
