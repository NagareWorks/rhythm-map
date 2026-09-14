# Native AdamW displacement boundary: authored v1

Scope: no musical input, fitted weights, holdout, product change, new model,
auxiliary supervision or automatic fit admission. Earlier studies remain frozen.

1. Preserve PhaseSyncReadout, original phase/count losses, complete-record
   weighting, aggregate-before-routing in the same three disjoint groups, and
   one global norm clip. Observe both loss transports independently. Only this
   deterministic, buffer-free model is supported by the recorder.
2. Run native PyTorch AdamW on an owned clone of the complete parameters and
   current optimizer state. This includes existing moments, diagonal scaling,
   epsilon, bias correction and decoupled weight decay. No guessed AdamW formula.
3. For each group with count gradient c and adverse actual proposal d, project
   d onto `c dot d <= 0`: `d' = d - (c dot d)/(c dot c) * c`. Apply to old native
   weights and round back to their float32/float64 dtype. Recheck actual values;
   an adverse rounded projection restores that group's old values. A zero c
   means undefined protection, not evidence of correct tempo. No tolerances or
   repeated line-search/repair steps are used.
4. Certify the sign for the stored gradient mantissas and actual native endpoints
   with exact binary-integer accumulation. This avoids falsely passing a zero
   produced by rounding/underflow/cancellation. It does not improve gradient
   precision or guarantee finite-step count-loss reduction, let alone BPM.
5. A committed proposal advances ALL its AdamW moments/step counters, including
   vetoed/nonmoving groups. This is projected AdamW, not the old unconstrained
   optimizer and not per-group skipping of AdamW time. Report returned calls,
   committed proposals and proposals that actually moved weights separately.
6. Reuse FitRecorder's update lifecycle. Persist complete pre-update state before
   zero_grad; persist proposed weights/moments, candidate, count and receipts
   before live commit. A proposal/persistence fault leaves live state unchanged.
   A caught copy failure attempts rollback of parameters and state, then closes
   the recorder; failure evidence distinguishes restored from unknown rollback.
   Neither multi-tensor writes nor a process/device death are crash-atomic.
   Retain the existing entered/returned ambiguity, never silently resume/retry.
7. Authored regression coverage: real preconditioner/momentum/decay witnesses,
   native rounding, zero count/no movement, positive-exponent gradients, exact
   cancellation/underflow sign, unmodified safe AdamW parity, complete batch,
   partial gradient/record evidence, proposal/persistence/copy/postcommit faults,
   identity/unsupported ownership rejection, and real CNN/clock VJP integration
   on CPU and CUDA. Replaying captured evidence must not invoke an optimizer.
8. One fixed synthetic learning observation: existing teacher_case seed 1729,
   61 frames, 40 recorded steps; AdamW lr 0.001, decay 0.0001, clip 1; no sweep,
   no restart from best. Report initial/final phase and native count loss,
   all group actions and finite-count regressions, elapsed time and peak RSS.
   The reachability gate is both final losses strictly below their initial
   values and all committed native first-order signs nonpositive. A failure
   stays a failure; do not rewrite this criterion or infer musical success.

Ownership is single-caller/exclusive. One complete ordered AdamW parameter group,
one device and native dtype, disjoint contiguous storage, no optimizer hooks,
AMP, distributed/captured/differentiable/fused execution, schedulers or resume.
The tiny CNN permits a shadow copy; cost must be measured, not assumed free.
