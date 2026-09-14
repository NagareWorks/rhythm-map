# Native offset versus variation v1

Freeze this source and contract before fitting. The closed tempo-conflict audit
found native-fit/development gradient opposition, but did not isolate its offset
component. Test that hypothesis, not another coefficient/epoch search. No new
architecture, encoder calls, audio rendering, phase training, product settings,
release or holdout access. Old failed experiments remain immutable.

## One loss ablation

On one original FIT recording's valid outgoing cells, let residual e be predicted
log2 period minus native reference log2 period. The old MSE is exactly
mean(e squared) = mean(e) squared + mean((e - mean(e)) squared).
Call these offset and variation respectively. Center the residual over this
recording's valid cells only, not across recordings, works or a whole batch.
Keep the mean differentiable. Both terms use the original float64 target math.
Masks remain unchanged; no filling gaps, beat-level canonicalization or label
repair. These are recording/crop-level terms, not a claim about a full song.

Two arms, both initialized from the same pre-relative separated-natural tempo
head (23,937-parameter, unchanged 32-channel dilated CNN):

- native-retained control: original native MSE + pair MSE + retention MSE.
- variation-retained primary: native variation MSE + pair MSE + retention MSE.

Keep coefficients 1, the original detached initial-output teacher on natural
FIT cells, equal-work weighting, 20 epochs / 100 AdamW updates per arm, LR .0003,
weight decay .0001, clip norm 1, natural seed 142 and pair seed 8142. Identical
four-record batches and four replacement draws from each of 15 pair groups.
Use all 559 admitted old points; retain all 915 points and exclusions. No loss
rescaling to compensate for removing offset. Original full loss is evaluated
with the original function so the control can replay the closed retained arm.
Require all 21 control epoch state hashes to match it, otherwise stop before
the primary fit: the comparison would not be isolated. No automatic retry.

All 40 natural packets are byte-pinned: 20 FIT / 9 works, 5 development / 3
disjoint works and 15 exposed ART variants / 1 source. Only FIT enters gradients
or teacher targets. Selection uses development metrics and training pair MSE
at epoch 0 and each complete epoch, with exactly the old work/record 1.05 gates
and earliest tie policy. Finish all 100 updates; keep all epoch checkpoints,
selected and terminal models. Epoch-zero fallback is explicitly not learning.

## Evaluation without oracle centering

No centering, reference-assisted output shift or post-hoc calibration at
inference/evaluation. Retain absolute native MSE and BPM median/p95 errors for
every record and work macro, all old pair metrics and identity controls.
Additionally report native MSE offset/variation decomposition as diagnostics;
these never select checkpoints. Neither evaluation nor test labels enter fit.

Reuse the six already encoded local-tempo v1 schedules only AFTER both arms'
selected/terminal exports freeze. Their 366-point ledger (224 admitted, 140
changed) and features remain byte-pinned. Relabel their role
`exposed_schedule_diagnostic`; they are NOT fresh or independent validation.
They never train or select. Same three source works and renderer cannot establish
domain generalization or rule out renderer shortcuts. No new schedule is made.

Apply the old numerical outcome checks without weakening them: primary selected
epoch >0; each identity MAE <=.05; each of six exposed schedules' changed RMSE
<=.75 zero-change and <=.8 previous-global, slope [.5,1.5], sign >=.8;
development AND ART work-macro MSE and median BPM error <=1.05 initial. Call this
an exposed exploratory gate, never independent acceptance. Report both arms'
selected AND terminal scores; do not choose the winning arm after results.
Even a pass requires subsequent genuinely unexposed evaluation before promotion.

## Observability and limits

Record full/offset/variation and active native loss per batch. Before every
update, persist exact batches, current weights, optimizer moments and RNG;
after backward save unclipped/clipped gradients; after step save actual weights
and moments. Persist entered/returned receipts and every epoch export. These
packets close the previous unsaved-intermediate-update gap and are not inputs
to gradient routing. No development gradients are computed in this experiment.

Deterministic FP32 model, inherited float64 native/retention math, TF32 off,
2 CPU threads, GPU allocation cap 40%. One run, no tuning after outcomes;
1800 s per arm, 5400 s outer watchdog. On failure preserve partial artifacts,
do not restart fitting. Verify control replay, matched draws and unchanged
teacher/initial/input/source identities. Replay saved predictions/selection
without optimizer steps, and independently check saved AdamW displacements.
The independent NumPy float64 update check requires maximum absolute weight
error <=2e-7 and moment agreement at rtol 2e-5 / atol 2e-8 against saved FP32
states; it is a numerical audit, not exact bitwise optimizer emulation.
Publish protocol, metrics and provenance, never private audio/features/weights.
