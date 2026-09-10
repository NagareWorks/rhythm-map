# Direct clock learning: one development experiment

This replaces the proposed candidate-ranking pilot for the current experiment.
It is not a production model, independent acceptance or a new user strategy.
The user accepted moving from repeated diagnostics to a small learning experiment
on 2026-09-10. No existing results, dataset locks or shipping outputs are rewritten.

Frozen scope before fitting:

- Reuse the 25 already exposed RUBATO recordings (12 works). A salted work-ID
  hash determines nine fit works and three development works. Repeated versions
  of one work stay together. Performer and encoder-recording independence are
  not certified. This is exploratory reuse of calibration, not a fresh holdout.
- All 15 ARTBeaT calibration inputs remain diagnostic only; they cannot select
  parameters or checkpoints. No sealed holdout or new audio acquisition.
- Use the first at most 60 seconds of each declared input, retaining shorter
  recordings and tail frames. These are explicitly crops when the original is
  longer; metrics are not the old full-recording results. Do not choose crops
  based on detector outcomes. Unbracketed annotation frames remain unavailable.
- Frozen Beat This final0 encoder, normalized 512-channel / 50 Hz features.
  Three direct per-frame outputs: log2(period in seconds), phase cosine/sine.
  Native annotation beat convention is a supervised target, not a claim that
  other levels are wrong. RUBATO gets no invented hard change/no-rhythm labels.
- A 512->32 projection, six residual depthwise-kernel-5/pointwise blocks with
  dilations 1/2/4/8/16/32, and 32->3 output: **24,003 trainable parameters**.
  Radius 126 frames. No candidate list, tempo prior input or annotation feature.
- One seed 142, one main fit and one identical zero-audio-feature control.
  At most 20 epochs / 30 minutes each; two fits / one hour total. AdamW lr 0.001,
  weight decay 0.0001, gradient norm cap 1, batch size 4. No sweep or seed picking.
  Loss is native log-period squared error plus mean squared cosine/sine error.
  Group-balanced training; checkpoint by lowest development loss, earlier tie.
- Feature preparation is separately capped at 120 CPU-core minutes. Reuse
  byte-identified captures only for identical crop/ownership inputs. Private
  features and fitted weights stay off the system drive and out of Git.
- Compare same-input frozen raw-event interpolation, direct learned output and
  fitted zero-audio control. Report every input, missing output, native tempo
  median/P95 and phase error separately. Report original calibration identities
  but do not label overlapping frames or exposed works independent trials.
- Continue this learned design only if macro development tempo-median error and
  phase error both improve by at least 10% over the fitted zero-audio control,
  neither worsens against raw-event interpolation on common available support,
  and prediction coverage is not reduced. Per-input regressions are mandatory
  output even if the gate passes. A failure closes this fixed model; no automatic
  widening, threshold repair, seed replacement or encoder fine-tuning.

The phase and period heads can disagree. Do not silently integrate one into the
other or turn phase crossings into detected beats. This experiment tests direct
temporal prediction; coherence, independent data and Rust/FFI/WASM integration
remain separate product requirements. A successful fit is not a release gate.
