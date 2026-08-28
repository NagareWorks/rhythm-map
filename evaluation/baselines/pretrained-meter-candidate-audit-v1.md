# Pretrained perceived-beat candidate audit v1

Date: 2026-08-28

This audit follows the failed BeatNet replacement and consensus experiments.
Its purpose is to find already-trained, commercially usable evidence that is
independent enough to resolve canonical half/double-time choices. It is not a
model leaderboard and did not open `rubato-holdout-v1`.

## Acceptance boundary

A candidate can enter calibration only if all of the following are true:

- source code and pretrained weights have explicit commercially usable terms;
- inference can yield backend-neutral pulse or meter evidence without adopting
  a decoder that forces a preferred BPM range or meter;
- the released weights match the released inference architecture;
- the dependency and artifact cost is plausible for an optional packaged
  component; and
- success would still feed the single time-map estimator rather than create a
  user-visible backend strategy.

## Findings

| Candidate | Released terms | Relevant evidence | Packaging finding | Decision |
| --- | --- | --- | --- | --- |
| madmom pretrained models | Model repository declares CC BY-NC-SA 4.0 | Beat/downbeat activations and DBN tracking | Non-commercial weights conflict with the intended library/product use | Reject |
| BeatNet+ | No license found in the repository root or package metadata at audit time | A non-percussive/classical checkpoint is advertised | Source and weight redistribution/use rights are not established | Do not adopt unless the authors publish explicit terms |
| BEAST | No license found with its released pretrained model at audit time | Beat tracking | Source and weight redistribution/use rights are not established | Do not adopt unless the authors publish explicit terms |
| Beat Transformer | MIT repository with eight released checkpoints | Beat/downbeat activations from a demixed transformer | Technically auditable, but the usable weights require five Spleeter stems; the lighter non-demixed architecture has no released matching weights | Do not integrate into the core or run the holdout |

Official project sources:

- [madmom model terms](https://github.com/CPJKU/madmom_models)
- [BeatNet+](https://github.com/mjhydri/BeatNet-Plus)
- [BEAST](https://github.com/WildHoneyPie/BEAST)
- [Beat Transformer](https://github.com/zhaojw1998/Beat-Transformer)
- [Beat Transformer license](https://github.com/zhaojw1998/Beat-Transformer/blob/main/LICENSE)

## Beat Transformer feasibility detail

Beat Transformer is the only candidate in this pass that clears the initial
license check, but the released artifact shape does not clear the packaging or
independence boundary:

- the public checkpoint directory contains only eight cross-validation fold
  files, each about 35.5 MB, for roughly 284 MB in total;
- the checkpoint-loading evaluation constructs the main model with five
  instruments and consumes `(batch, instrument, time, mel-bin)` input;
- the official preprocessing separates each recording into five Spleeter
  stems before computing 128-bin mel spectrograms at 44.1 kHz / 1024 hop;
- `code/ablation_models/non_demix_model.py` proves that a single-mixture
  architecture was studied, but no corresponding pretrained checkpoint is
  present in the released checkpoint directory; and
- the example decoder constrains tempo to 55--215 BPM and downbeat meter to
  3/4. Rhythm Map must not inherit either assumption. A future experiment could
  consume only activations and use the existing backend-neutral time-map
  estimator, but only after a matching practical checkpoint exists.

Using one arbitrary cross-validation fold as a universal model would not be a
reproducible substitute for a released all-data checkpoint. Running all eight
folds plus source separation would also turn an optional evidence source into
a large Python/TensorFlow/PyTorch preprocessing stack that is unsuitable for
the Rust/ONNX/WASM product boundary.

## Decision

Do not add another backend, selector, product strategy, or dependency in this
iteration. BeatNet remains useful evaluation evidence, and Beat Transformer is
recorded as a licensed research lead rather than integrated code. Keep the
RUBATO holdout sealed.

Under the current no-training constraint, no audited pretrained candidate
provides a complete safe selector for canonical beat level. The shipping
behavior therefore remains one zero-tuning estimator that exposes supported
metrical alternatives, localized ambiguity, provenance, and confidence. A
future candidate may reopen calibration only when it supplies independently
licensed, matching weights that can add meter evidence without forcing a BPM
band or inventing timestamps.
