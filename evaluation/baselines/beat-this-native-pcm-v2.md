# Native PCM decoder/resampler isolation — 2026-09-03

The remaining ARTBeaT 15 original-file parity mismatch follows the resampling
path, not the native decode/downmix path, in this controlled experiment.
This diagnoses the discrepancy; it does **not** fix it or establish which beat
sequence is musically correct.

Evidence: [`native-pcm-v2-audit.json`](../parity/native-pcm-v2-audit.json).
The experiment uses the previously frozen full calibration recording and v2
trace, the pinned official `final0` checkpoint, official frontend, and minimal
postprocessor. No annotations, holdout, threshold tuning, or alternate model
are involved. All five inference runs use the same official implementation.

## Controlled result

Both native decoders return 996,141 mono frames at 44,100 Hz after downmix;
both resamplers return 498,071 frames at 22,050 Hz. The diagnostic lag scan
finds zero offset in every pair. Native decode/downmix differs numerically
(full-waveform RMSE approximately `4.61e-8`, maximum `9.54e-7`) but not in the
selected beat or downbeat timestamps at either fixed resampler.

| Native decode/downmix | Current Rust resampler | Official soxr HQ resampler |
| --- | ---: | ---: |
| Rust | 37 beats, 8 downbeats | 36 beats, 8 downbeats |
| Official | 37 beats, 8 downbeats | 36 beats, 8 downbeats |

The table is an event-count summary, not the sole test: switching decoders
preserves **every timestamp**, while switching resamplers with either fixed
decoder produces exactly the same three event changes. Relative to soxr,
the current Rust path omits 19.08 s and adds 20.84 s and 22.26 s. Downbeats
are identical across all five runs.

At fixed native input, resampler-path differences have full-waveform RMSE
about `3.74e-4`, maximum sample difference `0.00654`, and maximum beat-logit
difference about `0.193`. Decode-only differences produce maximum beat-logit
differences below `6e-5` and no event changes. This separates the two paths;
it does not yet isolate cutoff, filter length, fractional phase, or internal
arithmetic within a resampler.

| Fixed probe | Official source probability | Current Rust PCM probability |
| --- | ---: | ---: |
| 19.08 s | 0.53959 | 0.49631 |
| 20.84 s | 0.49656 | 0.52960 |
| 22.26 s | 0.49855 | 0.51423 |

These are nearby peak probabilities, with event selection verified separately;
the production threshold remains strictly above 0.5.

## Controls and scope

The 2x2 matrix normalizes native mono inputs to float32. Rust consumes that
float32 input; soxr receives its float64 promotion, matching the official
path's input type. A fifth run preserves the official loader's unrounded
float64 mono input. For this recording, the normalization produces exactly
the same resampled float32 waveform, logits, and events as that fifth run.
Thus the matrix has not substituted a precision change for the original-file
control. This identity is measured, not assumed for other recordings.

The Rust adapter now shares an explicit native decode helper with its shipping
file path. Evaluation-only hooks expose the actual native stage and actual
mono resampling stage without loading a model or offering product strategies.
The native-stage reconstruction is bit-identical to shipping decode and to
the frozen v2 PCM trace. Same-PCM official/RTen logit and event budgets still
pass. The observation contract therefore stays at v2.

`controls_passed: true` means the diagnostic controls succeeded;
`source_event_parity_passed: false` deliberately preserves the unresolved
original-file mismatch. The earlier 63/64 report is not overwritten or turned
green. No accuracy suite is rerun or rescored here, and the preceding 30-case
mixed accuracy result remains the last full calibration evidence.

Local Windows verification: 177 workspace/all-target Rust tests and 20 Python
diagnostic tests pass, as do formatting, all-feature Clippy, doc tests,
no-default-feature adapter/model tests, the five generated core regression
cases, and the release-profile WASM build. The report's auditor/exporter/source
digests match the files used for this run. This is not a new remote CI or
macOS verification result.

## Next decision

Stop investigating codec replacement or adjusting the beat threshold for this
case. First characterize reference-compatible resampling on generated impulses,
sweeps, edges, and multiple rates, then evaluate one internal compatibility
candidate with the frozen reference checks and full paired calibration.
Any replacement must retain alignment/duration invariants, commercial-license
and embedding compatibility, and acceptable performance. Do not add Python or
another runtime to the product merely because it reproduces one recording.
Keep one public behavior, the holdout sealed, and releases paused.
