# Beat This time-aligned resampling v2

Date: 2026-09-03

This corrects an input time-axis defect; it is not a new musical-pulse selector.
The estimator, decoder, model weights, and acceptance thresholds are unchanged.
No holdout was opened and no package/release was published.

## Implementation boundary

The adapter now owns file decode/downmix and one shared mono resampling path
used by file, native PCM, and evaluation trace callers. It retains the existing
256-tap sinc parameters, processes fixed-size chunks, removes the initial
filter delay with the rubato 3.0 sampling-phase convention, flushes the delayed
tail, and returns the nearest integer output duration. It does not subtract
an offset from returned beats. See [`../../docs/ALGORITHM.md`](../../docs/ALGORITHM.md).

The raw-observation cache contract changes from
`beat-this-rten-observations-v1+decode-audio-v1` to
`beat-this-rten-observations-v2+decode-audio-v2`. Old entries remain intact but
cannot satisfy v2 requests. Traces additionally fingerprint the new audio module.
The adapted upstream decode scaffold retains its MIT notice in the source crate
and native-distribution staging logic. No new native/Python inference dependency
or caller-selectable policy was introduced.

Seven model-free audio tests cover native PCM identity, invalid PCM, safe
downmixing, generated WAV/PCM equivalence, short and exact/partial-chunk lengths,
chunk-size invariance, and beginning/interior/tail impulses at
8/16/44.1/48/96/192 kHz. Impulse peaks are within one output sample of the
expected location; 44.1 kHz test impulses match exactly. This is a bounded
alignment guarantee, not bit-exact equivalence to soxr at every sampling rate.

## Official-reference parity

The source revision, original `final0` checkpoint, ONNX models, cases, prefix
durations, and all numerical budgets are identical to the
[v1 audit](beat-this-reference-parity-v1.md). The new aggregate report is
[`../parity/baseline-v2.json`](../parity/baseline-v2.json); v1 remains historical.
Both cases pass all 16 stage checks (32 total).

| Measurement | ARTBeaT 05 before | ARTBeaT 05 after |
| --- | ---: | ---: |
| Output samples | 299,879 | 299,880 |
| Sample-count difference from official | -1 | 0 |
| Best waveform delay, first two seconds | 63 samples | 0 samples |
| Unshifted waveform RMSE | 0.0791307 | 0.000109644 |
| Maximum beat timestamp difference from official file path | about 20 ms | 0.000458 ms |
| Maximum downbeat timestamp difference from official file path | about 20 ms | 0.000306 ms |
| Beat/downbeat counts | 17 / 5 | 17 / 5 |

The 35-second native-22.05 kHz RUBATO Bach prefix still has 771,750 samples,
zero PCM difference, zero waveform delay, 50 beats and 24 downbeats. Its
timestamp difference remains below two microseconds (f32 representation).
These two cases establish numerical compatibility, not general music accuracy.

## ARTBeaT calibration regression

The compact per-case metrics, audio identities, executable digests, and full
report digests for both suites are retained in
[`../parity/resampling-v2-calibration.json`](../parity/resampling-v2-calibration.json).
This file contains neither PCM nor private filesystem paths.

All 15 cases were replayed with the existing v1 executable (15 cache hits) and
then inferred with v2 (15 cache misses). All oracle results are exactly equal;
the model pack, case IDs, audio hashes, and default policies match.

| Metric, unweighted case mean unless noted | v1 | v2 |
| --- | ---: | ---: |
| Beat F1 | 0.805161 | 0.807961 |
| Median tempo error (%) | 18.512812 | 19.050469 |
| P95 tempo error (%) | 66.480756 | 58.964548 |
| Cases passing all musical gates | 1 / 15 | 1 / 15 |

Beat F1 improves on five cases, regresses on three, and is unchanged on seven.
Change recall is unchanged on every case, but change precision is not: ARTBeaT
14 produces an extra false change and its tempo P95 worsens from 78.5716% to
108.3335%, despite better beat F1. A better aggregate number is not a no-regression
result. In particular:

| Regressing beat case | v1 F1 | v2 F1 | Observed event-level change |
| --- | ---: | ---: | --- |
| 13: 180 to 120 | 0.750000 | 0.730159 | 31 to 30 events; 24 to 23 truth matches; one additional leading miss |
| 15: 85 to 127.5 | 0.648649 | 0.623377 | 34 to 37 events; 24 matches unchanged; three additional extras |
| 18: piano rubato | 0.756757 | 0.722222 | 30 to 28 events; 28 to 26 matches; two additional interior misses |

These changes already exist in the raw backend events; analyzed counts equal
raw counts on all three cases. They are not merely matches crossing the scoring
window after a timestamp shift. Cache inspection localizes threshold crossings:

- Case 13's real 0.34 s candidate remains, but confidence changes from 0.6307
  to 0.4712, below the unchanged 0.5 peak threshold.
- Case 18 loses near-threshold peaks around 5.06, 14.14, and 15.14 s. Their v1
  confidences were 0.5049, 0.5271, and 0.5482; nearby v2 candidates are 0.3602,
  0.3687, and 0.4634. A new 8.34 s event partly offsets the lost count.
- Case 15's four additional peaks near 20.12--22.26 s cross upward: v1
  confidences 0.4567/0.4018/0.2453/0.4380 become
  0.6057/0.5296/0.5847/0.5142. Another old event disappears, giving a net three
  extras. Flushing the signal tail does not justify deleting these outputs
  solely because they are near the track end.

This is evidence of model/peak-threshold sensitivity to preprocessing, not yet
proof of why each neural confidence changed. Before changing the decoder,
extend official-file parity to these regressions and isolate input-phase/tail
sensitivity on calibration. Do not restore a known waveform delay to optimize
this corpus, lower a global threshold to recover selected missed beats, add a
legacy-resampling user switch, or treat the two-case numerical pass as a release
accuracy gate. The fix remains unreleased development work.

## FSLD tempo-only calibration regression

All 15 cases completed with 15 old-cache hits and 15 new-cache misses. Oracle
results are exactly unchanged. No previously passing case became a failure.

| Metric, unweighted case mean unless noted | v1 | v2 |
| --- | ---: | ---: |
| Median tempo error (%) | 29.431193 | 22.348798 |
| P95 tempo error (%) | 52.113052 | 50.187819 |
| Cases passing all tempo gates | 6 / 15 | 7 / 15 |

The 200 BPM case is the new pass: its roughly 50% half-time error falls below
0.00002%. The 128 BPM case also improves substantially but still fails. These
gains must not hide the 110 BPM case's P95 regression from 33.4811% to
109.7900%, or the smaller regressions listed in the compact report. FSLD has no
beat/downbeat phase truth, so this audit makes no beat-F1 claim for that suite.

## Development decision and checks

Keep the corrected time origin as unreleased development work, with one
preprocessing path and no legacy/user-selectable fallback. The accuracy gates
remain **1/15 ARTBeaT and 7/15 FSLD**, not release-ready results. Next inspect
official-source parity and phase/tail sensitivity on the three ARTBeaT beat
regressions and the FSLD 110 BPM tempo-tail regression before considering a
decoder change. Preserve the sealed RUBATO holdout.

Local validation passed: formatting, all-target/all-feature Clippy, 175 Rust
unit/integration/example tests, workspace doc tests, no-default-feature model
tests and Beat This check, seven Python parity tests, five generated core cases,
and the `wasm32-unknown-unknown` build. This does not claim a fresh macOS/Linux
native CI run, a package publication, or a Git push.

## Reproduce

Run the commands in [`../parity/README.md`](../parity/README.md) with new trace
and report filenames. Keep PCM traces and source audio outside Git. For each
calibration suite, compare an unchanged v1 executable/cache replay with the v2
executable using the same model pack, audio bytes, and default policies:

```bash
cargo xtask eval-backend --suite evaluation/suites/artbeat-v1.json \
  --model-dir /data/beat-this-full-v1 --audio-dir /data/artbeat-v1 \
  --observation-cache /data/observation-cache --no-fail \
  --report /data/reports/artbeat-resampling-after-v2.json

cargo xtask eval-backend --suite evaluation/suites/fsld-tempo-v1.json \
  --model-dir /data/beat-this-full-v1 --audio-dir /data/fsld-tempo-v1 \
  --observation-cache /data/observation-cache --no-fail \
  --report /data/reports/fsld-resampling-after-v2.json
```

`--no-fail` permits reporting the already-known musical accuracy failures; it
does not mark acceptance gates as passed or suppress decoding/inference errors.
FSLD is tempo-only and must not be treated as a beat/downbeat benchmark.
