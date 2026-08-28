# RUBATO Beat This observation baseline v1

Date: 2026-08-28

This baseline measures the shipping Beat This observation path on the 25-case
RUBATO real-performance calibration suite. It was completed before any model
inference on `rubato-holdout-v1`. The holdout remains sealed.

## Reproducible input

- suite: `evaluation/suites/rubato-calibration-v1.json` (`calibration`)
- model pack: `models/beat-this-full-v1.json`
- model manifest SHA-256:
  `ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`
- backend: Beat This through RTen, immutable upstream decoder
- estimator: shipping default
- analysis schema: 4
- report schema: 11
- observation cache: 8 hits and 17 misses during the completed run; 25/25
  observations are now cached
- report:
  `D:/rhythm-map-eval/reports/rubato-beat-this-calibration-v11.json`

```bash
cargo xtask eval-backend \
  --suite evaluation/suites/rubato-calibration-v1.json \
  --model-pack models/beat-this-full-v1.json \
  --model-dir D:/rhythm-map-models/beat-this-full-v1 \
  --audio-dir D:/rhythm-map-eval/rubato-calibration-v1 \
  --observation-cache D:/rhythm-map-eval/observation-cache \
  --report D:/rhythm-map-eval/reports/rubato-beat-this-calibration-v11.json \
  --no-fail
```

## End-to-end result

The deterministic estimator is not the failing component. Feeding the same 25
official beat/downbeat sequences directly to the estimator passes every case,
with exact beat and downbeat F1 and low tempo error. The audio observation path
passes only one case.

| Path | Passing cases | Mean beat precision | Mean beat recall | Mean beat F1 | Mean downbeat F1 | Mean tempo median error | Mean tempo P95 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Official observations -> estimator | 25/25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1544% | 4.4053% |
| Beat This -> estimator | 1/25 | 0.4870 | 0.5920 | 0.5213 | 0.4434 | 65.0667% | 212.9036% |

The one passing end-to-end case is
`rubato-handel-hwv040-1-01-ar-buttnerfmaj2025`. The weakest slices are organ
(0.1964 mean beat F1), choir (0.2568), clarinet (0.3648), strings (0.4095),
orchestra (0.4494), and restored arrangements (0.4175). These tags overlap and
are diagnostics, not runtime routing labels.

## Observation and hypothesis coverage

Across 6,726 annotated beats, the selected sequence emits 9,261 events and
matches 4,210. The backend's retained radius-one local maxima match 5,523 truth
beats, for 82.11% micro candidate recall. Of 2,510 truth beats missed by the
selected sequence, only 1,307 have candidate support, for 52.07% micro recall.
The remaining misses cannot be repaired without new observation evidence or
invented timestamps.

| Interpretation | Mean beat F1 | Cases improved | Cases regressed |
| --- | ---: | ---: | ---: |
| Shipping selected sequence | 0.5213 | - | - |
| Current truth-free top-1 rank | 0.5221 | 1 | 0 |
| Truth-assisted best top-K ceiling | 0.5232 | 3 | 0 |

The locally varying metrical path is emitted in only three cases. It improves
Vivaldi RV 269 / Modena Chamber 2022 by 0.0209 F1, but regresses Handel HWV 40
/ Buettner F major 2025 by 0.0198 and Vivaldi RV 269 / Intartaglia 2011 by
0.0210. The existing truth-free rank chooses the useful local path only for the
Modena case and otherwise retains the selected sequence.

Even a truth-assisted oracle over the fixed hypothesis vocabulary finds gains
in only three cases: half-time phase for Berlioz H 48 (+0.0132), the local path
for Vivaldi / Modena (+0.0209), and candidate midpoint augmentation for Vivaldi
/ Intartaglia (+0.0134). This ceiling is far too small to explain the broad
end-to-end failure.

## Decision

Do not promote a canonical selector or open `rubato-holdout-v1` from this
result. Selecting every emitted local path is unsafe, while the current
truth-free ranker's small RUBATO gain does not erase its known ARTBeaT
regressions. Keep the shipping selected time map unchanged and continue to
publish supported alternatives and localized ambiguity metadata.

The measured bottleneck is the observation path: missing candidate evidence,
extra events, and incorrect beat-level semantics dominate the residual. The
next training-free experiment is one fixed BeatNet run on this same calibration
suite. Its purpose is to decide whether an existing alternate backend supplies
materially better evidence; it is not permission to add a public backend
strategy, tune per-case rules, or inspect the sealed holdout.

That experiment is recorded in
[`rubato-beatnet-observation-v1.md`](rubato-beatnet-observation-v1.md). BeatNet
adds candidate coverage but fails both replacement and consensus gates.
