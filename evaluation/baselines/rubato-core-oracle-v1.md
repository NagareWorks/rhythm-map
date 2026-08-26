# RUBATO core-oracle calibration v1

Date: 2026-08-26

This baseline feeds the 25 official RUBATO beat/downbeat sequences directly to
the deterministic tempo-map estimator. It isolates estimator behavior from
Beat This inference and is calibration evidence, not a holdout result.

## Reproducible input

- suite: `evaluation/suites/rubato-calibration-v1.json` (`calibration`)
- source: RUBATO v0.3, DOI `10.5281/zenodo.21159596`
- 25 commercially compatible real performances across 12 works
- command:

```bash
cargo xtask eval \
  --suite evaluation/suites/rubato-calibration-v1.json \
  --no-fail
```

## Aggregate result

| Metric | Result |
| --- | ---: |
| Passing cases | 20/25 |
| Mean beat F1 | 1.0000 |
| Mean downbeat F1 | 1.0000 |
| Mean tempo median error | 2.6407% |
| Mean tempo P95 error | 16.1781% |

All beat and downbeat timestamps survive the ideal-observation path exactly.
Five cases fail only the locked tempo-error budgets:

| Case | Median tempo error | P95 tempo error |
| --- | ---: | ---: |
| `rubato-mozart-kv618-ar-papalin2012` | 9.79% | 51.14% |
| `rubato-mozart-kv618-ov-gliarmonici2015` | 40.38% | 90.11% |
| `rubato-verdi-nabucco-vapensiero-ar-r-operaphilia2022` | 3.45% | 43.77% |
| `rubato-verdi-nabucco-vapensiero-ov-garcia2015` | 1.77% | 30.52% |
| `rubato-verdi-nabucco-vapensiero-ov-secci2020` | 2.89% | 44.74% |

## Annotation audit

The importer retains official beat timestamps and never clamps or synthesizes
them. Seven sub-60-ms intervals across five tracks remain beat truth but are
omitted from BPM truth because they exceed the schema's 1000 BPM ceiling. One
Mozart terminal beat and matching measure marker occur 7.5 ms beyond decoded
audio due to frame quantization; both remain in the immutable CSVs and are
omitted from scored truth rather than moved to an invented timestamp.

## Decision

The five failures are deterministic estimator failures because the observation
timestamps are already exact. Do not weaken suite budgets or retrain Beat This
for them. Use this open calibration set to improve meter-aware tempo level and
local tempo representation, preserving explicit metrical ambiguity when the
evidence cannot choose safely.

A full Beat This run was not made part of this integration gate: on the
development VDI the first 128-second track remained in inference after more
than four minutes, projecting a multi-hour suite. Keep the full 1.8-hour
end-to-end baseline as a scheduled/offline evaluation with per-case progress.
