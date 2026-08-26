# ARTBeaT localized metrical ambiguity regions v1

Date: 2026-08-27

This baseline promotes the already-auditable selected/local disagreement
partition into product metadata. It does not reuse the rejected cross-backend
selector: no BeatNet evidence, annotations, or truth-assisted margin enters the
analysis result, and no region changes the selected time map.

## Reproducible input

- suite: `evaluation/suites/artbeat-v1.json` (`calibration`)
- model pack: `models/beat-this-full-v1.json`
- model manifest SHA-256:
  `ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`
- backend: Beat This through RTen, immutable upstream decoder
- estimator: shipping default
- analysis schema: 4
- report schema: 11
- observation cache: 15/15 hits
- report:
  `D:/rhythm-map-eval/reports/artbeat-local-ambiguity-regions-v11.json`
- comparison:
  `D:/rhythm-map-eval/reports/artbeat-default-local-hypothesis-v10.json`

## Region contract

Exact timestamps shared by the selected and locally varying hypotheses are
anchors. Each anchor-to-anchor span with at least one sequence-only real event
is returned. Audio zero and duration bound one-sided regions but are never
treated as beat timestamps.

| Anchor state | Meaning | Regions | Cases |
| --- | --- | ---: | ---: |
| right only | leading edge | 3 | 3 |
| both sides | bounded interior | 34 | 9 |
| left only | trailing edge | 3 | 3 |
| neither side | whole-track unanchored | 0 | 0 |
| **Total** |  | **40** | **10 distinct cases** |

Across the regions, the selected path contains 49 unique real timestamps and
the local path contains 26 unique real timestamps. These counts describe
disagreement; they are not votes for either path.

The six one-sided edge regions are:

| Case | Edge | Selected-only | Local-only |
| --- | --- | ---: | ---: |
| `artbeat-06-150-to-75` | trailing | 1 | 0 |
| `artbeat-13-180-to-120` | leading | 5 | 0 |
| `artbeat-14-240-to-96` | leading | 1 | 0 |
| `artbeat-15-85-to-127-5` | trailing | 2 | 0 |
| `artbeat-20-ramp-200-to-80` | trailing | 1 | 0 |
| `artbeat-21-polyrhythm-70-to-105` | leading | 2 | 1 |

## Compatibility and performance

Primary end-to-end metrics, every beat hypothesis and score, and all warnings
exactly match the schema-v3 comparison report. Only schema-v4 region metadata
and report-schema-v11 diagnostics are added. The hot-cache run averaged 54.16
ms per case for PCM enrichment, estimation, and region derivation; this remains
deterministic post-processing rather than neural inference.

Older serialized analyses remain readable because the new field has an empty
default. Consumers that require the new field can gate on analysis schema 4.

## Decision

Ship localized ambiguity regions across Rust, CLI, C ABI JSON, and WASM JSON.
Do not add another edge repair, extrapolate a grid from one anchor, or select a
local path from these regions. Their purpose is to let an editor highlight
uncertain timing spans and offer an informed manual choice while the automatic
result remains conservative and zero-tuning.
