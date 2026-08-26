# Metrical attribution v9

Date: 2026-08-26

This calibration report uses cached raw Beat This observations to distinguish
three different failure classes without changing the shipping analysis:

1. a wrong global half/double-time selection;
2. local curve or pulse-level errors despite a correct global level; and
3. timestamp errors at track edges versus inside matched anchors.

The diagnostics use truth only after inference and estimation. They are omitted
from holdout reports and never change a returned beat, BPM, hypothesis score, or
tempo segment.

## Reproducible inputs

- model pack: `models/beat-this-full-v1.json`
- model manifest SHA-256:
  `ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`
- suites: `evaluation/suites/fsld-tempo-v1.json` and
  `evaluation/suites/artbeat-v1.json`
- backend report schema: 9
- estimator: shipping default unless explicitly stated
- local cache root: `D:/rhythm-map-eval/observation-cache-v1`

## FSLD fixed-tempo coverage

The primary tempo map passes 6/15 cases. The already returned global
half/selected/double alternatives contain a BPM within five percent of the
fixed-tempo label in 14/15 cases. This is a truth-assisted coverage ceiling, not
a selector result.

| Expected BPM | Primary global error | Best top-K error | Best level |
| ---: | ---: | ---: | ---: |
| 41 | 99.57% | 0.21% | -1 |
| 60 | 96.12% | 1.94% | -1 |
| 70 | 0.31% | 0.31% | 0 |
| 80 | 0.45% | 0.45% | 0 |
| 90 | 0.01% | 0.01% | 0 |
| 100 | 0.00% | 0.00% | 0 |
| 110 | 0.21% | 0.21% | 0 |
| 120 | 0.05% | 0.05% | 0 |
| 128 | 95.31% | 2.34% | -1 |
| 130 | 50.19% | 0.38% | +1 |
| 140 | 43.59% | 12.82% | +1 |
| 150 | 0.00% | 0.00% | 0 |
| 160 | 0.48% | 0.48% | 0 |
| 180 | 0.04% | 0.04% | 0 |
| 200 | 50.00% | 0.00% | +1 |

The 90, 110, and 150 BPM cases demonstrate the other failure class: their
global level is already correct, but isolated local curve errors still fail the
P95 budget. `sequence-phase-v1` currently passes 8/15 by repairing the 60 BPM
edge-extra and rejecting the false half-time fold at 200 BPM. It remains an
evaluation candidate because FSLD has no beat timestamps and cannot validate
which events should be removed.

## ARTBeaT error location

Across 15 timestamped ARTBeaT calibration cases, selected-sequence misses are:

| Location | Missed truth beats | With backend candidate support | Selected extras |
| --- | ---: | ---: | ---: |
| Leading edge | 2 | 2 | 1 |
| Interior | 121 | 112 | 40 |
| Trailing edge | 5 | 4 | 3 |

Track edges account for only 7 of 128 misses. The dominant problem is therefore
interior pulse-level interpretation, especially local half/double-time changes,
not lack of bilateral smoothing at the beginning or end of these tracks.

## Decision

Do not add another edge heuristic or promote `sequence-phase-v1` from these
results. Whole-track alternatives already cover nearly every FSLD global BPM;
the remaining product problem is a truth-free selector plus locally varying
paths where one global octave choice is insufficient. Preserve the alternatives
as explicit metadata until new calibration evidence and a fresh precommitted
timestamped holdout support one unified selection rule.
