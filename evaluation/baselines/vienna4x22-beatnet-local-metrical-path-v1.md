# Vienna 4x22 BeatNet locally varying metrical path holdout v1

Date: 2026-08-25

This is the single permitted opening of the corpus-disjoint Vienna 4x22
expressive-piano holdout. The candidate algorithm and constants were frozen as
`local-metrical-path-v1` at commit
`03fc058b3e0607faec53ab0ab9cb6d6425b9d8a5` before any holdout inference.
These recordings are now historical evidence and must not be used for tuning.

## Reproducible input

- suite: `evaluation/suites/vienna4x22-holdout-v1.json` (`holdout`)
- 12 performances: four works, performers p01, p08, and p15
- model pack: `models/beatnet-v1.json`
- model manifest SHA-256:
  `dcc6aeb313fda31ab862d287976cc7d7bc996e1ee78fc77b028e8b0a9d69b1e5`
- estimator policy: `local-metrical-path-v1`
- candidate definition commit: `03fc058b3e0607faec53ab0ab9cb6d6425b9d8a5`
- report schema: 1
- local report:
  `D:/rhythm-map-eval/reports/vienna4x22-beatnet-local-metrical-path-v1.json`

The holdout command accepted only the named policy and emitted no raw
observations or policy sweep. `truth_free_choice` compares only the already
returned primary and local relative scores. `coverage_ceiling` uses truth to
choose the better member of that pair and is not a deployable result.

## Aggregate result

| Result | Mean precision | Mean recall | Mean beat F1 | Cases passing 0.8 |
| --- | ---: | ---: | ---: | ---: |
| Primary selected sequence | 0.3168 | 0.6858 | 0.4018 | 0/12 |
| Existing truth-free choice | 0.3168 | 0.6858 | 0.4018 | 0/12 |
| Truth-assisted primary/local ceiling | 0.3965 | 0.7388 | 0.4772 | 0/12 |

The local path was distinct on 6/12 cases and improved all six. Its mean F1 on
those emitted cases was 0.5865. However, its relative score was below the
primary score in every emitted case, so the existing truth-free choice selected
the primary sequence 12/12 times and produced no deployable improvement.

| Case | Primary F1 | Local F1 | Delta | Local relative score |
| --- | ---: | ---: | ---: | ---: |
| `chopin-op38-p01` | 0.3481 | 0.3951 | +0.0470 | 0.8136 |
| `chopin-op38-p15` | 0.2768 | 0.5113 | +0.2345 | 0.7766 |
| `mozart-k331-p08` | 0.3299 | 0.5433 | +0.2134 | 0.7764 |
| `mozart-k331-p15` | 0.3475 | 0.4981 | +0.1505 | 0.7066 |
| `schubert-d783-p08` | 0.7368 | 0.7910 | +0.0541 | 0.8510 |
| `schubert-d783-p15` | 0.5743 | 0.7802 | +0.2060 | 0.7792 |

The coverage ceiling improved from 0.3191 to 0.4267 on six 6/8 performances
and from 0.6719 to 0.7586 on three Schubert 3/4 performances. The three Chopin
2/4 performances received no distinct local path and stayed at 0.2970 mean F1.

## Decision

The locally varying hypothesis generalizes as additional coverage, so retain it
as a necessary internal parallel hypothesis. Do not promote it unconditionally:
on ARTBeaT it was not always better when emitted, and this holdout proves that
the existing evidence ranker never selects it. The overall absolute result also
fails the locked gate by a wide margin.

Do not tune the event cost, harmonic weight, switch cost, ranker, or 2/4 behavior
on Vienna. Future work requires new calibration data for a truth-free selector
and for missing local-path coverage, followed by a new independently
precommitted holdout.
