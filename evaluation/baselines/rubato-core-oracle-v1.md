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
| Passing cases | 25/25 |
| Mean beat F1 | 1.0000 |
| Mean downbeat F1 | 1.0000 |
| Mean tempo median error | 0.1544% |
| Mean tempo P95 error | 4.4053% |

All beat and downbeat timestamps survive the ideal-observation path exactly.
Every case now passes its locked tempo-error budgets.

## Diagnosed initial failure

The first run passed only 20/25 with mean tempo median/P95 error of
2.6407/16.1781 percent. The five failures were:

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

Timestamp-level diagnostics showed that all three existing estimator candidates
produced the same failures. The primary curve was clamping raw cadence to the
40--320 BPM range: every truth point in the slower Mozart Gli Armonici
performance is below 40 BPM, while the same floor distorted expressive slow
intervals in the other Mozart and Verdi recordings.

## Decision and resolution

The five failures were deterministic estimator failures because the observation
timestamps were already exact. The unified estimator now preserves any positive
finite cadence implied by accepted beat timestamps. Its 40--320 BPM bounds
remain only on publishable metrical alternatives, where they prevent unusable
hypotheses without rewriting the primary tempo curve. A 28 BPM or 360 BPM
observation therefore remains 28 or 360 BPM. No threshold was weakened, no
timestamp was changed or invented, and no new user-selectable strategy was
added.

The first unified correction passed 25/25, but timestamp-level diagnostics
still found 25 isolated points above 25 percent error. Every point had
near-zero output confidence because unconditional metrical smoothing had
rewritten a real expressive interval by approximately one octave.

Local metrical repair now also requires observation support. A slower interval
needs a real backend candidate at every implied missing pulse; a faster
interval may be regularized only for a backend that declares a fixed frame
rate. RUBATO's exact annotations therefore keep the original interval instead
of having strong rubato mistaken for a missed or duplicate beat. The suite
remains 25/25, mean tempo median/P95 error improves to 0.1544/4.4053 percent,
and none of 6,694 scored points remains above 25 percent error. No public
strategy or caller parameter was added.

A full Beat This run was not made part of this integration gate: on the
development VDI the first 128-second track remained in inference after more
than four minutes, projecting a multi-hour suite. Keep the full 1.8-hour
end-to-end baseline as a scheduled/offline evaluation with per-case progress.
