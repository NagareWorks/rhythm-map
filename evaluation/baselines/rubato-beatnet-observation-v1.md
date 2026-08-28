# RUBATO BeatNet observation baseline v1

Date: 2026-08-28

This baseline compares the pinned experimental BeatNet backend with Beat This
on the same 25-case RUBATO calibration suite. It also replays the two frozen
cross-backend consensus diagnostics. No holdout model inference was performed.

## Reproducible input

- suite: `evaluation/suites/rubato-calibration-v1.json` (`calibration`)
- model pack: `models/beatnet-v1.json`
- model manifest SHA-256:
  `dcc6aeb313fda31ab862d287976cc7d7bc996e1ee78fc77b028e8b0a9d69b1e5`
- backend: BeatNet BDA through RTen, guarded candidate graph
- estimator compatibility policy: `local-metrical-path-v1`, which now aliases
  the shipping estimator behavior
- report schema: 11
- BeatNet report:
  `D:/rhythm-map-eval/reports/rubato-beatnet-calibration-v11.json`
- local consensus report:
  `D:/rhythm-map-eval/reports/rubato-local-metrical-consensus-v1.json`
- global consensus report:
  `D:/rhythm-map-eval/reports/rubato-global-consensus-v3.json`

```bash
cargo xtask eval-beatnet \
  --suite evaluation/suites/rubato-calibration-v1.json \
  --model-pack models/beatnet-v1.json \
  --model-dir D:/rhythm-map-models/beatnet-v1 \
  --audio-dir D:/rhythm-map-eval/rubato-calibration-v1 \
  --report D:/rhythm-map-eval/reports/rubato-beatnet-calibration-v11.json \
  --no-fail

cargo xtask local-metrical-diagnose \
  --suite evaluation/suites/rubato-calibration-v1.json \
  --primary D:/rhythm-map-eval/reports/rubato-beat-this-calibration-v11.json \
  --secondary D:/rhythm-map-eval/reports/rubato-beatnet-calibration-v11.json \
  --tolerance-s 0.07 \
  --report D:/rhythm-map-eval/reports/rubato-local-metrical-consensus-v1.json

cargo xtask consensus-diagnose \
  --primary D:/rhythm-map-eval/reports/rubato-beat-this-calibration-v11.json \
  --secondary D:/rhythm-map-eval/reports/rubato-beatnet-calibration-v11.json \
  --tolerance-s 0.07 \
  --report D:/rhythm-map-eval/reports/rubato-global-consensus-v3.json
```

## Replacement-backend result

BeatNet is not a viable replacement for Beat This on this slice. It improves
three recordings and regresses 22, emits 17,051 selected events for 6,726 truth
beats, and passes no case.

| Backend | Passing cases | Mean beat precision | Mean beat recall | Mean beat F1 | Mean downbeat F1 | Mean tempo median error | Mean tempo P95 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Beat This | 1/25 | 0.4870 | 0.5920 | 0.5213 | 0.4434 | 65.0667% | 212.9036% |
| BeatNet | 0/25 | 0.2982 | 0.6207 | 0.3865 | 0.2478 | 198.8495% | 446.7589% |

BeatNet's existing truth-free top-1 rank reaches 0.4024 mean beat F1, and a
truth-assisted oracle over its fixed top-K hypotheses reaches 0.4232. Both
remain materially below the unchanged Beat This path.

## Complementary evidence

BeatNet does expose useful evidence before decoding. Its 70,870 real local
maxima cover 95.30% of all truth beats, compared with 82.11% for Beat This's
35,151 candidates.

Using Beat This's own selected sequence as the reference, it misses 2,510
annotated beats under the suite's 70 ms tolerance:

| Candidate support for Beat This misses | Beats | Fraction of misses |
| --- | ---: | ---: |
| Beat This candidates | 1,307 | 52.07% |
| BeatNet candidates | 2,259 | 90.00% |
| Either backend | 2,347 | 93.51% |
| BeatNet-only increment | 1,040 | 41.43% |
| Neither backend | 163 | 6.49% |

This is genuine complementary observation coverage, but not a deployable
selection rule. BeatNet's candidate set is about twice as large and its decoded
sequence is much too dense.

## Consensus results

The frozen local rule
`anchored-pareto-decoded-event-dense-pulse-v1` selects 51 bounded regions. It
improves two cases, regresses one, and lowers mean Beat This beat F1 from 0.5213
to 0.5203. The regression is Vivaldi RV 269 / Intartaglia 2011, where both
decoded-event and dense-pulse evidence endorse the wrong canonical beat level.

The existing global cross-backend meter gate is also rejected. BeatNet dense
downbeat periodicity changes three cases, improves none, and lowers mean F1 from
0.5221 to 0.5124. Adding that signal to the local Pareto rule would not be an
independent safety argument.

## Decision

Keep BeatNet evaluation-only. Do not replace Beat This, add a user-visible
backend strategy, ship a mandatory second model, or open the RUBATO holdout.
The alternative backend proves that much of the missing timestamp evidence can
be observed, but its pulse and downbeat semantics cannot safely select the
musical beat level.

The next reusable candidate must provide genuinely different perceived-beat or
meter evidence under a commercially usable license. If no such pretrained model
exists, canonical automatic selection remains blocked without training; the
shipping behavior continues to expose supported alternatives, localized
ambiguity, provenance, and confidence without inventing timestamps.
