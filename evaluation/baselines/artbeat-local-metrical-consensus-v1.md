# ARTBeaT local metrical consensus v1

Date: 2026-08-27

This calibration tests whether independent BeatNet evidence can safely select
only bounded parts of Beat This's locally varying metrical path. It does not
change the shipping estimator, add a user-visible strategy, or reopen a
holdout.

## Reproducible input

- suite: `evaluation/suites/artbeat-v1.json` (`calibration`)
- primary: cached Beat This report with `local-metrical-path-v1`, schema 8,
  `D:/rhythm-map-eval/reports/artbeat-cache-local-metrical-v8.json`
- primary model manifest SHA-256:
  `ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`
- secondary: BeatNet report retaining complete 50 Hz pulse/downbeat
  activations, schema 10,
  `D:/rhythm-map-eval/reports/artbeat-beatnet-dense-meter-v5.json`
- secondary model manifest SHA-256:
  `dcc6aeb313fda31ab862d287976cc7d7bc996e1ee78fc77b028e8b0a9d69b1e5`
- timestamp tolerance: 70 ms
- policy: `anchored-pareto-decoded-event-dense-pulse-v1`
- result:
  `D:/rhythm-map-eval/reports/artbeat-local-metrical-consensus-v1.json`
- holdout: not opened

Reproduce after the two backend reports exist:

```powershell
cargo run -p rhythm-map-eval -- local-metrical-diagnose `
  --suite evaluation/suites/artbeat-v1.json `
  --primary D:/rhythm-map-eval/reports/artbeat-cache-local-metrical-v8.json `
  --secondary D:/rhythm-map-eval/reports/artbeat-beatnet-dense-meter-v5.json `
  --report D:/rhythm-map-eval/reports/artbeat-local-metrical-consensus-v1.json
```

## Frozen rule

Exact timestamps shared by Beat This's `selected` and
`locally_varying_metrical_path` sequences bound a disagreement region. Leading
and trailing regions lack two anchors and are diagnosis-only. For every
timestamp unique to either path inside a bounded region, the candidate asks
whether choosing the local path improves both:

1. binary event decisions against BeatNet's decoded top-ranked sequence; and
2. Bernoulli log likelihood under BeatNet's undecoded dense pulse activation.

Both mean margins must be strictly positive. Ties preserve `selected`. The rule
has no learned weight, fitted score threshold, forced BPM band, fixed-duration
window, or invented timestamp.

## Result

| Measure | Beat This `selected` | Local candidate | Delta |
| --- | ---: | ---: | ---: |
| Mean beat F1 | 0.80516 | 0.80308 | -0.00208 |
| Improved cases | - | 3 | - |
| Regressed cases | - | 1 | - |
| Selected bounded regions | - | 9 | - |
| Calibration gate | - | fail | - |

The candidate improves `75-to-112.5`, `85-to-127.5`, and piano rubato. It
regresses `240-to-96` by about 0.11 beat F1. In that clip, four bounded regions
from 0.52 s through 6.04 s pass both truth-free checks even though the
annotations retain the faster pulse. For example, the decoded-event decision
margins in those regions are positive (0.33 to 1.0), and the dense pulse log
likelihood margins are also positive (0.93 to 4.59), while their annotated
decision margins are non-positive.

## Decision

Reject the candidate as a selector and do not spend a holdout on it. Localizing
the decision and removing decoder-selection bias are both necessary, but they
are not sufficient when Beat This and BeatNet share the same wrong metrical
interpretation. A future selector needs evidence whose target is perceived
beat/meter level rather than another view of beat presence. Until such evidence
exists, Rhythm Map should retain supported alternatives and explicit ambiguity
instead of silently enforcing one half/double-time convention.
