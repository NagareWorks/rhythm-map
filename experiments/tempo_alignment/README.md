# Real-music alignment: 11/15 pairs pass, full-population admission fails

The [fixed protocol](PROTOCOL-v1.md) was executed on all 15 retained Rubber Band
music pairs. No re-rendering, feature-axis stretching, neural model, optimizer,
holdout, new dependency or shipping change. The [complete report](results-v1.json)
retains 2,215 point measurements with both spectral resolutions, actual delayed-
PCM controls, wrong-map controls, abstentions, regions and unchanged denominators.
The old renderer report and its source closure remain byte-pinned.

## Result and limits

Both interior grid and interior beat-point support must reach 80% in every pair.
Support requires an aligned natural comparison **and** recovery of a separately
injected +120 ms delay from actual PCM. The full gate fails: 11 pairs pass, four
do not. No thresholds were changed after seeing music results.

| Recording / profile | Interior grid supported | Interior beat points supported |
| --- | ---: | ---: |
| Vivaldi, all five profiles | 100% | 100% |
| Bach, identity | 100% | 100% |
| Bach, faster | 100% | 98.33% |
| Bach, slower | 92.54% | 95.45% |
| Bach, local slow-fast | 98.11% | 98.41% |
| Bach, local fast-slow | 92.45% | 95.31% |
| Berlioz, identity | 100% | 100% |
| Berlioz, slower | 49.25% | 49.25% |
| Berlioz, faster | 40.00% | 40.63% |
| Berlioz, local slow-fast | 69.81% | 70.77% |
| Berlioz, local fast-slow | 67.92% | 69.70% |

Across fixed grid points, 694/795 interior, 54/60 seam and 25/60 edge points are
supported (773/915 overall). Points without a complete unpadded context remain
in the denominator. These are correlated time points from only three already
exposed recordings, not independent samples or a generalization accuracy score.
All 12 changed-profile wrong-map controls pass: 0/504 physically supported
changed-context points falsely classify untransformed source audio as aligned
with the requested speed map. Missing physical support is reported, not counted
as an extra successful negative example.

The four rejected profiles belong to the same orchestral recording. Its 91
rejected interior grid points all fail the 512-sample matching score; at 2048
samples, 74/91 qualify. Median best correlation is 0.70828 / 0.83296 respectively,
and median best lag is zero at both resolutions. These **unqualified** best lags
are not accepted alignment estimates. The evidence points to insufficient
robustness/observability of this matching representation on that material; it
does not establish a systematic renderer timing drift, corrupted beat labels,
or failed tempo learning. Do not reinterpret mismatch as a missing/weak beat.

All three processed-identity profiles pass at 100% interior support. Qualified
interior lag estimates on the transformed pairs have maxima of 5--10 ms (the
10 ms search grid and two-view averaging limit precision). Reporting only these
small lags would hide the abstentions and falsely suggest the full corpus passed.

## What this enables, and what it does not

There are now explicitly located, control-checked musical correspondence points,
not just a synthetic click test and requested transform factors. The checker can
reject wrong maps and recover deliberate shifts. It also exposes where that
evidence is insufficient. Preserve this v1 outcome rather than dropping the
orchestral recording, weakening correlation thresholds, or taking its long-window
result alone to rescue the gate.

No whole track or nominal label file has been admitted to training. Supported
points must not be interpolated into a dense mask, used to certify all context
samples, or advertised as proof of perceptually correct beat level. A follow-on
learning contract can investigate relative-speed supervision only at explicitly
verified units, retain every original source and report excluded/uncertain points,
and retain unchanged natural/no-change controls. That is a new scoped experiment,
not a retrospective pass for this full-population gate. It must not claim broader
music accuracy from this three-source pilot. No automatic fit is started here.

## Reproduce

```sh
python -m unittest discover -s experiments/tempo_alignment -p 'test_*.py' -v
timeout --signal=TERM --kill-after=10s 900s python -m experiments.tempo_alignment.run \
  --pairs RETAINED_PRIVATE_TEMPO_PAIRS --output FRESH_PRIVATE_OUTPUT
```

The report pins code/protocol and every opened source/preview/label hash from
the previous renderer report. Audio never enters the repository; attribution
and license provenance remain in the pinned source manifest/private labels.
Public tests recompute scalar decisions. Private validation separately reruns
the actual PCM measurement; a completed program alone is not a passing data gate.
