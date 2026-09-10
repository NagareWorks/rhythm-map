# Native-trajectory loss v1: a controlled tradeoff, not an admitted model

Date: 2026-09-10. Decision: **close this fixed loss-only experiment; do not ship**.

The [pre-fit protocol](../../experiments/trajectory_clock/PROTOCOL-v1.md),
[plan](../../experiments/trajectory_clock/plan-v1.json),
[execution](../../experiments/trajectory_clock/execution-v1.json) and
[all-case report](../../experiments/trajectory_clock/results-v1.json) retain the
actual paired fit. Previous direct and coupled experiments remain unchanged.

## First separate origin error from cumulative drift

Using only the retained coupled-v1 predictions, [post-fit oracle attribution](../../experiments/trajectory_clock/attribution-v1.json)
finds the exact circular-L1 best single phase offset per recording. This is an
optimistic anchor-only bound, not an automatic correction, listener annotation,
new prediction or independent evaluation. It does not change any previous score.

On the SAME reference/raw-supported points, the coupled-v1 work-macro phase error
is 0.2545 cycles on development and 0.2475 on ARTBeaT. Even allowing a perfect
constant phase shift only reduces these to 0.2227 / 0.2353, still worse than raw
0.1427 / 0.1078. A single origin repair therefore cannot recover the raw baseline.

Within each contiguous reference span, ordinary least squares decomposes count
error into a constant offset, a constant-rate component and a nonlinear residual.
No count constraint crosses an annotation hole. Macro-average RMSE values below
are separately averaged over recordings/works; **do not square these averages
and treat them as an additive energy decomposition**. Orthogonality holds within
each individual span, where the script checks it.

| Coupled-v1 cohort | Centered count RMSE | Constant-rate component RMSE | Nonlinear residual RMSE |
| --- | ---: | ---: | ---: |
| fit | 4.5595 | 4.4499 | 0.5207 |
| development | 4.5021 | 4.4134 | 0.5097 |
| diagnostic | 3.5923 | 3.4259 | 0.9696 |

All RMSE values are in native beats. Persistent rate bias is substantial, but
removing a global bias is not sufficient: nonlinear residuals also remain.
The first supported annotation point is not necessarily physical frame zero;
the diagnostic does not pretend to know an unannotated starting beat.

## The single changed factor

Same 23,970-parameter coupled CNN, positive integrated clock, frozen Beat This
features, padding, crop lengths, initial state and numeric precision. Same 20
RUBATO recordings / 9 works fit, 5 / 3 develop, and 15 ARTBeaT diagnose. Same
20 epochs / 100 updates, seed 142, AdamW 0.001 / weight decay 0.0001, clip 1,
four-recording accumulation, work weighting and deterministic execution.

Replace the circular-phase plus 1/25/100/200-frame relative-count objective with
whole-span cumulative-count MSE. For error e = predicted count - native count,
subtract ONE integer k = round(mean(e)) per annotation span, then minimize
mean((e-k)^2). Integer beat numbering is arbitrary; per-frame cycle slips are not.
Labels and registration are used only in the loss, NEVER to align reported
predictions. Complete-crop inference still uses the model's own origin and rates.

Training/development loop source is checked against the previous loop with only
the loss call changed. The checkpoint selection rule stays lowest work-macro
development loss, but its objective necessarily changes too. Compare the same
musical metrics, not the different numerical training-loss units.

The fit data still lacks audited omission, weak-attack, abrupt-change and
no-rhythm region labels. All cohorts were previously exposed; the attribution
also informed this design. ARTBeaT never trains or selects an epoch and is not
independent acceptance. No new audio, feature capture, encoder fine-tune, holdout,
perceived-level certification or model distribution.

## Actual execution

Both fits completed without numerical failure or budget exhaustion, with the
same initial state as coupled v1. The selected epochs were 6 / 7.

| Fit | Fit + development time | Peak allocated CUDA memory | Epochs / updates |
| --- | ---: | ---: | ---: |
| audio | 3.645 s | 160.81 MiB | 20 / 100 |
| zero-audio | 3.020 s | 166.87 MiB | 20 / 100 |

These T4 times exclude artifact transfer, preparation, CPU verification and CI.
Memory is PyTorch allocated memory, not total driver/host usage; the zero run
also retains the selected audio model. No end-to-end inference benchmark is claimed.

## Musical result on identical common support

All columns are lower-is-better work-macro recording summaries. BPM uses exact
50 Hz cell endpoint counts; raw is event interpolation, not the shipping
estimator. Independent v1 retains its declared trapezoidal cell-rate projection
and original phase. Every comparison keeps identical point/cell/4-second interval
denominators; the JSON retains every case, control, P95 and regression flag.

| Cohort | Method | Median BPM error % | P95 BPM error % | Phase error, cycles | 4 s count drift, beats |
| --- | --- | ---: | ---: | ---: | ---: |
| development | Raw | 41.5878 | 100.7129 | 0.1427 | 1.0488 |
| development | Independent v1 | 37.2602 | 67.2536 | 0.1393 | 1.0133 |
| development | Coupled v1 | 33.9147 | 67.6306 | 0.2545 | 1.0818 |
| development | Trajectory loss | 45.3405 | 97.3183 | 0.2687 | 1.0621 |
| diagnostic | Raw | 14.4462 | 49.4982 | 0.1078 | 1.7287 |
| diagnostic | Independent v1 | 31.0812 | 51.0475 | 0.1432 | 2.5825 |
| diagnostic | Coupled v1 | 38.0601 | 61.4559 | 0.2475 | 3.0393 |
| diagnostic | Trajectory loss | 28.1967 | 68.1944 | 0.2475 | 2.0500 |

On reference-only development support, audio also fails the matched zero-feature
test: median BPM error is 44.8379% versus zero 42.7816%; phase error is 0.2679
versus 0.2406 cycles. Neither achieves the required 10% improvement.

| Cohort | Baseline | Median-tempo regressions | Phase regressions |
| --- | --- | ---: | ---: |
| development | Raw | 4/5 | 4/5 |
| development | Independent v1 | 5/5 | 5/5 |
| development | Coupled v1 | 4/5 | 3/5 |
| diagnostic | Raw | 11/15 | 15/15 |
| diagnostic | Independent v1 | 4/15 | 15/15 |
| diagnostic | Coupled v1 | 3/15 | 8/15 |

Complete-fit and finite-coverage gates pass; signal, raw/v1 nonregression,
recording breadth and previous-coupled nonregression all fail. The ARTBeaT
median-tempo/count improvements cannot erase development regressions, increased
tail tempo errors, or unreliable phase. Nothing is admitted into the product.

## Interpretation and next boundary

This experiment causally changes the objective under the fixed harness; it does
not establish that cumulative-count supervision, CNNs or frozen features cannot
work generally. Native-count MSE weights errors in absolute beat units and
emphasizes complete-span agreement; it does not guarantee good local relative
tempo or phase. The observed tradeoff is evidence against this replacement alone,
not permission to quietly blend checkpoints or expose two internal strategies.

The next proposal must jointly use local acoustic phase evidence and native
count/tempo consistency, with one coherent output and a falsifiable comparison.
Do not just alternate scalar losses, extend the seed/epoch budget, or assert a
Transformer/encoder fine-tune is required. Representative supervision, rhythm
availability and perceived beat-level ambiguity remain unresolved. No new fit
or holdout access follows automatically from this failed gate.

## Evidence checks and reproduction boundary

All 80 selected-model predictions were replayed on CPU from retained private
weights. Maximum CPU/CUDA cumulative-clock difference was 2.833e-6 cycles
(atol 1e-5, rtol 2e-5). Recomputed musical metrics differed by at most 4.264e-14
across the supported CPU builds. Every clock was finite and strictly increasing;
all source, packet, prediction and weight identities were verified. These are
numerical reproducibility checks, not additional accuracy evidence.

The existing private coupled-v1 packets are reused without recomputing features.
From the repository root, the registered workflow was:

```sh
python -m experiments.trajectory_clock.register --output NEW_PRIVATE_PLAN
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.trajectory_clock.run \
  --inputs PINNED_COUPLED_INPUTS --plan NEW_PRIVATE_PLAN \
  --expected-plan-sha256 PINNED_PLAN_SHA --output NEW_PRIVATE_RESULTS --device cuda
```

This describes the completed run, not authorization for a replayed optimizer.
The exclusive output/plan creation guards prevent accidental overwrite. The
actual run used Python 3.12, PyTorch 2.10.0+cu129 and NumPy 2.2.6. Artifacts remain
private; public JSON contains hashes and diagnostics only.

Plan SHA256: f822174d6140df7e3eef3ab38aaa9bbe99a38b4faaeeee2c9355b4aaf3fb5ee4.
Execution SHA256: 9a0b8614cdf11e97efc9a259bfb97048f7d95b772da516b7173a766500d8b59a.
Report SHA256: a20e960c8fa7d3ffffdfab24e9e2d73fb2a0ab78069cc8a33d7e1cfd42d5a8e1.

## Every recording versus the previous coupled head

Asterisks mark strict regression versus coupled v1. Fit recordings are shown
for transparency, not counted as validation. Raw/v1/control comparisons and
every other metric are available in the machine-readable report.

| Role / recording | Coupled BPM error % | New BPM error % | Coupled phase | New phase |
| --- | ---: | ---: | ---: | ---: |
| fit / rubato-bach-bwv1007-01-ar-macleod2011 | 28.8584 | 18.7929 | 0.2477 | 0.2541 * |
| fit / rubato-bach-bwv1007-01-ov-milman2022 | 34.9296 | 22.1460 | 0.2464 | 0.2671 * |
| fit / rubato-beethoven-op072-0-ov-henning2017 | 22.7707 | 25.5483 * | 0.2491 | 0.2093 |
| fit / rubato-berlioz-h048-04-ov-dennis2016 | 27.4538 | 16.9954 | 0.2565 | 0.2250 |
| development / rubato-boccherini-g275-03-ov-krux2022 | 26.8580 | 22.2862 | 0.2571 | 0.2506 |
| fit / rubato-brahms-op115-01-ov-schmitta2025 | 12.8637 | 20.9189 * | 0.3122 | 0.2485 |
| fit / rubato-brahms-op115-01-ov-steffens2008 | 16.2186 | 21.2555 * | 0.2043 | 0.2612 * |
| fit / rubato-handel-hwv040-1-01-ar-buttnerfmaj2025 | 7.1096 | 14.1453 * | 0.3674 | 0.2599 |
| fit / rubato-handel-hwv040-1-01-ar-buttnergmaj2025 | 11.2936 | 15.5588 * | 0.2333 | 0.2889 * |
| fit / rubato-handel-hwv040-1-01-ar-r-bra2020 | 11.0307 | 20.5148 * | 0.2741 | 0.2647 |
| development / rubato-mozart-kv618-ar-papalin2012 | 48.5729 | 81.5001 * | 0.2498 | 0.2552 * |
| development / rubato-mozart-kv618-ov-gliarmonici2015 | 74.1620 | 111.0900 * | 0.2479 | 0.2507 * |
| development / rubato-mussorgsky-picturesatanexhibition-10-ov-bertoglio2012 | 13.2631 | 18.7166 * | 0.3231 | 0.2858 |
| development / rubato-mussorgsky-picturesatanexhibition-10-ov-r-staab2016 | 13.7740 | 16.1637 * | 0.1919 | 0.3193 * |
| fit / rubato-schubert-d733-01-ar-staab2012 | 37.6221 | 34.7465 | 0.2503 | 0.2564 * |
| fit / rubato-schubert-d733-01-ov-buttnerweiss2025 | 38.6342 | 24.6248 | 0.2491 | 0.1930 |
| fit / rubato-schumann-op039-05-ov-buttner2025 | 12.8453 | 17.4849 * | 0.2901 | 0.2075 |
| fit / rubato-schumann-op039-05-ov-pizarro2012 | 17.6713 | 16.4219 | 0.2996 | 0.2663 |
| fit / rubato-verdi-nabucco-vapensiero-ar-r-operaphilia2022 | 31.6073 | 47.2945 * | 0.2636 | 0.2287 |
| fit / rubato-verdi-nabucco-vapensiero-ov-garcia2015 | 18.5427 | 22.9975 * | 0.2170 | 0.2450 * |
| fit / rubato-verdi-nabucco-vapensiero-ov-librosvivos2007 | 23.4481 | 28.0388 * | 0.2710 | 0.2292 |
| fit / rubato-verdi-nabucco-vapensiero-ov-secci2020 | 27.2680 | 34.9394 * | 0.2575 | 0.2427 |
| fit / rubato-vivaldi-rv269-01-ar-bra2021 | 40.2536 | 16.0022 | 0.2520 | 0.2265 |
| fit / rubato-vivaldi-rv269-01-ar-modenachamber2022 | 39.9598 | 21.0265 | 0.2519 | 0.2383 |
| fit / rubato-vivaldi-rv269-01-ar-r-intartaglia2011 | 49.8958 | 33.6227 | 0.2549 | 0.2475 |
| diagnostic / artbeat-05-75-to-150 | 44.8383 | 31.5074 | 0.2856 | 0.2585 |
| diagnostic / artbeat-06-150-to-75 | 47.3567 | 31.9987 | 0.2299 | 0.2308 * |
| diagnostic / artbeat-07-75-to-112-5 | 27.9694 | 22.0914 | 0.2348 | 0.2943 * |
| diagnostic / artbeat-08-112-5-to-75 | 29.0322 | 15.4858 | 0.2250 | 0.2560 * |
| diagnostic / artbeat-09-90-to-80 | 19.8744 | 26.4983 * | 0.2513 | 0.2235 |
| diagnostic / artbeat-10-90-to-120 | 33.7837 | 15.0372 | 0.2529 | 0.2203 |
| diagnostic / artbeat-11-60-to-80 | 18.0922 | 34.6364 * | 0.2414 | 0.2542 * |
| diagnostic / artbeat-12-80-to-150 | 43.3423 | 26.8000 | 0.2638 | 0.2129 |
| diagnostic / artbeat-13-180-to-120 | 43.8217 | 27.4114 | 0.2462 | 0.2377 |
| diagnostic / artbeat-14-240-to-96 | 38.0384 | 20.7536 | 0.2509 | 0.2160 |
| diagnostic / artbeat-15-85-to-127-5 | 39.6041 | 22.4479 | 0.2561 | 0.2401 |
| diagnostic / artbeat-18-piano-rubato | 45.1958 | 32.9931 | 0.2449 | 0.2610 * |
| diagnostic / artbeat-19-ramp-80-to-200 | 58.5734 | 48.5039 | 0.2438 | 0.2928 * |
| diagnostic / artbeat-20-ramp-200-to-80 | 58.0577 | 36.5938 | 0.2524 | 0.2639 * |
| diagnostic / artbeat-21-polyrhythm-70-to-105 | 23.3212 | 30.1916 * | 0.2332 | 0.2500 * |
