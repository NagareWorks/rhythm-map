# Coupled-clock learning v1: coherent, but not musically admitted

Date: 2026-09-10. Decision: **close this fixed experiment; do not ship or sweep**.

The [pre-fit protocol](../../experiments/coupled_clock/PROTOCOL-v1.md),
[input identities](../../experiments/coupled_clock/inputs-v1.json),
[execution identity](../../experiments/coupled_clock/execution-v1.json) and
[complete machine report](../../experiments/coupled_clock/results-v1.json) preserve
both actual fits, all forty cases and the failed gate. This is not another
unexecuted training proposal. No old report or score was rewritten.

## What ran

The 23,970-parameter CNN consumes reused, hash-pinned Beat This final0 hidden
features. One predicted anchor and positive cell rates define a single cumulative
clock over each complete crop; phase and tempo are derived from that same clock.
The encoder stays frozen and was not rerun. Neither v1 weights nor raw beat times
are inputs to the new model. The old selected head is replayed for comparison only.

Same exposed population as v1: 20 RUBATO recordings / 9 works fit, 5 recordings /
3 works develop, 15 ARTBeaT examples diagnose only. No new audio, annotation,
holdout access, performer-independence certification or encoder-overlap clearance.
The fitting data provides expressive native beat references, not audited
weak-attack/omission, abrupt-change or no-rhythm region labels. ARTBeaT retains
its existing step/ramp diagnostics and never chooses a checkpoint.

Both fits used seed 142, identical initialization and ordering, AdamW 0.001 /
weight decay 0.0001, gradient clipping at 1, complete-crop loss with four-recording
gradient accumulation, and work-balanced weighting. Both completed 20 epochs /
100 optimizer updates without budget exhaustion or numerical failure. The main
checkpoint was epoch 9; the fitted zero-feature control selected epoch 20, each
by its own lowest work-macro development loss, never ARTBeaT.

| Fit | Optimizer + development time | Peak allocated CUDA memory | Selected epoch |
| --- | ---: | ---: | ---: |
| audio | 11.876 s | 160.79 MiB | 9 |
| zero-audio | 4.781 s | 166.85 MiB | 20 |

Times are for the small readout fits on one T4, not end-to-end audio analysis.
They exclude local packet preparation, transfer, report verification and CI.
Allocated CUDA memory excludes driver context, reserved memory and host arrays.
The zero run's memory counter also includes the retained selected audio model.
There is no evidence of a training performance bottleneck here. Loss, crop
length and update count differ from v1, so this is **not an architecture-only
controlled ablation**.

## Outcome on the predeclared measurements

All values below use native reference beat level, not half/double-time minimum.
Cell-average BPM is measured between 50 Hz frame endpoints. Raw uses exact
interpolation of pinned detector timestamps, **not the shipping smoothed
estimator**. V1 uses the declared trapezoidal cell-rate projection while keeping
its independently predicted phase. Original instantaneous-frame v1 scores
are unchanged and should not be substituted into this table.

The table uses the identical reference/raw-supported intersection for every
method. Values are per-recording summaries, averaged within work and then across
works. Count drift is mean absolute native-beat advance error over fully supported
4-second intervals. Lower is better for every column.

| Cohort | Method | Median BPM error % | P95 BPM error % | Phase error, cycles | 4 s count drift, beats |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | Raw events | 41.5878 | 100.7129 | 0.1427 | 1.0488 |
| Development | Closed v1 | 37.2602 | 67.2536 | 0.1393 | 1.0133 |
| Development | Coupled audio | 33.9147 | 67.6306 | 0.2545 | 1.0818 |
| ARTBeaT | Raw events | 14.4462 | 49.4982 | 0.1078 | 1.7287 |
| ARTBeaT | Closed v1 | 31.0812 | 51.0475 | 0.1432 | 2.5825 |
| ARTBeaT | Coupled audio | 38.0601 | 61.4559 | 0.2475 | 3.0393 |

Against the fitted zero-feature control on **reference-only** development support,
median tempo error falls from 39.9987% to 33.5884%
(16.03% improvement), passing that half of the signal criterion.
Phase error instead rises from 0.2417 to 0.2534 cycles
(4.85% worse), failing the required 10% improvement.

Development regressions versus raw are 3/5 for median tempo and 3/5 for phase;
versus v1, 3/5 and 5/5. ARTBeaT regressions versus raw are 13/15 and 15/15;
versus v1, 14/15 and 15/15. A favorable development tempo aggregate therefore
cannot admit this head or conceal broad transition/phase regressions.

Both complete-fit and complete-coverage checks pass. Audio-signal, common-support
nonregression and per-recording breadth gates fail. All forty cases retain their
full required point/cell/interval denominators; no missing output was scored as
zero or removed to improve an average.

## What this establishes, and what it does not

The retained 80 predicted clocks (audio plus zero for 40 cases) are finite and
strictly increasing. Integrating a positive rate removes backward-clock
inconsistency, but **coherence is not alignment**:

Clock error = origin error + accumulated rate error.

For intuition, a persistent 1 BPM rate error accumulates one beat of count error
in 60 seconds. Monotonicity alone places no useful bound on that integral.
The measured phase errors near 0.25 cycles show poor sustained alignment here;
0.25 is also the expected wrapped absolute error for a uniform relative phase,
not evidence that these actual predictions are statistically uniform.

This result rejects the narrow claim that imposing this coherent readout and
native-count loss, with this fixed optimization and available supervision, is
sufficient. It does not isolate whether the main limit is initialization/
optimization, sustained phase alignment, the exposed fit population, or the
frozen representation. It does not prove that CNNs cannot work, that a larger
Transformer is required, or that more epochs would help. No such retry ran.

The next proposal must address sustained phase alignment **and** representative
continuation/change supervision, with a falsifiable comparison that separates
origin/accumulation errors from missing discriminative evidence. Do not promote
another aggregate, reset clocks at evaluation boundaries, or open the holdout
as a tuning shortcut. Rhythm availability, perceived beat-level ambiguity and
independent product admission remain unresolved. No inferred crossing is emitted
as an observed beat; no default, public strategy, model pack or release changed.

## Reproduction and evidence integrity

Preparation validates the old inputs/report, every hidden feature, source truth,
raw trace, and selected v1 weight by SHA256; it creates exclusive private packets.
The manifest pins protocol and all experiment Python sources before optimization.
The runner validates those hashes, logs each epoch, retains both private fitted
weights and each prediction digest, and never overwrites an existing output.
Run from the repository root, with the already-authorized private caches:

```sh
python -m experiments.coupled_clock.prepare --old-inputs PRIVATE_V1_INPUTS \
  --old-results PRIVATE_V1_RESULTS --trace-root PRIVATE_TRACES \
  --trace-root PRIVATE_REUSED_TRACES --output NEW_PRIVATE_INPUTS
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.coupled_clock.run \
  --inputs NEW_PRIVATE_INPUTS --expected-inputs-sha256 PINNED_MANIFEST_SHA \
  --output NEW_PRIVATE_RESULTS --device cuda
```

These commands describe the completed experiment, not authorization for another
fit. New preparation may differ numerically across supported CPU builds; its
manifest must be frozen again rather than pretending to reproduce identical bytes.
The actual run used Python 3.12, PyTorch 2.10.0+cu129 and NumPy 2.2.6. Do not use
this research Python pipeline as the Rust product's runtime dependency.

Private artifact verification recomputed all reported metrics and denominators;
maximum Windows/Linux numerical difference was 4.27e-14. Replaying both selected
checkpoints on CPU differed from saved CUDA cumulative clocks by at most
2.18e-6 cycles (verification atol 1e-5, rtol 2e-5). Replayed v1 fields differed
from its old saved GPU predictions by at most 5.97e-7. These are numerical
reproduction checks, not accuracy improvements or independent evaluation.

Input manifest SHA256: b4f05c10ebede84c394b0b8e06f6723b096d1f9a0a48e63024eb81d913fffd9c.
Execution SHA256: 63a82d56c464827c38296d810f47587daef6f88910b1802c59eaada39c3bdb0f.
Report SHA256: e33a44db4ce20a35f0676aa3f4fc47806da83f4a2ab80c1cfb555ce5343cda9a.

## Every recording on common support

Asterisks mark strict regression of the coupled output versus raw, independently
for tempo and phase. This table includes the fit recordings for transparency;
fit scores are not validation. P95, count drift, coverage, all control values and
regressions versus v1 are retained per case in the machine-readable report.

| Role / recording | Raw BPM error % | V1 BPM error % | Coupled BPM error % | Raw phase | V1 phase | Coupled phase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fit / rubato-bach-bwv1007-01-ar-macleod2011 | 6.1674 | 9.6751 | 28.8584 * | 0.1659 | 0.1674 | 0.2477 * |
| fit / rubato-bach-bwv1007-01-ov-milman2022 | 6.8798 | 22.9111 | 34.9296 * | 0.0882 | 0.0805 | 0.2464 * |
| fit / rubato-beethoven-op072-0-ov-henning2017 | 45.6705 | 19.9302 | 22.7707 | 0.2068 | 0.1582 | 0.2491 * |
| fit / rubato-berlioz-h048-04-ov-dennis2016 | 10.1062 | 12.7040 | 27.4538 * | 0.1314 | 0.0906 | 0.2565 * |
| development / rubato-boccherini-g275-03-ov-krux2022 | 11.6401 | 15.6880 | 26.8580 * | 0.0920 | 0.1060 | 0.2571 * |
| fit / rubato-brahms-op115-01-ov-schmitta2025 | 179.8365 | 10.0177 | 12.8637 | 0.2198 | 0.1416 | 0.3122 * |
| fit / rubato-brahms-op115-01-ov-steffens2008 | 21.6765 | 8.5564 | 16.2186 | 0.1330 | 0.1247 | 0.2043 * |
| fit / rubato-handel-hwv040-1-01-ar-buttnerfmaj2025 | 1.9217 | 15.5202 | 7.1096 * | 0.0292 | 0.0411 | 0.3674 * |
| fit / rubato-handel-hwv040-1-01-ar-buttnergmaj2025 | 7.4542 | 18.4908 | 11.2936 * | 0.0782 | 0.0871 | 0.2333 * |
| fit / rubato-handel-hwv040-1-01-ar-r-bra2020 | 7.4542 | 15.6328 | 11.0307 * | 0.0699 | 0.0900 | 0.2741 * |
| development / rubato-mozart-kv618-ar-papalin2012 | 113.5476 | 64.0155 | 48.5729 | 0.2501 | 0.2442 | 0.2498 |
| development / rubato-mozart-kv618-ov-gliarmonici2015 | 101.9970 | 103.5156 | 74.1620 | 0.2569 | 0.2460 | 0.2479 |
| development / rubato-mussorgsky-picturesatanexhibition-10-ov-bertoglio2012 | 8.2593 | 11.0885 | 13.2631 * | 0.1079 | 0.0854 | 0.3231 * |
| development / rubato-mussorgsky-picturesatanexhibition-10-ov-r-staab2016 | 2.4427 | 13.5655 | 13.7740 * | 0.0573 | 0.0481 | 0.1919 * |
| fit / rubato-schubert-d733-01-ar-staab2012 | 5.1075 | 16.8849 | 37.6221 * | 0.1744 | 0.0551 | 0.2503 * |
| fit / rubato-schubert-d733-01-ov-buttnerweiss2025 | 4.9030 | 20.0770 | 38.6342 * | 0.0806 | 0.0504 | 0.2491 * |
| fit / rubato-schumann-op039-05-ov-buttner2025 | 4.5771 | 9.1215 | 12.8453 * | 0.0535 | 0.0752 | 0.2901 * |
| fit / rubato-schumann-op039-05-ov-pizarro2012 | 33.2670 | 18.8052 | 17.6713 | 0.1764 | 0.1372 | 0.2996 * |
| fit / rubato-verdi-nabucco-vapensiero-ar-r-operaphilia2022 | 64.1920 | 27.1161 | 31.6073 | 0.2041 | 0.2013 | 0.2636 * |
| fit / rubato-verdi-nabucco-vapensiero-ov-garcia2015 | 33.6392 | 12.1623 | 18.5427 | 0.1890 | 0.2001 | 0.2170 * |
| fit / rubato-verdi-nabucco-vapensiero-ov-librosvivos2007 | 60.1282 | 14.9707 | 23.4481 | 0.2023 | 0.1493 | 0.2710 * |
| fit / rubato-verdi-nabucco-vapensiero-ov-secci2020 | 57.3121 | 27.2270 | 27.2680 | 0.2391 | 0.2082 | 0.2575 * |
| fit / rubato-vivaldi-rv269-01-ar-bra2021 | 4.9300 | 25.5449 | 40.2536 * | 0.0523 | 0.0880 | 0.2520 * |
| fit / rubato-vivaldi-rv269-01-ar-modenachamber2022 | 41.1637 | 24.0570 | 39.9598 | 0.1607 | 0.1246 | 0.2519 * |
| fit / rubato-vivaldi-rv269-01-ar-r-intartaglia2011 | 44.1200 | 28.5524 | 49.8958 * | 0.2318 | 0.2465 | 0.2549 * |
| diagnostic / artbeat-05-75-to-150 | 47.9166 | 43.9846 | 44.8383 | 0.1294 | 0.1586 | 0.2856 * |
| diagnostic / artbeat-06-150-to-75 | 1.2500 | 50.1967 | 47.3567 * | 0.1275 | 0.1815 | 0.2299 * |
| diagnostic / artbeat-07-75-to-112-5 | 2.1370 | 23.6394 | 27.9694 * | 0.0559 | 0.1194 | 0.2348 * |
| diagnostic / artbeat-08-112-5-to-75 | 1.9345 | 18.9759 | 29.0322 * | 0.0836 | 0.1199 | 0.2250 * |
| diagnostic / artbeat-09-90-to-80 | 1.8518 | 11.3279 | 19.8744 * | 0.0289 | 0.0923 | 0.2513 * |
| diagnostic / artbeat-10-90-to-120 | 1.5522 | 25.4172 | 33.7837 * | 0.0199 | 0.1000 | 0.2529 * |
| diagnostic / artbeat-11-60-to-80 | 0.6332 | 14.0072 | 18.0922 * | 0.0476 | 0.0890 | 0.2414 * |
| diagnostic / artbeat-12-80-to-150 | 2.3438 | 22.0904 | 43.3423 * | 0.1361 | 0.1594 | 0.2638 * |
| diagnostic / artbeat-13-180-to-120 | 2.6042 | 41.4112 | 43.8217 * | 0.0987 | 0.1795 | 0.2462 * |
| diagnostic / artbeat-14-240-to-96 | 2.4621 | 29.9722 | 38.0384 * | 0.0829 | 0.1201 | 0.2509 * |
| diagnostic / artbeat-15-85-to-127-5 | 1.9609 | 28.0061 | 39.6041 * | 0.1264 | 0.1595 | 0.2561 * |
| diagnostic / artbeat-18-piano-rubato | 48.5704 | 41.9980 | 45.1958 | 0.1853 | 0.1663 | 0.2449 * |
| diagnostic / artbeat-19-ramp-80-to-200 | 49.5895 | 49.7441 | 58.5734 * | 0.1972 | 0.1713 | 0.2438 * |
| diagnostic / artbeat-20-ramp-200-to-80 | 49.8456 | 44.8972 | 58.0577 * | 0.1830 | 0.1884 | 0.2524 * |
| diagnostic / artbeat-21-polyrhythm-70-to-105 | 2.0409 | 20.5497 | 23.3212 * | 0.1147 | 0.1425 | 0.2332 * |
