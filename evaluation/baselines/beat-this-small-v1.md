# Beat This small model: fast/accurate measurement baseline v1

Date: 2026-08-28

This baseline measures the `small1` Beat This checkpoint as a candidate "fast"
model pack against the shipping full model, addressing the Phase 2 roadmap item
"A measured fast/accurate model-pack policy; do not make the full model the
default merely because it is more accurate in isolation." Both model packs ran
through the identical shipping estimator and upstream decoder; no decoder,
estimator, or threshold was changed between runs.

The full-model runs reproduce the previously recorded baselines exactly: 0.8052
mean beat F1 and 6/15 end-to-end passes on the ARTBeaT calibration suite, and
6/15 tempo-gate passes on FSLD.

## Reproducible input

- small model pack: `models/beat-this-small-v1.json`
  (manifest SHA-256 `7634a20153a2093434330e2664241ea06c02fe06075efcb40e7cf80b7318dca7`)
- full model pack: `models/beat-this-full-v1.json`
  (manifest SHA-256 `ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`)
- suites: `generated-v1` (regression), `artbeat-v1` (calibration),
  `fsld-tempo-v1` (calibration)
- backend: Beat This through RTen, immutable upstream decoder
- estimator: shipping default
- audio: synthetic render plus fetched ARTBeaT and FSLD slices, all verified by
  SHA-256 before decoding
- machine: development PC, optimized non-LTO `evaluation` profile; runtime is
  machine-specific and is recorded only as a diagnostic, not an acceptance
  threshold

```bash
cargo xtask eval-backend \
  --suite evaluation/suites/<suite>.json \
  --model-pack models/<pack>.json \
  --model-dir <verified-model-dir> \
  --audio-dir <verified-audio-dir> \
  --report <report>.json \
  --no-fail
```

Model size is the primary deployment difference: the full beat model is
83,162,650 bytes and the small model is 10,555,592 bytes, a 7.9x reduction
that matters for embedded distributions and browser downloads even before
runtime is considered. The shared mel frontend is unchanged.

## generated-v1 regression suite

| Case | Full beat F1 | Small beat F1 | Full tempo P95 | Small tempo P95 | Full ms | Small ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant-120 | 1.0000 | 0.9846 | 1.30% | 2.82% | 876 | 633 |
| step-120-160 | 1.0000 | 0.6190 | 1.32% | 41.41% | 1156 | 831 |
| ramp-96-144 | 0.6406 | 0.6555 | 108.33% | 78.61% | 8810 | 4751 |
| gap-128 | 1.0000 | 1.0000 | 2.34% | 2.34% | 2318 | 1519 |
| subdivision-90 | 1.0000 | 0.9836 | 1.01% | 2.05% | 2895 | 1874 |

The full model passes 4/5 gates; the small model passes only `gap-128`. The
small model's `step-120-160` regression is the most serious: the deterministic
tempo jump is lost entirely (41.41% P95 tempo error). `constant-120` and
`subdivision-90` miss the 0.99 beat F1 gate marginally. This is a required CI
regression suite, so the small model cannot ship as the default while these
gates fail.

## ARTBeaT calibration suite

| Metric | Full | Small |
| --- | ---: | ---: |
| Mean beat precision | 0.8910 | 0.8823 |
| Mean beat recall | 0.7526 | 0.7622 |
| Mean beat F1 | 0.8052 | 0.8043 |
| Mean tempo median error | 18.51% | 20.64% |
| Mean tempo P95 error | 66.48% | 88.01% |
| End-to-end gate passes | 1/15 | 0/15 |
| Total analysis time | 16.11 s | 10.89 s |

| Case | Full beat F1 | Small beat F1 | Full tempo P95 | Small tempo P95 | Full ms | Small ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 05-75-to-150 | 0.7907 | 0.7727 | 50.41% | 50.40% | 784 | 597 |
| 06-150-to-75 | 0.7727 | 0.7727 | 50.00% | 50.41% | 775 | 526 |
| 07-75-to-112-5 | 0.8696 | 0.8889 | 185.71% | 100.00% | 779 | 519 |
| 08-112-5-to-75 | 0.8696 | 0.8462 | 90.48% | 106.96% | 852 | 578 |
| 09-90-to-80 | 0.9600 | 0.9796 | 3.78% | 3.85% | 1234 | 774 |
| 10-90-to-120 | 1.0000 | 0.9831 | 1.42% | 3.85% | 1099 | 715 |
| 11-60-to-80 | 0.9268 | 0.9524 | 39.52% | 63.04% | 1200 | 749 |
| 12-80-to-150 | 0.8000 | 0.8214 | 50.41% | 50.00% | 1159 | 739 |
| 13-180-to-120 | 0.7500 | 0.7667 | 177.78% | 49.49% | 793 | 540 |
| 14-240-to-96 | 0.7778 | 0.7778 | 78.57% | 108.33% | 1053 | 755 |
| 15-85-to-127-5 | 0.6486 | 0.6667 | 34.02% | 34.02% | 1663 | 1169 |
| 18-piano-rubato | 0.7568 | 0.8810 | 57.62% | 63.40% | 1804 | 1250 |
| 19-ramp-80-to-200 | 0.6667 | 0.6667 | 75.45% | 59.92% | 1115 | 770 |
| 20-ramp-200-to-80 | 0.7077 | 0.5085 | 50.75% | 525.00% | 1133 | 732 |
| 21-polyrhythm-70-to-105 | 0.7805 | 0.7805 | 51.28% | 51.41% | 667 | 475 |

Per case, the small model improves 7, regresses 4, and matches the full model
on 4. The largest improvement is `18-piano-rubato` (0.7568 to 0.8810); the
largest regression is `20-ramp-200-to-80` (0.7077 to 0.5085, with tempo P95
error exploding from 50.75% to 525.00%). The per-case churn on a
15-case calibration slice is not evidence for a runtime model selector; it is
evidence that the two models make different metrical mistakes on ambiguous
material, consistent with the roadmap's rejection of strategy selectors.

## FSLD tempo-only calibration suite

| Metric | Full | Small |
| --- | ---: | ---: |
| Mean tempo median error | 29.43% | 33.88% |
| Mean tempo P95 error | 52.11% | 118.08% |
| Tempo gate passes | 6/15 | 5/15 |
| Total analysis time | 15.99 s | 10.80 s |

| Case | Full median | Full P95 | Small median | Small P95 | Full ms | Small ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 476866-41-bpm | 99.57% | 101.40% | 99.57% | 106.14% | 648 | 492 |
| 404840-60-bpm | 96.12% | 102.74% | 100.00% | 102.85% | 5381 | 3409 |
| 210835-70-bpm | 0.33% | 0.47% | 100.94% | 100.94% | 870 | 548 |
| 131423-80-bpm | 0.45% | 1.35% | 0.43% | 0.45% | 687 | 459 |
| 219021-90-bpm | 0.98% | 100.08% | 0.01% | 1.01% | 1545 | 1066 |
| 360687-100-bpm | 0.04% | 2.36% | 0.00% | 1.17% | 469 | 332 |
| 19069-110-bpm | 1.01% | 33.48% | 1.06% | 109.79% | 409 | 309 |
| 418991-120-bpm | 1.25% | 1.94% | 0.00% | 0.00% | 176 | 125 |
| 124542-128-bpm | 95.31% | 113.07% | 2.34% | 485.94% | 436 | 244 |
| 486302-130-bpm | 50.19% | 50.89% | 49.83% | 49.83% | 347 | 280 |
| 271070-140-bpm | 44.08% | 167.86% | 38.78% | 435.71% | 331 | 263 |
| 330889-150-bpm | 1.61% | 49.57% | 50.00% | 50.00% | 2073 | 1348 |
| 322315-160-bpm | 0.48% | 2.31% | 49.77% | 50.22% | 1807 | 1343 |
| 348652-180-bpm | 0.04% | 4.17% | 0.04% | 2.08% | 591 | 413 |
| 439993-200-bpm | 50.00% | 50.00% | 15.38% | 275.00% | 221 | 171 |

The small model flips whole-track metrical level on clips the full model tracks
correctly (`210835-70-bpm` from 0.33% to 100.94% median error,
`322315-160-bpm` from 0.48% to 49.77%) while fixing others (`219021-90-bpm`
P95 from 100.08% to 1.01%, `418991-120-bpm` to an exact 0.00%/0.00%). Again
the direction of each whole-track metrical decision differs by case rather
than consistently improving or degrading.

## Conclusion

On this machine the small model is about 1.48--1.67x faster than the full
model (32--40% less analysis time) and its beat model is 7.9x smaller. It does
not, however, qualify as the shipping default or as a currently recommendable
"fast" option:

1. It fails 4 of 5 required `generated-v1` regression gates, including a
   complete loss of the deterministic `step-120-160` tempo jump.
2. It loses gate passes on both real-music calibration slices (ARTBeaT 1/15 to
   0/15, FSLD 6/15 to 5/15) with materially worse mean tempo P95 error
   (66.48% to 88.01% on ARTBeaT, 52.11% to 118.08% on FSLD).
3. Its near-identical mean beat F1 on ARTBeaT (0.8052 to 0.8043) hides large
   opposite-direction per-case changes, which is metrical-ambiguity churn, not
   robustness.

The measured decision for the fast/accurate policy is therefore: keep the full
model as the only shipping default. The small pack remains a verified,
manifest-pinned option for size-constrained evaluation and future embedded or
browser distributions, where its 10.6 MB footprint and ~1.5x speedup may
justify the measured accuracy cost, provided consumers receive the same
explicit ambiguity and confidence metadata. Do not add a runtime fast/accurate
selector on this evidence: the per-case failure directions differ, so a
selector would repeat the rejected strategy-selector pattern.
