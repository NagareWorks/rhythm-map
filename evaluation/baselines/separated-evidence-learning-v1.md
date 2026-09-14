# Four-arm tempo/phase evidence learning: closed outcome

Date: 2026-09-14. Runner source: `227977b25731755e6f9cf1a3857a76188ca5ed27`.
The [fixed protocol](../../experiments/separated_evidence_fit/PROTOCOL-v1.md)
was registered before this invocation. No architecture, budget, masks, population,
checkpoint selection or numerical gate changed during execution.

## Outcome

All four fits completed, but **the musical gate failed**. Both shared and
separated networks now pass the joint development audio-versus-fitted-zero
threshold: BPM median error improves by 16.58% / 16.00%, and phase error by
46.04% / 46.99%, respectively. Natural phase also passes the fixed temporal
controls. The reference-only support used for this signal test is not silently
substituted for common-support comparator measurements below.

Errors on unchanged common support; lower is better. These are work-macro
aggregates of per-record metrics, not accuracy percentages or beat-detection F1.

| Cohort / output | BPM median error % | BPM P95 error % | Phase MAE, cycles | 4 s count MAE, cycles |
| --- | ---: | ---: | ---: | ---: |
| Development / raw events | 41.5878 | 100.7129 | 0.142737 | 1.048782 |
| Development / earlier protected clock | 42.1886 | 86.1719 | 0.205368 | 0.852539 |
| Development / shared | 35.4250 | 70.0797 | 0.134427 | 0.973301 |
| Development / separated | 35.4772 | 67.4752 | 0.132226 | 0.873077 |
| ARTBeaT / raw events | 14.4462 | 49.4982 | 0.107802 | 1.728725 |
| ARTBeaT / earlier protected clock | 29.1187 | 53.6912 | 0.230896 | 2.216003 |
| ARTBeaT / shared | 29.3684 | 49.0909 | 0.129888 | 2.458479 |
| ARTBeaT / separated | 28.2056 | 48.0883 | 0.123641 | 2.267299 |

Separation improves development phase/P95/count and all four ARTBeaT aggregates
versus shared training, but development median BPM regresses. Per-record BPM
regressions against shared are 3/5 development and 6/15 ARTBeaT; phase regressions
are 1/5 and 4/15. The required nonregression and development breadth gates fail.
Against raw events, shared regresses on BPM/phase in 3/5 and 1/5 development
recordings, and 12/15 and 13/15 ARTBeaT recordings. Separated has 3/5 and 0/5
development regressions, and 11/15 on each ARTBeaT metric. Earlier coupled,
trajectory and v1 comparisons remain in the machine-readable results, including
unfavorable metrics; the protected result is an additional descriptive comparator.

## Execution and selection

Twenty epochs / 100 paired updates per arm, unchanged complete recordings and
work weighting. Shared arms each made 100 native AdamW calls; separated arms
each made 100 tempo plus 100 phase calls. Total: 600 returned optimizer calls,
1,600 logical training-record presentations, 80 complete epochs and 400 evaluated
evidence pairs. All native-call receipts, initial/before/after/best snapshots,
selected exports and prediction vectors were retained. No fit was resumed.

| Arm | Seconds including snapshots/selection/export | Selected tempo epoch | Selected phase epoch |
| --- | ---: | ---: | ---: |
| Shared natural | 32.41 | 5 | 17 |
| Shared fitted zero | 32.39 | 4 | 10 |
| Separated natural | 65.90 | 7 | 20 |
| Separated fitted zero | 65.81 | 4 | 18 |

Whole runner elapsed before report: 201.84 s; process peak RSS: 1.65 GiB; maximum
per-arm reported Torch CUDA allocation: 195.46 MiB. This uses already cached
features, not end-to-end decoding/encoder inference or a product latency result.
Each arm remained below its 1,800 s budget; the whole run stayed below 7,500 s.
Runtime: Python 3.12.13, Torch 2.10.0+cu129, NumPy 2.2.6, T4; deterministic FP32,
TF32 off, two CPU threads. A container command initially failed to find `python`
before entering Python or opening the output directory; its exit-127 log was
retained. Correcting that executable name to `python3` launched the sole musical
run, without changing code, runtime, initialization or the registered plan.

Both architectures select tempo and phase independently using complete-epoch
development task losses, with earliest ties. Shared training has 24,003 parameters,
separated 47,907; **both selected exports have 47,907 parameters and two trunk
evaluations**. This does not isolate interference from increased training capacity,
and a task-selected shared export is not one joint checkpoint.

## What this establishes, and the next boundary

The frozen features carry learnable phase information under this experiment, and
both networks beat separately fitted zero inputs on development tempo too. But
removing parameter sharing is insufficient for reliable native speed: the
separated diagnostic BPM error remains nearly twice raw's, and the development
BPM breadth test fails. Do not turn partial gains into an automatic architecture
promotion, extra strategy, hyperparameter sweep or release.

There is a concrete next diagnostic, requiring no fit or new inference: inspect
the retained tempo fields, reference native beat levels and control predictions.
Time-mean features have slightly *lower* development median BPM error than
natural features (shared 34.4265% vs 35.4250%; separated 35.1911% vs 35.4772%).
That observation does not prove tempo ignores time, but shows the aggregate
audio-versus-zero gain alone cannot certify local change tracking. Quantify
absolute speed/beat-level errors separately from local timing sensitivity and
existing supervision coverage before proposing another model or fusion mechanism.

These are reused research cohorts: 20 fit recordings / 9 works, 5 development
recordings / 3 works, and 15 ARTBeaT diagnostic cases from one shared source.
No fresh independent acceptance data, holdout, new audio or upstream encoder
training was used. Phase is modular; four-second count integrates the tempo
branch alone. Independent tempo/phase evidence does not establish agreement,
beat timestamps, silence availability, confidence or bars. The one-call,
zero-per-song-tuning shipping contract and release state remain unchanged.

## Evidence

- [Plan](../../experiments/separated_evidence_fit/plan-v1.json), SHA256
  `f7798c4b78f06af3eb378e21660fbda6f9b372b734b02ba855a557fafd940408`.
- [Execution](../../experiments/separated_evidence_fit/execution-v1.json),
  [all cases and gates](../../experiments/separated_evidence_fit/results-v1.json),
  report SHA256 `251dcb1f421d1588bf8b6d11709e05696a15af57624d659600e51a15059984a8`.
- [Epoch/evaluation journal](../../experiments/separated_evidence_fit/journal-v1.jsonl)
  and four native-call journals in `experiments/separated_evidence_fit/checks/`.
- Private archive `four-arm-music-227977b.tar.gz`, SHA256
  `0b82723ef31009d98541cd9f6ae8ab7a4493563f7949556ef600e838f9df01b5`;
  model/optimizer/RNG snapshots and feature-derived arrays stay private.
- A read-only audit verified all 20 checkpoint hashes, 600 update receipts and
  selected weight identities, then reconstructed every measurement from all 40
  retained prediction packets and pinned references: exact equality for all 400
  pairs, aggregates and gates, without model inference or optimization. Public
  outcome tests independently check source/population identity, support, frozen
  comparators, update schedules, selection and decision reconstruction.
