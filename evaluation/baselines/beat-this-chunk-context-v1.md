# Prefer-central-window stitching: rejected calibration candidate

Date: 2026-09-04. Same weights, mel input, chunk locations and decoder; no extra
model calls in the candidate, training, threshold search, or holdout access.

## Fixed hypothesis

The upstream predictor uses 1,500-frame chunks, discards six frames at each
edge, and advances by 1,488 frames. To avoid a short final chunk, it moves the
last window backwards. That last window can overlap much of its predecessor,
but `keep_first` always takes the earlier window's output.

The experiment changes only ownership of shared valid frames. Choose the
window maximizing `min(local_frame, chunk_length - 1 - local_frame)`; equal
distances retain the earlier window. Beat and downbeat logits come from the
same owner. There is no averaging, confidence-based selection, extra padding,
new window, or invented timestamp. Outside the final overlap the output is
unchanged; single-window inputs are unchanged everywhere.

## Feasibility screen versus complete-recording check

The two existing frozen reference traces were tested first. The same ONNX
Runtime predictions supplied both the upstream and candidate aggregation.
Upstream beat events exactly matched the RTen traces; the largest baseline
logit difference across both heads/cases was approximately `4.10e-5`, within
the existing reference-parity budget.

- The complete 13.6-second ARTBeaT `75-to-150` control uses one window. Its
  events and F1 are unchanged.
- The **35-second prefix**, not complete recording, of RUBATO Bach/MacLeod
  uses windows starting at frames -6 and 257. Ownership changes on 612 frames;
  matched beats rise from 31 to 35, unmatched predictions remain 19, and beat
  F1 rises from 0.65263 to 0.70707. This motivated the larger check, not promotion.

The RUBATO trace records the historical CRLF suite digest
`45994397adfe4cb343769b8f05effcf76cd8577c0b87bda086099ecc70534f3b`.
Reconstructing CRLF bytes from the current LF suite exactly reproduces it.
Both identities remain recorded; neither artifact was rewritten or relabeled.

## All 25 RUBATO recordings

The larger check fixed the same rule for all 25 calibration cases, covering
6,490 seconds of audio. Complete decoded native-rate float32 PCM matched each
frozen shipping PCM hash before inference. The unchanged frontend processes
that PCM; only the two original tail windows are inferred again. Earlier
events are retained from the locked cache.

Before scoring any candidate, reconstruct the default tail events and require
the complete spliced raw sequence to equal the cache exactly. The first four
frames at the partial-trace boundary remain from the cache, and candidate
logits there must be unchanged. A mismatch fails closed rather than being
treated as a window improvement. All 25 raw-event and chronological truth-pair
replays passed. This is partial neural recomputation with **complete-recording
raw-event scoring**, not full neural or end-to-end estimator re-execution.

| Complete-recording raw beat metric | Result |
| --- | ---: |
| Baseline mean F1 | 0.5214049901 |
| Candidate mean F1 | 0.5219194534 |
| Cases with improved / regressed F1 | 8 / 7 |
| Cases failing coverage/precision/F1/timing comparison | 10 |
| Previously missed truth identities recovered | 8 |
| Previously covered truth identities lost | 4 |
| Net change in unmatched predictions | -10 |
| Cases failing metrics **or** covered-truth identity preservation | 11 |

The default estimator's 0.5212633988 baseline is a different measurement, not
the raw baseline in this table. Downbeat, tempo, section and change-point
accuracy were not certified by this beat-only screen.

The complete Bach/MacLeod recording also illustrates why better aggregates are
insufficient: it gains four previously missed truth beats but loses one
previously covered truth beat. Its matched count, precision, recall and F1 all
increase, median timing error decreases, and P95 is unchanged. Thus even the
current metric-only comparison gate cannot certify preservation of every
covered truth identity. The separate identity audit catches that loss. This
report does not silently upgrade historical comparison schemas or claim that
the existing gate already checks identities.

All 15 ARTBeaT calibration inputs fit one window, so the candidate is identical
to baseline on them by construction. This is a checked single-window control,
not 15 new model runs or evidence of improvement on tempo-changing exercises.

## Decision and cost

Reject default adoption. The prefix gain does not transfer safely to complete
recordings. Greater distance from a window edge is not evidence that a model
has selected the correct musical beat level. Do not add a public stitching
option, confidence-weighted blend, or song-specific exception to salvage this
rule. This does not establish that every context/inference change is useless.

The complete-recording screen needed 50 beat-model calls versus 230 for one
full recomputation of the original windows. Measured frontend plus tail model
time totaled 309.78 seconds on the local CPU run. This excludes input hashing,
file/JSON loading and scoring; it is not product latency or a production
speedup. The candidate itself introduces no additional model calls.

Eight authored tests passed across the frozen private scripts: single-window
and non-overlap invariance, central ownership/earlier ties/head consistency,
invalid or uncovered windows, global float32 timestamps and retained prefix,
cache/context mismatch rejection, start geometry, and metric regression hidden
by higher F1. Dense inputs and per-event outputs remain outside Git.

## Immutable evidence identities

- Routing/screen script:
  `5fba78a58240d7d23c381ba679d3eae3ba8fcbb85c662cfd92779dcf0552c7e1`.
- Prefix-screen result:
  `ca67d5f136c7a8732af830bcc604ca213758d4c53b5343d0ab6b37cee6d87de5`.
- Complete-recording tail-screen script:
  `57dc7459059c72bee0aa401832833b93d9463c0193e14d2c68af36055a381489`.
- Complete-recording result:
  `c1c95b283ee8357671b9470c0cca4a3d496e7c0cc83efa52d376741c5403ebff`.
- RUBATO observation evidence:
  `ce5e678276888a0e430c004444dce4b27f0cfac0761767736abee2ec3fc05937`.
- Model manifest:
  `ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`.

See the [reference contract](beat-this-reference-parity-v1.md),
[full-PCM equivalence](beat-this-rubato-pcm-equivalence-v1.md), and
[raw-cache replay](beat-this-rubato-cache-replay-v1.md) for the input provenance.
