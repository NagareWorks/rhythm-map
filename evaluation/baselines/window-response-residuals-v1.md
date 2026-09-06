# Fixed-window response residual inventory

This is the first real-capture use of the exact response-packet ledger bridge.
It identifies where responses fail to cover supplied clocks, not which tempo an
automatic decoder should choose. Annotations supply the reference and prefix
clock; results are truth-assisted diagnostics, never beat accuracy or recovered
beats. No inference, production change, user strategy, holdout access or training.

## Frozen comparison contract

Every capture is partitioned from frame zero into nonoverlapping half-open
400-frame (eight-second) windows. All frames and source response centers belong
to exactly one window, including excluded windows. Both sources use the same
eligible windows and all five clock hypotheses. No outcome selects a window.

The last four annotated beats strictly before the window establish a prefix
period: their total span divided by three. The final prefix beat anchors phase.
Complete clocks over the same physical window are:

- annotated beats inside the window (an oracle reference);
- continuation of the prefix period;
- twice that period, with both alternating continuation phases retained;
- half that period, on the same prefix anchor.

Neither half phase is selected as the better one. There is no phase search or
future-response fitting. Annotation coordinates, including tempo-region bounds,
are rounded to the nearest microsecond with positive half ties upward, then
represented as exact rational 50 Hz coordinates (at most 0.5 microsecond error).
Backend packets keep their original exact nominal coordinates, score-frame marks,
publication and identities; whole-capture extraction is not rerun on each crop.

Partial windows, insufficient prefix, uncovered annotation tails, excess tick or
response counts, and packet support crossing a window boundary are explicit
exclusions. Do not truncate inputs or compare only the source/hypothesis that
fits. Limits remain 128 ticks, 128 responses per source and 16,641 states per
ledger. The rational ledger and provenance adapter are unchanged and hash-pinned.

Assignments never cross windows. A radius-three query extending beyond the
window is boundary-censored; its absence is not an interior miss. This is not an
edge-recovery benchmark. This audit's exact three-frame (60 ms) nominal-coordinate
radius differs from the older published-time 70 ms beat matching, so their missed
counts are not directly comparable. Short audio tails and missing annotation
coverage remain unknown, not acoustic silence.

## What is counted

Each ledger maximizes one-to-one matches, then minimizes exact absolute offset.
Ticks and responses retain never/sometimes/always matched states across all
optimal assignments. A witness is not treated as the only explanation. Response
strata retain boundary coverage, proximity to an annotation and original beat
logit sign. Off-annotation is a geometric label, not proof of musical irrelevance;
nonpositive is a source property, not a new confidence threshold.

For an annotated tick never covered by the raw source, distinguish a censored
boundary, a candidate assigned in every/some optimum, a nearby candidate lost to
one-to-one competition, and no candidate inside the radius. Candidate mark
provenance uses every optimal pair, retaining positive-only, nonpositive-only or
mixed support rather than reading only the witness.

Clock comparisons retain changes in matched responses and unmatched ticks as two
separate quantities. More candidate coverage does not automatically mean a more
plausible clock. There is no fitted omission penalty or scalar winner score.

## All-track results and exclusions

All 40 frozen capture/source/model identities and the previous packet inventory
were verified. The complete owned population remains 9,648 raw events and 36,710
candidates, but only the eligible subset below enters clock comparison.

| Exposure | ARTBeaT | RUBATO |
| --- | ---: | ---: |
| Input tracks | 15 | 25 |
| Tracks contributing eligible windows | 9 | 25 |
| All windows | 39 | 824 |
| Eligible windows | 9 | 766 |
| Insufficient-prefix windows | 15 | 25 |
| Partial-tail windows | 15 | 24 |
| Uncovered-annotation-tail windows | 0 | 9 |
| Eligible raw responses | 99 | 8,998 |
| Eligible candidate responses | 468 | 32,819 |
| Annotated ticks in eligible windows | 134 | 6,435 |
| Raw matches to annotated ticks | 93 | 3,792 |
| Candidate matches to annotated ticks | 129 | 4,948 |

There were no budget or packet-support exclusions. All real ledgers had one
optimal assignment; ambiguity is still implemented and tested on authored cases.
The largest table was 2,665 states, below the fixed cap.

Among raw-unmatched annotated ticks, the complete partition is:

| Residual cause | ARTBeaT | RUBATO |
| --- | ---: | ---: |
| Boundary-censored | 1 | 42 |
| Interior, candidate assigned in every optimum | 36 | 1,143 |
| Interior, candidate lost to assignment competition | 0 | 3 |
| Interior, no candidate within radius | 4 | 1,455 |
| Total raw-unmatched ticks | 41 | 2,643 |

All 36 ARTBeaT candidate-supported interior misses have nonpositive candidate
marks. Of 1,143 RUBATO misses with candidate support, 1,064 are nonpositive-only
and 79 positive-only. This does not make those candidates safe to add: relative
to the annotated clock, the same eligible pools also contain 314 ARTBeaT and
20,982 RUBATO interior, off-annotation, nonpositive candidates that never match.
These are hard background observations to preserve, not a justification for
lowering the default threshold.

Density alone still favors overly fine clocks. On RUBATO candidates, continuation
matches 3,793 responses with 2,716 unmatched ticks, while double matches 7,767
with 5,278 unmatched ticks. On ARTBeaT those pairs are 88/28 versus 177/59.
The tradeoff is visible rather than hidden in a chosen weight. Neither is a
tempo verdict; the reference clock and its phase were already annotation-assisted.

## Coverage limitation and next gate

The eligible ARTBeaT windows contain eight change-context windows and one rubato
window. All 766 RUBATO windows are rubato. **There are zero constant-context and
zero ramp-context windows** under this fixed prefix-plus-eight-second contract.
Thus this run cannot establish discrimination of constant tempo with omissions
versus true changes, or tune behavior for constant music. Six ARTBeaT tracks
contribute no eligible window; they are retained, not silently dropped from the
input count. Counts are pooled exposure inventories, not a macro accuracy score.

Next audit annotation-only availability for matched constant/change contexts,
without inspecting response quality to choose spans or offsets. Freeze the next
comparison protocol before replay; keep this eight-second result and its coverage
failure unchanged. Candidate-supported misses must then be paired with competing
off-annotation responses in the same exposure before judging whether existing
joint-head/context evidence distinguishes them. Do not fit a single-frame gate,
declare the algorithm solved, or infer that training is necessary from these
unmatched-count summaries.

## Reproduction

The contract and script were frozen before the first run. That run stopped before
report publication because RUBATO metadata lacks optional `tags`; only this field
access was fixed, with a whole-track mock regression and re-frozen script. No
window, clock, radius or statistic changed after viewing results.

```bash
python evaluation/parity/window_response_residual_audit.py \
  --artbeat-evidence /data/candidate-evidence-v1.private.json \
  --artbeat-captures /data/dense-artbeat-v1 \
  --rubato-evidence /data/rubato-cache-replay-final-v1.private.json \
  --rubato-captures /data/dense-rubato-v1 \
  --output /data/window-response-residuals-v1.json
python -m unittest discover -s evaluation/parity -p test_window_response_residuals.py -v
```

Output refuses overwrite. Real frame arrays, packet coordinates, published event
times and private paths remain outside Git. The aggregate report identifies all
captures, truth files, helpers and contracts. SHA-256:
`35eb07cfc8c96471710ecf0a40d3fae6da836638845090cc21583ba337769a8a`.
