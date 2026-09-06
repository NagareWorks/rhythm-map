# Backend response identity and exact-coordinate bridge

This evaluation-only step connects the unchanged backend's actual observations
to the supplied-clock ledger contract. It does not search clocks, select tempo,
recover beats, calibrate confidence, run inference, open holdout or train a model.
There is no new runtime dependency, product strategy or user parameter.

## What one response packet means

The adapter retains separate fields instead of deriving one from another:

| Field | Meaning |
| --- | --- |
| Source and source index | Identity within the original default/raw or candidate event sequence |
| Exact nominal coordinate | Integer or rational center in the 50 Hz model-frame domain |
| Published observation | Original float32 seconds widened to float64 and original confidence fields |
| Score frame | Actual integer frame used by the backend to read confidence |
| Paired marks | Original beat and downbeat logits at that score frame |
| Parent plateau and raw members | Provenance under the shipped extraction rules, not nearest-event matching |

Raw peaks use a strict positive-logit threshold and radius-three local maximum.
The running-mean merge width is one frame. Two adjacent maxima can therefore
publish a half-frame center. Publication through float32 seconds can change which
side of the half-frame rounding boundary supplies confidence: center 11/2 can
publish just below 0.11 seconds and read frame 5, not nominally rounded frame 6.
The adapter preserves all three coordinates rather than silently rounding them.

Candidate peaks use radius one with no logit floor. A contiguous maximum plateau
has one representative, its lower middle actual frame. Flat negative background
regions also contribute candidates. A long positive plateau can produce multiple
raw events sharing one candidate parent, even when their centers are more than
three frames apart. Parentage follows original peak membership, not distance;
these sources are not independent evidence and cannot be pooled in the bridge.

Default downbeat peaks are snapped to raw beat times. A raw beat within 70 ms of
a snapped event receives a downbeat confidence floored at 0.5; otherwise that
field is zero. Neither value is necessarily the sigmoid of its original downbeat
logit. Candidate downbeat confidence is the direct score-frame sigmoid. Keeping
these meanings distinct prevents a later observation model from treating a
postprocessing decision as a second independent neural response.

Candidate publication also filters times beyond the audio duration; raw peak
publication does not apply that filter here. Float32 publication can put an
endpoint nominally equal to duration just beyond it. Excluded candidate plateaus
and any raw descendants remain explicit in the provenance inventory.

## Exact bounded matching

`rational_response_ledger.py` extends the existing integer ledger without changing
its frozen bytes. It accepts exact integer/Fraction coordinates, not rounded
floats, and uses the same objective: maximize one-to-one ordered matches within
three frames, then minimize total absolute offset. The offset is rational; all
optimal assignment multiplicities remain exact integers. One deterministic
witness does not replace the other optimal assignments.

Both bracketing frames of a fractional response must be observed. Query coverage
is counted over actual integer frames: a radius-three integer-center query has
seven possible samples, a half-frame-center query has six. Missing samples are
not zeros. The source replay itself requires complete finite heads; it does not
pretend that the production detector has been replayed through missing regions.

Limits remain 4,096 frames, 128 ticks, 128 responses, and 16,641 DP states.
Rational denominators are capped at 1,000,000. The new dense table checks its full
allocation budget before execution and rejects excess input without a partial
result. Marks remain metadata, not likelihoods or matching weights. Real cohort
clock matching is deliberately not part of this identity audit.

## Frozen all-track inventory

The script, contract and rational kernel were hashed before the first cohort run.
All 15 ARTBeaT and 25 RUBATO calibration captures were checked, including their
capture, source, model and predecessor identities. No truth labels were used for
this inventory. Each published event count and timestamp must match exactly;
confidence formulas allow a predeclared eight float64 ULPs for library arithmetic
while preserving original values. Both cohorts actually matched at zero ULP.

| Count | ARTBeaT | RUBATO |
| --- | ---: | ---: |
| Frames per head | 12,328 | 324,515 |
| Raw/default events | 375 | 9,273 |
| Included candidate events | 1,559 | 35,151 |
| Candidate plateaus excluded by duration | 1 | 0 |
| Candidates with raw lineage | 375 | 9,273 |
| Candidates without raw lineage, logit <= 0 | 1,180 | 24,758 |
| Positive candidates excluded by radius-three competition | 4 | 1,120 |
| Raw downbeat field differs from direct frame sigmoid | 240 | 5,141 |
| Raw downbeat confidence floored from below 0.5 | 4 | 83 |

These 40 captures have no fractional raw centers, multiple-raw plateau parents,
or raw events lacking an included candidate parent. The authored controls cover
those legal backend edge cases without claiming they caused observed musical
errors. The one excluded ARTBeaT candidate plateau has no raw descendant.
The candidate-reason rows partition all included candidates; they are extraction
causes, not musical false-positive or missed-beat labels.

Controls verify the distinct coordinate/score semantics, long plateau lineage,
negative and zero plateaus, endpoint filtering, original record preservation,
the fixed numeric allowance, and rejection of mixed/duplicate packet identities.
All 70 previous integer clock ledgers retain the same objective, full assignment
counts, coverage and witness. Eighty fractional cases agree with independent
enumeration; unavailable-padding and 2^64 multiplicity controls also pass.

## Reproduction and limits

Run the adapter with private immutable captures outside the repository:

```bash
python evaluation/parity/backend_response_packet_audit.py \
  --artbeat-evidence /data/candidate-evidence-v1.private.json \
  --artbeat-captures /data/dense-artbeat-v1 \
  --rubato-evidence /data/rubato-cache-replay-final-v1.private.json \
  --rubato-captures /data/dense-rubato-v1 \
  --output /data/backend-response-packets-v1.json
python -m unittest discover -s evaluation/parity -p test_backend_response_packets.py -v
```

Output creation refuses overwrite. The public JSON contains only per-track
aggregate inventories and identities, not frame arrays, event times or private
paths. Report SHA-256:
`6bf7aa6b88aae92b7b91767b1e63a7fe78b4b4275080dba42edd4c39ceef23e0`.
Its embedded hashes identify the script, lock, helpers and frozen predecessor.

This closes the source-provenance prerequisite, not the tempo inference problem.
Next freeze matched bounded windows and supplied-clock hypotheses before using
the bridge to inventory real unmatched responses/ticks, separately by source.
Retain weak/subdivision responses, omissions and indistinguishable clocks rather
than fitting an omission penalty or promoting candidate counts as beat accuracy.
The original half/double and omission/change ambiguities remain unresolved;
there is still no demonstrated requirement to train a model.
