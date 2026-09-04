# Observation dropout and missing-step clock v1

## Decision

**Reject `missing-step-clock-v1` as a replacement for the default estimator.**
It can bridge missing observations without inserting beat events, but its
tempo-continuity prior erases real octave changes and regresses expressive
music. This is evidence against this timestamp-only reference model, not
evidence that clock-state inference is exhausted or training is necessary.
No production estimator, backend, selected beat, schema or public option changed.

The protocol and objective were fixed before running the candidate; see
[training decision protocol](../../docs/TRAINING-DECISION.md). No weight sweep,
audio transformation, model inference or holdout evaluation was performed.

## Scope and measurement

All 15 ARTBeaT and 25 RUBATO calibration cases were included. Each has five
fixed observation masks and two downbeat factors: 400 perturbation variants,
plus 30 unique authored-control variants (repeated in both reports). The
controls are generated musical truth; this is not 430 independent recordings.

Only ideal **observations** are removed. Annotated musical truth remains
unchanged; absent observations are not made available through candidates,
activations or acoustic features. Retained beat confidence is one. Thus these
are controlled detector deletions, not measured errors from Beat This.

Query locations are unchanged truth beat-interval midpoints. The table reports
the arithmetic mean of each recording's P95 absolute relative BPM error, in
percent. It is not pooled P95 and uses a different query contract from earlier
main-evaluator reports. Downbeat-zeroed results exactly match oracle-downbeat
results for the listed measurements; this does not establish that downbeat
evidence is generally useless. The candidate ignores that channel.

| Calibration | Observation mask | Default mean P95 % | Candidate mean P95 % | Cases with worse P95 |
| --- | --- | ---: | ---: | ---: |
| ARTBeaT (15) | Intact | 2.37 | 29.44 | 11 |
| ARTBeaT (15) | One missing every eight | 33.91 | 25.06 | 3 |
| ARTBeaT (15) | Alternating middle third | 51.00 | 26.94 | 3 |
| ARTBeaT (15) | Four missing centrally | 60.88 | 33.77 | 4 |
| ARTBeaT (15) | Eight missing at tail | 16.29 | 23.25 | 8 |
| RUBATO (25) | Intact | 5.57 | 116.00 | 15 |
| RUBATO (25) | One missing every eight | 39.23 | 101.42 | 18 |
| RUBATO (25) | Alternating middle third | 54.67 | 128.23 | 20 |
| RUBATO (25) | Four missing centrally | 7.65 | 117.16 | 16 |
| RUBATO (25) | Eight missing at tail | 8.18 | 108.00 | 14 |

"Worse" here means more than 1e-9 percentage points; it is a descriptive P95
comparison, not the complete promotion gate. Intact-case regressions matter:
repairing synthetic deletions cannot excuse damaging already correct input.

Tail rows additionally lose candidate numerical coverage: 120 ARTBeaT and 196
RUBATO midpoint queries per downbeat factor have no candidate tempo. Default
segments numerically cover those queries but they lie outside its returned
beat span. Neither absence nor unsupported extension can be hidden by averaging
only covered errors. Interior masks have no uncovered queries in either path.

## Authored discrimination controls

For constant 120 BPM, removing one observation every eight beats produces 17
default `TempoJump` outputs and 33.33% P95 error. The missing-step candidate
recovers 120 BPM throughout the observed span, with no invented event outputs.
It also recovers the constant-clock central alternating and four-event gaps.

However, an intact 120/60/120 BPM control has approximately zero default P95
error and **100% candidate P95 error**: the candidate chooses 120 BPM during
the genuine 60 BPM section. The intact 120/90/120 control remains correct.
This is a specific octave ambiguity, not evidence all transitions were smoothed
away. Counts of default change outputs are not matched boundary precision.

The strongest witness uses zeroed downbeats:

- constant 120 with alternating central observations missing;
- actual 120/60/120 with every musical beat observed.

These have exactly equal `RhythmObservations`, including metadata. Default
returns 60 BPM at 12 seconds for both; the candidate returns 120 for both.
Each picks a different explanation of the **same evidence**. A different
timestamp-only penalty cannot guarantee correctness for both. Additional audio
evidence might distinguish them; the audit does not claim their audio is equal.

## What follows

Oracle-only success did not establish robustness: we now have reproducible
evidence that removing observations can create tempo changes with no change in
musical truth. Yet a missing-step state model by itself is not sufficient.

Do not tune another penalty on these failures or promote the candidate only
for a handpicked slice. The next experiment should test a genuinely additional
observation likelihood (dense beat/downbeat evidence or independent acoustic
features) and explicitly separate observed events, inferred clock advancement,
and unavailable evidence. It must include real cached detector output, the
paired octave counterexample, genuine tempo changes and fixed coverage gates.
No sealed holdout should be spent on the rejected reference model.

## Reproduction and provenance

Run the evaluation-only example once per calibration suite with a fresh output
file on the configured artifact drive:

```sh
cargo run --locked -p rhythm-map-eval --example observation_dropout -- \
  --suite evaluation/suites/artbeat-v1.json --output <new-artbeat-report.json>
cargo run --locked -p rhythm-map-eval --example observation_dropout -- \
  --suite evaluation/suites/rubato-calibration-v1.json --output <new-rubato-report.json>
```

The command rejects non-calibration roles and refuses to overwrite reports.
Schema 2 includes the candidate source identity and its uncalibrated objective,
advancement histogram, fixed-query coverage and tempo errors for each variant.
Twelve unit tests cover input equivalence, no truth leakage, downbeat isolation,
coverage accounting, short/invalid inputs, and DP versus exhaustive search.
No audio, private paths or per-event truth are published in this note.

| Artifact | SHA-256 |
| --- | --- |
| ARTBeaT schema-2 aggregate report | `5bb58405df3ee6bd0f934e3fd38998ec0f089cf4220c24481b57383a17ff05cb` |
| RUBATO schema-2 aggregate report | `32d88fe4d29b15cbe2c7088d7f139145ee2f92702f0b3dd91582685b4e8ecc5f` |
| Audit source | `0fe85bb6e9e417975e2a98981e8bef400ce8dc770d40cae69eca03c71025eea8` |
| Candidate source | `444c8ed0e573de7eeb85499d289d44d966bace243408d635d757ad377cf9f5ec` |
| Unchanged estimator source | `3d2bc3ca875025b5d08e511dcecf38351fc8f62e27daf8d49147f9f8a68bf8f1` |
| ARTBeaT suite | `21f3d44bacbfe9c50dfbc889990c563d44e406d56558492627402d21e5a7e81b` |
| RUBATO suite | `c10c229bbf7b89ebd23dd2b4ff2a2d19aaec9b5f28d2b5eb6d121d950fb62653` |
