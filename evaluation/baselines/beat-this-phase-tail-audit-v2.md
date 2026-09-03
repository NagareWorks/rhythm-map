# Beat This regression origin/tail audit v2

Date: 2026-09-03

## Result and scope

The four regressions are predominantly sensitive to input origin/context,
not to the newly recovered final 64--65 PCM samples. Restoring the old origin
reproduces the old event changes even when retaining the complete tail.
Removing only that tail changes neither beat nor downbeat timestamps on any
of the four cases, at either tested origin. This is a measured result for
these recordings, not a universal claim that end context never matters.

The expanded official-reference audit is **63/64, not all green**. All same-PCM,
same-mel, conversion/runtime, and postprocessor checks pass. One original-file
beat check fails on ARTBeaT 15: official decode/downmix/soxr produces 36 events,
while the current Rust PCM produces 37. This residual is upstream of neural
execution, but codec and resampling-filter contributions are not yet isolated.

No production code, model weight, observation contract, threshold, or estimator
policy changed in this step. The holdout remains sealed; nothing was pushed or
published. The previous 30-case accuracy baseline remains unchanged.

## Locked inputs and controls

`../parity/regression-lock-v2.json` binds ARTBeaT 13/15/18 and FSLD 110 BPM to
their existing calibration suites, audio hashes, probe times, and the unchanged
base reference lock. All four are complete recordings shorter than the 35 s
trace cap, originally 44.1 kHz. Both old and current PCM are retained privately;
the role-gated exporter uses the exact `beat-this = 1.0.0` decoder for the old
input, as an evaluation-only dev-dependency.

The same pinned official frontend, checkpoint, chunking, and minimal decoder
process every counterfactual. Given current PCM `C`, legacy PCM `L`, and the
predeclared 63-sample origin difference `d`, the two factors are:

```text
current:             C
tail trimmed only:   C[..len(L)-d]
origin restored:     L[..d] + C
both restored:       L[..d] + C[..len(L)-d]
legacy control:      L
```

Both-restored input equals the actual legacy input **sample for sample on all
four cases**, with zero waveform and zero output-logit differences. This
checks that no unaccounted waveform change is needed to explain v1 versus v2.
The tail difference is 64 samples on case 13/FSLD and 65 on cases 15/18; the
experiment measures lengths rather than assuming every legacy output lost
exactly one duration sample.

Changing the origin includes pre-origin filter content, input length, and
frontend/padding context. It is not an isolated fractional-delay test with
identical endpoint padding. Source-time subtraction is diagnostic only and
does not modify product timestamps. See the detailed procedure in
[`../parity/README.md`](../parity/README.md).

## Counterfactual observations

| Case | Actual v1 beats | v2 beats | v2 with tail trimmed | v2 with old origin, full tail | Official original-file beats |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARTBeaT 13: 180 to 120 | 31 | 30 | 30 | 31 | 30 |
| ARTBeaT 15: 85 to 127.5 | 34 | 37 | 37 | 34 | 36 |
| ARTBeaT 18: piano rubato | 30 | 28 | 28 | 30 | 28 |
| FSLD 110 BPM | 16 | 18 | 18 | 16 | 18 |

These are diagnostic counts, not F1 or a preferred musical pulse. The
same-origin tail experiment preserves exact beat/downbeat timestamps, not
merely their counts. Tail changes can still alter logits: maximum beat-logit
differences reach about 0.0435, but do not cross a selected-event boundary here.

Restoring the old origin reproduces the previously identified changes:

- Case 13 regains the approximately 0.34 s candidate.
- Case 15 loses the four new events around 20.12/20.84/21.54/22.26 s and
  regains the old approximately 19.08 s event.
- Case 18 regains the three peaks around 5.06/14.14/15.14 s and loses the new
  approximately 8.34 s peak.
- FSLD loses the two new **interior** events around 3.56/4.10 s. They shorten
  nearby intervals and explain why a tempo P95 regression must not be read as
  a physical file-tail defect. FSLD has tempo truth only; these events are not
  independently labeled false beats by this experiment.

## The remaining original-file mismatch

A separate read-only official-model replay uses the same `LogMelSpect`,
`final0`, `split_predict_aggregate(1500, 6, keep_first)`, and minimal decoder
on (a) official original-file PCM and (b) the Rust trace PCM. It reproduces
36 versus 37 beats without RTen participating. Probe maxima within two frames
of the locked times show three threshold crossings, not just one unmatched
event from the net count:

| Time | Official-file probability | Official model on Rust PCM | Selected, file / Rust PCM |
| --- | ---: | ---: | --- |
| 19.08 s | 0.539585 | 0.496309 | yes / no |
| 20.84 s | 0.496565 | 0.529600 | no / yes |
| 22.26 s | 0.498550 | 0.514230 | no / yes |

Both source paths have identical sample counts and zero detected waveform lag.
Thus a constant timestamp offset or relaxed event tolerance cannot fix this
count mismatch. The evidence points to model/threshold sensitivity to the
remaining PCM differences; it does not yet distinguish codec decoding from
resampling filter/phase details. It also does not establish which event set is
musically correct. The spot-check data and exact inputs are in
`../parity/source-threshold-probes-v2.json`.

## Decision and next bounded check

Keep the physically aligned v2 preprocessing and the current default decoder.
Do not turn the defective legacy origin into a fallback, add a per-track
resampling choice, globally lower the confidence threshold, or infer complete
reference parity from same-PCM checks alone.

Next isolate the remaining original-file mismatch by decoding once at the
native rate, checking native PCM agreement, and feeding that identical native
PCM through the two resampling paths. Keep onset/endpoint alignment and lengths
explicit. Only after this separation should a calibration-only peak-stability
experiment be considered; it would still need full-suite accuracy and cost
gates before replacing any part of the one shipping decoder.

## Evidence and validation

- Main result: [`../parity/regression-v2-audit.json`](../parity/regression-v2-audit.json).
  `passed: false` is intentional evidence of the unresolved check, not suppressed.
- Ordinary numerical checks: 63 pass, one original-file beat check fails.
- Input/logit reconstruction controls: exact on all four cases.
- Local engineering checks: 175 Rust tests, 13 Python tests, Clippy, formatting,
  five generated core cases, and diff whitespace checks pass.
- No fresh full 30-case music rerun was needed: only evaluation tooling changed;
  adapter `lib.rs` and `audio.rs` digests are unchanged from the preceding audit.

Reproduce the main audit by exporting all locked cases with
`--include-legacy-pcm`, then supplying the four traces, original audio files,
the pinned checkpoint/model pack, and `--phase-tail-lock` as documented in the
parity README. Existing reports and private traces are never overwritten.
