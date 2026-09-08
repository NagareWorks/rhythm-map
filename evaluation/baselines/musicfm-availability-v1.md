# MusicFM: reuse of the shipping activity contract

This is a **separate retrospective composition**, designed after the known
[unconditioned silence failure](musicfm-temporal-v1.md). That report, its rule,
scores and failed decision remain unchanged. This is not independent musical
acceptance, a trained head or a new shipping strategy.

## What is reused, and what is new

The [frozen protocol](../parity/musicfm-availability-lock-v1.json) pins the
existing core engine, estimator, types, Cargo lock and diagnostic exporter.
No core file or default was changed. The native executable invokes the actual
`Engine::observe_pcm` activity extractor, not a Python approximation. It keeps:

- 24 kHz mono PCM, nominal 100 ms activity windows and 50 ms hops;
- RMS relative to the complete input's loudest activity window;
- the existing all-negligible-peak branch (`peak <= f64::EPSILON`) at -120 dB;
- the unchanged -40 dB threshold and 0.8-second minimum low-activity span.

An empty diagnostic backend produces no model events. To query whether a native
25 Hz feature center lies inside a shipping low-activity region, the exporter
passes **one diagnostic sentinel at a time** through `analyze_observations`.
With fewer than three beats, the existing early-return path cannot estimate a
tempo; the metrical selector also returns without changing a one-event sequence.
A retained timestamp must be bit-identical and a rejected one must have the
existing low-activity warning. Sentinels are membership queries, not observed or
inferred beats. They never enter a music result or the public report.

This deliberately reuses the real private filtering logic through its existing
public boundary without copying its thresholds/region algorithm or widening the
product API. Repeating the estimator is a bounded research diagnostic, not a
proposed production execution path. The generated onset envelope is not scored.

The new composition has three states:

| State | Meaning | Feature scoring |
| --- | --- | --- |
| `unknown_activity` | Required coverage is absent, gapped or outside covered cells | Not run |
| `low_activity` | An existing low-activity region rejects a required center | Not run |
| `available_for_periodic_test` | Required centers are covered and not rejected | May run the unchanged comparison |

The third state does **not** mean audible rhythm, known beats or confidence in a
musical beat unit. Uniform DC, steady tones and noise can pass an energy test.
Missing activity is unknown even though the core estimator treats an absent
optional envelope as no silence constraint. This is an explicit research
availability precondition, not a change to timestamp-only product behavior.

## Coverage and score preservation

The union of targets and both interpolation endpoints required by all eight
previous clocks must be covered and retained. Any unknown/low-activity member
rejects the **whole window before feature-file access**. No favorable target
subset, phase, boundary, chunk owner or clock is chosen. An unavailable member
still rejects its pair and stays in the denominator. Both native clock families
require 100 distinct centers here, including the 75 common scoring targets.

An initial authored unit expectation exposed an existing edge condition: on
two seconds of silence, the last activity center is at 1.95 s and the median-hop
half-cell ends near 1.975 s. A sentinel at 2.0 s is not removed by that region.
The new composition preserves that behavior and returns **unknown** outside
covered cells; it does not extend the quiet region to EOF. A 12.09-second input
also demonstrates an emitted native token at 12.08 s beyond the final half-cell.
Missing/irregular activity cells similarly cannot become affirmative evidence.
The coverage geometry was fixed and tested before the composed replay.

For uniform coverage the diagnostic uses the existing median-hop half-cell
geometry, clipped to the file, with a `1e-12` second numerical uniform-grid
check. That tolerance is not a new silence or musical-confidence threshold.
Coverage at nominal centers is not proof that a noncausal model's entire
receptive context is informative or that its representation is context-invariant.

If a window remains available, use the original context owners, 1,024-channel
features, full-cycle-minus-half-cycle formula, supplied clocks and numerical
budgets without modification. Verify bounded NPY file size and full SHA-256 on
the same opened handle **before** parsing with pickle disabled; then verify
shape, dtype, raw tensor hash and complete ownership. The five active original
cases replay old features; silence never opens its feature files. This avoids
new neural inference, not the need for independent validation later.

## Authored cases and measured result

The [complete report](../parity/musicfm-availability-v1.json) passes all 15
activity-contract cases. It preserves six old PCM identities and adds nine
fixed activity-only controls before cached feature replay:

| Condition | Availability result |
| --- | --- |
| Original clean, omitted, weak, accelerated and slowed PCM | Available; all original scores exact |
| Original complete silence | Low activity; both clock ledgers suppressed before feature access |
| Original constant PCM scaled by 2^-16 | Available; not erased just for low absolute amplitude |
| Half-second complete cue/pulse rest | Available under the unchanged minimum-span rule |
| Four-second rest, three-second prefix rest, final rest | Intersecting window rejected as low activity |
| Steady 440 Hz tone, constant DC, seeded noise | Available, **periodicity not evaluated** |
| Original PCM scaled by 2^-60 | Low activity via the existing absolute numerical floor |

Each full 12.01-second input produces 241 activity points and 301 native centers.
The long-rest case rejects 99 centers; prefix and suffix controls each reject 75.
None of the fifteen complete inputs has a coverage-unknown center; unknown/tail
and missing-cell behavior is covered separately by native and model-free tests.

All six original constant/change pairs retain both correct native shapes and
all density-rival exclusions, with every score exactly equal to the old report.
The known silence counterexample is now rejected before opening its archive.
This is **known-counterexample composition success**, not a new 6/6 accuracy
claim: those features and outcomes were already exposed when this was designed.
Gain/rest controls only establish availability behavior, not preservation of
MusicFM's learned representation under those transformations.

The end-to-end diagnostic took 13.92 seconds, including fifteen native activity
exports, source/file verification and replay of five cached feature sets. It ran
no pretrained forward and is not comparable to the earlier 54-forward timing or
to an end-to-end product throughput benchmark. Eight native tests and eighteen
initial Python controls passed before replay; three retained-report regressions
were added afterward. No real music, holdout or training was accessed.

## Next boundary

Before any music-feature evaluation, separately freeze neural negative/robustness
checks for the still-unscored steady/noisy controls and short-rest/gain variants.
Explicitly state the expected abstention/retention behavior before opening new
features. Energy availability alone cannot supply a no-rhythm hypothesis or a
confidence calibration. Do not adjust the old recurrence scores or claim the
original failure disappeared. This result still does not prove training necessary.

Keep all 40 existing calibration identities, the seven exposed ARTBeaT pairs,
untyped RUBATO and the sealed holdout unchanged. No product option, default,
schema, model pack, commercial clearance, native/WASM model deployment or
release is introduced.

## Reproduction

```text
cargo test --locked -p rhythm-map-eval --example activity_availability
cargo build --locked -p rhythm-map-eval --example activity_availability
python -m unittest discover -s evaluation/parity -p test_musicfm_availability.py -v
python evaluation/parity/musicfm_availability.py --native-executable <built-example> --features <verified-old-private-archives> --output <fresh-private-directory>
```

Use NumPy 2.2.6 and the byte-pinned previous archives; Torch and new checkpoint
loading are not required. The current orchestration output guard requires a
fresh Windows off-system path outside Git. Native unit checks run on all three
CI platforms. Each native export is limited to complete finite mono PCM of at
most 30 seconds and a 30-second subprocess timeout. The first contract mismatch
stops the replay; incomplete output is not a pass. Raw PCM and activity envelopes
stay private, while reports retain identities, counts and outcomes.
