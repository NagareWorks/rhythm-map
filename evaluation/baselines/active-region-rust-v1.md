# Active-interval Rust candidate generation v1

## Decision and ownership

Implemented in `rhythm-map-eval`, which is not published. This is candidate
generation, **not an accuracy improvement or an automatic adoption policy**.
The default estimator, serialized Analysis schema, CLI, C ABI and WASM behavior
are unchanged. No training, threshold search, additional user tuning parameters,
neural inference or holdout access is involved.

The unsafe adoption results in [the preceding audit](active-region-adoption-audit-v1.md)
still apply. A valid path can delete genuine beats. No ranking or local score is
assigned here, especially when the primary has fewer than three events.

## Algorithm

1. Apply the frozen default activity rule: relative level at most -40 dB for
   at least 0.8 seconds, window edges expanded by half the median activity hop.
   Silence endpoints are inclusive. This heuristic can mistake quiet music for
   silence; it is not a perceptual activity classifier.
2. Remove candidates inside those silence regions, then take the active
   complements. Split again at candidate gaps strictly longer than 1.5 seconds.
   Unsupported prefixes/suffixes longer than 1.5 seconds are recorded as unknown.
   No events means an unknown interval. Unknown interiors are not filled;
   supported endpoint events may belong to adjacent components.
3. Within each component, reuse the historical pair-state recurrence, requiring
   at least eight path events and inter-event intervals in [0.1875, 1.5] seconds
   (40--320 BPM). Keep the historical whole-filtered-case harmonic coverage gate:
   harmonic sample count must be at least half the remaining candidate count.
4. Event reward is `confidence + 0.1 * downbeat_confidence + 5 * harmonic - 0.95`.
   Harmonic evidence is the nearest sample within exactly 0.02 seconds, with the
   earlier sample winning a distance tie. For successive intervals, let
   `r = ln(next_interval / previous_interval)`; subtract
   `min(2*r*r, 0.5 + 2*(abs(r) - ln(2))^2)`.
5. First/last events must be within 1.5 seconds of the component boundaries.
   Preserve ascending event traversal and strictly-greater updates so ties and
   traceback match the frozen experiment. An unavailable or too-short path
   produces explicit fallback, not an empty replacement sequence.

Unlike the historical dense N-by-N table, the Rust implementation stores only
allowed successor pairs. For N events and at most K successors per event,
the DP uses O(NK) storage and O(NK^2) transition work. K depends on event density;
this is not an unconditional linear-time or constant-memory guarantee. Harmonic
lookup uses sorted adjacent samples instead of scanning the full evidence array.

Every proposal has original and candidate timestamps plus shared-anchor
disagreement geometry. The boundary flags and primary-only/alternative-only
counts follow `MetricalAmbiguityRegion` semantics. The evaluation-specific type
deliberately omits `alternative_relative_score`: the public type requires a
complete-sequence rank, which these unranked local proposals do not have. This
avoids a sentinel score masquerading as confidence and leaves the public schema
intact. Equal candidates remain visible proposals with no disagreement regions.

## Frozen calibration replay

Windows x86-64, Rust 1.98.1, optimized `evaluation` profile, one measured run.
Cached PCM-derived observations are reused; input parsing and the historical
default-analysis replay are timed separately from candidate generation.

| Cohort | Exact cases | Components | Valid proposals | Fallbacks | Generator total | Slowest case | Peak DP capacity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RUBATO calibration | 25/25 | 65 | 46 | 19 | 68.983 ms | 6.537 ms | 407,344 B |
| ARTBeat calibration | 15/15 | 15 | 15 | 0 | 4.094 ms | 0.832 ms | 35,616 B |

All component boundaries, silence regions, unknown spans, proposal timestamps,
and fallback decisions exactly match the frozen Python outputs. Forced-stitch
TP/FP/FN and F1 also match; this is a diagnostic comparison, never adoption.
All 40 historical selected timestamp arrays replay exactly. ARTBeat's auxiliary
probe is excluded. Public authored regressions additionally assert that the
complete default Analysis stays unchanged when candidate generation is called.

DP totals: RUBATO 258,816 pair states and 2,071,863 scored transitions;
ARTBeat 12,902 pair states and 110,751 transitions. Peak capacity counts the
simultaneously allocated row/score/backpointer vectors; it excludes allocator
bookkeeping, inputs, rewards, output paths and process RSS. Timings are a single
VDI measurement, not a throughput benchmark or end-to-end audio latency promise.

Identities (raw observations/timestamps remain private):

- Estimator source: `3d2bc3ca875025b5d08e511dcecf38351fc8f62e27daf8d49147f9f8a68bf8f1`.
- Rust generator source: `0871202d8e05418965a283c5a7d98d9f6d39e0d49a29511770f183662a89c257`.
- RUBATO frozen experiment: `49324840abfb1b8ecbe824bc9deab16aa0f88992f7a79d6639bb16b6c06d138e`.
- ARTBeat frozen experiment: `886fd8c3bec7e9834c0b0656c3692dca80daf690d331328c5d0dc38eedffa8f2`.
- RUBATO Rust private result: `98ca21198e2b81cf116ac0bf3fe317bfeec87f619ece13516e030ab39711e72c`.
- ARTBeat Rust private result: `f4543bb418d92f4297182ad2f9816090cacf5f10844a106487ba92b55a615457`.

## Reproduction

Public CI needs no audio or private evidence:

```sh
cargo test -p rhythm-map-eval active_regions --lib
cargo test -p rhythm-map-eval --examples
python -m unittest discover -s evaluation/parity -v
```

The dense independent reference covers 84 regular cases (spacing, offset,
negative/zero/positive reward, exact ties) and 32 nonuniform graphs. Other tests
cover pauses, padding, silence endpoints, missing evidence, short primaries,
true tempo changes, sparse storage and one-/two-sided disagreement geometry.

For maintainers with the immutable calibration artifacts, choose an output
outside Git that does not already exist; obtain the two input hashes from
`evaluation/parity/active_region_parity.py`'s cohort lock:

```sh
cargo run -p rhythm-map-eval --profile evaluation --example active_region_candidates -- \
  --evidence PRIVATE_EVIDENCE --evidence-sha256 EVIDENCE_SHA256 \
  --baseline HISTORICAL_REPORT --baseline-sha256 BASELINE_SHA256 --output NEW_PRIVATE_RESULT
python evaluation/parity/active_region_parity.py --cohort rubato \
  --evidence PRIVATE_EVIDENCE --baseline HISTORICAL_REPORT \
  --frozen FROZEN_PYTHON_RESULT --result NEW_PRIVATE_RESULT
```

Repeat with `--cohort artbeat` and its corresponding artifacts. The verifier
rejects changed identities, missing/duplicate cases, shifted paths, boundary or
fallback drift, inconsistent edit counts and mismatched forced-stitch metrics.
Do not relabel old observation contracts; the existing RUBATO native-PCM bridge
and ARTBeat evidence audit remain the provenance basis of these locked inputs.

## Next gate

Do not wire this generator into the product yet. Establish an auditable,
truth-free **abstention/adoption contract** for disagreements, including short
primaries, one-sided edges, genuine half/double-time changes and counterexamples
where a smoother candidate loses true beats. Any later ranking must survive both
calibration cohorts without silently erasing their demonstrated tradeoffs. Only
then consider an independent holdout gate and product integration.
