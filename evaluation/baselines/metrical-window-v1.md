# Temporally tolerant metrical-event windows

## Scope and frozen contract

This inventories existing neural evidence; it is not a new decoder, likelihood
adapter, recovery count or accuracy improvement. All 15 ARTBeaT and 25 RUBATO
calibration recordings reuse their complete frozen captures. No model inference,
fitted mapping, threshold search, production change or holdout evaluation runs.

The [contract](../parity/metrical-window-lock-v1.json) and extraction script were
frozen before the first cohort execution, after nine authored controls passed.
No extraction or ranking rule changed after viewing the results. The
[report](../parity/metrical-window-v1.json) hashes the contract, source, helper
code, earlier dense audit, capture summaries and every truth/capture file.
Its SHA-256 is `ffc006a15222dddf67cb912c3861dd9dcbb92ec30c8aa233979d333666b731a7`.

Each annotated beat is rounded to a 50 Hz center using nearest-integer,
ties-to-even rounding. A window always contains that center and three real
frames on each side, matching the pinned training tolerance. Relative to the
unquantized timestamp, quantization can add up to 10 ms. A partial or unavailable
window stays unavailable; it is not clipped, padded, shrunk or labeled silence.

The feature function receives only the two heads, integer center and optional
availability. It records the beat maximum, **downbeat value at that same frame**,
independent downbeat maximum, separation between the two peak frames, and beat
peak gain over the center. Peak ties prefer the nearest center, then earliest
frame. The two maxima are never added or treated as independent probabilities.

Annotations select query centers in this audit, so the complete procedure is
truth-assisted despite the observation-only feature function. No latent clock
or automatic event discovery is claimed.

## Labels and controls

- Metre: all 460 ARTBeaT beat labels have unknown downbeat status. The importer's
  false placeholders do not mean non-downbeats. RUBATO uses its official
  measure-aligned downbeat booleans.
- Acoustic presence: unknown for **every** calibration query. A missing model
  peak is not evidence of a missing sound. Neither dataset's beat labels provide
  event-level audible-attack/weakness/silence labels for this experiment.
- Tempo regime: for ARTBeaT, each existing change-point annotation anchors the
  nearest beat (ties earliest); that beat and two on either side form a change
  neighborhood. Other beats use the annotated ramp or constant region. ARTBeaT
  piano-rubato and all RUBATO performances remain `rubato`; per-interval constant
  storage in their truth files does not establish constant performed tempo.
  These regimes are contextual strata, not inferred change-point locations.
- Raw misses and candidate-absent misses replay the historical 70 ms one-to-one
  matching and candidate-nearness rules. These labels do not enter extraction.

The beat window is compared with a same-sized window at the following annotated
interval's midpoint. A pair is excluded if either window is unavailable, the
canonical window overlaps another annotated beat window, or the midpoint window
overlaps any annotated beat window. Closed radius-three windows overlap when
their integer centers differ by at most six frames. Precedence is fixed; final
beats have no following pair. Exclusions remain in the denominator inventory.
The midpoint is an annotation-relative control, **not** an absent-sound label.
Other windows can still share frames across queries; no independent-sample or
product-likelihood assumption is made.

## Measurements

All capture identities, full timelines, unchanged raw observations and default
event reconstructions pass. No dense arrays or event coordinates enter Git.

| Fixed-window diagnostic | ARTBeaT | RUBATO |
| --- | ---: | ---: |
| Annotated queries / full windows | 460 / 448 | 6,726 / 6,720 |
| Raw misses / full missed windows | 128 / 126 | 2,514 / 2,512 |
| Missed windows: positive center / positive peak | 0 / 2 | 62 / 500 |
| Missed canonical wins / eligible midpoint pairs | 77 / 111 (69.37%) | 1,542 / 2,455 (62.81%) |
| Candidate-absent wins / eligible pairs | 3 / 9 (33.33%) | 814 / 1,188 (68.52%) |
| Separated head peaks / full missed windows | 35 / 126 | 733 / 2,512 |

For RUBATO, 438 full missed windows have a nonpositive center but positive peak.
That shows temporal displacement matters; it does not recover 438 beats. A
window maximum can be the flank of a different peak outside the window, fail
the existing local-maximum selection, or belong to another metrical level.
ARTBeaT's 65 full constant-interior missed windows remain entirely nonpositive.
The same is true of its 25 missed ramp windows. Wider evidence is not a universal
substitute for a better sequence interpretation.

The ARTBeaT missed-pair exclusions are four final beats, two unavailable windows
and 11 midpoint overlaps. RUBATO excludes 25 finals, two unavailable, 12 midpoint
overlaps and 20 canonical/neighbor overlaps. Full-window feature statistics and
eligible-pair comparisons intentionally have different denominators.

ARTBeaT missed canonical wins by annotated regime are 32/52 constant interior,
14/17 change neighborhood, 17/25 ramp and 14/17 rubato. Constant-interior and
change-neighborhood missed beat-peak p10/median/p90 values are respectively
`[-8.615, -4.380, -1.498]` and `[-10.324, -7.086, -0.878]`. Their central ranges
overlap substantially. These unequal, correlated groups are not a matched
change-detection experiment and do not justify a local-strength threshold.

Per-track missed-win macro means are 71.37% on 13/15 ARTBeaT tracks and 63.16%
on 25/25 RUBATO tracks. Three contributing ARTBeaT tracks and four RUBATO tracks
fall below one half; all are retained. The earlier narrower-window audit used
different geometry and denominators: comparing its percentages with these is
not a matched algorithm improvement or regression measurement.

RUBATO downbeat ranking uses 2,625 full downbeat and 4,095 non-downbeat windows.
Higher downbeat-at-beat-peak gives pooled/macro-track AUC 0.7864/0.7824; the
independent downbeat maximum gives 0.7869/0.7830. Among raw misses (745/1,767
full class windows), these are 0.7025/0.7218 and 0.7043/0.7215. All 25 tracks
contribute. These small differences are not a reason to select a winner or
claim calibrated downbeat probabilities. ARTBeaT AUC remains null, not zero.

## Decision and next gate

Keep the paired-head window contract and its missing-label/coverage boundaries.
Do not promote a seven-frame maximum, lower the confidence gate, interpret
nonpositive values as acoustic absence, or calibrate using off-grid controls as
silence. Single-window evidence is useful but does not establish the distinction
between a constant clock with omitted events and a genuine tempo transition.
An authored identical-input witness preserves that ambiguity explicitly.

Next freeze a **shared-phase multi-beat context comparison** on the same
captures: a single timing displacement must apply across consecutive windows,
and constant-with-omissions versus changing-clock hypotheses must see the same
observation domain. Keep inferred ticks separate from observed events. This
should test whether temporal context adds discriminating evidence, not add a
mixture of case-specific policies or silently multiply overlapping windows as
independent likelihoods. Any truth-assisted result is still a representation
diagnostic before an observation adapter and automatic decoder acceptance.

No neural-retraining decision follows from this audit alone. Public one-call,
zero-tuning behavior stays unchanged. Eleven tests cover window semantics,
unknown labels, overlap geometry, identical-input ambiguity, ranking, identities,
all cohort/track denominators and privacy. Commands are in the
[parity guide](../parity/README.md#temporally-tolerant-metrical-windows).
