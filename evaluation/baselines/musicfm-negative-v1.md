# MusicFM: fixed neural negatives and robustness controls

This is a prospective neural check of five inputs whose **activity-only** results
were already exposed in the [availability composition](musicfm-availability-v1.md).
It is not an independent music evaluation, a new no-rhythm classifier, or a
revision of either previous report. No model, layer, score or threshold is fitted.

## Fixed before new features

The [protocol](../parity/musicfm-negative-lock-v1.json) byte-pins all five PCM
identities from the previous report, the availability helper/report and its
transitive core, temporal, serialization, checkpoint and runtime contracts.
Nineteen model-free controls passed before the new neural run.

| Input | Previously fixed construction | Required neural outcome |
| --- | --- | --- |
| `quiet-gain` | Constant clean PCM multiplied by 2^-16 | Retain native constant shape and beat density |
| `short-rest` | Zero the complete cue/pulse input for 0.5 s | Retain native constant shape and beat density |
| `steady-tone` | Constant-amplitude 440 Hz sine | No preferred supplied clock |
| `dc` | Constant float32 0.125 | No preferred supplied clock |
| `noise` | Fixed PCG64-seed-142 uniform noise | No preferred supplied clock |

Each 12.01-second input must first pass the **unchanged native availability**
precondition in both clock families. A different availability outcome is a
contract failure before model access, not a way to pass a neural negative. The
other previous activity-only controls are excluded by their original low-activity
rule, not selected after examining neural output; they are not newly inferred.

Each complete input uses three fixed contexts: 8-second contexts with 4-second
hops and unpadded tail. Fixed midpoint ownership produces 301 native 25 Hz tokens.
Every context executes the reference, hooked and repeated reference pass: fifteen
contexts and forty-five forwards, all twelve blocks and projection executed,
layer-7 features bit-exact between passes. The complete model state must match
the previous temporal run before and after extraction. PCM, metadata, eval mode,
RNG, context ownership and private archive round trips are checked separately.

Only the same physical [4, 8) second window is scored, using all previously
required centers, raw float64 endpoint interpolation, normalization and the
unchanged full-cycle-minus-half-cycle cosine rule. Periods and the candidate
change at 6 s are **oracle-supplied**. This does not discover BPM or a change point.

The two constant controls each pair with the two frozen previous change score
ledgers: four pairs. Both shape and all density rivals must resolve correctly.
The old change scores are reused evidence, not four independent music examples.
Exact feature/score equality across gain/rest transformations is not required.

Each of three nonrhythmic inputs is tested in both fast/slow clock families:
six negative ledgers. Unavailable numerical support is explicit abstention;
otherwise the maximum-minus-minimum score among **all eight clocks** must be at
most the existing 1e-6 numerical tie budget. A winning constant clock still fails
this negative: detecting "constant" is not detecting "no rhythm". A tie is not a
calibrated confidence and a non-tie is not proof of a musical beat.

The first fidelity/resource failure stops extraction and cannot count as success.
After valid extraction, all five cases are judged; any negative or robustness
failure closes this composed rule before music. No failed case/family is omitted,
no alternate noise seed tried, and no gain/threshold/metric/context/layer changed.

## Measured result

The [complete measured report](../parity/musicfm-negative-v1.json) retains a
**failed composed-rule decision**, separately from successful extraction:

- All five native availability checks passed in both clock families.
- All 45 forwards passed exact reference/hook/repeat checks; all fifteen context
  ledgers and five archive round trips passed, with unchanged complete model state.
- All four gain/rest-to-change pairs passed both native shape and density checks.
- None of the six negative clock-family ledgers abstained (0/6; three cases).

| Nonrhythmic input | Fast-family score spread | Slow-family score spread | Fixed budget |
| --- | ---: | ---: | ---: |
| Steady tone | 0.004242456925 | 0.004242456925 | 0.000001 |
| DC | 0.000431530585 | 0.000344164945 | 0.000001 |
| Seeded noise | 0.009138275068 | 0.008181500813 | 0.000001 |

These are raw differences between supplied-clock scores, **not probabilities**.
No null hypothesis is present in the eight-clock ledger. All negative rows were
numerically eligible, so none was hidden by missing support or the activity gate.
The measured decision is `close_composed_rule_before_music`. The earlier known
silence exclusion remains valid but does not generalize to "no audible rhythm".

The run took 468.04 seconds (7.80 minutes), with a 1,941,176,320-byte peak working
set under the four-GiB limit. This includes forty-five diagnostic forwards,
construction/loading, state hashing and native activity exports; it is not a
single production analysis latency or a GPU performance comparison. Nineteen
model-free controls preceded inference; three retained-report regressions were
added afterward (22 tests in the new suite).

## Consequence and next boundary

Keep the positive conditional discrimination and the negative failure together.
Activity availability plus relative representation recurrence is not sufficient
evidence that an input has a beat. The gain/rest results also mean this failure
cannot be summarized as "the model loses rhythm whenever a pulse is weak".

Do not sweep the 1e-6 numerical tie budget into a fitted confidence cutoff, choose
another noise seed, or promote only the four successful pairs. The current rule
and both preceding reports remain closed/preserved as recorded. Freeze a separate
audit of reusable rhythm-presence/rejection evidence before new features or any
music acceptance: it must reject these nonrhythmic controls while retaining the
gain/rest cases, without oracle tempo labels supplying the rejection decision.
First inventory what existing licensed model observations actually provide, and
reject mere renaming of an activity or pairwise-rank score as confidence. If no
reusable evidence survives such a fixed gate, a learned presence/confidence head
becomes a concrete next candidate; this single failed composition does not yet
prove training necessary or justify a larger model/architecture sweep.

The inventory must account for the already rejected background-copy/dropout
formulation and the existing prior-only family-preference counterexamples in the
[roadmap](../../ROADMAP.md). They are not new untested confidence hypotheses;
do not restart those same searches under another name.

## Reproduction and retained boundaries

```text
python -m unittest discover -s evaluation/parity -p test_musicfm_negative.py -v
python evaluation/parity/musicfm_negative.py --upstream <verified-source> --config <verified-config> --stats <verified-stats> --checkpoint <verified-FMA-file> --native-executable <built-activity-exporter> --output <fresh-private-directory>
```

Use the previous hash-pinned Windows CPU MusicFM runtime and native exporter.
The worker keeps the four-GiB job limit, two Torch threads, physical-memory,
commit-memory and disk-space preconditions, and offline guard. The parent timeout
is 900 seconds, a resource boundary rather than a product latency target.
Private PCM/features/checkpoint stay outside Git and off the Windows system drive.

Keep all forty calibration identities, seven exposed ARTBeaT pairs, untyped
RUBATO and sealed holdout unchanged. No product/API/default change, user strategy,
training, model pack distribution, commercial clearance or release is introduced.
