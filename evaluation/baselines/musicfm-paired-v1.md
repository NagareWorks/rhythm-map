# MusicFM: conditional comparison on the original seven pairs

This separate diagnostic asks whether the fixed representation/recurrence rule
distinguishes the original constant/change music windows **conditional on the
supplied clocks**. It does not reopen the rejected composed rule for admission.
The [negative experiment](musicfm-negative-v1.md) remains a failure: none of its
six nonrhythmic clock-family ledgers abstained. That observation answers a
different question from discrimination between clocks in rhythmic music.

## Frozen before this music-feature inspection

- Retain the original seven ARTBeaT pairs, fourteen physical four-second windows,
  all four prefix phases, eight shape/density names and the existing clock
  geometry guard. Keep all forty calibration identities in the report; the
  twenty-six expressive/untyped inputs remain untyped and are not inferred here.
- Reuse the byte-verified **complete** mono traces from the corrected pre-head
  audit. Their 22,050 Hz PCM is bridged to 24,000 Hz with pinned torchaudio
  2.8.0 CPU, `sinc_interp_hann`, width 6, rolloff 0.99. No offset fitting, new
  decode, audio download or selected-window inference. This frontend bridge is
  explicit: the comparison is not an isolated encoder ablation.
- Keep the FMA checkpoint, float32 CPU, layer 7, eight-second contexts,
  four-second hops, midpoint ownership and unpadded tail. All twelve blocks
  and projection execute. Check reference/hooked/repeated features exactly for
  every context, unchanged model/input/RNG and private archive round trips.
- Use the unchanged full-cycle-minus-half-cycle cosine rule at native 25 Hz.
  The pre-existing coordinate bridge retains the same half-open physical
  window; odd 50 Hz starts move the first native center 20 ms inside it.
  Preserve common-support exclusions and cross-context-owner counts.
- Keep the `1e-6` numerical tie budget and `1e-12` zero-norm boundary. Neither
  is confidence. Do not fit layers, distances, thresholds, phase, context,
  source blends or per-track exceptions. The four anchors cancel analytically
  in the lag rule and are not independent votes.

The [protocol](../parity/musicfm-paired-lock-v1.json) pins the adapter, runner,
reused implementations and both prior measured reports. Twelve new model-free
controls passed before any new neural execution. Authored resampling origin,
length, silence, repeatability and long-context hook checks run before neural
music capture. The existing four-GiB Windows job envelope, two Torch threads,
offline worker and minimum resource checks remain; the seven-input run has a
30-minute timeout. Private PCM/features and logs stay outside Git and off the
system drive. These are validation costs, not shipping latency.

## Decision contract

Both native shapes must be correct in both members at every registered phase.
Density resolution additionally requires excluding every rival. A missing or
invalid member rejects the pair, retains its denominator and suppresses orphan
scores. Report every original case, including false changes and erased changes.

A favorable conditional result cannot establish automatic BPM/change detection,
rhythm presence, general music accuracy, independent acceptance or a production
strategy. A failed comparison closes this fixed real-music recurrence test,
not every pretrained representation or training-free method. Training still
requires a separate target/data/budget proposal. No default, API, model pack,
user option, holdout, training or release changes are authorized by this gate.

## Measured result: shape failures remain even conditional on rhythm

The [complete report](../parity/musicfm-paired-v1.json) is byte-identical to the
private completed run, SHA-256
`e71ebd237e5c37c10496d2a58f51ef607788caf3ed2ed5630ccc6ec2f643b203`.
All seven pairs remain eligible under the original geometry guard. All 27
complete-input contexts (81 forwards) pass exact reference/hook/repeat and
full-block/projection checks. Model state matches the earlier negative run,
stays unchanged, and every private feature/owner archive round trip passes.

| ARTBeaT case | Both correct native shapes | All density rivals excluded | Retained native-shape failure |
| --- | --- | --- | --- |
| 06: 150 to 75 | No | No | Constant favors step; true step favors constant |
| 08: 112.5 to 75 | No | No | Constant favors step; true step favors constant |
| 09: 90 to 80 | Yes | No | None at native density |
| 10: 90 to 120 | Yes | No | None at native density |
| 12: 80 to 150 | No | No | Constant favors step |
| 14: 240 to 96 | No | No | Constant favors step |
| 15: 85 to 127.5 | Yes | No | None at native density |

The result is **3/7 native-shape pairs, 0/7 complete density resolutions**.
These are exposed conditional gates, not accuracy percentages or recovered
beat timestamps. Weak/omitted observations were not newly manipulated in this
music replay; the earlier authored pulse-loss checks remain separate evidence.
Compared with the prior pre-head recurrence experiment, case 15 gains native
shape support while 14 loses it. Do not combine per-track winners. Beat This's
prior 2/7 density result is not a paired, frontend-controlled superiority test.

Common native support ranges from 62 to 87 target frames, including 6--30
pre-boundary and 50--69 post-boundary frames. Eleven of fourteen windows use
at least one cross-owner comparison. Unlike the old Beat This captures, these
MusicFM contexts have seams inside the supplied windows. This is a recorded
frontend/context confound, not permission to select favorable seams or claim
the encoder alone caused every failure. Phase rows are analytically redundant;
none of these counts creates additional independent recordings.

The complete diagnostic took 840.27 seconds (14.00 minutes), peak working set
1,952,501,760 bytes (about 1.82 GiB), under the fixed four-GiB limit. It includes
loading, hashing, resampling and 81 validation forwards; it is not the latency
of one production analysis. Twelve new checks preceded inference. Three more
model-free tests (replay refusal and an independently authored continuous-phase
cue with missing pulses) were added during the unchanged run, bringing the new
suite to fifteen before reading its final scores. Retained features reproduce
every paired score, verdict and forty-ID disposition without neural execution.
Three subsequent retained-report regressions bring the new suite to eighteen.

Close this **fixed conditional MusicFM recurrence comparison**. The known
no-rhythm rejection failure is not its only missing prerequisite: false changes,
erased changes and density ambiguity occur even with supplied rhythmic clocks.
Do not restart a layer/distance/threshold/context search on these seven labels.
Nor does failure of this readout prove that all information is absent from the
encoder or that every training-free approach is impossible.

Together with the prior bounded evidence, this supports moving the next
investment decision to a **small learned timing-readout proposal**, not another
presence-only source inventory. Its target must distinguish clock continuation
through missed/weak beats from genuine transitions and perceived beat level;
rhythm presence alone cannot do that. Define independent labels, work/recording
groups, rights, held-out acceptance, runtime budget and stopping criteria first.
No fitting or holdout access starts without the separate authorization required
by the [training decision](../../docs/TRAINING-DECISION.md).

## Reproduction

Run model-free checks with the evaluation NumPy environment:

```text
python -m unittest discover -s evaluation/parity -p test_musicfm_paired.py -v
```

The original neural run uses the separately pinned Windows MusicFM environment
and pre-existing verified assets; the output directory must be new:

```text
python evaluation/parity/musicfm_paired_probe.py --upstream <verified-source> --config <verified-config> --stats <verified-stats> --checkpoint <verified-FMA-file> --trace-dir <corrected-prehead-traces> --output <fresh-private-directory>
```

For later scoring/accounting verification, NumPy alone suffices:

```text
python evaluation/parity/check_musicfm_paired.py --report <measured-report.json> --feature-directory <retained-private-directory>
```

The latter verifies feature hashes, original context ownership, all forty
identities and every paired ledger. It does **not** rerun the neural model or
independently verify its recorded hooked/reference parity from stitched arrays.
