# BRID reference adaptation and source-expansion design v1

2026-09-15. Follow-up to [complete source acquisition](brid-audio-pool-v1.md).
No downloads, encoder calls or musical optimizer updates. This prepares an
experiment; it does not report an accuracy improvement.

## Reference conversion

The adapter verifies all 279 source assets and writes 93 private NPZ packets:
original beats/positions, 50 Hz reference beat count and validity, and T-1
cell validity. Log2-period targets are derived by the unchanged runtime loss,
not serialized as canonical arrays. Each packet is read back without pickle
and checked by array shape/dtype/little-endian SHA-256. The
[public report](../datasets/brid-reference-v1.json) retains hashes and coverage,
not audio, original annotation files or private machine paths.

| Property | Result |
| --- | ---: |
| Recordings / conservative source groups | 93 / 1 |
| Planned full-audio frame points | 133,247 |
| Bracketed reference frame points | 126,363 |
| Fully supported tempo cells | 126,270 |
| Supported cell duration | 2,525.40 s (42:05.40) |
| Timestamp shifts / invented change labels | 0 / 0 |

Cell duration is smaller than the 2,527.192 s annotated spans because only full
20 ms cells inside half-open reference support qualify. Unknown targets are
NaN, not zero-error samples; both endpoints must be supported. No extrapolation
to audio edges, global-BPM replication, weak-beat deletion or smoothing fills
the difference. Beat-boundary cells integrate native beat-count advancement
rather than choosing the interval containing their center. CPU tests directly
compare this with the unchanged old reference-clock and tempo-loss functions.

Initial cross-platform replay exposed 495 derived log2 values across 34 records
with at most 2.22e-16 absolute difference; every other array matched exactly.
The unpushed candidate and failed check are retained privately. The corrected
packet freezes q and masks before transcendental evaluation, matching the old
training input contract. No annotation or training target was rounded to hide
the difference, and source/reference byte checks remain strict.

Original 1/2 beat positions are preserved. q=0 is an arbitrary count origin,
not a derived downbeat; future phase/bar supervision needs an explicit contract.
IBI variation includes microtiming and possible annotation noise, not a set of
ground-truth tempo-change events. Original CC BY 4.0 attribution stays in the
source fetch lock.

## Alignment recheck

For all 93 records, repeat the fixed +/-150 ms RMS-rise profile on the same
interior beat population. Add equal-peak-normalized per-beat profiles and two
temporal halves. Every old mean-profile result replays. Mean and equal-beat
profiles peak at +5 ms for 92 recordings and +25 ms for 0092. Both time halves
of 0092 also peak at +25 ms; its per-beat winning-offset quartiles are 20 / 25 /
25 ms, with full range -150 to +65 ms.

Thus a few loud attacks alone do not explain 0092. This is **not** independent
musical alignment certification: envelope shape, microtiming, nearby percussion
and annotation convention remain alternatives. No offset, mask or selection
depends on these scores. Keep 0092 in the frozen population and complete an
explicit source-alignment disposition before fitting. An actual annotation
defect requires a versioned correction, not an invisible shift or exclusion.

## Matched comparison and remaining work

The [protocol](../../experiments/tempo_source_expansion/PROTOCOL-v1.md) and
[100-update schedule](../../experiments/tempo_source_expansion/plan-v1.json)
freeze the same CNN, starting weights, four old shared recordings plus one
extra slot per update, old pair sampling and FIT-only teacher. Control extra
slots use work-balanced old FIT; primary slots cycle through all 93 BRID IDs
once, then seven twice. BRID contributes 10% of natural-supervision weight,
not 93 independent-work shares. No old development/diagnostic role changes.

Both arms have five natural forwards per update, versus four in the previous
closed experiment: this is a new matched control, not a promised bit-exact
replay of the old trajectory. Equal slot/update budgets do not imply equal
FLOPs or elapsed time; BRID clips are shorter. Wall caps and absolute-timing
gates remain unchanged. Existing frozen Beat This pre-head features are reused
as the representation; no new encoder or user-facing strategy is introduced.

Design freeze is not execution admission. Source disposition, all-93 feature
capture/replay with exact frame geometry, old packet identity replay and the
executable runner closure still precede fitting. Prepared references are not
trained weights; exposed development/schedules cannot prove independent
generalization. No public defaults or release artifacts change.

## Reproduction

Use new external output directories on a data drive:

```sh
python evaluation/parity/brid_reference.py \
  --audio-dir /external/brid-mixtures-v1 --output /external/brid-reference-v1
python evaluation/parity/brid_reference.py \
  --audio-dir /external/brid-mixtures-v1 --check evaluation/datasets/brid-reference-v1.json
python -m experiments.tempo_source_expansion.design \
  --check experiments/tempo_source_expansion/plan-v1.json
python -m unittest discover -s evaluation/parity -p test_brid_reference.py -v
python -m unittest discover -s evaluation/parity -p test_tempo_source_design.py -v
python -m unittest discover -s experiments/tempo_source_expansion -p 'test_*.py' -v
```

The first two commands verify the already acquired corpus. CI uses authored
fixtures and retained identities without downloading music. Only the final
test command requires CPU PyTorch; adapter/schedule generation needs NumPy.
