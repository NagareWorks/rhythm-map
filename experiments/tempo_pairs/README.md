# Actual audio tempo pairs: renderer audit, not a fitted model

The [predeclared protocol](PROTOCOL-v1.md) and [complete scalar result](results-v1.json)
test real PCM transformation before using its requested time map as supervision.
This follows the failed local-speed response in the
[frozen tempo attribution](../tempo_attribution/README.md); it does not reopen
that fit or change any old metric, threshold, data role or product strategy.

## What actually ran

Two existing FFmpeg filters, five fixed profiles (identity, slower, faster and
both local-step orders), three authored signals and three already exposed fit
recordings. All 180 piece renders completed on CPU. There are 30 authored checks
and 30 private music previews with nominal mapped beats/downbeat identities.
The unchanged source prefixes, licenses, tool/code identities, actual segment
lengths and output hashes are retained. No encoder, neural weights, optimizer,
development/holdout audio, shipping code or release was involved.

| Authored result | `atempo` | `rubberband` |
| --- | ---: | ---: |
| Signal/profile checks passing | 9/15 | 15/15 |
| Worst p95 click timing error | 90.795 ms | 9.201 ms |
| Worst individual click timing error | 91.366 ms | 10.040 ms |
| Worst cumulative segment-duration error | 92.789 ms | 0 ms |

All click counts were complete (118 each). Pure-tone tests stay at 440 Hz to the
0.5 Hz FFT-bin resolution; this does not certify pitch/quality on arbitrary music.
The unchanged-rate `atempo` isolated-click witness already accumulates about
69.7 ms drift across the three separately processed pieces. This is a finding
about **this build/configuration/pipeline**, not all time stretching or all audio.
The real previews' largest cumulative duration deviations are 44.762 ms and
0 ms respectively. Exact output duration does not establish internal alignment.

## Consequence for training

Reject naive `atempo` + requested-factor labels for this pipeline. Rubber Band
passes the authored gate and is the next renderer to validate on real music;
this is **not** evidence that the BPM head has learned anything. All 30 music
previews, including Rubber Band, remain `training_eligible: false`: their actual
transient/annotation alignment has not yet been measured. No permissive fallback
admits failed or uncertain examples.

Requested source-time maps own phase/rate semantics independently of a renderer.
For a speed r, BPM must scale by r while beat identities move by the inverse
time map. A speed step inside one original beat interval stays a step, rather
than being smeared by interpolation of transformed beat timestamps. Actual
rendered piece lengths are logged separately, never used to conceal timing drift.
Identity profiles retain the same seams and an untouched prefix is kept as a
natural control. These controls do not prove synthetic artifacts harmless.

Next: check music-specific alignment and uncertain regions, then capture real
transformed-audio frozen-encoder evidence and register a relative-rate learning
test with natural and no-change controls. Do not substitute cached hidden-axis
stretching, claim independent accuracy, or sweep old losses again.

## License and reproduction boundaries

Per-asset CC0/CC-BY provenance comes from the pinned RUBATO selection; attribution
and modification notices accompany private nominal-label files. Source bytes
are checked, not downloaded. This reuses the existing license audit rather than
claiming a new legal opinion. The audited FFmpeg build enables GPL components;
it is an external **private research tool**, not linked, vendored or shipped with
the Rust crates, CLI, GUI, DLL or WASM. FFmpeg's license depends on enabled
components ([official licensing](https://www.ffmpeg.org/legal.html)); future
distribution of any augmentation tooling/assets requires its own compliance
review. Do not mistake GPL for a non-commercial restriction.

```sh
python -m unittest discover -s experiments/tempo_pairs -p 'test_*.py' -v
# In an isolated CPU-only environment; use a fresh private output directory.
timeout --signal=TERM --kill-after=10s 900s python -m experiments.tempo_pairs.run \
  --audio-root PINNED_RUBATO_ROOT --ffmpeg /usr/bin/ffmpeg \
  --ffmpeg-sha256 36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45 \
  --output NEW_PRIVATE_OUTPUT
```

The report records full build flags. The private execution additionally pins its
container image and resource limits; a matching executable hash alone does not
pin dynamically linked libraries on a different machine. Public CI tests the
time-map/witness logic and recomputes the full recorded scalar gates without
audio downloads or FFmpeg installation. Private validation verifies the actual
audio, mapped-label files and checksums. No generated music is committed.
