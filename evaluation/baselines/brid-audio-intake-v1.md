# BRID source-audio smoke v1

2026-09-15. Follow-up to the frozen [source intake](training-source-intake-v1.md).
This completes acquisition and structural inspection of the four preregistered
styles; it does **not** admit a training pool or certify musical alignment.
No old experiment is reopened and no product option or model default changes.

## Verified result

The [immutable fetch lock](../datasets/brid-training-smoke-v1.json) contains the
four source WAVs and their eight original annotation files. Its SHA-256 is
`e7f495df75c1058f8d2c43f958fb3260aadaf2b78459bbd8c5daf0e2f838f158`.
The [offline audit](../datasets/brid-audio-smoke-v1.json) retains all four IDs
chosen before model output: 0001, 0003, 0010, 0028. No result-based selection.

| ID / source style | Native duration (s) | Beats / downbeats | Leading / trailing span without beat annotations (s) |
| --- | ---: | ---: | ---: |
| 0001 / samba | 36.000000 | 48 / 24 | 0.025 / 0.720 |
| 0003 / partido alto | 40.600000 | 61 / 31 | 0.522 / 0.643 |
| 0010 / samba-enredo | 28.215034 | 59 / 30 | 0.247 / 1.184034 |
| 0028 / marcha | 18.429070 | 41 / 21 | 0.248 / 0.654070 |

The source files total 21,743,028 audio bytes, 123.244104 seconds of native PCM,
209 beats and 106 downbeats. Annotated last-minus-first spans total 119.001
seconds, not the recording duration. All four are stereo PCM16 at 44.1 kHz.
All twelve file sizes and SHA-256 identities verify, source audio CRCs agree
with the frozen upstream ZIP directory, and annotation bytes agree with the
earlier inventory. There are no out-of-bounds beats, all-zero recordings or
full-scale PCM samples in this slice. Nothing is padded, shifted or relabeled.

The production Rust audio decoder independently reports the same input hashes
and sizes. Its 22.05 kHz decoded duration differs from the native duration by
at most one output sample; only 0010 requires sample-count rounding. This is a
decoder compatibility check, not neural inference.

## What the acoustic diagnostics do and do not establish

The source-only checker computes multichannel RMS energy in approximately 5 ms
blocks, positive energy rises, distance from annotated beats to nearby rise
peaks, and a fixed -150 to +150 ms offset profile. Every offset uses the same
interior beats, so changing edge coverage cannot win the comparison.

All four profiles peak at +5 ms. Their median nearest-rise-peak distances are
about 3.2-6.0 ms. These are descriptive, coarse-envelope measurements, **not a
measured annotation error, a correction to apply, or an accuracy gate**. Dense
percussion can put an acoustic peak near the wrong metrical position; weak or
syncopated beats need not have an onset. The test cannot certify duple-meter
phase, half/double-time interpretation, sustained tempo movement or every beat.
The full profile and unannotated edge spans are retained in the report.

Semantic annotation review remains pending. The checker neither turns global
`.bpm` into framewise truth nor derives change points from individual intervals.
The frozen `brid-corpus-v1` leakage group and training-candidate-only role remain;
independent work count is unknown and admitted training recordings remain zero.

## Full-pool acquisition attempt: retained failure

After the four-style structural smoke passed, the same source-only harness
attempted all 93 inventory mixtures, in numeric order, without model scores.
It acquired audio and annotation files for 0001-0004, then stopped on 0005:
HTTP 429 persisted through the existing fetcher's three range attempts for
bytes 93,746,934-94,009,077 of the audio archive. The twelve completed files
remain outside Git. This interrupted run did **not** produce a full-pool lock,
and its partial files are not claimed as a completed or admitted dataset.

No immediate restart, proxy rotation or concurrency increase was used after
rate limiting. The completed smoke lock and report are unaffected. Continue
full-pool acquisition only after source rate limits permit it, preserving and
verifying completed bytes before any reuse. A partial success must not become
a favorable four-record training pool. Before fitting, finish full-pool byte
verification and annotation review, freeze the audio/reference adapter, and
register the matched old-fit versus expanded-fit comparison described in the
source-intake protocol. Do not repeat rejected loss/epoch sweeps while waiting.

## Reproduction

Use external directories, not the checkout. No audio or original annotations
are vendored. The [BRID release](https://zenodo.org/records/14051323) remains
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); preserve attribution
and modification notices. The lock carries the full source attribution. The
project's Apache license does not relicense these third-party assets.

For ordinary reproduction of the completed smoke, prefer the immutable lock:

```sh
cargo xtask dataset-fetch \
  --manifest evaluation/datasets/brid-training-smoke-v1.json \
  --output /external/brid-smoke-v1 --with-annotations
for id in 0001 0003 0010 0028; do
  cargo run --locked --profile evaluation -p rhythm-map-eval -- \
    audio-inspect --input /external/brid-smoke-v1/audio/$id.wav \
    > /external/brid-smoke-v1/decoder-$id.json
done
python evaluation/parity/brid_audio_audit.py --audio-dir /external/brid-smoke-v1 \
  --check evaluation/datasets/brid-audio-smoke-v1.json
python -m unittest discover -s evaluation/parity -p test_brid_audio_audit.py -v
```

The internal source-intake harness can generate the initial lock, or request
the full cohort after the hash-pinned smoke gate. It requires a **fresh** output
directory and refuses existing directories; it is not an automatic resume tool.
Do not repeatedly rerun it after a rate-limit failure.

```sh
cargo run --locked --profile evaluation -p rhythm-map-eval --example brid_acquire -- \
  --output /external/new-brid-smoke
# Only when source request limits permit full-pool acquisition:
cargo run --locked --profile evaluation -p rhythm-map-eval --example brid_acquire -- \
  --all-mixtures --output /external/new-brid-mixtures
```

Both selections reuse the existing bounded ZIP fetcher. `--all-mixtures` is an
internal source-acquisition choice, not an end-user rhythm strategy. CI tests
authored PCM, integrity failures, bounds, silence, decoder identity/tolerance,
unchanged annotations and frozen selection; it never downloads the corpus or
runs a musical training job.
