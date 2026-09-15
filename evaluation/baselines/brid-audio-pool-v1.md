# BRID complete mixture-source intake v1

2026-09-15. Follow-up to the [preregistered four-style smoke](brid-audio-intake-v1.md),
not a new training run or an accuracy result. The previous HTTP 429 failure and
its partial bytes remain preserved. No source was selected using model output.

## Acquisition and integrity

The old source-independent ZIP fetcher uses parallel 256 KiB requests. The
resumed source response advertised a request limit of 133, with remaining count
132 and `Retry-After: 60`. Those observed headers support reducing request
frequency, not changing identities or increasing concurrency after throttling.
They are a historical observation, not a promised permanent service quota.

The new BRID-specific intake helper reads the already pinned ZIP directory and
annotation archive locally. It requests one complete, bounded local ZIP-member
range per missing WAV, sequentially, with at least two seconds between request
starts. It verifies HTTP 206, exact Content-Range/length, identity encoding,
local filename/header, deflate completion, extracted size and source CRC before
installing any audio. A source HTTP error stops the run; 429 has no automatic
retry. Existing complete files are checked before reuse; partial writes and
corrupt files are preserved as explicit failures, not overwritten.

Six earlier WAVs are reusable: 0001, 0002, 0003, 0004, 0010 and 0028. All
annotations come from the verified 84,989-byte source archive, with their exact
earlier SHA-256 identities. This avoids repeatedly requesting small annotation
files and the ZIP directory. No whole 944 MB audio archive is needed.

The [complete immutable lock](../datasets/brid-training-mixtures-v1.json) records
93 WAVs plus 186 original annotation files. A SHA-256 created at acquisition
freezes the bytes for future reproduction; the publisher's whole-audio-archive
MD5 is **not** claimed as verified from selected members. The original smoke
hashes remain unchanged. The complete lock retains the source's CC BY 4.0
license and full attribution; the project Apache license does not relicense
third-party data. No audio or original annotations are committed to Git.

## Full-pool result

| Verified property | Result |
| --- | ---: |
| Complete audio / annotation assets | 93 / 186 |
| Audio bytes | 469,990,540 |
| Sum of native PCM durations | 2,663.980635 s (44:23.981) |
| Sum of annotated last-minus-first spans | 2,527.192 s (42:07.192) |
| Beat / downbeat rows | 4,342 / 2,205 |
| Samba / partido alto / samba-enredo / marcha | 41 / 28 / 21 / 3 |
| Out-of-bounds beat / zero-channel / full-scale-sample recordings | 0 / 0 / 0 |
| Exact duplicate native PCM groups | 0 |
| Native / Rust decoder agreement | All 93 within one 22.05 kHz output sample |

All recordings are stereo PCM16 at 44.1 kHz. The total agrees with the source
description's rounded 44:24 duration; it must not be confused with annotated
coverage. This run reused six WAVs and downloaded the other 87 with one range
each, without another HTTP 429. The final helper's offline verification also
recomputed all 279 assets and the same complete lock without any downloads.
The complete lock SHA-256 is
`e19481eb14707777736b0087302924448b3c7bdf6338d004694664a59d8440f3`.

The [offline report](../datasets/brid-audio-pool-v1.json) retains each native
PCM identity, stereo-channel levels, silent/clipped channel indicators,
annotation coverage, all beat-bound findings and Rust-decoder identity/duration.
It also retains every fixed RMS-rise diagnostic, without applying an offset or
treating nearby transients as proof of metrical correctness. The four smoke
records are not a substitute for the other 89.

Structural success is not semantic annotation certification. Published global
`.bpm` values are descriptive source metadata, not framewise targets. Individual
inter-beat changes may contain microtiming or annotation noise, not sustained
tempo changes. No weak beat is deleted, no silent gap is filled with invented
beats, and no automatic half/double-time relabeling is performed at intake.

## Data role and next experiment boundary

All 93 recordings retain `training_candidate_only` and one conservative
`brid-corpus-v1` leakage group because performers and recording context overlap.
Exact PCM duplicate checks cannot prove musical independence. Independent work
count remains unknown; this is not a 93-work test set. The old holdout stays
sealed, and no existing development or exposed diagnostic recording moves to
fit. No features, optimizer steps, model outputs or public default changes are
part of this intake.

Before an actual fit, finish the source-reference alignment review and freeze
the adapter's valid intervals: no extrapolation beyond annotated support, no
global-BPM phase invention, and no treating every interval fluctuation as a
tempo-change event. Then register the old-fit versus expanded-fit comparison
with the same CNN, starting weights, compute budget and absolute-timing gates.
Keep the old rejected experiments frozen; more available data is a falsifiable
new hypothesis, not permission to promote rejected weights or sweep losses.

## Reproduction

The script needs the existing pinned annotation archive and audio-directory
tail from [source intake](training-source-intake-v1.md#reproduction-and-integrity).
Use external directories. Optional reuse roots must contain the same normalized
`audio/NNNN.wav` layout as the smoke or interrupted pool.

```sh
python evaluation/parity/brid_pool_acquire.py \
  --audio-tail /external/brid-audio-tail.bin \
  --annotations /external/brid-annotations.zip \
  --reuse /external/brid-smoke-v1 --reuse /external/interrupted-pool \
  --output /external/brid-mixtures-v1
```

After a source interruption, wait for the service's Retry-After period before
rerunning with the same output directory. Do not remove partial files blindly.
Once a complete lock exists, use the following read-only, offline verification:

```sh
python evaluation/parity/brid_pool_acquire.py --verify-only \
  --audio-tail /external/brid-audio-tail.bin \
  --annotations /external/brid-annotations.zip --output /external/brid-mixtures-v1
for id in $(seq -f '%04g' 1 93); do
  cargo run --locked --profile evaluation -p rhythm-map-eval -- \
    audio-inspect --input /external/brid-mixtures-v1/audio/$id.wav \
    > /external/brid-mixtures-v1/decoder-$id.json
done
python evaluation/parity/brid_pool_audit.py --audio-dir /external/brid-mixtures-v1 \
  --check evaluation/datasets/brid-audio-pool-v1.json
python -m unittest discover -s evaluation/parity -p 'test_brid_pool*.py' -v
```

The lock is also compatible with `cargo xtask dataset-fetch`, but its generic
network path still uses parallel small ranges; prefer the polite source helper
when acquiring BRID from Zenodo. These are research-data tooling choices, not
new end-user rhythm strategies. The Rust library, CLI, FFI and WASM interfaces
remain unchanged, and CI uses authored fixtures without downloading the corpus.
