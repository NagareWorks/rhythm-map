# Real-audio tempo-pair renderer audit v1

Registered before execution. This is data/renderer validation, not training,
encoder evidence, a new default, or an independent accuracy test. The previous
four-arm fit and attribution report stay frozen. No hidden-frame interpolation.

## Fixed population and intervention

Three already exposed RUBATO fit recordings: the first suite record in each of
the first three salted fit works in the old direct-clock plan. `protocol.IDS`
pins these IDs, source/truth hashes and per-asset CC0/CC-BY provenance are checked.
No development, ARTBeaT or holdout audio is opened. This small tooling pilot is
not genre coverage or evidence of generalization. Source metadata provenance is
in the existing selection manifest; copying it is not a new independent legal
audit or a claim of certified encoder-training recording disjointness.

Decode PCM16 mono 22050 Hz, take the first 60 seconds. Five profiles have source
boundaries 0/20/40/60 s: identity 1/1/1, slow .8/.8/.8, fast 1.25/1.25/1.25,
local 1/.8/1.25 and reversed local 1/1.25/.8. All profiles process the same three
pieces, including identity, so seams alone do not uniquely identify a speed
change. Keep an additional untouched source prefix. This does not eliminate all
synthetic seam or renderer artifacts; a future learner must pass natural audio
and unseen transformation controls too.

Both existing FFmpeg filters are audited, not selected by downstream BPM scores:
`atempo=RATE` and `rubberband=tempo=RATE:pitch=1`, otherwise defaults. Each piece
gets a separate process. Concatenate actual output; no silent padding, cropping,
crossfade, normalization or duration-based adjustment of requested rate. The
binary hash, full version/build flags, NumPy/Python versions and code closure are
recorded. A build with GPL options remains an **external private research tool**;
no executable/library, rendered music, or new runtime dependency is shipped.
See [FFmpeg filters](https://www.ffmpeg.org/ffmpeg-filters.html#atempo),
[rubberband filter](https://www.ffmpeg.org/ffmpeg-filters.html#rubberband), and
[build-dependent licensing](https://www.ffmpeg.org/legal.html).

## Requested map is not measured alignment

For source interval [a,b] at speed r and nominal output start u:
`source(t) = a + r*(t-u)`, `output(s) = u + (s-a)/r`.
Source beat phase q(s), linearly interpolated between original annotations,
becomes q(source(t)); instantaneous rate becomes r times the source rate.
This preserves a speed step even when it lies inside an original beat interval.
Interpolating the transformed beat list instead would incorrectly smear it.
Downbeat flags stay attached to the same mapped beat identities. No label beyond
bracketing native annotations, no assertion of a uniquely perceived beat level.
Final endpoints have no instantaneous-rate label; step boundaries are right-owned.

The actual renderer may shift transients or change length. Export mapped beat
lists as **nominal audit labels only**, never as verified ground truth. Actual
segment lengths and cumulative drift are separately retained. No quality result
on synthetic clicks certifies the timing of a real recording: music-specific
alignment checks are still required before encoder capture or paired fitting.

## Predeclared witnesses, controls and gates

Run all 2 filters x 5 profiles x 3 authored witnesses: isolated broadband bursts
every .5 s, the same bursts over a quiet 440 Hz tone, and a pure 440 Hz tone.
The detector enumerates all burst groups, checks their complete count, then
compares their ordered times against the requested map. Missing/extra events
fail; they are not dropped for better averages. Pure-tone frequency is measured
on the central two seconds of every piece (0.5 Hz FFT bins).

For every click witness/profile: p95 absolute timing error <=20 ms and maximum
<=40 ms. For all witnesses/profiles: maximum cumulative duration error at any
piece end <=40 ms. For every tone/profile: absolute pitch error <=10 cents in
all three pieces. All required checks must pass for a filter's authored gate;
preserve every failure, never tune a threshold after viewing results. Test
controls include exact map, no-change renderer, pitch-changing speedup, missing
and extra events, delayed output, and rate boundaries inside beat intervals.

Then render the 30 real-music previews (2 x 5 x 3) **even if authored gates fail**,
for explicit tooling diagnosis only. None is eligible for training in v1. Record
unmodified-source, rendered float-PCM and preview-WAV hashes, actual lengths,
clipping checks and provenance. No assertion of music quality from duration alone.

## Budget and rejection

CPU only, one render at a time, two CPU/container limit, 4 GiB RAM, 15 minutes
overall and 60 s per FFmpeg process. Maximum 180 FFmpeg renders: 90 authored and
90 music pieces. No network access in execution; no model import, forward pass,
checkpoint load, backward pass, or optimizer. Output only to a fresh private
directory; refuse overwrite. Exceptions preserve partial evidence and stop.

If authored timing fails, naive requested-factor label conversion is rejected
for that filter/configuration. If it passes, music alignment remains unverified.
Do not infer that tempo learning is impossible, that the model has improved, or
that either filter is generally unsuitable. Next work must resolve measured
audio-to-label alignment (including seams/edges), then encode real transformed
audio with the frozen encoder and test a predeclared relative-rate objective
against natural/no-change controls. No reopening the old optimizer sweep.
