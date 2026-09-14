# Retained real-music alignment audit v1

Freeze this protocol/code before opening music PCM. Read only the previous
`tempo_pairs/results-v1.json` and its private final WAV/nominal-label files.
Verify every opened artifact against that report and verify the old source
closure. Use all 3 fixed fit recordings x 5 Rubber Band profiles (15 pairs),
including processed identity and untouched source prefixes. Do not use atempo,
development, ARTBeaT, holdout, new downloads, FFT/model parameter searches, a new
renderer, neural inference or training. Do not modify the old renderer outcome.

## Measurement, controls and fixed numbers

This is a new authored diagnostic, not an established probability/calibration
model. Extract unpadded Hann STFTs at 512 and 2048 samples, hop 220, SR 22050 Hz.
Pool power over 32 geometric FFT-bin edge intervals (rounded duplicate edges
removed, DC omitted), then log1p(100 * RMS amplitude). For each output center,
compare source features at requested source(t) with observed features at t+lag.
The context is -1..+1 s in 20 ms steps; lags -250..+250 ms in 10 ms steps.
Subtract each band's temporal mean; use normalized flattened correlation.

Both resolutions must qualify independently: centered RMS spread >=.005,
correlation >=.8, best-vs-other-shifts margin >=.02 outside +/-40 ms of the best
lag, and best must not lie at the search boundary. Their best lags must agree
within 20 ms, and both must be within +/-20 ms for `aligned`. Retain disagreement,
mismatch, ambiguity, uninformative signal and unsupported edges separately.
Never extrapolate features, interpret low spread as silence/beat absence, or
claim that the best correlation locates each perceptual beat with this precision.
Two views share the same audio and are not independent statistical replications.

Measure a fixed 1 s grid at centers .5,1.5,... before duration, plus every nominal
mapped beat (same measurement, separately reported population). Beat labels
choose measurement positions only; they do not enter spectral features, matching
scores or time-map construction. Profile source boundaries are fixed wall times.
Within 2 s of a track edge classify `edge`; otherwise within 1.25 s of an internal
render seam classify `seam`; all remaining centers are `interior`. Keep all points
in their denominators, including abstentions and unmeasurable edge contexts.

The real negative control is the same rendered PCM shifted by exactly 2646
samples (+120 ms), zero-padded at the start and cropped back to original length.
Re-extract its STFTs from actual delayed PCM; do not shift cached feature axes.
`feature_capture_supported` requires natural alignment AND a qualified displaced
control with each resolution's lag increased by 120 +/-20 ms. A natural match
alone cannot admit a point. This control validates local delay sensitivity, not
all transformation artifacts or sample-independent timing truth.

For each nonidentity profile also compare the *untransformed* source PCM against
the requested changed-rate map on the same grid. Only centers whose full 2 s
context is within a non-unit-rate segment count toward the changed-context
negative population; retain all other centers in the report too. In each of the
12 changed-profile pairs, require at least 10 physically supported changed-context
points and <=5% aligned among those points. Unsupported points stay in coverage
counts, but must not improve this false-alignment denominator. This is a wrong-map negative control,
not an alternate model or a claim that unaltered music lacks tempo variation.

Authored tests require correct known identity/delay/gain matches, rejection of
wrong direction/map, lack of variation, repeated ambiguous events, search-limit
and feature-support errors, mixed-resolution disagreement, negative-control
failure and no input mutation. Also exercise STFT-to-match on authored changing
tone bursts and an actual delayed waveform. No music-derived threshold tuning.

## Decision, exposure and budget

A pair passes the exploratory interior gate only when >=80% of interior grid
points AND >=80% of interior beat points are feature-capture-supported. The
global negative gate above must pass too. Publish complete per-point and
per-region results, not only successful points or qualified-lag quantiles.
An empty population cannot pass. A supported point is evidence for an aligned
local context, not certification of an entire interval, whole recording, source
annotation correctness, pitch quality or perceived beat level. Do not interpolate
supported points into a dense mask or treat an unmeasured edge as accepted.

No existing nominal-label training_eligible flag is changed. If the interior gate
fails, report the actual unsupported/conflicting populations and close this
fixed diagnostic without loosening thresholds. If it passes, these points may
support a separately scoped frozen-encoder/paired-objective experiment; it still
needs natural controls and its own data/compute contract. No automatic fit,
accuracy claim, public strategy, default change, or release.

CPU-only NumPy, at most 15 pairs and three spectral extractions per pair plus
reused source spectra; 15 minutes wall budget, streamed small FFT batches, no
network, no new runtime library. Outputs go to a fresh private directory. Require
an outer 900 s timeout when executing remotely, preserve partial report on a
Python exception, and record runtime, source hashes and every input identity.
