# Pre-head recurrence discrimination gate v1

This is one fixed use of the [validated normalized pre-head sequence](prehead-capture-v1.md),
not an automatic tempo estimator, probability model, layer search or product
option. The comparison and 18 authored/model-free controls were initially frozen
before new real feature inspection; two implementation corrections below were
then checked with 20 pre-run controls before a full corrected replay. Existing
published reports, checkpoints and default behavior are unchanged.

## What the score asks

For a supplied phase-continuous clock `u(t)`, define `q1` as the time one
candidate cycle before `t`, and `qhalf` as half a cycle before `t`. Evaluate:

```text
score(clock) = mean_t [cosine(h(t), h(q1)) - cosine(h(t), h(qhalf))]
```

`h` is the original 512-dimensional normalized pre-head vector. Interpolate raw
feature vectors linearly in float64 at rational query coordinates, then unit
normalize for cosine. No channel selection, fitted projection, PCA, per-track
normalization fit, score blending or learned distance is used. Higher score is
declared favorable before looking at results. The `1e-6` difference budget only
defines numerical ties; it is not calibrated confidence or a user parameter.

The subtraction matters: a repeating pattern can match both one and two cycles
ago. Rewarding recurrence alone would systematically leave integer multiples
unpenalized. Contrasting the intervening half-cycle tests a specific periodic
structure, but still cannot tell whether that structure represents a beat,
subdivision, bar, timbre pattern or phrasing. It is a hypothesis about evidence,
not an inference that the learned representation contains a musical clock.

## Frozen geometry and support

- Reuse the same seven constant/change pairs, fourteen 200-frame windows,
  nominal pre/post periods, transferred boundary and four supplied prefix phases
  from the [oracle-period experiment](oracle-period-paired-v1.md).
- Keep constant and step shapes at native, both half phases and double density.
  No boundary, period, phase, window or source is selected from feature scores.
- Scoring does not receive truth or role labels, but the supplied clocks and
  selected windows remain **oracle-assisted**. This is not label-free discovery
  of tempo or change locations.
- Targets are integer frames inside the original half-open four-second window.
  Every hypothesis uses the intersection of all eight clocks' interpolation
  support. No interpolation endpoint may leave the window. Unsupported edges
  are counted rather than extrapolated, wrapped or replaced by zero.
- A required feature with norm at most `1e-12` makes the whole window unavailable.
  Any unavailable member rejects the pair and suppresses both score ledgers;
  the original pair remains in the denominator and is never replaced.
- Report pre/post-boundary support counts and per-clock cross-owner counts.
  Keep original complete-recording chunk ownership; do not re-infer a crop or
  silently discard a seam. Very few post-boundary frames are weak coverage,
  not strong evidence merely because a numerical ordering exists.
- Retain the old same-ordinal, two-direction 60ms geometry guard. A failed
  geometry condition cannot support a discrimination verdict.

For this lag-based rule, phase offsets cancel analytically, even across the
continuous change. The two half-phase alternatives therefore tie by design,
and all four prefix anchors produce the same query coordinates. They remain in
the ledger for contract continuity, but are not four votes or four independent
successes. Eight competing names likewise do not represent eight independent
pieces of evidence.

Correct native shape must beat the opposite native shape in both windows at
every registered phase. Density resolution additionally requires beating all
seven rivals. An eligible failure forbids promotion of **this fixed recurrence
rule**; it does not prove every pretrained representation or training-free
algorithm incapable. Positive calibration would still require an observation-only
decoder and independent work-disjoint acceptance.

## Authored controls and capture identity

The artificial vectors explicitly encode a continuous phase circle in two
channels and a small pulse in a third, with the other 509 channels zero. A
25-frame pre-period changes to either 12.5 or 40 frames. Alternating pulse
omissions and one-eighth pulse amplitudes retain the independently authored
phase cue. Clean and these weakened-pulse cases pass both shape and density
checks. This establishes algebraic behavior **if** that cue exists, not that a
real encoder preserves it through weak or absent beats.

Flat vectors give ties, zero vectors give unavailable evidence. Half/double
feature trajectories also demonstrate the limitation: identical observations
cannot distinguish two externally assigned musical beat units. The score can
prefer the wrong semantic unit; these are counterexamples, not successful music
classification. Other controls cover exact interpolation, phase cancellation,
partial edges, immutable inputs, owner seams, geometry reuse and no orphan wins.

The all-ID check retains 40 calibration identities: seven selected pairs,
26 expressive/untyped cases (25 RUBATO plus ARTBeaT's piano rubato), and seven
other ARTBeaT cases without a registered pair. It does not open their hidden
features or relabel expressive timing as hard changes. An initial test assertion
mistakenly counted only the 25 RUBATO entries; it was corrected from the unchanged
annotation protocol before freeze, without altering selection or comparison.

Before hidden-feature inference, newly traced complete PCM, sample count, frame
count, dense heads and observation contents must equal the frozen older
captures. The only provenance translation is explicit and exact: the old dense
exporter's `beat_this.onnx`/null source becomes the trace exporter's locked model
pack ID/hash; both spellings are verified, not dropped without checks. The first
optimized native replay stopped at that metadata difference before any hidden
capture; this input-adapter correction was refrozen with unchanged score,
population and numerical budgets. Trace-embedded implementation hashes must
match current source. The
same pinned model, frontend, hook checks, head reconstruction, RTen tolerances,
event identities and private archive round trips from the capture gate apply.
No raw PCM, feature arrays or event timestamps enter the public report.

## Measured result: do not promote this rule

Dedicated pre-push review found two implementation defects after the initial
private run. The right interpolation endpoint was multiplied in float32 before
addition to the float64 left term; a cancellation could become a small nonzero
vector and then a false direction. Both endpoints now cast before multiplication,
as the original mathematical contract required. The runner also now verifies
and records every dependency pinned by the reused oracle, including its geometry
helper. New cancellation and dependency-mutation regressions cover the findings.
Neither fix changes the score definition, direction, numerical budgets or sample
selection. Initial private reports and the unpushed candidate bundle are retained;
their successful replay does not count as final validation of the corrected code.
The complete corrected replay retains identical feature hashes and every paired
verdict; the maximum score change from the initial implementation is about
`2.44e-10`. The initial private report remains identified by SHA-256
`4a082aecaafd391552e215421187ef6de8a99bfb2c083d65b76a0d757d392657`.

All seven original pairs passed the frozen geometry guard and remained eligible.
Complete PCM, frames, dense heads and non-provenance observation fields were
unchanged; all seven feature captures passed hooked/unhooked, reconstruction,
frontend, old-head and ordered-event checks. Model state stayed unchanged. The
[complete aggregate report](../parity/prehead-recurrence-v1.json) preserves all
four phases and eight clocks for both roles, plus all 40 input identities.
Its SHA-256 is `81b3d8a98c66d9362bf073187583c44d9f21b9f695bf178261ac24f6c5ee1c09`;
the public file is byte-identical to the corrected private run's report. Three
additional report regressions bring the new model-free test count to 23.

| ARTBeaT case | Both correct native shapes, all phases | All density rivals excluded |
| --- | --- | --- |
| 06: 150 to 75 | No: true step favors constant | No |
| 08: 112.5 to 75 | No: true step favors constant | No |
| 09: 90 to 80 | Yes | Yes |
| 10: 90 to 120 | Yes | Yes |
| 12: 80 to 150 | No: constant control favors step over native constant | No |
| 14: 240 to 96 | Yes | No: constant control favors half density |
| 15: 85 to 127.5 | No: true step favors constant | No |

Thus shape support is 3/7 and density resolution is 2/7. These are seven paired
calibration gates, **not accuracy percentages**; the four phase rows are
analytically redundant for this metric. Common support ranges from 125 to 175
frames per window, with 13 to 60 pre-boundary frames and 99 to 138 post-boundary
frames. All selected recordings use one original chunk, so their cross-owner
counts are zero; seam behavior is only authored coverage here.

The old count/sign test resolved density in 0/7 pairs; this different evidence
comparison resolves 09 and 10 but still has false changes and erased changes.
That is a useful bounded contrast, not a general superiority claim or permission
to mix per-track winners. Higher-dimensional features do not automatically
produce robust musical timing. No direction, threshold, layer or metric changed
after this result; close this fixed recurrence rule for default adoption.

The corrected capture-and-comparison portion took 64.91 seconds on CPU, excluding native
trace generation, imports, verification and model initialization. It includes
duplicate hook-parity forwards and is not shipping latency. An earlier debug
exporter attempt was stopped before completing its first native trace; using an
optimized build addressed that diagnostic execution issue, not an algorithmic
gain. Failure logs remain private and neither incomplete attempt is a music
result or a successful run.

Next inspect the one reserved MusicFM fallback's exact weight/license/runtime
boundary and input compatibility before any download/inference. Do not run an
unbounded layer/distance search on these labels. This failure alone does not
establish training as necessary. No production behavior, API, model pack, user
parameter or release changed.

## Reproduction

Build the native diagnostic exporter with optimizations; a debug build is not
a representative inference workload. All large output goes to a private path:

```bash
cargo build --locked --release -p rhythm-map-eval --example beat_this_trace
python -m unittest discover -s evaluation/parity -p test_prehead_recurrence.py -v
python evaluation/parity/prehead_recurrence_audit.py \
  --trace-executable /data/build/release/examples/beat_this_trace \
  --audio-dir /data/artbeat-v1 --model-dir /data/beat-this-full-v1 \
  --upstream /data/reference/beat_this --checkpoint /data/reference/final0.ckpt \
  --artbeat-evidence /data/candidate-evidence-v1.private.json \
  --dense-captures /data/dense-artbeat-v1 \
  --output-dir /data/private/prehead-recurrence-new
```

The output directory must be new, outside Git and, on Windows, off the system
drive. No automatic model acquisition or dependency installation occurs. A
failure leaves no completion report; partial artifacts are not accepted results.
