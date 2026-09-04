# Time-exposure prior intervention, v1

This is a controlled evaluation experiment, not a product strategy or a trained
model. The frozen [joint-clock reference](joint-clock-v1.md) is unchanged. The
new example copies its graph, emissions, meter prior, frame availability, null
reference and edge search domain; only the duration prior is replaced, including
its terminal survival factor. No truth paths enter decoding. All 15 control
head pairs are byte-identical to the preceding experiment.

## A duration cost measured in time

The old row-normalized prior assigned constant-period cells the probability
`1/Z(p)`, so a longer list of beats accumulated more `-log Z(p)` terms. The
previous decomposition exposed a -17.49 duration-prior disadvantage for the
weak genuine doubling, even though its beat evidence was +4.85 better.

Let `P = {10,...,75}` frames, `a = ln(100)`, and:

```
w(p,q) = exp(-a * abs(log2(p/q)))
Zoff(p) = sum_{q != p} w(p,q)
Z(p) = 1 + Zoff(p)
r = mean_{p in P} log(Z(p)) / p

T(p,p) = exp(-r*p)
T(p,q) = (1-exp(-r*p)) * w(p,q)/Zoff(p), q != p
```

Each transition row sums to one. Its self-survival probability depends on the
elapsed duration, not how often the program visits a beat state. `r` preserves
the old prior's uniform-domain average log-survival cost per frame; it is not
chosen against a tempo, recording, control outcome, or target accuracy. The old
distance shape and its `ln(100)` factor are held fixed. A one-duration domain
is handled as deterministic survival with no jumps.

Here `r = 0.06385311418611306` per frame, or about `3.1926557093` per second at
50 Hz. **This inherited reference rate is not an estimate of how often music
changes tempo**, nor a recommended production prior. Preserving it makes this
a one-factor intervention rather than another weight sweep.

For durations `p1,...,pk`, the duration factor is:

```
(1/|P|) * product_{i<k} T(pi,p{i+1}) * exp(-r*pk)
```

The last factor charges survival through the last complete cell; without it,
the final cell's duration would escape the time cost. For a constant clock
covering `N` frames, this is exactly `exp(-r*N)/|P|`, independent of how many
beats partition `N`. At 1152 frames the survival log cost is -73.5587875424 for
12-, 24- and 48-frame periods alike. Tests also change the time unit while
holding the same set of hypotheses, verifying unchanged transition probabilities.
This is not invariance to adding new hypotheses by refining the period grid.

For a changing clock the same factor can be decomposed into:

```
-log|P| - r*N
  + sum_{jumps p->q} [log(expm1(r*p)) - log Zoff(p) - a*abs(log2(p/q))]
```

The jump terms remain. Time-unit consistency does not by itself make weak
changes win, eliminate the per-bar meter prior, resolve ambiguous observations,
or provide calibrated confidence.

## Exact search and uncertainty

Two L1 sweeps sum the off-diagonal transitions. Self-survival is merged
separately, so it is neither double-counted nor approximated by a self-jump.
Max-product traceback and sum-product evidence/reference partitions share the
same graph. All-pairs transition checks, exhaustive small-graph enumeration and
independent traceback reconstruction pass. Terminal survival is included in
both partitions; the full model remains normalized by its finite-graph `Z0`.
It is a discrete, beat-boundary/coarsened-hazard reference, not an exact
continuous-time tempo process allowing arbitrarily many within-cell jumps.

The former model's limitations are deliberately not changed here: meter may
change between complete bars, unknown edges use the reference distribution,
unavailable frames split runs, flat cells provide no local beat evidence, and
the zero-cutoff component preference is not a calibrated detection decision.
MAP traces remain inferred diagnostics, not observed beats. In particular the
new terminal survival factor is **not** a solution for partial edge bars.

## Reproduction and decision gate

Run `cargo run --locked --profile evaluation -p rhythm-map-eval --example
time_clock` for `time-clock-v1.json`. The source-identified report includes the
derived rate and all prior controls, without model inference or private music.
Run `python evaluation/parity/time_clock_diagnosis.py` for
`time-clock-diagnosis-v1.json`. That independent implementation reconstructs
every timing-control input, verifies the selected MAP weight, and checks that
the exact search did not miss the former MAP or an authored diagnostic path.
The latter two competitors share the former MAP's edge domain; their scores
must not be mislabeled as new decoded paths or new whole-track accuracy.

## Frozen result: local progress, still rejected for promotion

| Control or property | Former joint reference | Time-exposure reference |
| --- | --- | --- |
| Duration survival at equal covered time | Depends on beat count and period | Exactly equal |
| Flat-middle period states | 24 and 72 frames | 24 frames only |
| Flat-middle inferred ticks | 41 | 47, including 7 flat-cell prior-only ticks |
| Weak true doubling | 47/64 exact matches; no doubling | Same failure and exact MAP trace |
| Other six timing controls | Strong changes/all-weak timing retained, final beat missed | Same exact MAP traces |
| Noise component log ratio | +0.377662 | +0.548502, worse at the unchanged zero cutoff |
| Whole-input flat heads | No supported clock | No supported clock |
| Unknown edge bars and meter artifacts | Present | Still present |

The flat-middle control retains 125 BPM rather than inventing a slowdown to
about 41.67 BPM. This is an inferred clock result, **not seven recovered detector
events**: those seven ticks explicitly have no local evidence. The same path
still contains a six-beat meter artifact in the gap and a final three-beat bar.
All seven main timing-control traces are byte-for-byte unchanged. The existing
intra-bar-change, distractor, three-beat-meter and frame-zero results remain
bounded authored checks, not real-music accuracy gains.

Independent decomposition on the original weak-doubling competitor pair,
sharing `[4,1132)` coverage, gives authored minus constant MAP:

| Component | Former prior | New prior |
| --- | ---: | ---: |
| Beat evidence | +4.853289 | +4.853289 |
| Bar evidence, meter prior and edge prior | 0 | 0 |
| Duration prior | -17.491781 | -10.569344 |
| Total path weight | -12.638492 | -5.716054 |

Thus removing the beat-count survival disadvantage is not sufficient. Under the
new factor the two octave jumps still cost 10.569344 relative to the constant
path: 9.210340 comes from the unchanged log-period distance term, and the rest
from hazard/destination normalization. The exact search cannot select this
authored competitor over the constant path. This is a measured limitation of
the current objective, not a reason to silently weaken its jump coefficient.

The single noise draw now has approximately 1.73:1 reference odds for a clock,
up from about 1.46:1. These weak odds are not calibrated confidence or an
estimated false-positive rate. Its MAP covers only 115 of 1152 available frames;
the other 1037 are explicitly reference-modeled edges. The frame-domain
normalizer is still mathematically valid, but this is not a useful production
decision or satisfactory coverage.

Keep time-consistent duration accounting as a tested prerequisite, **not this
complete reference as an additional strategy**. Next address partially observed
bar hypotheses and the decision over competing clock interpretations, including
their probability mass and unsupported states. A MAP trace or a small positive
component ratio must not become confident metadata by itself. Do not run a
jump-weight sweep or retained-cohort replay to rescue this failed authored gate.
No private music, holdout, model training, default-estimator change or release
was used. The remaining errors do not establish a need to train a neural model.
