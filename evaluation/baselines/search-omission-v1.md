# Unknown-clock omission search: bounded exact reference, v1

## Decision

The explicit pulse/accent omissions from [omission-clock v1](omission-clock-v1.md)
can now coexist with unknown initial beat position, per-tick tempo changes,
changing meter, and partial bars in one exact search. No supplied clock templates,
authored timestamps, or change boundaries enter inference. A separate exhaustive
enumerator verifies all clock-tick, label and tempo/meter-change marginals.

**Do not promote this reference or extend it to full-song inference yet.** In two
short change controls, the authored path has better feature evidence but loses
after clock priors. One MAP path invents a pulse at a flat feature pair. The
32-frame resource probe also exceeds the fixed 250,000-state safety budget.
These are useful objective and complexity failures, not a production accuracy
gain, a reason to fit another weight, or proof that model training is necessary.

Default analysis, public APIs, user parameters and packaging are unchanged. No
weights, audio, training, real-music replay, holdout or release is involved.

## What this gate does and does not compare

The ten main inputs are **18-frame authored feature arrays**, not recordings,
neural logits, or the previous twenty rank fixtures. Their fixed period domain
is every integer 3..6, and their meter domain is 2..3. Every initial tick offset
and meter phase is searched; every subsequent tick can change period, and each
completed bar can change meter. The implementation accepts meters 2..7 and has
normalization tests for that domain, but these ten cases do not establish meter-
diverse music performance. Frame units deliberately have no claimed musical BPM.

Paired full-frame normalization is reused unchanged. The rank preprocessing
pipeline is **not** exercised; no comparison to the former seven-control pass
rate is meaningful. Feature values 4 and 3 simply define algebraic test inputs,
not newly fitted detector strengths or a production emission scale.

The reference searches all permitted paths, not a beam or a preselected candidate
list. There are no chosen cut points or case-ID branches. Its inference result
contains full-model marginals as well as a separate joint MAP path; it does not
yet enumerate same-emission equivalence classes across this larger graph.

## Clock, meter and edge semantics

A clock state is `(frame, outgoing period, meter, beat-in-bar)`. At frame t, a
state with period p places the next tick at t+p. The next period is chosen there,
so a change can happen inside a bar. Initial periods and meters are uniform;
conditional on them, the first tick offset is uniform on `0..p-1` and the initial
bar phase is uniform on `0..m-1`. Beginning and ending partial bars are allowed.

Tempo transitions reuse `time_prior::Prior`'s row-stochastic, time-exposure
transition law, including its unchanged log-period distance scale. A transition
is taken only when the next tick is inside the recording. If it lies outside,
all unobserved future decisions are marginalized to one: there is **no extra
terminal transition or survival charge**. This finite-window generative boundary
differs from the older supplied-clock duration-weight calculation; the old
reports are untouched. It must not be described as just adding search to an
otherwise identical model. Period at the last tick is an inferred outgoing
state, not an observed complete inter-beat interval.

At a bar wrap, changing-meter probability h has one run-wide `Beta(1,1)` prior.
Conditional on a change, each other meter is equally likely. Integrating h
contributes `Beta(C+1,U-C+1)/(M-1)^C`, where U counts in-window bar decisions, C
counts changes and M is the number of possible meters. A singleton meter domain
has no trials or changes. No decisions beyond the recording are counted.

The general joint tempo/meter state-space framing is also described by
[Krebs, Boeck and Widmer (ISMIR 2015)](https://www.cp.jku.at/research/papers/Krebs_etal_ISMIR_2015.pdf).
This is an original evaluation implementation using existing repository factors,
not copied upstream decoder code or weights; the paper is background, not
validation of these omission, normalization or boundary choices.

## Integrated omissions and exact messages

The omission model retains the previous independent run-wide pulse and accent
retention priors `q,r ~ Beta(1,1)`. Available ticks allow inferred omitted/plain
labels, plus accented labels at latent bar starts. Unavailable frames contribute
no observation or retention trial but **the latent clock and meter keep advancing**.
Bridging an unavailable interval is a model assumption, not recovered evidence.

In addition to clock/meter state, forward messages retain six sufficient counts:
available tick trials N, retained pulses B, retained bar pulses Z, emitted accents
D, in-window meter decisions U, and meter changes C. At a terminal path, apply
exactly once:

```
Beta(B+1,N-B+1) * Beta(D+1,Z-D+1) * Beta(C+1,U-C+1) / Z_pair[B-D,D].
```

Changed-meter destination factors and normalized clock priors are accumulated
along edges; emitted labels accumulate the centered paired-feature numerator.
No independent beat/downbeat normalization or local repeated normalizer is added.
Log-space sum-product messages yield the partition. Separate max-product messages
and predecessor links yield the integrated joint MAP path. Reverse messages yield
per-frame tick, omitted/plain/accented-label, unavailable-tick, tempo-change and
meter-change probabilities. Marginally most likely labels are not spliced into
a purported MAP path.

These are **model probabilities, not calibrated audio confidence**. Label 0 is
an inferred omitted pulse; null is unavailable input. Labels 1 and 2 are inferred
emissions, not detected Beat events. No event-acceptance rule is implemented.

## Frozen outcomes, including failures

- Constant tempo and a shifted initial phase recover the intended clock and
  three-beat meter in their joint MAP paths.
- In the flat-middle and unavailable-middle controls, the clock remains at
  period 3. At frames 7 and 10, the former emits omission labels and the latter
  null labels. This distinguishes silence-like features from missing input.
- The changing-meter control contains a 2-to-3 transition at frame 7. At the
  final accented tick it selects meter 2 again; without a subsequent full bar,
  this ending meter is not an observed meter-change event.
- The true-half and erased-constant controls have identical features and exactly
  identical inference. Both select the globally slow ticks 1,7,13, dropping the
  extra authored pulse at frame 4. No input-only method can separate the two
  contradictory authored interpretations here.
- The true-double control selects constant period 3, including an inferred plain
  pulse at frame 4 whose input feature pair is `[0,0]`. Its MAP also accents frame
  7 despite zero accent feature there. Omission states alone do not prevent
  unsupported event insertion.
- Fully flat and fully unavailable inputs both have log ratio zero. Their
  marginalized clock-tick and change probabilities are equal, although joint
  MAP label assignments have different prior probabilities. A preferred MAP
  clock in either case is not signal evidence.

Post-inference authored-path audits identify the direction of failure. The
following are **authored minus selected** integrated joint-path log scores, not
posterior odds between tempo families:

| Control | Feature term | Clock prior | Omission prior | Meter prior | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Half | +2.3734 | -4.0721 | -0.2231 | 0 | -1.9219 |
| Double | +2.0408 | -5.3232 | +0.4418 | +0.4055 | -2.4351 |

The feature direction is correct in these witnesses; the selected path wins
after priors. This does not establish that only a prior change is needed across
music, nor does it license tuning the jump penalty on these fixtures. It is
evidence against declaring an immediate need for neural training.

## Reproduction, exactness and resource gate

```
cargo run --locked --profile evaluation -p rhythm-map-eval --example search_omission
python -m unittest discover -s evaluation/parity -p test_search_omission.py -v
```

The Rust command regenerates `evaluation/parity/search-omission-v1.json`. Python
rebuilds all ten partitions and maxima in probability space, independently
reconstructs each reported MAP path's score and validity, and enumerates all
**4,532 complete paths** of a separate eight-frame control to verify every label
and change marginal. That control includes missing observations and multiple
bar decisions. Rust also checks paired-permutation normalization, full/singleton
domains, no-data probability conservation, invalid inputs and budget errors.

Each fully available 18-frame case uses 17,802 states and 99,162 label/transition
branches. These counts depend on graph/availability, not observed feature values;
there is no evidence-based pruning. A 32-frame flat probe with the same domains
exceeds 250,000 states and returns an explicit error, **no partial result**.
Neither wall-clock speed on these tiny arrays nor their bounded resource usage
is a full-song performance claim. The count-augmented reference is not a viable
production search implementation yet.

Next check a coherent clock-prior/boundary formulation and support semantics
using matched, controlled interventions and these frozen failures. Do not just
increase the state budget, sweep transition weights, splice in an event threshold,
replay the closed holdout or begin long cohort inference. Search scalability and
supported-event provenance remain separate gates after objective correctness.
