# Real-audio calibration protocol

This protocol determines whether a failure belongs to Beat This observations or
to Rhythm Map's deterministic estimator. It is an evaluation workflow, not a
training dataset pipeline.

## Initial corpus slices

Start with short, legally held excerpts and keep each capability independently
searchable in manifest tags:

- `drumless-control`: stable-tempo harmonic music without drums;
- `drumless-ramp`: sustained accelerando or ritardando;
- `drumless-step`: an abrupt tempo change without a strong percussion cue;
- `rubato`: local expressive timing that should not become many tempo jumps;
- `compound-meter`: a meter where half-bar assumptions are unsafe; and
- `percussive-control`: a matched control for the same tempo behavior.

The first useful calibration set should contain more than one source and
arrangement per failing slice. Do not change a default threshold or algorithm
from a single track.

## Rights boundary

Audio may be locally purchased, commissioned, self-recorded, or obtained under
an explicit evaluation-compatible license. Record the audio and annotation
rights separately in the suite manifest. Possession or streaming access alone
does not grant redistribution rights.

Private audio stays outside the repository. The manifest stores a SHA-256 of
the exact encoded file bytes and an optional filename hint. Reports contain no
local path or audio bytes. Independently authored annotations may be committed
only when their chosen license permits it; otherwise keep them with the ignored
private suite.

## Annotation procedure

1. Fix the exact audio file and inspect it before annotation. Re-encoding creates
   a different asset identity.
2. Mark beat timestamps and downbeats in an audio editor or DAW without looking
   at Rhythm Map's prediction.
3. Describe constant and ramp regions with ordered, non-overlapping tempo
   segments. Silence may remain uncovered by tempo segments.
4. Mark only musically meaningful `tempo_jump`, `ramp_boundary`, and
   `rhythm_discontinuity` events. Do not label every rubato fluctuation as a
   change point.
5. Have a second pass review bar phase and every change boundary. When reviewers
   disagree, retain the case as ambiguous metadata rather than tuning against
   one person's preference.
6. Run the oracle path first. A failing oracle case indicates invalid or
   inconsistent truth, or an estimator limitation; it is not evidence against
   Beat This.
7. Run the end-to-end path only after the oracle case is accepted. Compare raw
   observations, analyzed events, and metric deltas by slice.

Do not use model output to initialize hidden holdout annotations. Calibration
cases used during algorithm development and untouched holdout cases should be
separate manifests even when their audio shares the same storage directory.

Mark those manifests with `"purpose": "calibration"` and
`"purpose": "holdout"`. The distinction is enforced rather than descriptive:
truth-assisted `decoder-sweep` and `decoder-recoverability` commands accept only
a calibration suite. Select one registered policy using calibration results, record
its ID, then open the holdout only through `decoder-eval --policy <id>`. That
command compares the one candidate with the immutable upstream baseline but
does not reveal any other candidate or policy oracle. Do not
rename already inspected ARTBeaT cases as holdout; they are calibration evidence
because their per-case results have already influenced decoder design.

The fixed-policy report aggregates every manifest tag. A candidate must be
examined by capability slice, not only by overall mean: a gain on percussive
tempo jumps does not excuse a regression on `rubato`, `drumless`, or
`metric-ambiguity`. Populate each required slice with more than one independent
source before treating the result as a product-default decision.

## Decision rule

- Oracle passes, end to end fails: inspect Beat This events and bounded
  deterministic recovery before considering another model.
- Oracle and end to end fail in the same way: improve truth consistency or the
  deterministic estimator.
- Both pass across multiple sources in the slice: preserve the training-free
  path and tighten gates only from a documented product requirement.

Aggregate averages do not override slice failures. In particular, a strong
percussive score cannot compensate for a failing `drumless-ramp` slice.
