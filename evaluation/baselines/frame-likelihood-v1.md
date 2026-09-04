# Complete-frame scoring correctness checkpoint

This is a structural prerequisite for a future sequence decoder, **not** a new
audio-analysis candidate or a replacement for the rejected
[renewal clock v1](dense-sequence-v1.md). That experiment and its source hashes
remain unchanged. No real capture, calibration truth, holdout or model training
is used here. Production analysis and user-facing options are unchanged.

## Scoring contract

For every available frame t, a proposed state contains binary beat B and
downbeat D labels, with D implying B. With the two given logits b and d:

`L(t) = log sigmoid((2B-1)b) + log sigmoid((2D-1)d)`.

Sum this over **all** available frames on the same domain for every hypothesis.
There is no division by predicted beat/bar count and no positive-only reward.
An unsupported extra bar pays for labeling negative downbeat evidence positive;
an omitted or wrongly phased bar pays for labeling positive evidence negative.
At logit magnitude 8, a single incorrect binary label costs exactly 8 relative
to the correct label. This removes the zero-cost extra-bar defect of v1.

The implementation uses stable log-sigmoid, rejects non-finite or inconsistent
inputs, and reports scored/unavailable frame counts. Explicitly unavailable
frames contribute neither positive nor negative evidence; their values cannot
create observed variation. Chunk sums agree with the full-domain score.

This is an **independent-head pseudo-likelihood**, not a learned observation
likelihood, calibrated confidence or proof that the two neural heads are
independent. Head calibration, time correlation, support shape and observation
dropout remain assumptions to resolve before real decoding. The scorer owns
none of those policies and receives no truth or case identity.

## Authored checks

The audit supplies a fixed beat grid: 1,200 frames at 50 Hz, period 24 frames
(125 BPM), origin frame 5, and radius-one rectangular support around each beat.
Period 24 permits exact integer half/double-period comparisons. These are
authored logit masks (+8 inside support, -8 outside), not measured audio or
general-purpose pulse templates. Known timing deliberately isolates scoring
correctness; no recovered beat timestamps or BPM accuracy are claimed.

- Enumerate all 27 meter/phase hypotheses for meters two through seven. For
  each of the 27 authored combinations, require the correct hypothesis to rank
  strictly first on identical evidence, including nonzero starting phases.
- Require strictly lower scores for extra halfway bars, omitted alternate
  bars, wrong bar phase, doubled beat density and halved beat density, with
  identical frame-count denominators.
- Keep the correct four-beat interpretation after one false halfway bar or one
  erased true bar. These isolated cases do not establish sustained-dropout
  robustness.
- With an exactly constant downbeat head at -8, 0 or +8, do not identify a
  meter. Raw scores can prefer a bar density even without phase information;
  the audit returns no ranking in this case. Exact constancy is only a
  diagnostic guard, not uncertainty calibration or a silence detector.
- Separately report repeated weak downbeat peaks of -2 on background -8. Do
  not change logits, weights or dropout assumptions to make this diagnostic
  pass. Strong-mask success must not conceal weak-evidence limitations.

## Result and bounded next step

The [authored report](../parity/frame-likelihood-v1.json) passes all 27 ideal
meter/phase cases, five contradictory-path checks, two isolated corruptions
and three flat-head abstentions. Six Rust tests also cover score additivity,
extreme logits, input validation, unavailable evidence and the weak diagnostic.

Repeated weak peaks rank four beats per bar first **only within meters two
through seven**. The correct path scores -83.74209, while dropping alternate
bars scores -47.74209 and dropping all bars scores -5.74209 (higher is better).
Those omission paths were diagnostic alternatives, not new meter options.
Their advantage follows directly from negative logits: the score interprets
each weak expected event as less likely than its absence. Restricting the
candidate set would hide this failure, not solve observation loss. The same
issue applies to weak beat-head evidence, not only bar labels.

The initial structural audit was extended with these explicit omission
comparisons to expose that limitation; no scoring formula, input geometry,
weight or acceptance threshold was changed after results. This is an authored
development checkpoint, not a preregistered real-music generalization result.
The former frozen sequence-v1 sources and reports remain byte-unchanged.

These checks run in `cargo test -p rhythm-map-eval --examples`. To reproduce
the authored report (stdout contains no real audio or event coordinates):

```bash
cargo run --locked -p rhythm-map-eval --example frame_likelihood
```

The result records scorer/audit source hashes. No real-cohort run is justified
by passing this checkpoint alone. Next define a normalized visible/missing
observation model and its uncertainty boundaries, then test weak evidence,
genuine tempo changes and unavailable spans before freezing a joint decoder.
Do not substitute known beats for the full-frame clock in that next experiment.
