# Rhythm-support null checks and initial data worksheet v1

## Result and limits

The model-free assumption checks pass: second-order-preserving transforms
cannot provide new evidence to a statistic consisting only of those preserved
quantities, and a stationary dependent sequence is not automatically
exchangeable under arbitrary permutations. These are algebraic witnesses, not
new model outcomes or music accuracy. The
[eleven-input producer gate](../parity/rhythm-presence-audit-v1.json) remains
`not_run`, with `passes: null` and no unknown count. No production change,
training, feature access, new audio or sealed-holdout access occurred.

The [data worksheet](../datasets/rhythm-support-worksheet-v1.json) is a
**publication-only draft**, not a completed clip inventory. Official metadata
downloads returned gateway timeouts; an alternate execution route was denied
before running. No metadata payload was verified or interpreted. The worksheet
keeps these missing facts null, rather than claiming there are no eligible
recordings. Zero *admitted* groups and zero collected labels describe our
current work, not the potential contents of the source.

This completes the algebraic checks and the worksheet structure, but not the
planned per-clip rights/independence inventory. Neither an unsuitable null
shortcut nor unavailable downloads establishes a need to train.

## What was actually checked

`rhythm_support_null.py` uses short, authored abstract arrays; they are not PCM
fixtures, recordings, encoder features or listening labels. There is no RNG,
seed search, trained network, audio decoding or network dependency.

| Check | Witness and consequence |
| --- | --- |
| Fourier phase offsets | A changed 64-element real sequence preserves its power spectrum and every direct time-domain circular autocorrelation lag. A power-only or circular-autocorrelation-only statistic cannot separate these inputs. |
| Feature circular shift | All whole-sequence circular dot-product lags of a 16-by-4 abstract matrix survive a seven-frame shift. Moving a fixed sequence is not independent counterevidence for this statistic. |
| Dependent-process permutation | Uniform rotations of `[1,1,1,1,-1,-1,-1,-1]` define a stationary process on the finite circle. The alternating permutation lies outside its eight-element support: probability 1/8 becomes zero. Stationarity alone cannot justify exchangeability. |
| Non-stationary reference | Independent symmetric random signs multiplied by an envelope with one amplitude-4 location have variance 16 there and 1 elsewhere, with zero distinct-time covariance. Non-stationarity does not itself supply a rhythm label; no perceptual classification is asserted. |
| Rank arithmetic | For 99 hypothetical surrogates, the finite-rank floor is 0.01 and all ties give 1. A separate authored score table shows why maximizing the original but not the surrogates changes the answer. No significance test was performed. |

The fixed absolute numerical check is `1e-10`. Reported residuals round to zero
at twelve decimal places; that is portable serialization, **not bit-exact
equality or a rhythm-confidence threshold**. Direct time-domain sums provide an
independent check on the Fourier construction. Tests include odd/even lengths,
every circular shift, finite/real/shape bounds and DC/Nyquist ownership.

The scope of invariance matters. Linear, truncated or fixed-physical-window
statistics need not be invariant to circular shifts; a test explicitly retains
that counterexample. This is not a reproduction or a new failure measurement of
the earlier MusicFM scorer. A nonlinear statistic on correctly regenerated
input-level surrogates is not ruled out by these witnesses, but still needs an
appropriate no-support reference and exchangeability argument.

This agrees with the original surrogate literature's explicit null semantics:
[Schreiber and Schmitz, improved surrogate tests](https://arxiv.org/abs/chao-dyn/9909041v1)
preserve specified statistical properties, while their
[non-stationary-signal analysis](https://arxiv.org/abs/chao-dyn/9904023v1)
explains why rejecting a stationary null need not isolate the desired cause.
Our arrays illustrate the limited assumptions above; they do not implement the
papers' algorithms or establish a null distribution for this product.

## Source worksheet and missing labels

The [official FSD50K v1 record](https://zenodo.org/records/4060432) reports
40,966 development clips: 14,959 CC0, 20,017 CC BY, 4,616 CC BY-NC and 1,374
Sampling+. Thus 34,976 fall nominally in the project's initial CC0/CC BY
screen. These are **publication counts**, not independently recomputed clip
permissions or a selected training set. Dataset-level terms and the authors'
commercial-contact note remain unresolved review items. Attribution is retained
in the worksheet. No third party was contacted and no eval metadata was read.

The worksheet separates:

- publication evidence from verified archive/member and PCM hashes;
- clip, annotation and dataset terms, with training/evaluation/redistribution
  permissions recorded separately and backed by evidence;
- uploader/creator hints from work, recording, shared-asset and generator
  identity, plus explicit frozen-encoder overlap review;
- an unassigned split from any later train/dev/calibration/acceptance group;
- a label not yet collected from an actual listener vote of `unknown`.

Existing forty calibration identities and eleven authored controls remain
regression-only. Newly authored recordings are a planned route, with no new
recordings or grants in this step. All eight proposed coverage slices currently
have zero newly labeled independent groups. No music/non-music tag was converted
to rhythm truth and no uploader count was presented as recording independence.

Candidate/window templates are intentionally empty; they are not synthetic
records or an executable acquisition manifest. Before populating them, freeze
physical query/context policy and preserve independent votes, disagreement
reasons and adjudication history. Work/recording/creator/derived-asset/generator
connections must keep related items in the same split, including transitive
connections. A structural schema check cannot grant rights, identify unseen
duplicates or certify honest independent annotation.

## Reproduction and next boundary

```sh
python evaluation/parity/rhythm_support_null.py --check evaluation/parity/rhythm-support-null-v1.json
python -m unittest discover -s evaluation/parity -p 'test_rhythm_support*.py' -v
```

All 22 new checks pass: fourteen numerical/semantic controls and eight frozen
worksheet-boundary checks. They run with the existing NumPy-only CI environment.
The worksheet tests preserve the draft and its unknowns; they **do not** verify
unavailable upstream clip bytes or certify dataset rights.

Close the tested second-order/circular-shift and naive permutation shortcuts.
Do not spend encoder forwards on them, interpret rank arithmetic as confidence,
or reopen the previous recurrence threshold/seed search. A broader surrogate
proposal remains unqualified until it provides a defensible null; this step
does not make another round of model search the default next action.

The concrete remaining task is to obtain and verify the official **development
metadata only**, then populate per-clip rights/provenance rows and count reviewed
groups against the eight missing label slices. Keep the original published
archive checksum and record a new SHA-256/member identity before parsing; do
not silently substitute a third-party mirror or assume schema fidelity. If
the source remains unavailable, retain this incomplete inventory and ask for
direction before replacing the data route. The
[minimal-component proposal](../../docs/RHYTHM-SUPPORT-PROPOSAL.md) still requires
separate authorization before audio acquisition, human-label coordination,
fitting, new neural execution or production integration.
