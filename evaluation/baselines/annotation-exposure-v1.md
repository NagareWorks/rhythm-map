# Annotation-only matched constant/step exposure v1

This is an **availability audit and a frozen future replay plan**, not a detector
accuracy result. It repairs the comparison design gap identified by the
[fixed eight-second residual inventory](window-response-residuals-v1.md), without
changing that report or selecting windows based on detector outcomes.

## Pre-run contract

The script and [lock](../parity/annotation-exposure-lock-v1.json) were frozen after
15 authored controls passed, before the first cohort availability run. The run
reads only public annotations and an explicit allowlist of public predecessor
metadata: cohort, suite hash, case ID, truth hash, capture hash and frame count.
Model-derived inventories in that predecessor are not selection inputs. Suite
purpose, ordered case identities, annotation file hashes, duration compatibility
and truth-directory confinement are checked before use. No audio, private capture,
model, numerical ML dependency, inference or decoder is accessed.

- Candidate durations are fixed globally at 100, 200 and 400 frames (2, 4 and
  8 seconds at 50 Hz). Enumerate every whole-frame start with a complete window;
  retain every rejected start and every track. These overlapping starts and their
  pair combinations are **availability counts, not independent observations**.
- Each window needs four annotated beats strictly before its start, all in the
  same constant segment that contains the start. No change may intervene in the
  prefix. The last annotated beat must reach the window end.
- A constant window stays inside that segment, contains no change and has at
  least four full-query annotated beats. A step window contains exactly one
  `tempo_jump` joining adjacent constant segments with different annotated BPM,
  stays inside those segments, and has at least two full-query beats on each
  side. A beat exactly on the change belongs to the after side.
- Time is quantized once to the nearest microsecond, with positive half ties
  upward. Full-query eligibility preserves the previous rational ledger rule:
  `ceil(center - 3) >= 0 && floor(center + 3) < window_frames`. Equivalently, in
  frame coordinates, `2 < center < window_frames - 3`; both bounds are strict.
  Boundary-censored beats cannot satisfy the interior minima.
- Annotated, prefix-continuation, both half-phase and double clocks must each
  fit the existing 128-tick budget. **Response counts and response support are
  not checked here**; annotation availability does not guarantee replay eligibility.
- A pair uses the same track, prechange constant segment and duration, with
  `constant_end <= change_start`. The two audio windows do not overlap; their
  annotation-prefix footprints may overlap. Independence is not asserted.
- Choose one pair per track by earliest pairable change, most time-balanced
  change window, earlier change start, then latest nonoverlapping constant start.
  Choose one global duration by maximum pairable track count, longer on ties.
  With no pairs, return a null duration and empty plan: no per-track fallback.
- RUBATO and the tagged ARTBeaT rubato item are untyped for this discrete
  constant-versus-step comparison. Local interval regularity does not create
  constant or step labels. Ramp cases remain excluded from step pairs.

## Complete availability result

All 15 ARTBeaT and 25 RUBATO inputs remain in every profile. The
[aggregate report](../parity/annotation-exposure-v1.json) contains per-case reasons,
identities and pair hashes, without raw response arrays or chosen coordinates.

| Window | ARTBeaT candidate starts | Constant starts | Step starts | Pairable ARTBeaT tracks | Pairable step starts | Legal pair combinations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 s | 10,843 | 1,858 | 95 | 4 / 15 | 92 | 9,129 |
| 4 s | 9,343 | 2,206 | 1,165 | 7 / 15 | 328 | 19,102 |
| 8 s | 6,343 | 136 | 2,228 | 0 / 15 | 0 | 0 |

RUBATO contributes respectively 322,040 / 319,540 / 314,540 untyped candidate
starts and zero typed pairs. These are **not failed detector predictions**.

The registered rule selects **four seconds, seven pairs** (seven tracks, fourteen
windows, 56 seconds of paired audio exposure). Selected cases:

- `artbeat-06-150-to-75`
- `artbeat-08-112-5-to-75`
- `artbeat-09-90-to-80`
- `artbeat-10-90-to-120`
- `artbeat-12-80-to-150`
- `artbeat-14-240-to-96`
- `artbeat-15-85-to-127-5`

The other eight ARTBeaT tracks and all 25 RUBATO tracks stay in the denominator;
none receives a shorter bespoke fallback. Eight-second isolated constant or step
starts do exist under this new sliding annotation inventory, but no legal
same-presegment nonoverlapping **pair** exists. This does not contradict the
previous fixed-grid eight-second report's zero eligible constant context: its
sampling rule is different and its result remains frozen.

## What this does and does not settle

The available calibration annotations can support a small matched constant/step
comparison. This is seven tracks, not 19,102 independent examples, and cannot
establish generalization to rubato, other genres or the full product domain.
Selecting duration by annotation coverage is an explicit study-design decision,
not evidence that four seconds is a better production analysis window.

Next regenerate and verify the selected plan hash, then replay the existing
unchanged response-packet and rational-ledger adapters on exactly these fourteen
windows. Freeze that replay's metrics first. Retain every selected pair, even if
response budgets or support reject one member; do not replace it or let only one
side enter a paired comparison. Inventory missed annotated ticks, candidate
support, off-annotation responses, assignment ambiguity and half/double density
tradeoffs separately by source and context. Only then ask whether existing
joint-head/context evidence distinguishes omissions from true changes.

No fitted penalties, new user strategy, default output change, holdout access,
training or accuracy improvement is implied. The timing API stays one-call and
zero-tuning; future metadata packs remain a separate product horizon.

## Reproduction and identity

```bash
python evaluation/parity/annotation_exposure_audit.py \
  --output /data/annotation-exposure-v1.json
python -m unittest discover -s evaluation/parity -p test_annotation_exposure.py -v
```

Output refuses overwrite. The report reproduces from public repository files
alone; its plan hash commits to the deterministically regenerated coordinates.
Authored controls cover positive and empty plans, nonoverlap and same-segment
pairing, brute-force pair counting/ranking, exact fractional query edges,
minimum context, annotation conflicts, clock budgets, untyped labels and poisoned
model-derived metadata. Frozen controls verify all 40 denominators, reproduction,
selection, file-access scope, and rejection of changed identities or suite roles.

SHA-256:

- Script: `824832bb891637a8eafb40bd05c05e4fb5c5000577c7f08a81a10bdc2892c681`
- Lock: `2f9fcbc4a75ac6c30acad4d927d681d686a6a2e9d93531831f8e8e9f934c1271`
- Report: `e9bb2086e906bda366e0b8d2d4dd87b9ff9d7c44441696d223a75a96471b1346`
- Metadata projection: `b93984c19e761873ea3c79144e35add6474cc0dd85677a67cbca90a4bfe8e13d`
- Selected plan: `5c15e677d0277dcac111aaebfb2be0b788996cbd44019a315c171fcf39bfe4f2`
