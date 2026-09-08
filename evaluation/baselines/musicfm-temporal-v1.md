# MusicFM complete-input temporal gate v1

This is a research-only gate, not a new decoder, user strategy or shipping pack.
The [protocol](../parity/musicfm-temporal-lock-v1.json) freezes one comparison
before new authored temporal or real-music feature inspection. It follows the
[seven short-input extraction checks](musicfm-fidelity-v1.md), which did not
establish a complete-song context policy or musical discrimination.

## One formula, a different representation

Keep the previous pre-head recurrence hypothesis unchanged mathematically:

```text
score(clock) = mean_t [cos(h(t), h(one candidate cycle before t))
                    - cos(h(t), h(half a candidate cycle before t))]
```

Only the fixed representation, its native grid and its context change: FMA,
non-Flash, float32, layer 7, 1,024 channels at nominal 25 Hz. No projection,
channel/layer selection, centering fit, metric sweep, phase search or per-track
blending follows. This controls the formula when asking whether another
representation supplies useful periodic evidence. It is not a claim that the
[failed pre-head rule](prehead-recurrence-v1.md) was secretly successful, nor a
clean representation-only ablation: encoder context and frontend also differ.

Clocks still come from externally supplied periods and a phase-continuous
constant/step boundary. Labels are absent from the score function, but those
inputs are oracle assistance, not automatic BPM or change-point detection.
Keep native, two half-phase names and double density for each shape. Absolute
phase and the two half phases cancel analytically; they are not extra votes.

Targets are the native 100 feature centers in a physical four-second window.
All eight clocks use the same intersection of supported targets. Both raw
interpolation endpoints must lie inside that window; convert both to float64
before interpolation, then unit-normalize. No extrapolation, rounding a lag,
wrapping or zero padding. A required norm <= `1e-12` rejects the entire window,
and either rejected member rejects a paired verdict. The `1e-6` score margin
only identifies numerical ties, not calibrated confidence.

## Complete input and ownership

Use 24 kHz mono PCM, with no additional per-recording normalization. For this
gate PCM is entirely self-authored: no decoder, resampler or music-file input.

- Contexts start at 0, 4, 8, ... seconds and include at most eight seconds.
  Stop when the final context covers the complete input; do not pad its tail.
  Multi-context final inputs are longer than four seconds and at most eight.
- Adjacent contexts split their overlap at its fixed midpoint: ownership seams
  are 6, 10, 14, ... seconds. The later context owns the exact seam. The first
  and final contexts own the remaining file edges. No averaging or feature-
  selected owner; every nominal token has exactly one owner.
- The existing frontend's token count is `ceil(floor(N/240)/4)`, with nominal
  centers at `960*k` samples. Aligned context starts preserve that whole-input
  count, including sub-hop file tails. All PCM participates in a context even
  when there is no additional nominal token for its final samples.
- Inputs shorter than 1,025 samples or longer than 30 minutes are refused by
  this bounded research helper. The actual authored experiment uses 12.01 s.
  These are not newly added product limits. The previous <=2 s probe is unchanged.
- Context is noncausal. Chunked features need not equal an unbounded whole-song
  forward, and two contexts' shared nominal centers need not be equal. Measure
  overlap differences and count seam-crossing score queries; do not declare
  either a numerical parity failure or silently re-infer a favorable crop.
- Nominal 40 ms spacing is not measured acoustic latency and cannot recover
  missing 50 Hz detail. For the eventual original 50 Hz calibration windows,
  keep the exact physical interval and select its native centers. An odd old
  start produces an explicit 20 ms first-target offset; do not round the
  supplied physical boundary or reinterpret the existing 60 ms geometry guard.

Each context uses all twelve original blocks and the final projection. Require
bit-exact unhooked reference, observer-hooked extraction and unhooked repeat;
the selected seventh block, hidden tuple index 7 and returned feature must agree.
Full parameter/buffer hashes, config, stats, mode and input bytes remain unchanged;
transient rotary cache and Python/CPU RNG are restored as in the previous gate.
Private feature/owner archives round-trip exactly. No learned state is repaired.

## Authored controls and stopping rule

Model-free phase-circle trajectories first check the algebra conditional on a
persistent clock cue. Clean, weak and omitted pulse channels must retain correct
constant/step shapes and reject density rivals. Flat vectors must tie, zero
vectors must be unavailable, and half/double trajectories remain explicit
semantic-density counterexamples. These are authored vectors, not neural outputs.

The separate pretrained experiment takes six complete 12.01 s PCM waveforms:
constant clean, alternating omitted pulses, alternating one-eighth pulses,
120-to-240 BPM, 120-to-75 BPM and silence. Changes occur at six seconds, exactly
on the first ownership seam. Each non-silent recipe has a weak 220 Hz carrier
whose amplitude follows the continuous phase, plus a decaying 880 Hz percussive
pulse. Omissions remove pulses, not the independently specified carrier cue.
The formulas, sample counts and amplitudes are frozen in the protocol. These
simple signals test this explicit conditional cue, not all real weak-beat music.

Each of the three constant conditions pairs with both changes: six complete
comparisons, with shared cases not counted as six independent songs. Shape
support requires both correct native shapes; density resolution additionally
requires both to beat every rival. Silence must abstain or remain tied across
all clocks. Any failure closes this fixed rule **before real music**. Do not
change the context, threshold, recipe or score after inspecting these features.
Extraction fidelity and discrimination are separate results.

The existing seven exposed ARTBeaT pairs and all 40 calibration identities remain
unchanged; RUBATO stays expressive/untyped and the holdout stays sealed. Even a
positive authored result would still need byte/provenance-checked 24 kHz music
inputs, the old geometry guards and independent work-disjoint acceptance. A
failure does not prove all pretrained reuse impossible or training necessary.
Native/WASM feasibility and model/training-provenance rights remain unapproved.

## Measured result: useful authored cue, failed abstention gate

The [complete report](../parity/musicfm-temporal-v1.json) retains all six PCM
inputs, eighteen contexts, 54 forwards, six paired verdicts and both silence
comparisons. All reference/hook/repeat features are bit-exact; each complete
input owns 301 tokens without gaps, both seams remain visible, and full model
state and input bytes are unchanged. All private archives round-trip exactly.

| Constant condition paired with change | 120 to 240 | 120 to 75 |
| --- | --- | --- |
| Clean | Shape and density pass | Shape and density pass |
| Alternating omissions | Shape and density pass | Shape and density pass |
| One-eighth weak pulses | Shape and density pass | Shape and density pass |

Every comparison has 75 common targets: 25 before and 50 after the boundary.
Thus paired shape support is 6/6 and density resolution is 6/6 **on these
authored signals with an explicit persistent cue**. Shared inputs, externally
specified clocks and synthetic audio make these neither six independent songs
nor an accuracy percentage, and they do not replace the seven real-music pairs.

Silence fails the preregistered abstention condition. With the fast rival, the
eight scores range from approximately -0.00111649 to -0.00023552, a spread of
0.00088097; the native step beats native constant by approximately 0.00024617.
With the slow rival, the spread is approximately 0.00067372. Both exceed the
unchanged `1e-6` numerical tie budget. These small negative values are not
calibrated probabilities, and selecting their maximum still invents a preference.
No sign gate or newly fitted silence threshold is added after seeing them.

All twelve adjacent-context overlap comparisons differ, including both silence
overlaps. Maximum absolute feature differences range from about 3.16 to 4.86.
This establishes context dependence, not its unique cause: the experiment does
not isolate frontend edge effects, positional responses and global attention.
Fixed ownership makes the output reproducible but does not make features
context-invariant or prove those differences caused every discrimination score.

The frozen decision is therefore **close this unconditioned fixed rule before
music**, despite successful extraction and non-silent authored comparisons.
Do not discard silence from the denominator or claim MusicFM itself is useless.
There is evidence of a usable authored timing cue and evidence that this raw
readout cannot provide its own no-evidence decision. Neither establishes that
training is necessary. Next inspect whether the already existing audio-evidence
availability contract can be reused unchanged in a separately specified
composition; do not reopen this report, tune its scores or access music features
under a silently revised rule.

Worker time was 703.58 seconds (11.73 minutes); process-lifetime peak RSS was
1,938,317,312 bytes (1.81 GiB). This includes loading, three forwards per context,
hashes, archive round trips and scoring on the VDI, with independent model-free
review checks overlapping part of the run. It is not isolated inference
throughput or product latency. The first attempt stopped at the physical-memory
preflight, before model loading or inference; its private log is retained.
After memory recovered, the unchanged resource thresholds passed on retry.

## Reproduction

Run the model-free tests in the existing NumPy-only CI environment:

```text
python -m unittest discover -s evaluation/parity -p test_musicfm_temporal.py -v
```

For actual authored extraction use the separately pinned Windows CPU runtime,
verified source, sidecars and FMA checkpoint, with a fresh private off-system
output directory. No downloads or installation occur inside this probe:

```text
<private-python> evaluation/parity/musicfm_temporal_probe.py --upstream <pinned-source> --config <verified-config.json> --stats <verified-fma_stats.json> --checkpoint <verified-pretrained_fma.pt> --output <fresh-private-directory>
```

The original 4 GiB job cap, two threads and free-resource thresholds apply. A
900-second parent timeout bounds all 54 forwards, loading, hashing, archives and
scoring; it is not a shipping latency target. The first execution/integrity failure
stops the run. Partial output or a failed preflight is not a completed result.
