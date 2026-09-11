# Distributed phase synchronization (experimental component)

This implements the [temporal-evidence follow-up contract](../../evaluation/baselines/clock-temporal-evidence-v1.md).
It is NOT a fitted or admitted music model, and does not change the Rust/default
analyzer. No new audio, labels, holdouts, optimizer, or old checkpoint is used
in the component checks. Musical accuracy is still unmeasured.

Status, 2026-09-11: the component and fixed CPU cost gate pass. The 29 numerical,
synthetic and neural-path checks also pass on the T4 (including its device/gradient
check); four report-audit tests pin the sources, budget and arithmetic. Existing
native-count supervision reaches the new phase heads and rejects a phase-compatible
octave error; this proves compatibility, not selection of the next fit's objective.

## One advancing clock, corrected throughout the sequence

At 50 frame points/second, a cell has prior advance `a = 2^(-ell)/50`, where
`ell` is predicted log2(period/seconds). With incoming state `q`, predict the
right endpoint `p = q + a`. Normalize that endpoint's local phase observation
`(u,v)` by `s = sqrt(1 + u*u + v*v)` to get `(x,y)`. Then:

```
e = y*cos(2*pi*p) - x*sin(2*pi*p)
k = ln(2) * sigmoid(learned_gain_logit)
d = a * exp(k*e)
q_next = q + d
```

Phase enters the FORWARD clock, not only its training loss. The correction
changes speed over a cell, never jumps state: `d` stays positive and between
half and twice the prior advance. This relative correction bound is not an
absolute BPM clamp. Clock phase and returned cell BPM both derive from `q`;
the prior tempo is not incorrectly returned as the corrected tempo.

The 24,037-parameter CNN reuses the old 512-to-32 projection and dilated blocks,
predicts period plus two local phase components, and adds one origin head and
one shared gain. Only the origin at physical frame zero is consumed. Every later
frame's phase observes its incoming cell. No reference or availability mask is
an inference input. Convolution halo ownership is unchanged; processing clock
chunks carries the preceding state, including its gradient, without resets.

Zero observation means zero phase correction, not a detected silent interval or
a calibrated lack of rhythm. Correct rate can coast across weak observations;
a wrong rate still drifts there. Periodic phase cannot identify integer beat
counts or half/double tempo, and biased rate can leave a steady-state phase
offset even while feedback prevents unbounded drift. Convergence is not global:
antipodal observations can have expanding gradients. Native count/rate evidence
and representative music supervision remain necessary. We do not create beat
events from this always-positive oscillator or infer rhythm presence.

## Backpropagation and execution boundary

`scan.py` uses a CPU float64 scan plus an analytic first-order reverse scan.
For `h = -2*pi*(y*sin(2*pi*p) + x*cos(2*pi*p))`, its local Jacobians are:

```
dq_next/dq   = 1 + d*k*h
dq_next/dell = (exp(k*e) + d*k*h) * (-ln(2)*a)
dq_next/du   = d*k*(-sin(2*pi*p) - e*x)/s
dq_next/dv   = d*k*( cos(2*pi*p) - e*y)/s
dq_next/dk   = d*e
```

The reverse scan carries the total future-state gradient back through these
Jacobians, adding each point's direct loss gradient. Incoming gradients are
never modified. Time and saved-Jacobian memory are O(batch * cells); no dense
time-by-time matrix is constructed. Invalid/nonrepresentable clocks or gradients
fail explicitly instead of being clipped into plausible output.

The custom function follows PyTorch's [extension and gradcheck contract](https://docs.pytorch.org/docs/2.10/notes/extending.html),
including `save_for_backward` and `once_differentiable`. Tests compare the adjoint
with an independent ordinary-PyTorch unroll and numerical finite differences.
Higher-order differentiation is deliberately unsupported. CUDA inputs entail
CPU transfers and synchronization; this is not a GPU-native or ONNX-exportable
scan. Production admission will need an equivalent portable Rust scan and a
separately exported neural evidence head, not a Python runtime dependency.

## Readiness gate fixed before measurement

Run `python -m unittest discover -s experiments/phase_sync -p 'test_*.py' -v`.
Authored cases cover causal evidence use, correct rate/phase, drift, phase gaps,
abrupt changes, octave ambiguity, gradient correctness, state-carry and CNN
chunk equivalence, and invalid numerics. These are component tests, not musical
accuracy measurements or tuning examples.

Before any new musical fit, measure complete 60-second sequences (3,000 cells),
batch 1, float32 CNN/float64 clock, four CPU threads, one warmup and five timed
forward+backward iterations. Compare direct, integrated, and synchronized heads
on the SAME synthetic features/device, with no optimizer. The private Linux CPU
gate is median synchronized forward+backward <1 second and process peak RSS
<2 GiB. Record overhead versus both baselines without requiring a speedup.
Do not shorten the crop or silently switch to GPU to pass. A CUDA measurement is
additional portability/cost evidence, not a replacement for the CPU gate.

Even passing this gate only permits planning a bounded, matched-control musical
fit. Initialization, objective, budget and musical stop/admission criteria must
be fixed before that separate experiment. No claim that a larger encoder or a
Transformer is required follows from these component checks.

## Measured complete-crop cost

Same private Linux host, four CPU threads, same source/feature hashes, no competing
validation jobs; CPU-only PyTorch 2.10.0 and a separate PyTorch 2.10.0 CUDA run:

| Head | Parameters | CPU median forward + backward | T4 median forward + backward |
| --- | ---: | ---: | ---: |
| Independent phase/period | 24,003 | 27.28 ms | 6.39 ms |
| Old integrated clock | 23,970 | 30.37 ms | 8.15 ms |
| New synchronized clock | 24,037 | 64.34 ms | 42.14 ms |

The CPU process peaks at 397 MiB including the three heads and imported runtime,
below the pre-registered 2 GiB ceiling. The 64.34 ms median is below the 1 s ceiling
but is 2.12x the old integrated head's cost, not a speedup. T4's host-scan overhead
limits the accelerator benefit; CUDA peak allocated tensor memory is reported
separately from host RSS. The read-only CUDA container disabled its unavailable
kernel disk cache; one warmup precedes all five timed rounds.

Full samples, environment versions, hashes and recomputable gates:
[CPU](cost-cpu-v1.json), [CUDA](cost-cuda-v1.json). Both cost measurements use
authored features and a simple dense squared objective, not a musical training
batch or all possible future losses. They include validation checks and backward,
but exclude audio decode and frozen-encoder feature extraction. They are not
end-to-end audio latency. The local Windows VDI precheck was approximately 138 ms
with PyTorch 2.8; it is not the registered Linux gate or a hardware comparison.

Reproduce with `python -m experiments.phase_sync.benchmark --device cpu --output
NEW_PRIVATE_JSON` (or `--device cuda`). The output is exclusive and the seven
source/contract hashes are checked before and after execution. Timing is not
rerun as a flaky CI assertion; CI audits the retained report and numerical tests.

Next: pre-register ONE musical comparison on the same exposed roles, with matched
fitted-zero and same-checkpoint temporal controls, the failed baselines, native
tempo/count coverage and the unchanged nonregression gate. Do not silently load
whichever old checkpoint wins a corpus. Initialization, objective and fixed budget
need an explicit decision before training. Holdouts stay sealed; representative
weak-attack/change/no-rhythm supervision and independent admission remain open.
