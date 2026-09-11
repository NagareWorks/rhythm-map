# Range-safe transport to the existing global gradient clip

Status, 2026-09-11: numerical experiment, **not a musical training run** or a
product strategy. The failed [phase-sync fit](../phase_sync_fit/README.md) and
all its source pins remain closed and unchanged. Its missing state cannot be
reconstructed by these authored tests.

## What changes, and what does not

The 24,037-parameter frozen-feature CNN, native float32 neural operations,
float64 phase-clock forward and native beat-count loss are reused unchanged.
This component changes how the first derivative is represented on its way to
the already-required single global gradient clip. It neither learns a new
architecture nor clamps, detaches, resets or shortens the clock recurrence.
There are no song-dependent settings, data loaders or automatic retries here.

For a clock cell `q[t+1] = F(q[t], fields[t])`, reverse mode computes
`a[t] = upstream[t] + F_q[t] * a[t+1]`. A positive, finite forward clock can
still multiply many derivatives above one into an enormous `a`. The old native
float32 field-gradient cast can overflow **before** clipping has a chance to
act. Merely clipping the resulting infinities cannot recover their direction.

The new path is:

1. Run the same CNN and traced clock forward, retaining the actual local
   Jacobians and complete loss gradient with respect to the returned clock.
2. Evaluate that same adjoint with float64 mantissas and separate integer
   powers of two. Multiplication adds exponents instead of materializing an
   unrepresentably large number. Upstream contributions are still added at
   every cell; this is not truncated backpropagation.
3. Partition the clock-field gradients into 64-binary-exponent bands. For each
   band, send values in `[2^-64, 1)` through the **unchanged native CNN VJP**,
   then restore its exponent on the resulting parameter-gradient packet.
   Linearity of a VJP justifies summing these contributions. One band alone
   could turn small but representable contributions into float32 zeros.
4. Apply the registered recording/work weights, sum **all records of the
   original batch**, then compute `g * min(1, cap / (||g|| + 1e-6))` once in
   scaled form. Only the finite result is installed as native parameter grads.
   The component does not call the optimizer.

Clipping separate records or bands would generally change the batch direction;
that is intentionally forbidden by the integration contract. `record_gradient`
does not modify model parameters or existing `.grad` buffers. `install_clipped`
validates every parameter conversion before assigning any `.grad` buffer.

## Evidence and explicit limits

Authored tests compare all four scan-input gradients against an independent
Torch unroll, and native CNN gradients against ordinary backpropagation.
Forward values are bit-identical in these paired checks; gradients match within
declared dtype-appropriate tolerances, **not** universally bit for bit. Each
path's tiny first AdamW update is additionally checked against the explicit
zero-moment formula using its own gradients and a dtype-rounding bound. Near
zero gradients, AdamW's `g/(abs(g)+epsilon)` can amplify otherwise acceptable
gradient rounding: an initial cross-path weight tolerance failed on one float32
CPU component and one CUDA component (about `2.9e-7` and `3.9e-7` respectively).
Do not call those weight updates identical or loosen an accuracy gate to conceal
the difference. The gradient tolerances remain unchanged; the explicit optimizer
check tests the update operation rather than assuming conditioning. CUDA uses
the same reference and formula checks on-device.

The existing authored 3,000-cell opposed-phase witness has an origin derivative
around `10^78`: old float32 backward fails, whereas exponent transport survives.
Its origin derivative agrees with an independent 80-digit Decimal product to
relative error below `1e-12`. An affine native-float32 readout additionally tests
the whole parameter-VJP/clip path against a float64 reference. A separate
artificial Jacobian chain tests values beyond float64's exponent range.
None is a replay of the unsaved failed music state or an accuracy score.

This extends **range, not precision or conditioning**. A huge derivative is
still huge and can make the clipped direction dominated by early states; global
clipping does not establish useful learning. Cancellation/rounding remain
finite-precision operations. Zero-producing alignment losses are counted and
rejected before parameter installation, but this is not an arbitrary-precision
or componentwise relative-error guarantee. Native CNN VJPs and final native
gradient casts still have their ordinary dtype's rounding and underflow limits.
Finite final numbers do not prove every small contribution was preserved.

The component supports first-order, clock-only objectives and live float32 or
float64 parameters, not AMP, higher derivatives, or an auxiliary loss bypassing
the clock. It is not a drop-in `loss.backward()` implementation. Internal
exponent/band budgets fail closed; they are numerical implementation bounds,
not policies to tune per recording. A future fitter must preserve complete
record/batch weighting and explicitly connect this path to the failure recorder.
No existing fit runner has been switched to it in this commit.

## Cost contract and remaining gate

`contract-v1.json` fixes one 60 s / 3,000-cell input, 512-dimensional float32
Gaussian features, a 120 BPM reference, seed 777, four CPU threads, one warmup
and five measured repetitions. The Linux CPU component gate stays **median
below 1 s and process peak RSS below 2 GiB**, declared before measurement.
It includes the native loss, trace, all CNN VJP bands, record weighting,
accumulation and global clip. It excludes decoding, feature extraction,
optimizer and recorder snapshot I/O. CUDA is additional device/cost evidence,
not a substitute for the CPU gate. No optimizer steps or music inputs occur.

Measured [CPU report](cost-cpu-v1.json): **283.44 ms** median, **397.35 MiB**
process peak RSS, passing that fixed component gate. The [T4 report](cost-cuda-v1.json)
measures **265.46 ms**, with **1,381.45 MiB** host process peak RSS (including
the CUDA runtime, not GPU tensor memory). Both used the same authored feature
bytes and one exponent band on this fixture. These are **not a multi-band
worst-case cost bound**; the separate overflow test exercises at least four
bands but is not a representative full-CNN timing gate. Reports pin source and
input hashes, all five timing samples, and the actual band/gradient diagnostics.

This workload is broader and uses a different loss than the old phase-sync
component benchmark; its numbers must not be presented as a like-for-like
speedup or slowdown. Reproduce with `python -m experiments.scaled_adjoint.benchmark
--device cpu --output <new-private-path.json>` (or `cuda`). Run unit tests with
`python -m unittest discover -s experiments/scaled_adjoint -p 'test_*.py' -v`.

Next: integrate range-safe accumulation with durable failure observation and
measure complete observed-update and multi-band cost. Establish a separately frozen fit
protocol and explicit conditioning/failure gates before touching musical data.
The old failure remains a failure; holdouts, shipping Rust behavior, model packs
and release status are unchanged. Successful numerical transport alone does not
authorize restarting or promoting the closed experiment.
