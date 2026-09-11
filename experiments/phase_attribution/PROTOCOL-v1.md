# Fixed prediction attribution v1

Declared before inspecting the retained per-frame values. This is a descriptive
diagnostic of the closed scaled_phase_fit v1, NOT a new fit, checkpoint selection,
acceptance gate or independent evaluation. Reuse all 40 pinned packets and all
five exported predictions; keep fit/development/diagnostic roles and work macros.
Never load weights, hidden features, audio or run a model/optimizer. Old source,
results, metrics and failures remain immutable. Derived output must be new and
outside the repository; only sanitized summaries belong in Git.

## Identities and denominators

Pin the prior plan, report and input manifest by SHA256. Check all source hashes
in the old plan and every input/prediction archive hash before loading arrays.
Use native reference support and, separately, its intersection with raw support;
intervals require every included frame, not only endpoints. Recompute every
available original audio/zero/control/raw metric as an audit before attribution.
Never omit a bad supported prediction to improve a denominator. Require finite
positive rates, reference geometry and finite local fields. Empty supports are
null metrics with explicit zero counts, not perfect accuracy.

## Fixed decompositions (not alternate estimators)

1. **Beat level and local rate:** for each supported cell let
   `z = log2(predicted_rate/reference_rate)`, `k = floor(z + 0.5)`.
   Retain median signed z, median absolute z, and nearest-octave residual
   `abs(z-k)`. Categories are native (k=0), half (k=-1), double (k=1), other
   octave, or off-grid. An octave category additionally requires
   `abs(z-k) <= log2(1.1)` (symmetric multiplicative factor 1.1, not +/-10%
   linear error); all other cells are off-grid. Report each fraction and longest
   contiguous run without joining missing support. These are descriptive bins,
   not ground-truth beat-level labels or a new pass threshold.
   Also report the residual median percentage error after ONE oracle recording
   level `floor(median(z)+0.5)`, and the cell-wise oracle folded residual. Both
   use reference labels, are explanatory lower-information diagnostics, and
   MUST NOT replace the original scores or become output corrections. A small
   folded residual does not establish perceptually acceptable tempo.
2. **Count versus circular phase:** retain original wrapped phase MAE and 4 s
   count MAE. In each contiguous support component, subtract its starting
   prediction/reference offset ONLY for count diagnosis. Decompose relative
   count error `d` into `floor(d+0.5)` plus a residual in [-0.5,0.5). Retain
   component endpoints, native/predicted total advances, terminal signed drift,
   relative drift MAE, integer-offset range and transitions. Integer transitions
   are crossings of diagnostic rounding boundaries, NOT detected missing beats.
   Never bridge annotation gaps or reinterpret a constant origin error as drift.
3. **Local prior versus feedback:** compare exported prior cell rate
   `2**(-fields[:-1,0])` with the actual saved clock's cell rate. Check the local
   feedback equation algebraically using saved incoming q and end-frame vectors;
   do not rerun the clock. Retain correction log2 size, prior error, actual error,
   phase-vector availability/phase error and per-cell theoretical correction
   bound `gain * norm(u,v)/hypot(1,norm(u,v)) / ln(2)`. Report how often the native
   rate is outside that bound. This concerns fixed local fields, not what another
   trained model could do; vector magnitude is not calibrated confidence.
4. **Loss accounting:** recompute circular loss and native log-count MSE at the
   existing 1/25/100/200-frame lags. Show each term separately, without changing
   weights or claiming scalar loss magnitudes prove gradient competition.

All cohort summaries average recordings within work, then works, with no pooled
frame/record shortcut. Retain per-record values and support geometry, including
unfavorable cases. Fit is descriptive only. Report associations and algebraic
constraints, not a proven training/data cause. No seed/loss/epoch sweep, data
role changes, holdout access, model promotion or production changes follow
automatically. A proposed next mechanism needs a distinct falsifiable test.
