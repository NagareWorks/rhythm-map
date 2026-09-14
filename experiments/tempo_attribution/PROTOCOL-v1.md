# Retained tempo evidence attribution v1

Declared before inspecting the four-arm per-cell outputs. This is descriptive
analysis of the closed `separated_evidence_fit` outcome, not training, inference,
selection, a new accuracy gate or a replacement estimator. Keep all 40 records,
all ten saved evidence variants, raw/v1 comparators, original masks and work
weights. Never open hidden features, weights, audio, holdouts or new music.

Pin the previous plan/report/input manifest, their source closure, and all 80
input/prediction packets before array access; verify again after analysis.
Lazily read only reference/valid/raw/raw_valid/v1 and saved prediction fields.
Recompute all applicable original metrics on native and common support using
the unchanged implementation. Never discard invalid required cells; empty
support has explicit zero counts and null measurements. Do not trim edges,
align time, choose windows or remove failing cases.

## Fixed calculations

1. Reuse `phase_attribution.diagnostic.rate_diagnosis` unchanged for log-rate
   error, factor-1.1 octave bands, longest contiguous runs and reference-informed
   recording/per-cell octave diagnostics. These use the answer and are NOT
   repaired accuracy or a perceptual-octave label. No folded output is exported.
2. For every supported cell, let `x=log2(reference_rate)`, `y=log2(saved_rate)`,
   `e=y-x`, and `b=mean(e)`. Report MSE, `b`, `b*b`, and `mean((e-b)**2)`;
   verify MSE = squared bias + residual MSE. Every cell is 1/50 s, so the within-
   record mean is duration-weighted. Bias is a reference-informed constant
   offset for accounting only, not an estimated correction. Residual error may
   contain annotation variation/noise and is not automatically a change-point
   error. Retain reference/predicted log-rate standard deviations and covariance
   response slope; a constant reference has null slope, not perfect tracking.
3. At fixed 50/200-cell lags (1/4 s), compare `dy=y[t+lag]-y[t]` with
   `dx=x[t+lag]-x[t]`. Require every intervening cell, including both endpoints,
   to be supported: never bridge gaps. Retain pair counts, `sqrt(mean(dx**2))`
   (the zero-change predictor's RMSE), `sqrt(mean((dy-dx)**2))`, response slope
   `sum(dx*dy)/sum(dx*dx)`, and direction agreement where reference magnitude is
   at least `log2(1.1)`. Report all pairs, reference-only large-change pairs
   (`abs(dx)>=log2(1.1)`) and remaining small-change pairs. The latter are NOT
   silence, missing-beat or stable-tempo ground-truth labels. A reference RMS
   <=1e-10 makes slope undefined; counts/other measurements remain. No lag scan,
   smoothing, relative time shift, new policy or model output is introduced.
4. Report reference-only BPM quantiles and duration fractions in fixed bands
   `<60`, `60..90`, `90..120`, `120..180`, `>=180`, plus all lag-pair counts.
   Fit/development/diagnostic populations stay separate; these summaries can
   reveal coverage differences, not prove a causal training-data deficiency.

Apply all calculations on native reference support and separately on unchanged
common support (raw/v1 only common). Summaries average recordings within each
work then works, never pool frames/recordings across works. For every scalar
retain missing-record counts; a missing required record leaves the complete
macro null. Ratios of aggregated squared bias to aggregated MSE are accounting
shares, not mean per-record ratios. Retain all per-record values so minority
regressions remain visible. No p-values or independent-generalization claims.

Compare natural with fitted-zero and fixed-weight zero/time-mean/half-roll
controls without selecting a winner or revising any old gate. Reference-only
decompositions explain error geometry; neither speed-prior, architecture,
optimization nor supervision causality follows from them alone. All old
decisions, one-call/default behavior and release state remain unchanged.
