# BRID source-alignment disposition v1

2026-09-15, before feature capture or source-expansion optimizer updates.

Disposition: retain all 93 published annotation pairs, including 0092, for
**exploratory native-tempo supervision only**. Residual alignment uncertainty
is accepted explicitly; musical alignment is not independently certified.
This completes the source-review decision, not execution admission or an
accuracy gate. The frozen two-arm design, source weight and membership stand.

Evidence: the frozen `brid-reference-v1.json` reproduces all 93 original RMS
rise profiles. Mean/equal-beat peaks are +5 ms for 92 records, +25 ms for 0092;
both halves of 0092 retain +25 ms. These are acoustic-envelope diagnostics,
not reference beat detectors. No strong evidence establishes which attack,
perceptual pulse or annotation convention is correct. Neither assuming exact
labels nor aligning them to this proxy is justified.

The original [dataset description](https://zenodo.org/records/14051323)
(`description.pdf`, SHA-256
`74c8cd5dab7881d4abef52fc7b14dfc905bb35291ec4d00bfa44daec88296ab5`,
page 2, mixture table) identifies 0092 / M2-44-SA as Reco-reco 4 + Surdo 2.
The table establishes composition, not the cause of the offset. Attack shape
is only one possible explanation. The original paper's annotation-method
description does not independently certify every file in this archive.

A constant annotation translation preserves inter-beat durations mathematically,
but shifts the placement of local rate transitions and support on the audio
timeline. Thus native-tempo training is not immune to alignment uncertainty.
The 25 ms proxy offset must not be called a measured annotation error, ignored
as irrelevant, or used to repair labels. Microtiming/annotation noise likewise
does not supply ground-truth tempo-change events.

No timestamp shift, extra mask, dropped recording, smoothed target, phase/bar
label, confidence weighting or source-weight adjustment is applied. BRID stays
one conservative corpus group with 10% native-loss weight and no teacher,
development, selector or acceptance role. Preserve original source bytes,
the exact q/mask reference packets and all uncertainty in the input report.
An independently established defect would require a versioned correction and
new preregistration, not an invisible edit to this experiment.

Stop this source-proxy investigation here. Next gates are mechanical input
capture/replay, old input identities and executable runner closure, not more
per-song offset tuning. A successful fit still needs independent acceptance
before any product promotion.
