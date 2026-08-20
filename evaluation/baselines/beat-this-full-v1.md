# Beat This full model: generated-v1 baseline

Measured on 2026-08-20 with an optimized build and the checked-in
`beat-this-full-v1.json` manifest. The verified manifest SHA-256 was
`ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`.
Runtime is machine-specific and is recorded only as a diagnostic, not an
acceptance threshold.

## Initial baseline

| Case | Oracle | End to end | Beat F1 | Tempo median error | Tempo P95 error | Change recall | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant-120 | pass | pass | 1.0000 | 0.00% | 0.00% | 1.00 | 2.87 s |
| step-120-160 | pass | fail | 0.8989 | 1.32% | 19.63% | 0.00 | 3.39 s |
| ramp-96-144 | pass | fail | 0.6406 | 2.80% | 50.30% | 0.00 | 14.67 s |
| gap-128 | pass | fail | 0.9412 | 1.90% | 2.34% | 0.00 | 2.74 s |
| subdivision-90 | pass | fail | 0.6667 | 96.08% | 104.17% | 1.00 | 3.28 s |

Total model-backed analysis time was about 26.94 seconds for 102 seconds of
generated audio on this machine. All five oracle paths passed. The suite-level
decision is therefore `observation_path`: failures emerge after audio is
converted into beat observations, but the paired test alone does not prove that
the neural network should be replaced.

## Observation-recovery baseline

The same model and suite were rerun after adding raw observation diagnostics,
PCM activity, evidence-based half-time selection, silence-event rejection, and
short smeared-jump recovery.

| Case | Raw / analyzed beats | End to end | Beat F1 | Tempo median error | Tempo P95 error | Change recall | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant-120 | 32 / 32 | pass | 1.0000 | 0.00% | 0.00% | 1.00 | 2.37 s |
| step-120-160 | 45 / 45 | fail | 0.8989 | 1.32% | 19.63% | 1.00 | 2.72 s |
| ramp-96-144 | 68 / 68 | fail | 0.6406 | 2.80% | 50.30% | 0.00 | 12.16 s |
| gap-128 | 36 / 32 | pass | 1.0000 | 1.90% | 2.34% | 1.00 | 3.23 s |
| subdivision-90 | 60 / 30 | pass | 1.0000 | 1.01% | 1.01% | 1.00 | 3.63 s |

Total model-backed analysis time was about 24.11 seconds. The acceptance
thresholds were unchanged. `gap-128` and `subdivision-90` now pass completely,
and `step-120-160` recovers its change point. Its remaining beat and P95 tempo
failures come from incorrect events during the model's roughly three-second
transition; the drumless ramp remains an unresolved observation-path case.

That baseline left two engineering targets:

1. distinguish isolated duplicate/missed events around abrupt tempo changes
   without suppressing real subdivisions;
2. calibrate the drumless ramp profile against licensed real examples so a
   synthetic timbre mismatch is not mistaken for a general model limitation.

## Guarded transition-grid baseline

The abrupt-transition failure was rerun after adding grid recovery guarded by
stable plateaus and explicit duplicate/missed-event evidence.

| Case | Raw / analyzed beats | End to end | Beat F1 | Tempo median error | Tempo P95 error | Change recall | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant-120 | 32 / 32 | pass | 1.0000 | 0.00% | 0.00% | 1.00 | 2.73 s |
| step-120-160 | 45 / 44 | pass | 1.0000 | 0.29% | 1.32% | 1.00 | 3.37 s |
| ramp-96-144 | 68 / 68 | fail | 0.6406 | 2.80% | 50.30% | 0.00 | 13.80 s |
| gap-128 | 36 / 32 | pass | 1.0000 | 1.90% | 2.34% | 1.00 | 3.80 s |
| subdivision-90 | 60 / 30 | pass | 1.0000 | 1.01% | 1.01% | 1.00 | 4.31 s |

The final run took about 28.00 seconds. Repeated candidate runs with identical
accuracy metrics ranged from 21.77 to 48.50 seconds, so runtime is retained as a
host-load-sensitive diagnostic rather than an acceptance claim. Thresholds were
again unchanged.
`step-120-160` now passes completely, while constant tempo, silence,
subdivision, and the genuine ramp path retain their prior accuracy. The only
remaining failure in this generated suite is the drumless ramp.

The next engineering targets are:

1. validate repaired downbeat phase, not only beat timestamps, around tempo
   changes; and
2. calibrate the drumless ramp profile against licensed real examples before
   changing the observation backend or adding a learned head.

Regenerate the full JSON report with the command documented in
`evaluation/README.md`. Do not copy model weights or private evaluation audio
into the repository.

## Downbeat phase baseline

The suite was extended with an independent downbeat F1 gate of 0.95, then rerun
after evidence-based half-bar candidate selection and recovered-grid boundary
realignment. No beat, tempo, or change-point threshold was relaxed.

| Case | Raw / analyzed beats | End to end | Beat F1 | Downbeat F1 | Tempo median error | Tempo P95 error | Change recall | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant-120 | 32 / 32 | pass | 1.0000 | 1.0000 | 0.00% | 0.00% | 1.00 | 2.31 s |
| step-120-160 | 45 / 44 | pass | 1.0000 | 1.0000 | 0.29% | 1.32% | 1.00 | 3.23 s |
| ramp-96-144 | 68 / 68 | fail | 0.6406 | 0.2917 | 2.80% | 50.30% | 0.00 | 10.46 s |
| gap-128 | 36 / 32 | pass | 1.0000 | 1.0000 | 1.90% | 2.34% | 1.00 | 2.66 s |
| subdivision-90 | 60 / 30 | pass | 1.0000 | 1.0000 | 1.01% | 1.01% | 1.00 | 3.49 s |

The model-backed analysis time was 22.15 seconds. Before repair, downbeat F1
was 0.6667 for `constant-120`, 0.6250 for `step-120-160`, and 0.6957 for
`subdivision-90`; `gap-128` was already 1.0000. The repaired cases now reach
1.0000 without changing their beat, tempo, or change-point metrics. The
drumless ramp remains the only failing generated case and is unchanged because
its primary problem is the upstream beat observation sequence, not an isolated
bar-phase ambiguity.
