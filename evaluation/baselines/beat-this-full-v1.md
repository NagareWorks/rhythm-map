# Beat This full model: generated-v1 baseline

Measured on 2026-08-20 with an optimized build and the checked-in
`beat-this-full-v1.json` manifest. The verified manifest SHA-256 was
`ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`.
Runtime is machine-specific and is recorded only as a diagnostic, not an
acceptance threshold.

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

The first engineering targets are:

1. add metrical-level normalization for the clear half/double-time failure in
   `subdivision-90`;
2. inspect raw observations around jumps and gaps before changing change-point
   thresholds;
3. calibrate the drumless ramp profile against licensed real examples so a
   synthetic timbre mismatch is not mistaken for a general model limitation.

Regenerate the full JSON report with the command documented in
`evaluation/README.md`. Do not copy model weights or private evaluation audio
into the repository.
