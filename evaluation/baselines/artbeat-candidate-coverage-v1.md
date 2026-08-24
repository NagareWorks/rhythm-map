# ARTBeaT candidate-evidence baseline v1

This calibration run asks two separate questions before changing the shipping
estimator:

1. Did the observation backend emit a real local maximum near the annotated
   beat?
2. If it did, can a fixed pulse/phase hypothesis set contain a better sequence
   without inventing timestamps?

The source report is intentionally kept outside Git at
`D:/rhythm-map-eval/reports/artbeat-candidate-coverage-v1.json` because it
contains every evaluated timestamp. It was produced from the 15-case
`artbeat-v1` calibration suite with report schema 3 and model manifest SHA-256
`ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`.

| Case | Candidate recall | Selected F1 | Naive top-1 F1 | Best top-K F1 | Best hypothesis | Best rank |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 05 75 to 150 | 1.0000 | 0.7907 | 0.5143 | 0.8475 | midpoint augmented | 4 |
| 06 150 to 75 | 1.0000 | 0.7727 | 0.5143 | 0.8525 | midpoint augmented | 4 |
| 07 75 to 112.5 | 1.0000 | 0.8696 | 0.6471 | 0.8696 | selected | 2 |
| 08 112.5 to 75 | 1.0000 | 0.8696 | 0.5882 | 0.8696 | selected | 3 |
| 09 90 to 80 | 0.9600 | 0.9600 | 0.6486 | 0.9600 | selected | 2 |
| 10 90 to 120 | 1.0000 | 1.0000 | 0.6512 | 1.0000 | selected | 3 |
| 11 60 to 80 | 1.0000 | 0.9268 | 0.5806 | 0.9268 | selected | 3 |
| 12 80 to 150 | 1.0000 | 0.8000 | 0.5000 | 0.8421 | midpoint augmented | 4 |
| 13 180 to 120 | 0.9697 | 0.7500 | 0.4167 | 0.7500 | selected | 3 |
| 14 240 to 96 | 0.9756 | 0.7778 | 0.4643 | 0.7778 | selected | 3 |
| 15 85 to 127.5 | 0.8250 | 0.6486 | 0.4912 | 0.6486 | selected | 3 |
| 18 piano rubato | 1.0000 | 0.7568 | 0.4407 | 0.7647 | midpoint augmented | 4 |
| 19 ramp 80 to 200 | 1.0000 | 0.6667 | 0.3922 | 0.8471 | midpoint augmented | 4 |
| 20 ramp 200 to 80 | 1.0000 | 0.7077 | 0.4528 | 0.8182 | midpoint augmented | 4 |
| 21 polyrhythm 70 to 105 | 1.0000 | 0.7805 | 0.5333 | 0.7805 | selected | 3 |
| **Mean** | **0.9820** | **0.8052** | **0.5224** | **0.8370** |  |  |

The backend therefore supplies candidate evidence for nearly every annotated
beat. Case 15 is the main evidence-limited exception. The fixed hypothesis set
raises the oracle ceiling only modestly, from 0.8052 to 0.8370, but the naive
ranking is invalid: its top-ranked hypothesis is never the truth-best member.
It rewards the regularity of alternating half-time subsets without charging
them for discarding strong selected events.

This result does **not** justify a product strategy or per-song oracle routing.
It rejects the naive score and motivates one truth-free, auditable evidence
score that accounts for retained and discarded backend evidence. Any resulting
candidate remains evaluation-only until it passes a new precommitted holdout.
