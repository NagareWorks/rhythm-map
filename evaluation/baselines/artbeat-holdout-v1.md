# ARTBeaT disjoint holdout v1

The holdout definition was committed as
`7c56bfe3dbb6ed5001cd7093ce4aca8b13b047f3` before model inference. It fixes
nine cases, a 70 ms matching tolerance, a minimum per-case beat F1 of 0.80, and
the registered `viterbi-edge-logit-minus-3.0-bias-2.0` candidate. The ideal
observation/oracle evaluation passed all nine cases before the holdout was
opened.

The one-time decoder evaluation used model-pack manifest SHA-256
`ccedbfeb35b4f584834df3aca1ea41899ed39fbaf2efad9e2cc71426aed9e23d`.
The candidate failed the no-regression and absolute-quality gates:

| Metric | Upstream | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Mean beat F1 | 0.67915 | 0.67771 | -0.00144 |
| Mean precision | 0.67648 | 0.66981 | -0.00667 |
| Mean recall | 0.71913 | 0.72126 | +0.00214 |
| Mean predicted beats | 45.22 | 46.22 | +1.00 |
| Cases meeting F1 >= 0.80 | 4 / 9 | 4 / 9 | 0 |

The first three arrangement-entry cases and the 7/8 drum pattern passed. Five
cases did not:

| Case | Upstream F1 | Candidate F1 | Interpretation |
| --- | ---: | ---: | --- |
| `artbeat-holdout-17-doom7` | 0.61111 | 0.61111 | Whole-track double-event ambiguity; 73 predictions for 35 truth beats. |
| `artbeat-holdout-22-polyrhythm-94` | 0.47826 | 0.47826 | Competing 4:3 pulse; the edge-only path makes no decision. |
| `artbeat-holdout-23-deception-102` | 0.00000 | 0.00000 | Correct-sized sequence at the wrong intended phase/pulse. |
| `artbeat-holdout-24-syncopated-94` | 0.35052 | 0.33663 | Regression: four extra candidate events without additional matches. |
| `artbeat-holdout-25-metal11` | 0.76744 | 0.76836 | Two extra matches, but still below the locked quality gate. |

This result rejects promotion of the edge-connected decoder. It is useful for
recovering a long weak sequence attached to a known path edge, but it does not
resolve whole-track pulse level or phase. The next comparison must change the
observation evidence or represent competing beat-phase hypotheses explicitly;
lowering the threshold or tuning another policy on this opened holdout would
invalidate its role.

The suite is case-disjoint from ARTBeaT calibration, not corpus-disjoint. Its
failure is sufficient to reject this candidate, but a future positive claim
still requires a separate timestamped source.
