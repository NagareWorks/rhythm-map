//! Evaluation-only complete-frame scoring, not a clock or a confidence model.
use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Clone, Copy, Default)]
pub struct State {
    pub beat: bool,
    pub downbeat: bool,
}

pub struct Evidence<'a> {
    pub beat: &'a [f32],
    pub downbeat: &'a [f32],
    pub available: Option<&'a [bool]>,
}

#[derive(Debug, Serialize)]
pub struct Evaluation {
    /// Independent-head pseudo-likelihood, not a calibrated probability.
    pub log_score: f64,
    pub scored_frames: usize,
    pub unavailable_frames: usize,
    /// Exact constancy diagnostics only; these are not silence classifiers.
    pub beat_varies: bool,
    pub downbeat_varies: bool,
}

fn log_sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        -(-value).exp().ln_1p()
    } else {
        value - value.exp().ln_1p()
    }
}

/// Every available frame is scored exactly once under each head. Comparing
/// hypotheses is valid only on identical evidence/availability/frame domains.
/// A downbeat is also a beat. The caller, not this scorer, owns clock dynamics,
/// support-window shape, alternative reporting and any missing-event model.
#[allow(clippy::float_cmp)] // Exact absence of variation, not near-equality.
pub fn score(evidence: &Evidence<'_>, states: &[State]) -> Result<Evaluation> {
    let n = evidence.beat.len();
    ensure!(
        n > 0 && evidence.downbeat.len() == n && states.len() == n,
        "frame length mismatch"
    );
    ensure!(
        evidence.available.is_none_or(|a| a.len() == n),
        "availability length mismatch"
    );
    ensure!(
        evidence
            .beat
            .iter()
            .chain(evidence.downbeat)
            .all(|v| v.is_finite()),
        "non-finite evidence"
    );
    ensure!(
        states.iter().all(|s| !s.downbeat || s.beat),
        "downbeat without beat"
    );
    let mut result = Evaluation {
        log_score: 0.0,
        scored_frames: 0,
        unavailable_frames: 0,
        beat_varies: false,
        downbeat_varies: false,
    };
    let mut first = None;
    for (i, state) in states.iter().enumerate() {
        if evidence.available.is_some_and(|a| !a[i]) {
            result.unavailable_frames += 1;
            continue;
        }
        let b = evidence.beat[i];
        let d = evidence.downbeat[i];
        let (first_b, first_d) = *first.get_or_insert((b, d));
        result.beat_varies |= b != first_b;
        result.downbeat_varies |= d != first_d;
        result.log_score += log_sigmoid(if state.beat {
            f64::from(b)
        } else {
            -f64::from(b)
        });
        result.log_score += log_sigmoid(if state.downbeat {
            f64::from(d)
        } else {
            -f64::from(d)
        });
        result.scored_frames += 1;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_both_absence_and_presence_without_candidate_count_normalization() {
        let b = [8., -8., 8., -8.];
        let d = [8., -8., -8., -8.];
        let evidence = Evidence {
            beat: &b,
            downbeat: &d,
            available: None,
        };
        let mut states = [State::default(); 4];
        states[0] = State {
            beat: true,
            downbeat: true,
        };
        states[2].beat = true;
        let correct = score(&evidence, &states).unwrap();
        states[2].downbeat = true;
        let extra = score(&evidence, &states).unwrap();
        assert!((correct.log_score - extra.log_score - 8.).abs() < 1e-12);
        states[2].downbeat = false;
        states[0].downbeat = false;
        let missing = score(&evidence, &states).unwrap();
        assert!((correct.log_score - missing.log_score - 8.).abs() < 1e-12);
        assert_eq!(correct.scored_frames, extra.scored_frames);
    }

    #[test]
    fn full_frame_score_is_additive_and_stable_at_extreme_logits() {
        let b = [1000., -1000., 8., -8.];
        let d = [1000., -1000., -8., -8.];
        let states = [
            State {
                beat: true,
                downbeat: true,
            },
            State::default(),
            State {
                beat: true,
                downbeat: false,
            },
            State::default(),
        ];
        let whole = score(
            &Evidence {
                beat: &b,
                downbeat: &d,
                available: None,
            },
            &states,
        )
        .unwrap();
        let left = score(
            &Evidence {
                beat: &b[..2],
                downbeat: &d[..2],
                available: None,
            },
            &states[..2],
        )
        .unwrap();
        let right = score(
            &Evidence {
                beat: &b[2..],
                downbeat: &d[2..],
                available: None,
            },
            &states[2..],
        )
        .unwrap();
        assert!(whole.log_score.is_finite());
        assert!((whole.log_score - left.log_score - right.log_score).abs() < 1e-12);
    }

    #[test]
    fn unavailable_is_not_negative_evidence_or_observed_variation() {
        let b = [-8., 8., -8.];
        let available = [true, false, true];
        let states = [State::default(); 3];
        let result = score(
            &Evidence {
                beat: &b,
                downbeat: &b,
                available: Some(&available),
            },
            &states,
        )
        .unwrap();
        assert_eq!((result.scored_frames, result.unavailable_frames), (2, 1));
        assert!(!result.beat_varies && !result.downbeat_varies);
        let unavailable = [false; 3];
        let result = score(
            &Evidence {
                beat: &b,
                downbeat: &b,
                available: Some(&unavailable),
            },
            &states,
        )
        .unwrap();
        assert_eq!(result.scored_frames, 0);
        assert!(!result.beat_varies && !result.downbeat_varies);
    }

    #[test]
    fn invalid_input_fails_before_scoring() {
        let b = [8., -8.];
        let evidence = Evidence {
            beat: &b,
            downbeat: &b,
            available: None,
        };
        assert!(score(&evidence, &[State::default()]).is_err());
        assert!(
            score(
                &evidence,
                &[State {
                    beat: false,
                    downbeat: true
                }; 2]
            )
            .is_err()
        );
        assert!(
            score(
                &Evidence {
                    beat: &[f32::NAN],
                    downbeat: &[0.],
                    available: None
                },
                &[State::default()]
            )
            .is_err()
        );
        assert!(
            score(
                &Evidence {
                    available: Some(&[true]),
                    ..evidence
                },
                &[State::default(); 2]
            )
            .is_err()
        );
    }
}
