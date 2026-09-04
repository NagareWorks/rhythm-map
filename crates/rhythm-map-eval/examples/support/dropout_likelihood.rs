//! Normalized reference densities for an evaluation-only dropout diagnostic.
use anyhow::{Result, ensure};
use serde::Serialize;

fn log_sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        -(-z).exp().ln_1p()
    } else {
        z - z.exp().ln_1p()
    }
}

fn log_add(a: f64, b: f64) -> f64 {
    let high = a.max(b);
    if high == f64::NEG_INFINITY {
        high
    } else {
        high + ((a.min(b) - high).exp()).ln_1p()
    }
}

#[derive(Debug, Serialize)]
pub struct Emission {
    pub absent_log_density: f64,
    pub present_log_density: f64,
    pub log_likelihood_ratio: f64,
    /// Conditional on a hypothesized event; NOT the probability of that event.
    pub missing_given_present: f64,
}

/// Let q=sigmoid(z), h(z)=q(1-q). Densities on the REAL LOGIT axis:
/// visible=2qh, background=2(1-q)h. Both integrate to one by dq=h dz.
/// Present events mix visible/background with normalized weights (1-r,r).
/// These are explicit reference assumptions, not fitted Beat This densities.
pub fn emission(z: f64, missing_rate: f64) -> Result<Emission> {
    ensure!(z.is_finite(), "non-finite logit");
    ensure!(
        missing_rate.is_finite() && (0.0..=1.0).contains(&missing_rate),
        "invalid missing rate"
    );
    let positive = log_sigmoid(z);
    let negative = log_sigmoid(-z);
    let common = 2.0_f64.ln() + positive + negative;
    let absent = common + negative;
    // Ratio form avoids cancellation of large common density terms.
    let ratio = log_add(missing_rate.ln(), (-missing_rate).ln_1p() + z);
    Ok(Emission {
        absent_log_density: absent,
        present_log_density: absent + ratio,
        log_likelihood_ratio: ratio,
        missing_given_present: (missing_rate.ln() - ratio).exp(),
    })
}

#[derive(Debug, Serialize)]
pub struct PathScore {
    pub log_density: f64,
    pub log_ratio_to_all_absent: f64,
    pub scored_frames: usize,
    pub unavailable_frames: usize,
    pub scored_hypothesized_frames: usize,
    pub conditional_expected_missing_frames: f64,
}

/// Scores GIVEN state masks, never infers a clock or observed timestamps.
pub fn score(
    logits: &[f64],
    states: &[bool],
    available: Option<&[bool]>,
    rate: f64,
) -> Result<PathScore> {
    ensure!(
        !logits.is_empty() && logits.len() == states.len(),
        "frame length mismatch"
    );
    ensure!(
        available.is_none_or(|a| a.len() == logits.len()),
        "availability length mismatch"
    );
    let mut result = PathScore {
        log_density: 0.0,
        log_ratio_to_all_absent: 0.0,
        scored_frames: 0,
        unavailable_frames: 0,
        scored_hypothesized_frames: 0,
        conditional_expected_missing_frames: 0.0,
    };
    for (i, (&z, &present)) in logits.iter().zip(states).enumerate() {
        let value = emission(z, rate)?;
        if available.is_some_and(|a| !a[i]) {
            result.unavailable_frames += 1;
            continue;
        }
        result.scored_frames += 1;
        result.log_density += if present {
            value.present_log_density
        } else {
            value.absent_log_density
        };
        if present {
            result.log_ratio_to_all_absent += value.log_likelihood_ratio;
            result.scored_hypothesized_frames += 1;
            result.conditional_expected_missing_frames += value.missing_given_present;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_background_and_mixture_are_normalized_densities() {
        let mut sums = [0.0; 3];
        // Midpoint quadrature, z in [-24,24], independent of any music labels.
        for i in 0..48_000 {
            let z = -24.0 + (f64::from(i) + 0.5) * 0.001;
            sums[0] += emission(z, 0.0).unwrap().present_log_density.exp() * 0.001;
            sums[1] += emission(z, 1.0).unwrap().present_log_density.exp() * 0.001;
            sums[2] += emission(z, 0.1).unwrap().present_log_density.exp() * 0.001;
        }
        for sum in sums {
            assert!((sum - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn normalized_dropout_cannot_reverse_negative_logit_evidence() {
        // Boundary/property checks, not parameter fitting or candidate search.
        for rate in [0.0, 0.1, 0.5, 0.99] {
            for z in [-1000.0, -8.0, -2.0, -0.001] {
                let value = emission(z, rate).unwrap();
                assert!(value.log_likelihood_ratio < 0.0);
                assert!(value.present_log_density.is_finite());
            }
            assert!(emission(8.0, rate).unwrap().log_likelihood_ratio > 0.0);
        }
        assert!(emission(-8.0, 1.0).unwrap().log_likelihood_ratio.abs() < 1e-12);
    }

    #[test]
    fn posterior_missing_is_not_beat_confidence() {
        let background = emission(-8.0, 0.1).unwrap();
        assert!(background.missing_given_present > 0.99);
        assert!(background.log_likelihood_ratio < 0.0);
        let strong = emission(8.0, 0.1).unwrap();
        assert!(strong.missing_given_present < 0.001);
        let direct = (0.1 + 0.9 * (-2.0_f64).exp()).ln();
        assert!((emission(-2.0, 0.1).unwrap().log_likelihood_ratio - direct).abs() < 1e-12);
    }

    #[test]
    fn unavailable_does_not_support_or_penalize_invented_events() {
        let a = score(&[-8.0, 8.0], &[true, false], Some(&[false, true]), 0.1).unwrap();
        let b = score(&[-8.0, 8.0], &[false, false], Some(&[false, true]), 0.1).unwrap();
        assert!((a.log_density - b.log_density).abs() < 1e-12);
        assert_eq!((a.scored_frames, a.unavailable_frames), (1, 1));
        assert_eq!(a.scored_hypothesized_frames, 0);
        assert_eq!(
            score(&[-8.0], &[true], Some(&[false]), 0.1)
                .unwrap()
                .scored_frames,
            0
        );
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert!(emission(f64::NAN, 0.1).is_err());
        assert!(emission(0.0, -0.1).is_err());
        assert!(emission(0.0, 1.1).is_err());
        assert!(score(&[], &[], None, 0.1).is_err());
        assert!(score(&[0.0], &[], None, 0.1).is_err());
        assert!(score(&[0.0], &[true], Some(&[]), 0.1).is_err());
    }
}
