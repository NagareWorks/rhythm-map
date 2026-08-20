use std::cmp::Ordering;

use rhythm_map_core::Analysis;
use serde::{Deserialize, Serialize};

use crate::{AcceptanceThresholds, GeneratedTruth};

/// One-to-one beat matching metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BeatMetrics {
    /// Number of matched beats.
    pub matched: usize,
    /// Fraction of predictions matched to truth.
    pub precision: f64,
    /// Fraction of truth beats recovered.
    pub recall: f64,
    /// Harmonic mean of precision and recall.
    pub f1: f64,
    /// Median absolute timestamp error among matched beats.
    pub median_absolute_error_ms: Option<f64>,
    /// 95th-percentile absolute timestamp error among matched beats.
    pub p95_absolute_error_ms: Option<f64>,
}

/// Relative tempo-curve errors sampled at predicted curve timestamps.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TempoMetrics {
    /// Number of points with a defined ground-truth tempo.
    pub sample_count: usize,
    /// Median absolute relative tempo error.
    pub median_absolute_error_percent: Option<f64>,
    /// 95th-percentile absolute relative tempo error.
    pub p95_absolute_error_percent: Option<f64>,
}

/// One-to-one, same-kind transition matching metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChangeMetrics {
    /// Number of matched expected transitions.
    pub matched: usize,
    /// Fraction of predictions matched to truth.
    pub precision: f64,
    /// Fraction of expected transitions recovered.
    pub recall: f64,
    /// Harmonic mean of precision and recall.
    pub f1: f64,
}

/// Metrics emitted for one analysis result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvaluationMetrics {
    /// Beat timestamp quality.
    pub beats: BeatMetrics,
    /// Tempo curve quality.
    pub tempo: TempoMetrics,
    /// Timing transition quality.
    pub changes: ChangeMetrics,
}

/// Scored case and acceptance decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaseEvaluation {
    /// Stable case identifier.
    pub id: String,
    /// Computed metrics.
    pub metrics: EvaluationMetrics,
    /// Whether every configured acceptance budget passed.
    pub passed: bool,
    /// Human-readable failed budgets.
    pub failures: Vec<String>,
}

/// Score one product analysis against deterministic ground truth.
#[must_use]
pub fn evaluate_analysis(
    id: impl Into<String>,
    analysis: &Analysis,
    truth: &GeneratedTruth,
    thresholds: &AcceptanceThresholds,
) -> CaseEvaluation {
    let beats = score_beats(
        &analysis
            .beats
            .iter()
            .map(|beat| beat.time_s)
            .collect::<Vec<_>>(),
        &truth
            .beats
            .iter()
            .map(|beat| beat.time_s)
            .collect::<Vec<_>>(),
        thresholds.beat_tolerance_ms / 1000.0,
    );
    let tempo = score_tempo(analysis, truth);
    let changes = score_changes(analysis, truth, thresholds.change_tolerance_s);
    let mut failures = Vec::new();
    if beats.f1 < thresholds.min_beat_f1 {
        failures.push(format!(
            "beat F1 {:.4} is below {:.4}",
            beats.f1, thresholds.min_beat_f1
        ));
    }
    match tempo.median_absolute_error_percent {
        Some(value) if value <= thresholds.max_tempo_median_error_percent => {}
        Some(value) => failures.push(format!(
            "median tempo error {value:.3}% exceeds {:.3}%",
            thresholds.max_tempo_median_error_percent
        )),
        None => failures.push("tempo curve has no scorable points".to_string()),
    }
    match tempo.p95_absolute_error_percent {
        Some(value) if value <= thresholds.max_tempo_p95_error_percent => {}
        Some(value) => failures.push(format!(
            "p95 tempo error {value:.3}% exceeds {:.3}%",
            thresholds.max_tempo_p95_error_percent
        )),
        None => failures.push("tempo curve has no scorable points".to_string()),
    }
    if changes.recall < thresholds.min_change_recall {
        failures.push(format!(
            "change recall {:.4} is below {:.4}",
            changes.recall, thresholds.min_change_recall
        ));
    }
    CaseEvaluation {
        id: id.into(),
        metrics: EvaluationMetrics {
            beats,
            tempo,
            changes,
        },
        passed: failures.is_empty(),
        failures,
    }
}

fn score_beats(predicted: &[f64], expected: &[f64], tolerance_s: f64) -> BeatMetrics {
    let mut predicted_index = 0;
    let mut expected_index = 0;
    let mut errors = Vec::new();
    while predicted_index < predicted.len() && expected_index < expected.len() {
        let difference = predicted[predicted_index] - expected[expected_index];
        if difference.abs() <= tolerance_s {
            errors.push(difference.abs() * 1000.0);
            predicted_index += 1;
            expected_index += 1;
        } else if difference < 0.0 {
            predicted_index += 1;
        } else {
            expected_index += 1;
        }
    }
    let matched = errors.len();
    let precision = safe_ratio(matched, predicted.len());
    let recall = safe_ratio(matched, expected.len());
    BeatMetrics {
        matched,
        precision,
        recall,
        f1: f1(precision, recall),
        median_absolute_error_ms: percentile(errors.clone(), 0.5),
        p95_absolute_error_ms: percentile(errors, 0.95),
    }
}

fn score_tempo(analysis: &Analysis, truth: &GeneratedTruth) -> TempoMetrics {
    let errors = analysis
        .tempo_curve
        .iter()
        .filter_map(|point| {
            truth
                .tempo_at(point.time_s)
                .map(|expected| ((point.bpm - expected) / expected).abs() * 100.0)
        })
        .collect::<Vec<_>>();
    TempoMetrics {
        sample_count: errors.len(),
        median_absolute_error_percent: percentile(errors.clone(), 0.5),
        p95_absolute_error_percent: percentile(errors, 0.95),
    }
}

fn score_changes(analysis: &Analysis, truth: &GeneratedTruth, tolerance_s: f64) -> ChangeMetrics {
    let mut candidates = analysis
        .change_points
        .iter()
        .enumerate()
        .flat_map(|(predicted_index, predicted)| {
            truth
                .change_points
                .iter()
                .enumerate()
                .filter(move |(_, expected)| expected.kind == predicted.kind)
                .map(move |(expected_index, expected)| {
                    (
                        (predicted.time_s - expected.time_s).abs(),
                        predicted_index,
                        expected_index,
                    )
                })
        })
        .filter(|(distance, _, _)| *distance <= tolerance_s)
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| float_order(left.0, right.0));
    let mut used_predictions = vec![false; analysis.change_points.len()];
    let mut used_expected = vec![false; truth.change_points.len()];
    let mut matched = 0;
    for (_, predicted_index, expected_index) in candidates {
        if !used_predictions[predicted_index] && !used_expected[expected_index] {
            used_predictions[predicted_index] = true;
            used_expected[expected_index] = true;
            matched += 1;
        }
    }
    let precision = safe_ratio(matched, analysis.change_points.len());
    let recall = safe_ratio(matched, truth.change_points.len());
    ChangeMetrics {
        matched,
        precision,
        recall,
        f1: f1(precision, recall),
    }
}

// Evaluation vectors are memory-bounded, and quantiles are validated call-site
// constants in [0, 1], so the computed index is representable as usize.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn percentile(mut values: Vec<f64>, quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| float_order(*left, *right));
    let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
    values.get(index).copied()
}

#[allow(clippy::cast_precision_loss)]
fn safe_ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn f1(precision: f64, recall: f64) -> f64 {
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn float_order(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beat_matching_is_one_to_one() {
        let metrics = score_beats(&[0.01, 0.02, 0.51], &[0.0, 0.5], 0.07);
        assert_eq!(metrics.matched, 2);
        assert!((metrics.precision - 2.0 / 3.0).abs() < 1e-9);
        assert!((metrics.recall - 1.0).abs() < f64::EPSILON);
    }
}
