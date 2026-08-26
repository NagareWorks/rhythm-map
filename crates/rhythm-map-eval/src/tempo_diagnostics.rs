use std::{collections::BTreeMap, path::Path};

use anyhow::{Result, bail};
use rhythm_map_core::Analysis;
use serde::{Deserialize, Serialize};

use crate::{
    GeneratedTruth, SuitePurpose,
    runner::{estimator_for_policy, load_case_truth, load_suite},
};

/// One estimator tempo sample paired with exact calibration truth.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TempoDiagnosticPoint {
    /// Midpoint timestamp of the estimated inter-beat interval.
    pub time_s: f64,
    /// Tempo directly implied by the two accepted beat timestamps.
    pub observed_interval_bpm: f64,
    /// Exact local tempo from the independent annotation.
    pub truth_bpm: f64,
    /// Regularized tempo emitted by the estimator.
    pub estimated_bpm: f64,
    /// Regularized tempo divided by the accepted-timestamp interval tempo.
    pub regularized_to_observed_ratio: f64,
    /// Estimated tempo divided by truth tempo.
    pub estimated_to_truth_ratio: f64,
    /// Base-two logarithm of `estimated_to_truth_ratio`.
    pub log2_ratio: f64,
    /// Nearest power-of-two metrical level relative to truth.
    pub nearest_octave_shift: i8,
    /// Relative error after removing the nearest power-of-two shift.
    pub octave_residual_percent: f64,
    /// Absolute relative error before metrical normalization.
    pub absolute_error_percent: f64,
    /// Estimator confidence attached to this tempo point.
    pub confidence: f64,
}

/// One contiguous run whose tempo error exceeds the diagnostic threshold.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TempoErrorRun {
    /// First failing tempo-point timestamp.
    pub start_s: f64,
    /// Last failing tempo-point timestamp.
    pub end_s: f64,
    /// Number of failing points in the run.
    pub sample_count: usize,
    /// Most frequent nearest power-of-two metrical level in the run.
    pub dominant_octave_shift: i8,
    /// Median estimated-to-truth tempo ratio.
    pub median_estimated_to_truth_ratio: f64,
    /// Median absolute relative error before metrical normalization.
    pub median_absolute_error_percent: f64,
    /// Largest absolute relative error in the run.
    pub maximum_absolute_error_percent: f64,
    /// Median error after removing the nearest power-of-two shift.
    pub median_octave_residual_percent: f64,
}

/// Truth-assisted local-tempo diagnosis for one calibration case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TempoDiagnosticCase {
    /// Stable evaluation case identifier.
    pub id: String,
    /// Estimator-wide median tempo summary.
    pub global_bpm: Option<f64>,
    /// Number of estimator points with defined truth.
    pub scored_point_count: usize,
    /// Number of points at or above the diagnostic error threshold.
    pub error_point_count: usize,
    /// Consecutive high-error regions.
    pub error_runs: Vec<TempoErrorRun>,
    /// Every scorable local tempo point.
    pub points: Vec<TempoDiagnosticPoint>,
    /// Estimator warnings emitted for the case.
    pub warnings: Vec<String>,
}

/// Truth-assisted local-tempo diagnosis for one calibration suite.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TempoDiagnosticEvaluation {
    /// Report schema version.
    pub schema_version: u32,
    /// Stable suite identifier.
    pub suite_id: String,
    /// Calibration is required because the report exposes truth per timestamp.
    pub suite_purpose: SuitePurpose,
    /// Stable estimator policy label.
    pub estimator_policy: String,
    /// Inclusive absolute-error threshold used to form error runs.
    pub minimum_error_percent: f64,
    /// Requested cases, or every case when no filter was supplied.
    pub cases: Vec<TempoDiagnosticCase>,
}

/// Pair an estimator's local tempo curve with independent calibration truth.
///
/// # Errors
///
/// Returns an error for a non-calibration suite, unknown case filter, invalid
/// threshold or estimator policy, malformed truth, or estimator failure.
pub fn diagnose_core_tempo_suite(
    suite_path: &Path,
    estimator_policy: Option<&str>,
    case_ids: &[String],
    minimum_error_percent: f64,
) -> Result<TempoDiagnosticEvaluation> {
    if !minimum_error_percent.is_finite() || minimum_error_percent < 0.0 {
        bail!("tempo diagnostic error threshold must be finite and non-negative");
    }
    let (suite, root) = load_suite(suite_path)?;
    if suite.purpose != SuitePurpose::Calibration {
        bail!("tempo diagnostics may inspect only calibration suites");
    }
    for requested in case_ids {
        if !suite.cases.iter().any(|case| case.id == *requested) {
            bail!("tempo diagnostic case {requested} is not present in the suite");
        }
    }

    let estimator = estimator_for_policy(estimator_policy)?;
    let mut cases = Vec::new();
    for case in &suite.cases {
        if !case_ids.is_empty() && !case_ids.contains(&case.id) {
            continue;
        }
        let truth = load_case_truth(case, &root)?;
        let analysis = estimator.estimate(&truth.ideal_observations())?;
        cases.push(diagnose_analysis(
            case.id.clone(),
            &analysis,
            &truth,
            minimum_error_percent,
        )?);
    }

    Ok(TempoDiagnosticEvaluation {
        schema_version: 2,
        suite_id: suite.id,
        suite_purpose: suite.purpose,
        estimator_policy: estimator_policy.unwrap_or("shipping-default").to_string(),
        minimum_error_percent,
        cases,
    })
}

fn diagnose_analysis(
    id: String,
    analysis: &Analysis,
    truth: &GeneratedTruth,
    minimum_error_percent: f64,
) -> Result<TempoDiagnosticCase> {
    let mut points = Vec::new();
    for (index, point) in analysis.tempo_curve.iter().enumerate() {
        let Some(truth_bpm) = truth.tempo_at(point.time_s) else {
            continue;
        };
        let Some(pair) = analysis.beats.get(index..=index + 1) else {
            bail!("tempo diagnostic curve does not align with accepted beat intervals");
        };
        let observed_interval_bpm = 60.0 / (pair[1].time_s - pair[0].time_s);
        if !observed_interval_bpm.is_finite() || observed_interval_bpm <= 0.0 {
            bail!("tempo diagnostic encountered an invalid accepted beat interval");
        }
        let ratio = point.bpm / truth_bpm;
        let (nearest_octave_shift, normalized_ratio) = normalize_octave_ratio(ratio)?;
        points.push(TempoDiagnosticPoint {
            time_s: point.time_s,
            observed_interval_bpm,
            truth_bpm,
            estimated_bpm: point.bpm,
            regularized_to_observed_ratio: point.bpm / observed_interval_bpm,
            estimated_to_truth_ratio: ratio,
            log2_ratio: ratio.log2(),
            nearest_octave_shift,
            octave_residual_percent: (normalized_ratio - 1.0).abs() * 100.0,
            absolute_error_percent: (ratio - 1.0).abs() * 100.0,
            confidence: point.confidence,
        });
    }
    let error_runs = collect_error_runs(&points, minimum_error_percent);
    let error_point_count = error_runs.iter().map(|run| run.sample_count).sum();
    Ok(TempoDiagnosticCase {
        id,
        global_bpm: analysis.global_bpm,
        scored_point_count: points.len(),
        error_point_count,
        error_runs,
        points,
        warnings: analysis.warnings.clone(),
    })
}

fn normalize_octave_ratio(ratio: f64) -> Result<(i8, f64)> {
    if !ratio.is_finite() || ratio <= 0.0 {
        bail!("tempo diagnostic encountered a non-finite or non-positive tempo ratio");
    }
    let octave_boundary = f64::sqrt(2.0);
    let mut normalized_ratio = ratio;
    let mut shift = 0_i8;
    while normalized_ratio >= octave_boundary {
        shift = shift
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("tempo diagnostic metrical shift is too large"))?;
        normalized_ratio /= 2.0;
    }
    while normalized_ratio < 1.0 / octave_boundary {
        shift = shift
            .checked_sub(1)
            .ok_or_else(|| anyhow::anyhow!("tempo diagnostic metrical shift is too small"))?;
        normalized_ratio *= 2.0;
    }
    Ok((shift, normalized_ratio))
}

fn collect_error_runs(
    points: &[TempoDiagnosticPoint],
    minimum_error_percent: f64,
) -> Vec<TempoErrorRun> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < points.len() {
        if points[start].absolute_error_percent < minimum_error_percent {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < points.len() && points[end].absolute_error_percent >= minimum_error_percent {
            end += 1;
        }
        runs.push(summarize_run(&points[start..end]));
        start = end;
    }
    runs
}

fn summarize_run(points: &[TempoDiagnosticPoint]) -> TempoErrorRun {
    let mut level_counts = BTreeMap::<i8, usize>::new();
    for point in points {
        *level_counts.entry(point.nearest_octave_shift).or_default() += 1;
    }
    let dominant_octave_shift = level_counts
        .into_iter()
        .max_by_key(|&(level, count)| (count, -i16::from(level).abs()))
        .map_or(0, |(level, _)| level);
    TempoErrorRun {
        start_s: points.first().expect("non-empty diagnostic run").time_s,
        end_s: points.last().expect("non-empty diagnostic run").time_s,
        sample_count: points.len(),
        dominant_octave_shift,
        median_estimated_to_truth_ratio: median(
            points
                .iter()
                .map(|point| point.estimated_to_truth_ratio)
                .collect(),
        ),
        median_absolute_error_percent: median(
            points
                .iter()
                .map(|point| point.absolute_error_percent)
                .collect(),
        ),
        maximum_absolute_error_percent: points
            .iter()
            .map(|point| point.absolute_error_percent)
            .fold(0.0, f64::max),
        median_octave_residual_percent: median(
            points
                .iter()
                .map(|point| point.octave_residual_percent)
                .collect(),
        ),
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        f64::midpoint(values[middle - 1], values[middle])
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use rhythm_map_core::{Analysis, BeatEvent, ModelInfo, TempoPoint, TempoSegmentKind};

    use super::{diagnose_analysis, diagnose_core_tempo_suite};
    use crate::{GeneratedTruth, TruthBeat, TruthTempoSegment};

    #[test]
    fn groups_contiguous_octave_errors_and_reports_residual() {
        let truth = GeneratedTruth {
            schema_version: 1,
            id: "tempo-diagnostic".to_string(),
            duration_s: 4.0,
            beats: vec![
                TruthBeat {
                    time_s: 0.0,
                    downbeat: true,
                },
                TruthBeat {
                    time_s: 1.0,
                    downbeat: false,
                },
                TruthBeat {
                    time_s: 2.0,
                    downbeat: false,
                },
            ],
            tempo_segments: vec![TruthTempoSegment {
                start_s: 0.0,
                end_s: 4.0,
                kind: TempoSegmentKind::Constant,
                start_bpm: 60.0,
                end_bpm: 60.0,
            }],
            change_points: Vec::new(),
        };
        let analysis = Analysis {
            schema_version: 1,
            duration_s: 4.0,
            source: ModelInfo {
                backend: "test".to_string(),
                model: "test".to_string(),
                version: None,
                frame_rate_hz: None,
            },
            beats: vec![
                BeatEvent {
                    time_s: 0.0,
                    confidence: 1.0,
                    downbeat: true,
                    downbeat_confidence: 1.0,
                },
                BeatEvent {
                    time_s: 1.0,
                    confidence: 1.0,
                    downbeat: false,
                    downbeat_confidence: 0.0,
                },
                BeatEvent {
                    time_s: 2.0,
                    confidence: 1.0,
                    downbeat: false,
                    downbeat_confidence: 0.0,
                },
                BeatEvent {
                    time_s: 3.0,
                    confidence: 1.0,
                    downbeat: false,
                    downbeat_confidence: 0.0,
                },
            ],
            beat_hypotheses: Vec::new(),
            metrical_ambiguity_regions: Vec::new(),
            global_bpm: Some(120.0),
            tempo_hypotheses: Vec::new(),
            tempo_curve: vec![
                TempoPoint {
                    time_s: 0.5,
                    bpm: 120.0,
                    confidence: 1.0,
                },
                TempoPoint {
                    time_s: 1.5,
                    bpm: 118.0,
                    confidence: 0.9,
                },
                TempoPoint {
                    time_s: 2.5,
                    bpm: 60.0,
                    confidence: 1.0,
                },
            ],
            tempo_segments: Vec::new(),
            change_points: Vec::new(),
            rhythm_sections: Vec::new(),
            warnings: Vec::new(),
        };

        let report =
            diagnose_analysis("tempo-diagnostic".to_string(), &analysis, &truth, 25.0).unwrap();
        assert_eq!(report.error_point_count, 2);
        assert_eq!(report.error_runs.len(), 1);
        assert_eq!(report.error_runs[0].dominant_octave_shift, 1);
        assert!(report.error_runs[0].median_octave_residual_percent < 2.0);
        assert!((report.points[0].observed_interval_bpm - 60.0).abs() < 0.001);
        assert!((report.points[0].regularized_to_observed_ratio - 2.0).abs() < 0.001);
    }

    #[test]
    fn rejects_truth_assisted_diagnostics_on_regression_suite() {
        let suite =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../evaluation/suites/generated-v1.json");
        let error = diagnose_core_tempo_suite(&suite, None, &[], 25.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("only calibration suites"));
    }
}
