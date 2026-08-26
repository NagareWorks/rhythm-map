use std::path::Path;

use anyhow::{Context, Result, bail};
use rhythm_map_core::RhythmActivationSeries;
use serde::{Deserialize, Serialize};

use crate::{
    BeatMetrics, BottleneckEvaluation, ModelPackIdentity, PulseHypothesisEvaluation, SuitePurpose,
    consensus::validate_inputs,
    metrics::score_beats,
    runner::{load_case_truth, load_suite},
};

const SUPPORTED_TIME_TOLERANCE_S: f64 = 1e-9;
const PROBABILITY_FLOOR: f64 = 0.01;
const LOCAL_POLICY_ID: &str = "anchored-pareto-decoded-event-dense-pulse-v1";

/// Calibration-only diagnosis of local metrical-path substitutions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalMetricalConsensusDiagnosis {
    /// Independent schema for this report-to-report experiment.
    pub schema_version: u32,
    /// Calibration suite shared by the reports and timestamp truth.
    pub suite_id: String,
    /// Backend whose selected and locally varying paths are compared.
    pub primary_model_pack: ModelPackIdentity,
    /// Independent backend supplying decoded and dense pulse evidence.
    pub secondary_model_pack: ModelPackIdentity,
    /// One-to-one timestamp tolerance used for independent event support.
    pub agreement_tolerance_s: f64,
    /// Frozen truth-free local substitution rule.
    pub policy_id: String,
    /// Mean annotated beat F1 of the primary `selected` paths.
    pub primary_mean_beat_f1: f64,
    /// Mean annotated beat F1 after locally applying the frozen rule.
    pub candidate_mean_beat_f1: f64,
    /// Directional candidate difference from the unchanged primary paths.
    pub candidate_delta: f64,
    /// Number of calibration cases improved by the local rule.
    pub improved_cases: usize,
    /// Number of calibration cases degraded by the local rule.
    pub regressed_cases: usize,
    /// True only for a positive aggregate gain with no case regression.
    pub passes_calibration_gate: bool,
    /// Number of bounded disagreement regions changed by the rule.
    pub selected_region_count: usize,
    /// Per-case evidence and truth-assisted attribution.
    pub cases: Vec<LocalMetricalConsensusCase>,
}

/// One case in a local metrical-path consensus diagnosis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalMetricalConsensusCase {
    /// Stable suite case identifier.
    pub id: String,
    /// Whether the primary report exposed a distinct locally varying path.
    pub local_path_available: bool,
    /// Annotated beat score of the unchanged `selected` path.
    pub primary_truth_beats: BeatMetrics,
    /// Annotated beat score of the complete local path, for attribution only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_path_truth_beats: Option<BeatMetrics>,
    /// Annotated beat score after applying eligible bounded substitutions.
    pub candidate_truth_beats: BeatMetrics,
    /// Directional annotated F1 change; never used for selection.
    pub candidate_truth_beat_f1_delta: f64,
    /// Number of bounded regions selected from the local path.
    pub selected_region_count: usize,
    /// Auditable bounded and edge disagreement regions.
    pub regions: Vec<LocalMetricalConsensusRegion>,
}

/// One maximal disagreement span between common backend-supported anchors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalMetricalConsensusRegion {
    /// Region start; a common timestamp when `left_anchored` is true.
    pub start_s: f64,
    /// Region end; a common timestamp when `right_anchored` is true.
    pub end_s: f64,
    /// Whether both paths share the timestamp at the left boundary.
    pub left_anchored: bool,
    /// Whether both paths share the timestamp at the right boundary.
    pub right_anchored: bool,
    /// Real timestamps present only in the primary selected path.
    pub primary_only_times_s: Vec<f64>,
    /// Real timestamps present only in the locally varying path.
    pub local_only_times_s: Vec<f64>,
    /// Mean binary-decision advantage against independently decoded events.
    pub decoded_event_decision_margin: f64,
    /// Mean Bernoulli log-likelihood advantage from dense pulse activations.
    pub dense_pulse_log_likelihood_margin: f64,
    /// Mean binary-decision advantage against annotations; attribution only.
    pub truth_decision_margin: f64,
    /// Whether the frozen rule substitutes this region from the local path.
    pub selected: bool,
}

/// Diagnose a conservative local selector over one calibration suite.
///
/// Only disagreement regions bounded on both sides by timestamps shared by the
/// primary selected and local paths are eligible. A local region must strictly
/// improve both binary agreement with the independent decoded sequence and
/// Bernoulli likelihood under its dense pulse activations. Edge regions remain
/// explicit but cannot change the candidate path.
///
/// # Errors
///
/// Returns an error when inputs are not distinct calibration reports for the
/// supplied timestamped suite, or dense secondary pulse activations are absent.
pub fn diagnose_local_metrical_consensus(
    primary: &BottleneckEvaluation,
    secondary: &BottleneckEvaluation,
    suite_path: &Path,
    agreement_tolerance_s: f64,
) -> Result<LocalMetricalConsensusDiagnosis> {
    validate_inputs(primary, secondary, agreement_tolerance_s)?;
    let (suite_id, cases) =
        diagnose_suite_cases(primary, secondary, suite_path, agreement_tolerance_s)?;

    let case_count = f64::from(
        u32::try_from(cases.len()).context("local consensus report contains too many cases")?,
    );
    let primary_mean_beat_f1 = cases
        .iter()
        .map(|case| case.primary_truth_beats.f1)
        .sum::<f64>()
        / case_count;
    let candidate_mean_beat_f1 = cases
        .iter()
        .map(|case| case.candidate_truth_beats.f1)
        .sum::<f64>()
        / case_count;
    let candidate_delta = candidate_mean_beat_f1 - primary_mean_beat_f1;
    let improved_cases = cases
        .iter()
        .filter(|case| case.candidate_truth_beat_f1_delta > f64::EPSILON)
        .count();
    let regressed_cases = cases
        .iter()
        .filter(|case| case.candidate_truth_beat_f1_delta < -f64::EPSILON)
        .count();
    let selected_region_count = cases.iter().map(|case| case.selected_region_count).sum();

    Ok(LocalMetricalConsensusDiagnosis {
        schema_version: 1,
        suite_id,
        primary_model_pack: primary.model_pack.clone(),
        secondary_model_pack: secondary.model_pack.clone(),
        agreement_tolerance_s,
        policy_id: LOCAL_POLICY_ID.to_string(),
        primary_mean_beat_f1,
        candidate_mean_beat_f1,
        candidate_delta,
        improved_cases,
        regressed_cases,
        passes_calibration_gate: candidate_delta > f64::EPSILON && regressed_cases == 0,
        selected_region_count,
        cases,
    })
}

fn diagnose_suite_cases(
    primary: &BottleneckEvaluation,
    secondary: &BottleneckEvaluation,
    suite_path: &Path,
    agreement_tolerance_s: f64,
) -> Result<(String, Vec<LocalMetricalConsensusCase>)> {
    let (suite, root) = load_suite(suite_path)?;
    if suite.purpose != SuitePurpose::Calibration {
        bail!("local metrical consensus diagnosis accepts calibration suites only");
    }
    if suite.id != primary.suite_id {
        bail!("local metrical consensus suite differs from backend reports");
    }
    if suite.cases.len() != primary.cases.len() || suite.cases.len() != secondary.cases.len() {
        bail!("local metrical consensus suite and reports contain different case sets");
    }

    let mut cases = Vec::with_capacity(suite.cases.len());
    for suite_case in &suite.cases {
        let truth = load_case_truth(suite_case, &root)?;
        if truth.beats.is_empty() {
            bail!(
                "local metrical consensus requires timestamped beat truth; case {} is empty",
                suite_case.id
            );
        }
        let primary_case = primary
            .cases
            .iter()
            .find(|case| case.id == suite_case.id)
            .with_context(|| format!("primary report is missing case {}", suite_case.id))?;
        let secondary_case = secondary
            .cases
            .iter()
            .find(|case| case.id == suite_case.id)
            .with_context(|| format!("secondary report is missing case {}", suite_case.id))?;
        if primary_case.audio_sha256 != secondary_case.audio_sha256 {
            bail!("audio identity differs for case {}", suite_case.id);
        }
        let primary_coverage = primary_case
            .pulse_hypothesis_coverage
            .as_ref()
            .with_context(|| {
                format!("primary case {} has no hypothesis coverage", suite_case.id)
            })?;
        let secondary_coverage = secondary_case
            .pulse_hypothesis_coverage
            .as_ref()
            .with_context(|| {
                format!(
                    "secondary case {} has no hypothesis coverage",
                    suite_case.id
                )
            })?;
        let selected = hypothesis_by_id(&primary_coverage.hypotheses, "selected", &suite_case.id)?;
        let local = primary_coverage
            .hypotheses
            .iter()
            .find(|hypothesis| hypothesis.id == "locally_varying_metrical_path");
        let secondary_selected = secondary_coverage
            .hypotheses
            .iter()
            .min_by_key(|hypothesis| hypothesis.rank)
            .with_context(|| format!("secondary case {} has no hypotheses", suite_case.id))?;
        let activations = secondary_case
            .observations
            .activations
            .as_ref()
            .with_context(|| {
                format!(
                    "secondary case {} has no dense pulse activations",
                    suite_case.id
                )
            })?;
        let truth_times = truth
            .beats
            .iter()
            .map(|beat| beat.time_s)
            .collect::<Vec<_>>();
        cases.push(diagnose_case(
            &suite_case.id,
            truth.duration_s,
            &truth_times,
            selected,
            local,
            secondary_selected,
            activations,
            agreement_tolerance_s,
        ));
    }

    Ok((suite.id, cases))
}

#[allow(clippy::too_many_arguments)]
fn diagnose_case(
    id: &str,
    duration_s: f64,
    truth_times: &[f64],
    selected: &PulseHypothesisEvaluation,
    local: Option<&PulseHypothesisEvaluation>,
    secondary: &PulseHypothesisEvaluation,
    activations: &RhythmActivationSeries,
    agreement_tolerance_s: f64,
) -> LocalMetricalConsensusCase {
    let primary_truth_beats =
        score_beats(&selected.beat_times_s, truth_times, agreement_tolerance_s);
    let Some(local) = local else {
        return LocalMetricalConsensusCase {
            id: id.to_string(),
            local_path_available: false,
            primary_truth_beats: primary_truth_beats.clone(),
            local_path_truth_beats: None,
            candidate_truth_beats: primary_truth_beats,
            candidate_truth_beat_f1_delta: 0.0,
            selected_region_count: 0,
            regions: Vec::new(),
        };
    };

    let mut candidate_times = selected.beat_times_s.clone();
    let mut regions = disagreement_regions(
        &selected.beat_times_s,
        &local.beat_times_s,
        duration_s,
        &secondary.beat_times_s,
        activations,
        truth_times,
        agreement_tolerance_s,
    );
    for region in regions.iter_mut().filter(|region| region.selected) {
        candidate_times.retain(|time_s| {
            !region
                .primary_only_times_s
                .iter()
                .any(|removed| same_supported_time(*time_s, *removed))
        });
        candidate_times.extend(&region.local_only_times_s);
    }
    candidate_times.sort_by(f64::total_cmp);
    candidate_times.dedup_by(|left, right| same_supported_time(*left, *right));

    let candidate_truth_beats = score_beats(&candidate_times, truth_times, agreement_tolerance_s);
    let selected_region_count = regions.iter().filter(|region| region.selected).count();
    LocalMetricalConsensusCase {
        id: id.to_string(),
        local_path_available: true,
        primary_truth_beats: primary_truth_beats.clone(),
        local_path_truth_beats: Some(score_beats(
            &local.beat_times_s,
            truth_times,
            agreement_tolerance_s,
        )),
        candidate_truth_beat_f1_delta: candidate_truth_beats.f1 - primary_truth_beats.f1,
        candidate_truth_beats,
        selected_region_count,
        regions,
    }
}

#[allow(clippy::too_many_arguments)]
fn disagreement_regions(
    selected: &[f64],
    local: &[f64],
    duration_s: f64,
    secondary: &[f64],
    activations: &RhythmActivationSeries,
    truth: &[f64],
    tolerance_s: f64,
) -> Vec<LocalMetricalConsensusRegion> {
    let anchors = common_supported_times(selected, local);
    let mut bounds = Vec::with_capacity(anchors.len() + 1);
    if let Some(&first) = anchors.first() {
        bounds.push((0.0, first, false, true));
        bounds.extend(
            anchors
                .windows(2)
                .map(|pair| (pair[0], pair[1], true, true)),
        );
        bounds.push((
            *anchors.last().expect("first anchor exists"),
            duration_s,
            true,
            false,
        ));
    } else {
        bounds.push((0.0, duration_s, false, false));
    }

    bounds
        .into_iter()
        .filter_map(|(start_s, end_s, left_anchored, right_anchored)| {
            let primary_only_times_s = unsupported_times_in_region(
                selected,
                local,
                start_s,
                end_s,
                left_anchored,
                right_anchored,
            );
            let local_only_times_s = unsupported_times_in_region(
                local,
                selected,
                start_s,
                end_s,
                left_anchored,
                right_anchored,
            );
            if primary_only_times_s.is_empty() && local_only_times_s.is_empty() {
                return None;
            }
            let decoded_event_decision_margin = binary_decision_margin(
                &primary_only_times_s,
                &local_only_times_s,
                secondary,
                tolerance_s,
            );
            let dense_pulse_log_likelihood_margin = dense_pulse_margin(
                &primary_only_times_s,
                &local_only_times_s,
                activations,
                tolerance_s,
            );
            let truth_decision_margin = binary_decision_margin(
                &primary_only_times_s,
                &local_only_times_s,
                truth,
                tolerance_s,
            );
            let selected = left_anchored
                && right_anchored
                && decoded_event_decision_margin > f64::EPSILON
                && dense_pulse_log_likelihood_margin > f64::EPSILON;
            Some(LocalMetricalConsensusRegion {
                start_s,
                end_s,
                left_anchored,
                right_anchored,
                primary_only_times_s,
                local_only_times_s,
                decoded_event_decision_margin,
                dense_pulse_log_likelihood_margin,
                truth_decision_margin,
                selected,
            })
        })
        .collect()
}

fn common_supported_times(left: &[f64], right: &[f64]) -> Vec<f64> {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut common = Vec::new();
    while left_index < left.len() && right_index < right.len() {
        let difference = left[left_index] - right[right_index];
        if difference.abs() <= SUPPORTED_TIME_TOLERANCE_S {
            common.push(left[left_index]);
            left_index += 1;
            right_index += 1;
        } else if difference < 0.0 {
            left_index += 1;
        } else {
            right_index += 1;
        }
    }
    common
}

fn unsupported_times_in_region(
    source: &[f64],
    other: &[f64],
    start_s: f64,
    end_s: f64,
    left_anchored: bool,
    right_anchored: bool,
) -> Vec<f64> {
    source
        .iter()
        .copied()
        .filter(|&time_s| {
            (time_s > start_s || (!left_anchored && time_s >= start_s))
                && (time_s < end_s || (!right_anchored && time_s <= end_s))
                && !other
                    .iter()
                    .any(|&candidate| same_supported_time(time_s, candidate))
        })
        .collect()
}

fn binary_decision_margin(
    primary_only: &[f64],
    local_only: &[f64],
    reference: &[f64],
    tolerance_s: f64,
) -> f64 {
    let decision_count = primary_only.len() + local_only.len();
    if decision_count == 0 {
        return 0.0;
    }
    let primary_margin = primary_only.iter().map(|&time_s| {
        if has_nearby_time(reference, time_s, tolerance_s) {
            -1.0
        } else {
            1.0
        }
    });
    let local_margin = local_only.iter().map(|&time_s| {
        if has_nearby_time(reference, time_s, tolerance_s) {
            1.0
        } else {
            -1.0
        }
    });
    (primary_margin.chain(local_margin).sum::<f64>()) / usize_to_f64(decision_count)
}

fn dense_pulse_margin(
    primary_only: &[f64],
    local_only: &[f64],
    activations: &RhythmActivationSeries,
    tolerance_s: f64,
) -> f64 {
    let decision_count = primary_only.len() + local_only.len();
    if decision_count == 0 {
        return 0.0;
    }
    let primary_margin = primary_only.iter().map(|&time_s| {
        let probability = pulse_probability_at_frame(time_s, activations, tolerance_s);
        (1.0 - probability).ln() - probability.ln()
    });
    let local_margin = local_only.iter().map(|&time_s| {
        let probability = pulse_probability_at_frame(time_s, activations, tolerance_s);
        probability.ln() - (1.0 - probability).ln()
    });
    (primary_margin.chain(local_margin).sum::<f64>()) / usize_to_f64(decision_count)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn pulse_probability_at_frame(
    time_s: f64,
    activations: &RhythmActivationSeries,
    tolerance_s: f64,
) -> f64 {
    let position = (time_s - activations.start_time_s) * activations.frame_rate_hz;
    if !position.is_finite() || position < 0.0 {
        return PROBABILITY_FLOOR;
    }
    let frame = position.round() as usize;
    let Some(frame_u32) = u32::try_from(frame).ok() else {
        return PROBABILITY_FLOOR;
    };
    let frame_time = activations.start_time_s + f64::from(frame_u32) / activations.frame_rate_hz;
    activations
        .pulse_confidences
        .get(frame)
        .filter(|_| (frame_time - time_s).abs() <= tolerance_s)
        .map_or(PROBABILITY_FLOOR, |&probability| {
            f64::from(probability).clamp(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
        })
}

fn has_nearby_time(events: &[f64], time_s: f64, tolerance_s: f64) -> bool {
    events
        .iter()
        .any(|candidate| (*candidate - time_s).abs() <= tolerance_s)
}

fn same_supported_time(left: f64, right: f64) -> bool {
    (left - right).abs() <= SUPPORTED_TIME_TOLERANCE_S
}

fn hypothesis_by_id<'a>(
    hypotheses: &'a [PulseHypothesisEvaluation],
    id: &str,
    case_id: &str,
) -> Result<&'a PulseHypothesisEvaluation> {
    hypotheses
        .iter()
        .find(|hypothesis| hypothesis.id == id)
        .with_context(|| format!("primary case {case_id} has no {id} hypothesis"))
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("local disagreement count fits in u32"))
}

#[cfg(test)]
mod tests {
    use rhythm_map_core::RhythmActivationSeries;

    use super::disagreement_regions;

    fn activations(pulses: &[(usize, f32)]) -> RhythmActivationSeries {
        let mut pulse_confidences = vec![0.5; 151];
        for &(frame, probability) in pulses {
            pulse_confidences[frame] = probability;
        }
        RhythmActivationSeries {
            start_time_s: 0.0,
            frame_rate_hz: 50.0,
            pulse_confidences,
            downbeat_confidences: vec![0.1; 151],
        }
    }

    #[test]
    fn bounded_region_requires_both_independent_signals() {
        let low_pulse = activations(&[(75, 0.1)]);
        let regions = disagreement_regions(
            &[0.0, 1.0, 1.5, 2.0],
            &[0.0, 1.0, 2.0],
            2.0,
            &[0.0, 1.0, 2.0],
            &low_pulse,
            &[0.0, 1.0, 2.0],
            0.01,
        );

        assert_eq!(regions.len(), 1);
        assert!(regions[0].left_anchored && regions[0].right_anchored);
        assert!(regions[0].decoded_event_decision_margin > 0.0);
        assert!(regions[0].dense_pulse_log_likelihood_margin > 0.0);
        assert!(regions[0].selected);

        let high_pulse = activations(&[(75, 0.9)]);
        let dense_conflict = disagreement_regions(
            &[0.0, 1.0, 1.5, 2.0],
            &[0.0, 1.0, 2.0],
            2.0,
            &[0.0, 1.0, 2.0],
            &high_pulse,
            &[0.0, 1.0, 2.0],
            0.01,
        );
        assert!(dense_conflict[0].decoded_event_decision_margin > 0.0);
        assert!(dense_conflict[0].dense_pulse_log_likelihood_margin < 0.0);
        assert!(!dense_conflict[0].selected);
    }

    #[test]
    fn leading_and_trailing_disagreements_remain_unselected() {
        let regions = disagreement_regions(
            &[0.25, 1.0, 2.0, 2.75],
            &[1.0, 2.0],
            3.0,
            &[1.0, 2.0],
            &activations(&[(13, 0.1), (138, 0.1)]),
            &[1.0, 2.0],
            0.02,
        );

        assert_eq!(regions.len(), 2);
        assert!(regions.iter().all(|region| !region.selected));
        assert!(!regions[0].left_anchored && regions[0].right_anchored);
        assert!(regions[1].left_anchored && !regions[1].right_anchored);
    }
}
