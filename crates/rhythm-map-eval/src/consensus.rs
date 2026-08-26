use anyhow::{Context, Result, bail};
use rhythm_map_core::{ObservedBeat, RhythmActivationSeries};
use serde::{Deserialize, Serialize};

use crate::{
    AttributionCase, BeatMetrics, BottleneckEvaluation, ModelPackIdentity,
    PulseHypothesisEvaluation, SuitePurpose, metrics::score_beats,
};

const AGREEMENT_WINDOW_COUNT: u32 = 4;
const METER_PULSE_COUNTS: [usize; 3] = [2, 3, 4];
const DOWNBEAT_PROBABILITY_FLOOR: f64 = 0.01;
const METER_GATE_POLICY_ID: &str = "pareto-beat-agreement-dense-downbeat-meter-v2";

/// Secondary-backend evidence used to score bar phase for one case.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MeterEvidenceSource {
    /// Uniform frame-level activations before the backend decoded events.
    DenseActivations,
    /// Compatibility fallback for an older report containing events only.
    DecodedEvents,
}

/// Reproducible diagnosis of a naive cross-backend hypothesis selector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConsensusDiagnosis {
    /// Independent schema for this report-to-report experiment.
    pub schema_version: u32,
    /// Calibration suite shared by both backend reports.
    pub suite_id: String,
    /// Model pack whose hypotheses are being ranked.
    pub primary_model_pack: ModelPackIdentity,
    /// Independent model pack whose top-ranked sequence supplies consensus evidence.
    pub secondary_model_pack: ModelPackIdentity,
    /// One-to-one timestamp tolerance used only for backend agreement.
    pub agreement_tolerance_s: f64,
    /// Frozen truth-free conjunction evaluated by the meter-gated fields.
    pub meter_gate_policy_id: String,
    /// Fixed pulse-cycle vocabulary evaluated for every hypothesis and phase.
    pub meter_pulse_counts: Vec<usize>,
    /// Downbeat probability assigned where the secondary report has no event.
    pub missing_downbeat_probability: f64,
    /// Mean annotated beat F1 of the primary backend's own top-ranked hypothesis.
    pub primary_mean_beat_f1: f64,
    /// Mean annotated beat F1 after choosing the hypothesis with greatest global agreement.
    pub naive_consensus_mean_beat_f1: f64,
    /// Directional difference from the unchanged primary selection.
    pub naive_consensus_delta: f64,
    /// Number of calibration cases improved by the naive selector.
    pub improved_cases: usize,
    /// Number of calibration cases degraded by the naive selector.
    pub regressed_cases: usize,
    /// True only when calibration has positive mean gain and no case regression.
    /// A separate precommitted holdout would still be required for promotion.
    pub passes_calibration_gate: bool,
    /// Mean annotated beat F1 when agreement and meter evidence must both improve.
    pub meter_gated_consensus_mean_beat_f1: f64,
    /// Directional difference from the unchanged primary selection.
    pub meter_gated_consensus_delta: f64,
    /// Number of calibration cases improved by the meter-gated selector.
    pub meter_gated_improved_cases: usize,
    /// Number of calibration cases degraded by the meter-gated selector.
    pub meter_gated_regressed_cases: usize,
    /// Calibration-only gate; a precommitted holdout is still required.
    pub meter_gated_passes_calibration_gate: bool,
    /// Per-case agreement and truth-assisted attribution.
    pub cases: Vec<ConsensusDiagnosisCase>,
}

/// One case in a cross-backend consensus diagnosis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConsensusDiagnosisCase {
    /// Stable suite case identifier.
    pub id: String,
    /// Top-ranked primary hypothesis before consensus.
    pub primary_hypothesis_id: String,
    /// Top-ranked sequence from the independent backend.
    pub secondary_hypothesis_id: String,
    /// Primary hypothesis selected by greatest global timestamp agreement.
    pub naive_consensus_hypothesis_id: String,
    /// Agreement of the unchanged primary sequence with the secondary sequence.
    pub primary_agreement: BeatMetrics,
    /// Agreement of the naive consensus choice with the secondary sequence.
    pub naive_consensus_agreement: BeatMetrics,
    /// Truth-free F1 advantage used by the naive global selector.
    pub agreement_margin: f64,
    /// Quarter-track agreement advantage over the unchanged primary sequence.
    pub window_agreement_margins: Vec<f64>,
    /// Number of material sign reversals in the quarter-track agreement advantage.
    pub material_support_reversals: usize,
    /// Secondary-backend representation used for downbeat periodicity.
    pub meter_evidence_source: MeterEvidenceSource,
    /// Auditable beat-agreement and downbeat-meter scores for every primary hypothesis.
    pub hypotheses: Vec<ConsensusHypothesisDiagnosis>,
    /// Hypothesis retained after requiring agreement and meter evidence to both improve.
    pub meter_gated_consensus_hypothesis_id: String,
    /// Annotated F1 before consensus; calibration attribution only.
    pub primary_truth_beat_f1: f64,
    /// Annotated F1 after naive consensus; calibration attribution only.
    pub naive_consensus_truth_beat_f1: f64,
    /// Directional annotated F1 change; never used for selection.
    pub truth_beat_f1_delta: f64,
    /// Annotated F1 after meter-gated consensus; calibration attribution only.
    pub meter_gated_consensus_truth_beat_f1: f64,
    /// Directional annotated F1 change; never used for selection.
    pub meter_gated_truth_beat_f1_delta: f64,
}

/// Truth-free cross-backend scores for one primary-backend hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConsensusHypothesisDiagnosis {
    /// Stable construction label from the primary report.
    pub id: String,
    /// Original truth-free primary-backend rank.
    pub rank: usize,
    /// One-to-one agreement with the secondary backend's top-ranked sequence.
    pub agreement: BeatMetrics,
    /// Agreement advantage over the unchanged primary hypothesis.
    pub agreement_margin: f64,
    /// Best class-balanced downbeat periodicity over common 2/3/4-pulse bars.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meter_evidence: Option<MeterPatternEvidence>,
    /// Meter log-likelihood advantage over the unchanged primary hypothesis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meter_log_likelihood_margin: Option<f64>,
    /// Whether both truth-free scores strictly dominate the unchanged primary.
    pub meter_gate_eligible: bool,
}

/// Best downbeat phase pattern available for one beat-sequence hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeterPatternEvidence {
    /// Pulses per bar in the best of the fixed 2/3/4-pulse candidates.
    pub pulses_per_bar: usize,
    /// Zero-based downbeat phase within that pulse cycle.
    pub phase: usize,
    /// Balanced mean log likelihood for downbeat and non-downbeat positions.
    pub log_likelihood: f64,
}

/// Diagnose whether a second observation backend can safely choose among the
/// first backend's already published hypotheses using global timestamp agreement.
///
/// Both inputs must be reports for the same calibration suite. Ground truth is
/// used only after the truth-free choice, to attribute gains and regressions.
///
/// # Errors
///
/// Returns an error when the reports are not distinct, non-empty calibration
/// results for the same suite and audio identities, or when hypothesis coverage
/// is absent.
pub fn diagnose_backend_consensus(
    primary: &BottleneckEvaluation,
    secondary: &BottleneckEvaluation,
    agreement_tolerance_s: f64,
) -> Result<ConsensusDiagnosis> {
    validate_inputs(primary, secondary, agreement_tolerance_s)?;

    let mut cases = Vec::with_capacity(primary.cases.len());
    for primary_case in &primary.cases {
        let secondary_case = secondary
            .cases
            .iter()
            .find(|candidate| candidate.id == primary_case.id)
            .with_context(|| format!("secondary report is missing case {}", primary_case.id))?;
        cases.push(diagnose_case(
            primary_case,
            secondary_case,
            agreement_tolerance_s,
        )?);
    }
    if cases.len() != secondary.cases.len() {
        bail!("primary and secondary reports contain different case sets");
    }

    let case_count =
        f64::from(u32::try_from(cases.len()).context("consensus report contains too many cases")?);
    let primary_mean_beat_f1 = cases
        .iter()
        .map(|case| case.primary_truth_beat_f1)
        .sum::<f64>()
        / case_count;
    let naive_consensus_mean_beat_f1 = cases
        .iter()
        .map(|case| case.naive_consensus_truth_beat_f1)
        .sum::<f64>()
        / case_count;
    let improved_cases = cases
        .iter()
        .filter(|case| case.truth_beat_f1_delta > f64::EPSILON)
        .count();
    let regressed_cases = cases
        .iter()
        .filter(|case| case.truth_beat_f1_delta < -f64::EPSILON)
        .count();
    let naive_consensus_delta = naive_consensus_mean_beat_f1 - primary_mean_beat_f1;
    let meter_gated_consensus_mean_beat_f1 = cases
        .iter()
        .map(|case| case.meter_gated_consensus_truth_beat_f1)
        .sum::<f64>()
        / case_count;
    let meter_gated_improved_cases = cases
        .iter()
        .filter(|case| case.meter_gated_truth_beat_f1_delta > f64::EPSILON)
        .count();
    let meter_gated_regressed_cases = cases
        .iter()
        .filter(|case| case.meter_gated_truth_beat_f1_delta < -f64::EPSILON)
        .count();
    let meter_gated_consensus_delta = meter_gated_consensus_mean_beat_f1 - primary_mean_beat_f1;

    Ok(ConsensusDiagnosis {
        schema_version: 3,
        suite_id: primary.suite_id.clone(),
        primary_model_pack: primary.model_pack.clone(),
        secondary_model_pack: secondary.model_pack.clone(),
        agreement_tolerance_s,
        meter_gate_policy_id: METER_GATE_POLICY_ID.to_string(),
        meter_pulse_counts: METER_PULSE_COUNTS.to_vec(),
        missing_downbeat_probability: DOWNBEAT_PROBABILITY_FLOOR,
        primary_mean_beat_f1,
        naive_consensus_mean_beat_f1,
        naive_consensus_delta,
        improved_cases,
        regressed_cases,
        passes_calibration_gate: naive_consensus_delta > f64::EPSILON && regressed_cases == 0,
        meter_gated_consensus_mean_beat_f1,
        meter_gated_consensus_delta,
        meter_gated_improved_cases,
        meter_gated_regressed_cases,
        meter_gated_passes_calibration_gate: meter_gated_consensus_delta > f64::EPSILON
            && meter_gated_regressed_cases == 0,
        cases,
    })
}

fn diagnose_case(
    primary_case: &AttributionCase,
    secondary_case: &AttributionCase,
    agreement_tolerance_s: f64,
) -> Result<ConsensusDiagnosisCase> {
    if primary_case.audio_sha256 != secondary_case.audio_sha256 {
        bail!("audio identity differs for case {}", primary_case.id);
    }
    let primary_coverage = primary_case
        .pulse_hypothesis_coverage
        .as_ref()
        .with_context(|| {
            format!(
                "primary case {} has no hypothesis coverage",
                primary_case.id
            )
        })?;
    let secondary_coverage = secondary_case
        .pulse_hypothesis_coverage
        .as_ref()
        .with_context(|| {
            format!(
                "secondary case {} has no hypothesis coverage",
                primary_case.id
            )
        })?;
    let primary_hypothesis = primary_coverage
        .hypotheses
        .iter()
        .min_by_key(|hypothesis| hypothesis.rank)
        .with_context(|| format!("primary case {} has no hypotheses", primary_case.id))?;
    let secondary_hypothesis = secondary_coverage
        .hypotheses
        .iter()
        .min_by_key(|hypothesis| hypothesis.rank)
        .with_context(|| format!("secondary case {} has no hypotheses", primary_case.id))?;
    let candidates = score_consensus_candidates(
        &primary_coverage.hypotheses,
        secondary_hypothesis,
        &secondary_case.observations.raw_beats,
        secondary_case.observations.activations.as_ref(),
        agreement_tolerance_s,
    )
    .collect::<Vec<_>>();
    let primary_candidate = candidates
        .iter()
        .find(|candidate| candidate.hypothesis.rank == primary_hypothesis.rank)
        .with_context(|| format!("primary case {} has no hypotheses", primary_case.id))?;
    let choice = candidates
        .iter()
        .max_by(|left, right| compare_consensus_candidates(left, right))
        .with_context(|| format!("primary case {} has no hypotheses", primary_case.id))?;
    let meter_gated_choice = candidates
        .iter()
        .filter(|candidate| meter_gate_eligible(candidate, primary_candidate))
        .max_by(|left, right| compare_consensus_candidates(left, right))
        .unwrap_or(primary_candidate);
    let primary_agreement = primary_candidate.agreement.clone();
    let window_agreement_margins = window_agreement_margins(
        primary_hypothesis,
        choice.hypothesis,
        secondary_hypothesis,
        agreement_tolerance_s,
    );
    let hypothesis_diagnostics = candidates
        .iter()
        .map(|candidate| hypothesis_diagnosis(candidate, primary_candidate))
        .collect();
    Ok(ConsensusDiagnosisCase {
        id: primary_case.id.clone(),
        primary_hypothesis_id: primary_hypothesis.id.clone(),
        secondary_hypothesis_id: secondary_hypothesis.id.clone(),
        naive_consensus_hypothesis_id: choice.hypothesis.id.clone(),
        agreement_margin: choice.agreement.f1 - primary_agreement.f1,
        primary_agreement,
        naive_consensus_agreement: choice.agreement.clone(),
        material_support_reversals: material_support_reversals(&window_agreement_margins),
        window_agreement_margins,
        meter_evidence_source: if secondary_case.observations.activations.is_some() {
            MeterEvidenceSource::DenseActivations
        } else {
            MeterEvidenceSource::DecodedEvents
        },
        hypotheses: hypothesis_diagnostics,
        meter_gated_consensus_hypothesis_id: meter_gated_choice.hypothesis.id.clone(),
        primary_truth_beat_f1: primary_hypothesis.beats.f1,
        naive_consensus_truth_beat_f1: choice.hypothesis.beats.f1,
        truth_beat_f1_delta: choice.hypothesis.beats.f1 - primary_hypothesis.beats.f1,
        meter_gated_consensus_truth_beat_f1: meter_gated_choice.hypothesis.beats.f1,
        meter_gated_truth_beat_f1_delta: meter_gated_choice.hypothesis.beats.f1
            - primary_hypothesis.beats.f1,
    })
}

struct ScoredConsensusCandidate<'a> {
    hypothesis: &'a PulseHypothesisEvaluation,
    agreement: BeatMetrics,
    meter_evidence: Option<MeterPatternEvidence>,
}

fn score_consensus_candidates<'a>(
    hypotheses: &'a [PulseHypothesisEvaluation],
    secondary: &'a PulseHypothesisEvaluation,
    secondary_beats: &'a [ObservedBeat],
    secondary_activations: Option<&'a RhythmActivationSeries>,
    tolerance_s: f64,
) -> impl Iterator<Item = ScoredConsensusCandidate<'a>> {
    hypotheses.iter().map(move |hypothesis| {
        let agreement = score_beats(
            &hypothesis.beat_times_s,
            &secondary.beat_times_s,
            tolerance_s,
        );
        let meter_evidence = best_meter_evidence(
            &hypothesis.beat_times_s,
            secondary_beats,
            secondary_activations,
            tolerance_s,
        );
        ScoredConsensusCandidate {
            hypothesis,
            agreement,
            meter_evidence,
        }
    })
}

fn compare_consensus_candidates(
    left: &ScoredConsensusCandidate<'_>,
    right: &ScoredConsensusCandidate<'_>,
) -> std::cmp::Ordering {
    left.agreement
        .f1
        .total_cmp(&right.agreement.f1)
        .then_with(|| right.hypothesis.rank.cmp(&left.hypothesis.rank))
}

fn meter_gate_eligible(
    candidate: &ScoredConsensusCandidate<'_>,
    primary: &ScoredConsensusCandidate<'_>,
) -> bool {
    candidate.agreement.f1 > primary.agreement.f1
        && candidate
            .meter_evidence
            .as_ref()
            .zip(primary.meter_evidence.as_ref())
            .is_some_and(|(candidate_meter, primary_meter)| {
                candidate_meter.log_likelihood > primary_meter.log_likelihood
            })
}

fn hypothesis_diagnosis(
    candidate: &ScoredConsensusCandidate<'_>,
    primary: &ScoredConsensusCandidate<'_>,
) -> ConsensusHypothesisDiagnosis {
    ConsensusHypothesisDiagnosis {
        id: candidate.hypothesis.id.clone(),
        rank: candidate.hypothesis.rank,
        agreement: candidate.agreement.clone(),
        agreement_margin: candidate.agreement.f1 - primary.agreement.f1,
        meter_log_likelihood_margin: candidate
            .meter_evidence
            .as_ref()
            .zip(primary.meter_evidence.as_ref())
            .map(|(candidate_meter, primary_meter)| {
                candidate_meter.log_likelihood - primary_meter.log_likelihood
            }),
        meter_evidence: candidate.meter_evidence.clone(),
        meter_gate_eligible: meter_gate_eligible(candidate, primary),
    }
}

#[cfg(test)]
fn most_agreeing_hypothesis<'a>(
    hypotheses: &'a [PulseHypothesisEvaluation],
    secondary: &PulseHypothesisEvaluation,
    tolerance_s: f64,
) -> Option<(&'a PulseHypothesisEvaluation, BeatMetrics)> {
    hypotheses
        .iter()
        .map(|hypothesis| {
            let agreement = score_beats(
                &hypothesis.beat_times_s,
                &secondary.beat_times_s,
                tolerance_s,
            );
            (hypothesis, agreement)
        })
        .max_by(
            |(left_hypothesis, left_agreement), (right_hypothesis, right_agreement)| {
                left_agreement
                    .f1
                    .total_cmp(&right_agreement.f1)
                    .then_with(|| right_hypothesis.rank.cmp(&left_hypothesis.rank))
            },
        )
}

fn best_meter_evidence(
    hypothesis_times: &[f64],
    secondary_beats: &[ObservedBeat],
    secondary_activations: Option<&RhythmActivationSeries>,
    tolerance_s: f64,
) -> Option<MeterPatternEvidence> {
    let probabilities = hypothesis_times
        .iter()
        .map(|&time_s| {
            secondary_activations.map_or_else(
                || downbeat_probability_at_event(time_s, secondary_beats, tolerance_s),
                |activations| {
                    downbeat_probability_at_frame(time_s, activations, tolerance_s)
                        .unwrap_or(DOWNBEAT_PROBABILITY_FLOOR)
                },
            )
        })
        .collect::<Vec<_>>();
    METER_PULSE_COUNTS
        .into_iter()
        .flat_map(|pulses_per_bar| {
            (0..pulses_per_bar).filter_map({
                let probabilities = &probabilities;
                move |phase| meter_pattern_evidence(probabilities, pulses_per_bar, phase)
            })
        })
        .max_by(|left, right| {
            left.log_likelihood
                .total_cmp(&right.log_likelihood)
                .then_with(|| right.pulses_per_bar.cmp(&left.pulses_per_bar))
                .then_with(|| right.phase.cmp(&left.phase))
        })
}

fn downbeat_probability_at_event(
    time_s: f64,
    secondary_beats: &[ObservedBeat],
    tolerance_s: f64,
) -> f64 {
    secondary_beats
        .iter()
        .min_by(|left, right| {
            (left.time_s - time_s)
                .abs()
                .total_cmp(&(right.time_s - time_s).abs())
        })
        .filter(|beat| (beat.time_s - time_s).abs() <= tolerance_s)
        .map_or(DOWNBEAT_PROBABILITY_FLOOR, |beat| {
            beat.downbeat_confidence
                .clamp(DOWNBEAT_PROBABILITY_FLOOR, 1.0 - DOWNBEAT_PROBABILITY_FLOOR)
        })
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn downbeat_probability_at_frame(
    time_s: f64,
    activations: &RhythmActivationSeries,
    tolerance_s: f64,
) -> Option<f64> {
    let position = (time_s - activations.start_time_s) * activations.frame_rate_hz;
    if !position.is_finite() || position < 0.0 {
        return None;
    }
    let frame = position.round() as usize;
    let frame_time = activations.start_time_s
        + f64::from(u32::try_from(frame).ok()?) / activations.frame_rate_hz;
    activations
        .downbeat_confidences
        .get(frame)
        .filter(|_| (frame_time - time_s).abs() <= tolerance_s)
        .map(|&probability| {
            f64::from(probability)
                .clamp(DOWNBEAT_PROBABILITY_FLOOR, 1.0 - DOWNBEAT_PROBABILITY_FLOOR)
        })
}

fn meter_pattern_evidence(
    probabilities: &[f64],
    pulses_per_bar: usize,
    phase: usize,
) -> Option<MeterPatternEvidence> {
    let mut downbeat_log_sum = 0.0;
    let mut downbeat_count = 0_u32;
    let mut ordinary_log_sum = 0.0;
    let mut ordinary_count = 0_u32;
    for (index, &probability) in probabilities.iter().enumerate() {
        if index % pulses_per_bar == phase {
            downbeat_log_sum += probability.ln();
            downbeat_count += 1;
        } else {
            ordinary_log_sum += (1.0 - probability).ln();
            ordinary_count += 1;
        }
    }
    if downbeat_count == 0 || ordinary_count == 0 {
        return None;
    }
    Some(MeterPatternEvidence {
        pulses_per_bar,
        phase,
        log_likelihood: 0.5 * downbeat_log_sum / f64::from(downbeat_count)
            + 0.5 * ordinary_log_sum / f64::from(ordinary_count),
    })
}

pub(crate) fn validate_inputs(
    primary: &BottleneckEvaluation,
    secondary: &BottleneckEvaluation,
    agreement_tolerance_s: f64,
) -> Result<()> {
    if primary.suite_purpose != SuitePurpose::Calibration
        || secondary.suite_purpose != SuitePurpose::Calibration
    {
        bail!("consensus diagnosis accepts calibration reports only");
    }
    if primary.suite_id != secondary.suite_id {
        bail!("primary and secondary reports use different suites");
    }
    if primary.model_pack.manifest_sha256 == secondary.model_pack.manifest_sha256 {
        bail!("consensus diagnosis requires distinct model packs");
    }
    if primary.model_pack.backend == secondary.model_pack.backend {
        bail!("consensus diagnosis requires distinct observation backends");
    }
    if primary.cases.is_empty() || secondary.cases.is_empty() {
        bail!("consensus diagnosis requires non-empty reports");
    }
    if !agreement_tolerance_s.is_finite() || agreement_tolerance_s <= 0.0 {
        bail!("agreement tolerance must be finite and greater than zero");
    }
    for case in &secondary.cases {
        if let Some(activations) = case.observations.activations.as_ref() {
            validate_activation_series(&case.id, activations)?;
        }
    }
    Ok(())
}

fn validate_activation_series(case_id: &str, activations: &RhythmActivationSeries) -> Result<()> {
    if !activations.start_time_s.is_finite()
        || !activations.frame_rate_hz.is_finite()
        || activations.frame_rate_hz <= 0.0
    {
        bail!("secondary case {case_id} has an invalid activation time base");
    }
    if activations.pulse_confidences.is_empty()
        || activations.pulse_confidences.len() != activations.downbeat_confidences.len()
    {
        bail!("secondary case {case_id} has incompatible activation channels");
    }
    if activations
        .pulse_confidences
        .iter()
        .chain(&activations.downbeat_confidences)
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        bail!("secondary case {case_id} has an invalid activation probability");
    }
    Ok(())
}

fn window_agreement_margins(
    primary: &PulseHypothesisEvaluation,
    choice: &PulseHypothesisEvaluation,
    secondary: &PulseHypothesisEvaluation,
    tolerance_s: f64,
) -> Vec<f64> {
    let end_s = primary
        .beat_times_s
        .iter()
        .chain(&choice.beat_times_s)
        .chain(&secondary.beat_times_s)
        .copied()
        .fold(0.0_f64, f64::max);
    if end_s <= 0.0 {
        return vec![0.0; 4];
    }
    (0..AGREEMENT_WINDOW_COUNT)
        .map(|index| {
            let start_s = end_s * f64::from(index) / f64::from(AGREEMENT_WINDOW_COUNT);
            let window_end_s = end_s * f64::from(index + 1) / f64::from(AGREEMENT_WINDOW_COUNT);
            let include_end = index + 1 == AGREEMENT_WINDOW_COUNT;
            let primary_times =
                times_in_window(&primary.beat_times_s, start_s, window_end_s, include_end);
            let choice_times =
                times_in_window(&choice.beat_times_s, start_s, window_end_s, include_end);
            let secondary_times =
                times_in_window(&secondary.beat_times_s, start_s, window_end_s, include_end);
            score_beats(&choice_times, &secondary_times, tolerance_s).f1
                - score_beats(&primary_times, &secondary_times, tolerance_s).f1
        })
        .collect()
}

fn times_in_window(times: &[f64], start_s: f64, end_s: f64, include_end: bool) -> Vec<f64> {
    times
        .iter()
        .copied()
        .filter(|time_s| {
            *time_s >= start_s && (*time_s < end_s || (include_end && *time_s <= end_s))
        })
        .collect()
}

fn material_support_reversals(margins: &[f64]) -> usize {
    const MATERIAL_MARGIN: f64 = 0.05;
    margins
        .iter()
        .copied()
        .filter(|margin| margin.abs() >= MATERIAL_MARGIN)
        .map(|margin| {
            if margin.is_sign_positive() {
                1_i8
            } else {
                -1_i8
            }
        })
        .collect::<Vec<_>>()
        .windows(2)
        .filter(|pair| pair[0] != pair[1])
        .count()
}

#[cfg(test)]
mod tests {
    use rhythm_map_core::{ObservedBeat, RhythmActivationSeries};

    use crate::{BeatMetrics, PulseEvidenceBreakdown, PulseHypothesisEvaluation};

    use super::{
        MeterPatternEvidence, ScoredConsensusCandidate, best_meter_evidence,
        material_support_reversals, meter_gate_eligible, most_agreeing_hypothesis,
        window_agreement_margins,
    };

    fn hypothesis(id: &str, rank: usize, times: &[f64]) -> PulseHypothesisEvaluation {
        PulseHypothesisEvaluation {
            id: id.to_string(),
            rank,
            evidence_score: 1.0,
            evidence: PulseEvidenceBreakdown::default(),
            metrical_level: 0,
            phase: None,
            beat_times_s: times.to_vec(),
            beats: BeatMetrics {
                matched: 0,
                precision: 0.0,
                recall: 0.0,
                f1: 0.0,
                median_absolute_error_ms: None,
                p95_absolute_error_ms: None,
            },
        }
    }

    fn agreement(f1: f64) -> BeatMetrics {
        BeatMetrics {
            matched: 0,
            precision: f1,
            recall: f1,
            f1,
            median_absolute_error_ms: None,
            p95_absolute_error_ms: None,
        }
    }

    #[test]
    fn meter_evidence_recovers_four_pulse_downbeat_phase() {
        let times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let secondary = times
            .iter()
            .enumerate()
            .map(|(index, &time_s)| ObservedBeat {
                time_s,
                confidence: 0.9,
                downbeat_confidence: if index.is_multiple_of(4) { 0.9 } else { 0.1 },
            })
            .collect::<Vec<_>>();

        let evidence = best_meter_evidence(&times, &secondary, None, 0.01).unwrap();

        assert_eq!(evidence.pulses_per_bar, 4);
        assert_eq!(evidence.phase, 0);
    }

    #[test]
    fn dense_meter_evidence_is_not_limited_to_decoded_events() {
        let times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let mut downbeat_confidences = vec![0.1; 176];
        for frame in [0, 100] {
            downbeat_confidences[frame] = 0.9;
        }
        let activations = RhythmActivationSeries {
            start_time_s: 0.0,
            frame_rate_hz: 50.0,
            pulse_confidences: vec![0.5; 176],
            downbeat_confidences,
        };

        let evidence = best_meter_evidence(&times, &[], Some(&activations), 0.01).unwrap();

        assert_eq!(evidence.pulses_per_bar, 4);
        assert_eq!(evidence.phase, 0);
    }

    #[test]
    fn meter_gate_requires_both_scores_to_improve() {
        let primary_hypothesis = hypothesis("primary", 1, &[0.0, 1.0]);
        let candidate_hypothesis = hypothesis("candidate", 2, &[0.0, 0.5, 1.0]);
        let primary = ScoredConsensusCandidate {
            hypothesis: &primary_hypothesis,
            agreement: agreement(0.8),
            meter_evidence: Some(MeterPatternEvidence {
                pulses_per_bar: 4,
                phase: 0,
                log_likelihood: -0.5,
            }),
        };
        let mut candidate = ScoredConsensusCandidate {
            hypothesis: &candidate_hypothesis,
            agreement: agreement(0.9),
            meter_evidence: Some(MeterPatternEvidence {
                pulses_per_bar: 4,
                phase: 0,
                log_likelihood: -0.6,
            }),
        };

        assert!(!meter_gate_eligible(&candidate, &primary));
        candidate.meter_evidence.as_mut().unwrap().log_likelihood = -0.4;
        assert!(meter_gate_eligible(&candidate, &primary));
    }

    #[test]
    fn global_agreement_can_choose_a_lower_ranked_dense_hypothesis() {
        let selected = hypothesis("selected", 1, &[0.0, 1.0, 2.0]);
        let dense = hypothesis("dense", 2, &[0.0, 0.5, 1.0, 1.5, 2.0]);
        let secondary = hypothesis("secondary", 1, &[0.0, 0.5, 1.0, 1.5, 2.0]);
        let hypotheses = [selected, dense];

        let (choice, agreement) = most_agreeing_hypothesis(&hypotheses, &secondary, 0.01).unwrap();

        assert_eq!(choice.id, "dense");
        assert!((agreement.f1 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn global_agreement_tie_preserves_the_earlier_primary_rank() {
        let later = hypothesis("later", 2, &[0.0, 1.0]);
        let earlier = hypothesis("earlier", 1, &[0.0, 1.0]);
        let secondary = hypothesis("secondary", 1, &[0.0, 1.0]);
        let hypotheses = [later, earlier];

        let (choice, _) = most_agreeing_hypothesis(&hypotheses, &secondary, 0.01).unwrap();

        assert_eq!(choice.id, "earlier");
    }

    #[test]
    fn final_agreement_window_includes_the_last_timestamp() {
        let selected = hypothesis("selected", 1, &[0.0, 1.0]);
        let dense = hypothesis("dense", 2, &[0.0, 1.0, 2.0]);
        let secondary = hypothesis("secondary", 1, &[0.0, 1.0, 2.0]);

        let margins = window_agreement_margins(&selected, &dense, &secondary, 0.01);

        assert!(margins[3] > 0.9);
    }

    #[test]
    fn material_reversals_ignore_small_window_noise() {
        assert_eq!(material_support_reversals(&[0.30, 0.02, -0.10, -0.32]), 1);
        assert_eq!(material_support_reversals(&[0.30, 0.25, 0.24, -0.04]), 0);
        assert_eq!(material_support_reversals(&[-0.33, 0.24, 0.38, -0.17]), 2);
    }
}
