use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::{
    AttributionCase, BeatMetrics, BottleneckEvaluation, ModelPackIdentity,
    PulseHypothesisEvaluation, SuitePurpose, metrics::score_beats,
};

const AGREEMENT_WINDOW_COUNT: u32 = 4;

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
    /// Annotated F1 before consensus; calibration attribution only.
    pub primary_truth_beat_f1: f64,
    /// Annotated F1 after naive consensus; calibration attribution only.
    pub naive_consensus_truth_beat_f1: f64,
    /// Directional annotated F1 change; never used for selection.
    pub truth_beat_f1_delta: f64,
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

    Ok(ConsensusDiagnosis {
        schema_version: 1,
        suite_id: primary.suite_id.clone(),
        primary_model_pack: primary.model_pack.clone(),
        secondary_model_pack: secondary.model_pack.clone(),
        agreement_tolerance_s,
        primary_mean_beat_f1,
        naive_consensus_mean_beat_f1,
        naive_consensus_delta,
        improved_cases,
        regressed_cases,
        passes_calibration_gate: naive_consensus_delta > f64::EPSILON && regressed_cases == 0,
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
    let (choice, choice_agreement) = most_agreeing_hypothesis(
        &primary_coverage.hypotheses,
        secondary_hypothesis,
        agreement_tolerance_s,
    )
    .with_context(|| format!("primary case {} has no hypotheses", primary_case.id))?;
    let primary_agreement = score_beats(
        &primary_hypothesis.beat_times_s,
        &secondary_hypothesis.beat_times_s,
        agreement_tolerance_s,
    );
    let window_agreement_margins = window_agreement_margins(
        primary_hypothesis,
        choice,
        secondary_hypothesis,
        agreement_tolerance_s,
    );
    Ok(ConsensusDiagnosisCase {
        id: primary_case.id.clone(),
        primary_hypothesis_id: primary_hypothesis.id.clone(),
        secondary_hypothesis_id: secondary_hypothesis.id.clone(),
        naive_consensus_hypothesis_id: choice.id.clone(),
        agreement_margin: choice_agreement.f1 - primary_agreement.f1,
        primary_agreement,
        naive_consensus_agreement: choice_agreement,
        material_support_reversals: material_support_reversals(&window_agreement_margins),
        window_agreement_margins,
        primary_truth_beat_f1: primary_hypothesis.beats.f1,
        naive_consensus_truth_beat_f1: choice.beats.f1,
        truth_beat_f1_delta: choice.beats.f1 - primary_hypothesis.beats.f1,
    })
}

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

fn validate_inputs(
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
    use crate::{BeatMetrics, PulseEvidenceBreakdown, PulseHypothesisEvaluation};

    use super::{material_support_reversals, most_agreeing_hypothesis, window_agreement_margins};

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
