//! Controlled loss of observations, never deletion of musical truth or audio.

mod latent_clock;

use std::{fs, path::Path};

use anyhow::{Result, ensure};
use rhythm_map_core::{Analysis, ChangeKind, RhythmObservations, analyze_observations};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    GeneratedTruth, RecipeSegment, SegmentShape, SuitePurpose, SyntheticAudioProfile,
    SyntheticRecipe, generate_truth,
    runner::{load_case_truth, load_suite},
};

#[derive(Clone, Copy, Debug)]
enum Mask {
    Intact,
    EveryEighth,
    MiddleAlternating,
    MiddleFour,
    TrailingEight,
}

const MASKS: [Mask; 5] = [
    Mask::Intact,
    Mask::EveryEighth,
    Mask::MiddleAlternating,
    Mask::MiddleFour,
    Mask::TrailingEight,
];

impl Mask {
    fn name(self) -> &'static str {
        match self {
            Self::Intact => "intact",
            Self::EveryEighth => "one_missing_every_eight",
            Self::MiddleAlternating => "alternate_missing_in_middle_third",
            Self::MiddleFour => "four_missing_at_middle",
            Self::TrailingEight => "eight_missing_at_tail",
        }
    }

    fn remove(self, index: usize, count: usize) -> bool {
        if count < 8 || index < 2 {
            return false;
        }
        match self {
            Self::Intact => false,
            Self::EveryEighth => index % 8 == 4 && index + 2 < count,
            Self::MiddleAlternating => {
                (count / 3..2 * count / 3).contains(&index) && index % 2 == 1
            }
            Self::MiddleFour => (count / 2..(count / 2 + 4).min(count - 2)).contains(&index),
            Self::TrailingEight => index >= count.saturating_sub(8).max(2),
        }
    }
}

fn masked(truth: &GeneratedTruth, mask: Mask, downbeats: bool) -> RhythmObservations {
    let mut observations = truth.ideal_observations();
    // No missing timestamp is leaked via candidates, dense activations or audio
    // features. This is a best-case detector with only deletions, not a model.
    observations.beats = observations
        .beats
        .into_iter()
        .enumerate()
        .filter_map(|(index, mut beat)| {
            if mask.remove(index, truth.beats.len()) {
                return None;
            }
            if !downbeats {
                beat.downbeat_confidence = 0.0;
            }
            Some(beat)
        })
        .collect();
    observations
}

fn tempo_at(analysis: &Analysis, time_s: f64) -> Option<f64> {
    analysis
        .tempo_segments
        .iter()
        .find(|s| s.start_s <= time_s && time_s < s.end_s)
        .map(|s| {
            s.start_bpm + (s.end_bpm - s.start_bpm) * ((time_s - s.start_s) / (s.end_s - s.start_s))
        })
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn quantile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(sorted[((sorted.len() - 1) as f64 * fraction).ceil() as usize])
}

fn tempo_measure(truth: &GeneratedTruth, lookup: impl Fn(f64) -> Option<f64>) -> Value {
    // All variants use the SAME truth-interval midpoint queries, not the
    // estimator's shrinking set of output points after observations disappear.
    let queries = truth
        .beats
        .windows(2)
        .filter_map(|b| {
            let t = f64::midpoint(b[0].time_s, b[1].time_s);
            truth.tempo_at(t).map(|tempo| (t, tempo))
        })
        .collect::<Vec<_>>();
    let errors = queries
        .iter()
        .filter_map(|&(t, expected)| {
            lookup(t).map(|actual| (actual / expected - 1.0).abs() * 100.0)
        })
        .collect::<Vec<_>>();
    json!({
        "truth_query_count": queries.len(), "scored_query_count": errors.len(),
        "uncovered_query_count": queries.len() - errors.len(),
        "median_error_percent": quantile(&errors, 0.5),
        "p95_error_percent": quantile(&errors, 0.95),
        "maximum_error_percent": quantile(&errors, 1.0),
    })
}

fn measure(analysis: &Analysis, truth: &GeneratedTruth) -> Value {
    let mut result = tempo_measure(truth, |t| tempo_at(analysis, t));
    let outside_beat_span = truth
        .beats
        .windows(2)
        .map(|b| f64::midpoint(b[0].time_s, b[1].time_s))
        .filter(|&t| truth.tempo_at(t).is_some())
        .collect::<Vec<_>>()
        .iter()
        .filter(|&&t| {
            analysis
                .beats
                .first()
                .zip(analysis.beats.last())
                .is_none_or(|(first, last)| t < first.time_s || t > last.time_s)
        })
        .count();
    result["outside_returned_beat_span_query_count"] = json!(outside_beat_span);
    result["tempo_jump_count"] = json!(
        analysis
            .change_points
            .iter()
            .filter(|c| c.kind == ChangeKind::TempoJump)
            .count()
    );
    result["evidence_discontinuity_count"] = json!(
        analysis
            .change_points
            .iter()
            .filter(|c| c.kind == ChangeKind::RhythmDiscontinuity)
            .count()
    );
    result["returned_beat_count"] = json!(analysis.beats.len());
    result
}

fn case_audit(truth: &GeneratedTruth) -> Result<Value> {
    let mut variants = Vec::new();
    for downbeats in [true, false] {
        for mask in MASKS {
            let observations = masked(truth, mask, downbeats);
            let analysis = analyze_observations(&observations)?;
            let clock = latent_clock::decode(&observations)?;
            let mut advancements = [0usize; 8];
            for interval in &clock.intervals {
                advancements[interval.advancement - 1] += 1;
            }
            let unsupported = analysis
                .beats
                .iter()
                .filter(|b| {
                    !observations
                        .beats
                        .iter()
                        .any(|o| o.time_s.to_bits() == b.time_s.to_bits())
                })
                .count();
            variants.push(json!({
                "mask": mask.name(), "downbeat_evidence": if downbeats {"oracle"} else {"zeroed"},
                "truth_beat_count": truth.beats.len(), "input_beat_count": observations.beats.len(),
                "removed_observation_count": truth.beats.len() - observations.beats.len(),
                "unsupported_returned_beat_count": unsupported,
                "measurement": measure(&analysis, truth),
                "missing_step_clock": {
                    "measurement": tempo_measure(truth, |t| clock.tempo_at(t)),
                    "objective_not_confidence": clock.objective,
                    "advancement_histogram_1_through_8": advancements,
                    "generated_beat_event_count": 0,
                },
            }));
        }
    }
    Ok(json!({"id": truth.id, "variants": variants}))
}

fn ensure_calibration(purpose: SuitePurpose) -> Result<()> {
    ensure!(
        purpose == SuitePurpose::Calibration,
        "observation audit requires calibration"
    );
    Ok(())
}

/// Diagnose the default estimator under five fixed observation-loss patterns.
///
/// No audio is decoded or altered, no model runs, and no rule is selected. Truth
/// remains intact; reports cover fixed query positions even when output shrinks.
/// Missing downbeat evidence is a separate controlled factor, not a user option.
///
/// # Errors
/// Rejects non-calibration suites before loading truth, invalid truth, or an
/// estimator failure. This diagnostic cannot authorize training or promotion.
pub fn audit_observation_dropout(suite_path: &Path) -> Result<Value> {
    let (suite, root) = load_suite(suite_path)?;
    ensure_calibration(suite.purpose)?;
    let mut cases = Vec::new();
    for case in &suite.cases {
        let truth = load_case_truth(case, &root)?;
        eprintln!("observation audit: {}", case.id);
        cases.push(case_audit(&truth)?);
    }
    Ok(
        json!({"schema_version": 2, "purpose": "controlled_observation_loss_not_model_accuracy",
        "suite_id": suite.id, "suite_sha256": format!("{:x}", Sha256::digest(fs::read(suite_path)?)),
        "audit_source_sha256": format!("{:x}", Sha256::digest(include_bytes!("observability.rs"))),
        "candidate_id": "missing-step-clock-v1",
        "candidate_source_sha256": format!("{:x}", Sha256::digest(include_bytes!("observability/latent_clock.rs"))),
        "estimator_source_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../rhythm-map-core/src/estimator.rs"))),
        "query_contract": "unchanged truth-interval midpoints; no extrapolation outside returned segments",
        "inference_run": false, "audio_modified": false,
        "authored_controls": authored_controls()?, "cases": cases}),
    )
}

fn fixture(tempos: &[f64]) -> GeneratedTruth {
    generate_truth(&SyntheticRecipe {
        schema_version: 1,
        id: "observability-fixture".into(),
        sample_rate: 22050,
        beats_per_bar: 4,
        audio_profile: SyntheticAudioProfile::Click,
        segments: tempos
            .iter()
            .map(|&bpm| RecipeSegment {
                duration_s: 8.0,
                shape: SegmentShape::Constant { bpm },
            })
            .collect(),
    })
    .expect("authored valid fixture")
}

fn authored_controls() -> Result<Value> {
    let mut constant = fixture(&[120.0; 3]);
    constant.id = "constant-120".into();
    let mut step = fixture(&[120.0, 60.0, 120.0]);
    step.id = "real-120-60-120".into();
    let mut non_octave = fixture(&[120.0, 90.0, 120.0]);
    non_octave.id = "real-120-90-120".into();
    let sparse = masked(&constant, Mask::MiddleAlternating, false);
    let slow = masked(&step, Mask::Intact, false);
    ensure!(sparse == slow, "authored input-equivalence witness changed");
    let analysis = analyze_observations(&sparse)?;
    let clock = latent_clock::decode(&sparse)?;
    Ok(
        json!({"cases": [case_audit(&constant)?, case_audit(&step)?, case_audit(&non_octave)?],
        "indistinguishable_pair": {"observations_equal": true,
            "constant_truth_bpm_at_12_s": constant.tempo_at(12.0),
            "step_truth_bpm_at_12_s": step.tempo_at(12.0),
            "same_input_estimated_bpm_at_12_s": tempo_at(&analysis, 12.0),
            "same_input_candidate_bpm_at_12_s": clock.tempo_at(12.0),
            "scope": "identical timestamp-only observations, not identical audio"}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_calibration_roles_are_rejected() {
        assert!(ensure_calibration(SuitePurpose::Calibration).is_ok());
        assert!(ensure_calibration(SuitePurpose::Holdout).is_err());
        assert!(ensure_calibration(SuitePurpose::Regression).is_err());
    }

    #[test]
    fn downbeat_factor_does_not_leak_into_timestamp_only_candidate() {
        let truth = fixture(&[120.0, 90.0, 120.0]);
        for mask in MASKS {
            assert_eq!(
                latent_clock::decode(&masked(&truth, mask, true)).unwrap(),
                latent_clock::decode(&masked(&truth, mask, false)).unwrap()
            );
        }
    }

    #[test]
    fn masks_delete_observations_not_truth_or_hidden_evidence() {
        let truth = fixture(&[120.0; 3]);
        let original = truth.clone();
        for mask in MASKS {
            let observations = masked(&truth, mask, false);
            assert!(
                observations
                    .beats
                    .iter()
                    .all(|b| b.downbeat_confidence == 0.0)
            );
            assert!(observations.beat_candidates.is_empty() && observations.activations.is_none());
            assert!(observations.activity.is_empty() && observations.onsets.is_empty());
            assert!(observations.beats.len() >= 2);
        }
        assert_eq!(truth, original);
        assert_eq!(
            masked(&truth, Mask::MiddleFour, true).beats.len(),
            truth.beats.len() - 4
        );
        assert_eq!(
            masked(&truth, Mask::TrailingEight, true).beats.len(),
            truth.beats.len() - 8
        );
    }

    #[test]
    fn fixed_query_count_cannot_shrink_with_missing_output() {
        let truth = fixture(&[120.0; 3]);
        let intact = analyze_observations(&masked(&truth, Mask::Intact, true)).unwrap();
        let tail = analyze_observations(&masked(&truth, Mask::TrailingEight, true)).unwrap();
        let a = measure(&intact, &truth);
        let b = measure(&tail, &truth);
        assert_eq!(a["truth_query_count"], b["truth_query_count"]);
        assert!(
            b["outside_returned_beat_span_query_count"]
                .as_u64()
                .unwrap()
                > 0
        );
        let mut no_segments = tail;
        no_segments.tempo_segments.clear();
        let absent = measure(&no_segments, &truth);
        assert_eq!(absent["uncovered_query_count"], a["truth_query_count"]);
        assert!(absent["median_error_percent"].is_null());
    }

    #[test]
    fn constant_clock_dropout_and_real_slowdown_can_have_identical_observations() {
        let constant = fixture(&[120.0; 3]);
        let slowdown = fixture(&[120.0, 60.0, 120.0]);
        let sparse = masked(&constant, Mask::MiddleAlternating, false);
        let slow = masked(&slowdown, Mask::Intact, false);
        assert_eq!(sparse, slow);
        assert_eq!(
            analyze_observations(&sparse).unwrap(),
            analyze_observations(&slow).unwrap()
        );
        assert_eq!(constant.tempo_at(12.0), Some(120.0));
        assert_eq!(slowdown.tempo_at(12.0), Some(60.0));
        // This is an observation-boundary impossibility witness, not evidence
        // that the AUDIO is identical or that a trained model could not help.
    }

    #[test]
    fn short_inputs_are_not_destroyed_by_masks() {
        let mut truth = fixture(&[120.0]);
        truth.beats.truncate(5);
        for mask in MASKS {
            assert_eq!(masked(&truth, mask, true).beats.len(), 5);
        }
    }

    #[test]
    fn quantiles_do_not_report_missing_samples_as_zero_error() {
        assert_eq!(quantile(&[], 0.5), None);
        assert_eq!(quantile(&[0.0, 10.0, 20.0, 30.0], 0.5), Some(20.0));
    }
}
