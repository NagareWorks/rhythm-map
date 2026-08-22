use std::{collections::HashSet, path::Path};

use serde::{Deserialize, Serialize};

/// Acceptance budgets shared by a suite or overridden by one case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AcceptanceThresholds {
    /// Maximum distance for matching a predicted beat to ground truth.
    pub beat_tolerance_ms: f64,
    /// Maximum distance for matching change points of the same kind.
    pub change_tolerance_s: f64,
    /// Lowest acceptable one-to-one beat F1 score.
    pub min_beat_f1: f64,
    /// Lowest acceptable one-to-one downbeat F1 score.
    #[serde(default = "default_min_downbeat_f1")]
    pub min_downbeat_f1: f64,
    /// Highest acceptable median relative tempo error, as a percentage.
    pub max_tempo_median_error_percent: f64,
    /// Highest acceptable 95th-percentile relative tempo error, as a percentage.
    pub max_tempo_p95_error_percent: f64,
    /// Lowest acceptable recall over expected change points.
    pub min_change_recall: f64,
}

impl Default for AcceptanceThresholds {
    fn default() -> Self {
        Self {
            beat_tolerance_ms: 70.0,
            change_tolerance_s: 1.0,
            min_beat_f1: 0.99,
            min_downbeat_f1: default_min_downbeat_f1(),
            max_tempo_median_error_percent: 5.0,
            max_tempo_p95_error_percent: 15.0,
            min_change_recall: 0.5,
        }
    }
}

const fn default_min_downbeat_f1() -> f64 {
    0.0
}

/// How an evaluation asset is obtained.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AssetKind {
    /// Created deterministically from a checked-in recipe.
    Generated,
    /// Publicly obtainable under the recorded license.
    Public,
    /// Locally supplied and never redistributed by this repository.
    Private,
}

/// How a suite may be used during algorithm development.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SuitePurpose {
    /// Stable product regression coverage, not a parameter-selection corpus.
    #[default]
    Regression,
    /// Truth may be inspected and decoder candidates may be compared.
    Calibration,
    /// Reserved evidence for evaluating one policy selected elsewhere.
    Holdout,
}

/// Rights and origin of audio and annotations are recorded separately.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AssetProvenance {
    /// Acquisition class for the audio.
    pub kind: AssetKind,
    /// SPDX identifier or a precise non-SPDX license label for the audio.
    pub audio_license: String,
    /// SPDX identifier or a precise non-SPDX license label for annotations.
    pub annotation_license: String,
    /// Whether this repository may redistribute the audio bytes.
    pub redistributable: bool,
    /// Whether the asset permits commercial evaluation use.
    pub commercial_evaluation_allowed: bool,
    /// Required credit text, if any.
    #[serde(default)]
    pub attribution: Option<String>,
    /// Canonical source page, if the asset came from elsewhere.
    #[serde(default)]
    pub source_url: Option<String>,
}

/// Content-addressed audio reference that never embeds audio bytes in a suite.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AudioReference {
    /// Lowercase SHA-256 of the exact encoded audio file bytes.
    pub sha256: String,
    /// Non-authoritative filename hint for a local dataset resolver.
    #[serde(default)]
    pub local_file_hint: Option<String>,
}

/// Source of audio and reference annotations for one case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CaseInput {
    /// Deterministic truth and optional audio rendered from a recipe.
    Generated {
        /// Recipe path relative to the suite manifest.
        recipe: String,
    },
    /// Real audio resolved locally by hash, with checked-in or local truth.
    External {
        /// Truth JSON path relative to the suite manifest.
        truth: String,
        /// Immutable identity for audio held outside this repository.
        audio: AudioReference,
    },
}

/// One independently scored evaluation case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvaluationCase {
    /// Stable case identifier used for reports and prediction filenames.
    pub id: String,
    /// Generated or external evaluation input.
    pub input: CaseInput,
    /// Searchable capability labels.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Asset rights and origin.
    pub provenance: AssetProvenance,
    /// Optional per-case acceptance budget.
    #[serde(default)]
    pub thresholds: Option<AcceptanceThresholds>,
}

/// Versioned list of evaluation cases and their default acceptance budget.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvaluationSuite {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Stable suite identifier.
    pub id: String,
    /// Human-readable purpose and scope.
    pub description: String,
    /// Whether truth may be used for tuning or only for a locked evaluation.
    #[serde(default)]
    pub purpose: SuitePurpose,
    /// Default acceptance budget.
    #[serde(default)]
    pub thresholds: AcceptanceThresholds,
    /// Cases in the suite.
    pub cases: Vec<EvaluationCase>,
}

impl EvaluationSuite {
    /// Validate identifiers, paths, numeric budgets, and provenance invariants.
    ///
    /// # Errors
    ///
    /// Returns a description when the manifest cannot be evaluated safely.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported evaluation manifest schema {}",
                self.schema_version
            ));
        }
        if self.id.trim().is_empty() || self.cases.is_empty() {
            return Err("suite id and at least one case are required".to_string());
        }
        validate_thresholds(&self.thresholds)?;
        let mut ids = HashSet::new();
        for case in &self.cases {
            if case.id.trim().is_empty() || !ids.insert(case.id.as_str()) {
                return Err(format!("empty or duplicate case id: {}", case.id));
            }
            match &case.input {
                CaseInput::Generated { recipe } => {
                    validate_relative_path(&case.id, recipe)?;
                    if case.provenance.kind != AssetKind::Generated {
                        return Err(format!(
                            "generated case {} must use generated provenance",
                            case.id
                        ));
                    }
                }
                CaseInput::External { truth, audio } => {
                    validate_relative_path(&case.id, truth)?;
                    if let Some(hint) = &audio.local_file_hint {
                        validate_relative_path(&case.id, hint)?;
                    }
                    if case.provenance.kind == AssetKind::Generated {
                        return Err(format!(
                            "external case {} cannot use generated provenance",
                            case.id
                        ));
                    }
                    if audio.sha256.len() != 64
                        || !audio
                            .sha256
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                    {
                        return Err(format!(
                            "case {} audio SHA-256 must be 64 lowercase hexadecimal characters",
                            case.id
                        ));
                    }
                }
            }
            if let Some(thresholds) = &case.thresholds {
                validate_thresholds(thresholds)?;
            }
            if case.provenance.kind == AssetKind::Private && case.provenance.redistributable {
                return Err(format!(
                    "private case {} cannot be marked redistributable",
                    case.id
                ));
            }
        }
        Ok(())
    }
}

fn validate_relative_path(case_id: &str, value: &str) -> Result<(), String> {
    let path = Path::new(value);
    if value.trim().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        return Err(format!(
            "case {case_id} input paths must stay below the suite directory"
        ));
    }
    Ok(())
}

fn validate_thresholds(thresholds: &AcceptanceThresholds) -> Result<(), String> {
    if !thresholds.beat_tolerance_ms.is_finite() || thresholds.beat_tolerance_ms <= 0.0 {
        return Err("beat tolerance must be finite and positive".to_string());
    }
    if !thresholds.change_tolerance_s.is_finite() || thresholds.change_tolerance_s <= 0.0 {
        return Err("change tolerance must be finite and positive".to_string());
    }
    for (name, value) in [
        ("minimum beat F1", thresholds.min_beat_f1),
        ("minimum downbeat F1", thresholds.min_downbeat_f1),
        ("minimum change recall", thresholds.min_change_recall),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(format!("{name} must be within [0, 1]"));
        }
    }
    for (name, value) in [
        (
            "maximum median tempo error",
            thresholds.max_tempo_median_error_percent,
        ),
        (
            "maximum p95 tempo error",
            thresholds.max_tempo_p95_error_percent,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{name} must be finite and non-negative"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_audio_cannot_be_redistributable() {
        let suite = EvaluationSuite {
            schema_version: 1,
            id: "private".to_string(),
            description: "test".to_string(),
            purpose: SuitePurpose::Regression,
            thresholds: AcceptanceThresholds::default(),
            cases: vec![EvaluationCase {
                id: "track".to_string(),
                input: CaseInput::External {
                    truth: "track.truth.json".to_string(),
                    audio: AudioReference {
                        sha256: "0".repeat(64),
                        local_file_hint: None,
                    },
                },
                tags: Vec::new(),
                provenance: AssetProvenance {
                    kind: AssetKind::Private,
                    audio_license: "local-use-only".to_string(),
                    annotation_license: "proprietary".to_string(),
                    redistributable: true,
                    commercial_evaluation_allowed: false,
                    attribution: None,
                    source_url: None,
                },
                thresholds: None,
            }],
        };
        assert!(suite.validate().is_err());
    }

    #[test]
    fn external_audio_hint_cannot_escape_resolver_root() {
        let suite = EvaluationSuite {
            schema_version: 1,
            id: "private".to_string(),
            description: "test".to_string(),
            purpose: SuitePurpose::Regression,
            thresholds: AcceptanceThresholds::default(),
            cases: vec![EvaluationCase {
                id: "track".to_string(),
                input: CaseInput::External {
                    truth: "track.truth.json".to_string(),
                    audio: AudioReference {
                        sha256: "0".repeat(64),
                        local_file_hint: Some("../track.wav".to_string()),
                    },
                },
                tags: Vec::new(),
                provenance: AssetProvenance {
                    kind: AssetKind::Private,
                    audio_license: "local-use-only".to_string(),
                    annotation_license: "proprietary".to_string(),
                    redistributable: false,
                    commercial_evaluation_allowed: false,
                    attribution: None,
                    source_url: None,
                },
                thresholds: None,
            }],
        };
        assert!(suite.validate().is_err());
    }

    #[test]
    fn omitted_suite_purpose_defaults_to_regression() {
        let suite: EvaluationSuite = serde_json::from_str(
            r#"{
                "schema_version": 1,
                "id": "legacy",
                "description": "legacy schema-one suite",
                "cases": []
            }"#,
        )
        .unwrap();

        assert_eq!(suite.purpose, SuitePurpose::Regression);
    }
}
