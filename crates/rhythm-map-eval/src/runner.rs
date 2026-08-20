use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use rhythm_map_beat_this::BeatThisBackend;
use rhythm_map_core::{Analysis, Engine, TempoMapEstimator};
use rhythm_map_models::{ModelArtifactRole, VerifiedModelPack, verify_model_pack};
use serde::{Deserialize, Serialize};

use crate::{
    CaseEvaluation, CaseInput, EvaluationCase, EvaluationSuite, GeneratedTruth, SyntheticRecipe,
    evaluate_analysis, generate_truth,
    wav::{render_synthetic_audio, synthesize_audio},
};

/// Complete evaluation report for one suite.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SuiteEvaluation {
    /// Evaluation report schema version.
    pub schema_version: u32,
    /// Stable suite identifier.
    pub suite_id: String,
    /// True only when every case passes.
    pub passed: bool,
    /// Per-case metrics and failed budgets.
    pub cases: Vec<CaseEvaluation>,
}

/// Model-pack identity recorded in an end-to-end evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelPackIdentity {
    /// Stable model-pack identifier.
    pub id: String,
    /// Immutable model-pack version.
    pub version: String,
    /// Observation backend identity.
    pub backend: String,
    /// SHA-256 of the exact manifest bytes.
    pub manifest_sha256: String,
}

/// End-to-end metric difference relative to oracle observations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaseMetricDelta {
    /// End-to-end minus oracle beat F1.
    pub beat_f1: f64,
    /// End-to-end minus oracle median tempo error percentage.
    pub tempo_median_error_percent: Option<f64>,
    /// End-to-end minus oracle p95 tempo error percentage.
    pub tempo_p95_error_percent: Option<f64>,
    /// End-to-end minus oracle change-point recall.
    pub change_recall: f64,
}

/// Oracle and end-to-end results for one case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttributionCase {
    /// Stable case identifier.
    pub id: String,
    /// Estimator result from exact beat observations.
    pub oracle: CaseEvaluation,
    /// Product result from rendered audio and the observation backend.
    pub end_to_end: CaseEvaluation,
    /// Directional end-to-end metric difference.
    pub delta: CaseMetricDelta,
    /// Wall-clock time spent in end-to-end audio analysis.
    pub end_to_end_runtime_ms: f64,
}

/// Coarse bottleneck conclusion supported by suite acceptance gates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AttributionDecision {
    /// Exact beats pass while the full audio-to-result path fails.
    ///
    /// This includes observation errors and estimator robustness to plausible
    /// but metrically ambiguous observations; it does not by itself prove the
    /// neural backend must be replaced.
    ObservationPath,
    /// The deterministic estimator fails even with exact beats.
    DeterministicEstimator,
    /// Both paths satisfy the current acceptance gates.
    NoMeasuredBottleneck,
}

/// Side-by-side oracle and end-to-end evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BottleneckEvaluation {
    /// Evaluation report schema version.
    pub schema_version: u32,
    /// Stable suite identifier.
    pub suite_id: String,
    /// Verified model-pack identity.
    pub model_pack: ModelPackIdentity,
    /// True only when both oracle and end-to-end paths pass.
    pub passed: bool,
    /// Acceptance-gate bottleneck decision.
    pub attribution: AttributionDecision,
    /// Per-case paired measurements.
    pub cases: Vec<AttributionCase>,
}

/// Evaluate the timing estimator with ideal observations from every recipe.
///
/// # Errors
///
/// Returns an error for invalid manifests, recipes, or estimator failures.
pub fn evaluate_core_suite(suite_path: &Path) -> Result<SuiteEvaluation> {
    let (suite, root) = load_suite(suite_path)?;
    let estimator = TempoMapEstimator::default();
    let mut cases = Vec::with_capacity(suite.cases.len());
    for case in &suite.cases {
        let CaseInput::Generated { recipe } = &case.input else {
            bail!(
                "core evaluation suite {} contains external case {}",
                suite.id,
                case.id
            );
        };
        let recipe = load_recipe(&root.join(recipe))?;
        if recipe.id != case.id {
            bail!("case {} points to recipe with id {}", case.id, recipe.id);
        }
        let truth = generate_truth(&recipe).map_err(anyhow::Error::msg)?;
        let analysis = estimator
            .estimate(&truth.ideal_observations())
            .with_context(|| format!("estimating case {}", case.id))?;
        let thresholds = case.thresholds.as_ref().unwrap_or(&suite.thresholds);
        cases.push(evaluate_analysis(&case.id, &analysis, &truth, thresholds));
    }
    Ok(suite_report(suite.id, cases))
}

/// Run rendered synthetic audio through Beat This and compare it with oracle
/// observations using the same truth and thresholds.
///
/// # Errors
///
/// Returns an error for invalid suites, model packs, model loading, rendering,
/// inference, or estimator failures.
pub fn evaluate_backend_suite(
    suite_path: &Path,
    model_pack_path: &Path,
    model_root: &Path,
) -> Result<BottleneckEvaluation> {
    let verified = verify_model_pack(model_pack_path, model_root)
        .with_context(|| format!("verifying model pack {}", model_pack_path.display()))?;
    validate_beat_this_contract(&verified)?;
    let mel_model = required_model_path(&verified, ModelArtifactRole::MelFrontend)?;
    let beat_model = required_model_path(&verified, ModelArtifactRole::BeatModel)?;
    let backend = BeatThisBackend::load(&mel_model, &beat_model)?;
    let mut engine = Engine::new(backend, TempoMapEstimator::default());
    let estimator = TempoMapEstimator::default();
    let (suite, root) = load_suite(suite_path)?;
    let mut cases = Vec::with_capacity(suite.cases.len());

    for case in &suite.cases {
        let CaseInput::Generated { recipe } = &case.input else {
            bail!(
                "backend evaluation suite {} contains external case {}",
                suite.id,
                case.id
            );
        };
        let recipe = load_recipe(&root.join(recipe))?;
        if recipe.id != case.id {
            bail!("case {} points to recipe with id {}", case.id, recipe.id);
        }
        let truth = generate_truth(&recipe).map_err(anyhow::Error::msg)?;
        let thresholds = case.thresholds.as_ref().unwrap_or(&suite.thresholds);
        let oracle_analysis = estimator
            .estimate(&truth.ideal_observations())
            .with_context(|| format!("estimating oracle case {}", case.id))?;
        let oracle = evaluate_analysis(&case.id, &oracle_analysis, &truth, thresholds);

        let samples = synthesize_audio(&recipe, &truth)
            .with_context(|| format!("synthesizing case {}", case.id))?;
        let started = Instant::now();
        let analysis = engine
            .analyze_pcm(&samples, recipe.sample_rate, 1)
            .with_context(|| format!("running backend case {}", case.id))?;
        let runtime_ms = started.elapsed().as_secs_f64() * 1000.0;
        let end_to_end = evaluate_analysis(&case.id, &analysis, &truth, thresholds);
        let delta = metric_delta(&oracle, &end_to_end);
        cases.push(AttributionCase {
            id: case.id.clone(),
            oracle,
            end_to_end,
            delta,
            end_to_end_runtime_ms: runtime_ms,
        });
    }

    let oracle_passed = cases.iter().all(|case| case.oracle.passed);
    let end_to_end_passed = cases.iter().all(|case| case.end_to_end.passed);
    let attribution = if !oracle_passed {
        AttributionDecision::DeterministicEstimator
    } else if !end_to_end_passed {
        AttributionDecision::ObservationPath
    } else {
        AttributionDecision::NoMeasuredBottleneck
    };
    let manifest = verified.manifest();
    Ok(BottleneckEvaluation {
        schema_version: 1,
        suite_id: suite.id,
        model_pack: ModelPackIdentity {
            id: manifest.id.clone(),
            version: manifest.version.clone(),
            backend: manifest.backend.clone(),
            manifest_sha256: verified.manifest_sha256().to_string(),
        },
        passed: oracle_passed && end_to_end_passed,
        attribution,
        cases,
    })
}

fn validate_beat_this_contract(model_pack: &VerifiedModelPack) -> Result<()> {
    let manifest = model_pack.manifest();
    if manifest.backend != "beat-this-rten" {
        bail!(
            "model pack {} declares backend {}, expected beat-this-rten",
            manifest.id,
            manifest.backend
        );
    }

    let contract = &manifest.feature_contract;
    if contract.sample_rate_hz != 22_050
        || contract.mel_bands != 128
        || (contract.frame_rate_hz - 50.0).abs() > f64::EPSILON
    {
        bail!(
            "model pack {} has incompatible Beat This feature contract: {} Hz, {} mel bands, {} fps",
            manifest.id,
            contract.sample_rate_hz,
            contract.mel_bands,
            contract.frame_rate_hz
        );
    }
    Ok(())
}

/// Render each recipe to a WAV and exact truth JSON in an explicit output path.
///
/// # Errors
///
/// Returns an error for invalid input or filesystem failures.
pub fn render_suite(suite_path: &Path, output: &Path) -> Result<Vec<PathBuf>> {
    let (suite, root) = load_suite(suite_path)?;
    fs::create_dir_all(output)
        .with_context(|| format!("creating output directory {}", output.display()))?;
    let mut rendered = Vec::new();
    for case in &suite.cases {
        let CaseInput::Generated { recipe } = &case.input else {
            bail!(
                "render suite {} contains external case {}",
                suite.id,
                case.id
            );
        };
        let recipe = load_recipe(&root.join(recipe))?;
        let truth = generate_truth(&recipe).map_err(anyhow::Error::msg)?;
        let wav_path = output.join(format!("{}.wav", case.id));
        render_synthetic_audio(&recipe, &truth, &wav_path)
            .with_context(|| format!("rendering {}", wav_path.display()))?;
        let truth_path = output.join(format!("{}.truth.json", case.id));
        fs::write(&truth_path, serde_json::to_vec_pretty(&truth)?)
            .with_context(|| format!("writing {}", truth_path.display()))?;
        rendered.push(wav_path);
        rendered.push(truth_path);
    }
    Ok(rendered)
}

/// Score externally produced analyses named `<case-id>.json`.
///
/// # Errors
///
/// Returns an error for missing predictions or invalid manifests and JSON.
pub fn score_prediction_directory(
    suite_path: &Path,
    prediction_directory: &Path,
) -> Result<SuiteEvaluation> {
    let (suite, root) = load_suite(suite_path)?;
    let mut cases = Vec::with_capacity(suite.cases.len());
    for case in &suite.cases {
        let truth = load_case_truth(case, &root)?;
        let prediction_path = prediction_directory.join(format!("{}.json", case.id));
        let analysis: Analysis = serde_json::from_slice(
            &fs::read(&prediction_path)
                .with_context(|| format!("reading {}", prediction_path.display()))?,
        )
        .with_context(|| format!("parsing {}", prediction_path.display()))?;
        let thresholds = case.thresholds.as_ref().unwrap_or(&suite.thresholds);
        cases.push(evaluate_analysis(&case.id, &analysis, &truth, thresholds));
    }
    Ok(suite_report(suite.id, cases))
}

fn load_case_truth(case: &EvaluationCase, root: &Path) -> Result<GeneratedTruth> {
    match &case.input {
        CaseInput::Generated { recipe } => {
            let recipe = load_recipe(&root.join(recipe))?;
            generate_truth(&recipe).map_err(anyhow::Error::msg)
        }
        CaseInput::External { truth, .. } => {
            let path = root.join(truth);
            serde_json::from_slice(
                &fs::read(&path).with_context(|| format!("reading truth {}", path.display()))?,
            )
            .with_context(|| format!("parsing truth {}", path.display()))
        }
    }
}

fn load_suite(path: &Path) -> Result<(EvaluationSuite, PathBuf)> {
    let suite: EvaluationSuite = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("reading suite {}", path.display()))?,
    )
    .with_context(|| format!("parsing suite {}", path.display()))?;
    suite.validate().map_err(anyhow::Error::msg)?;
    let root = path
        .parent()
        .context("suite path must have a parent directory")?
        .to_path_buf();
    Ok((suite, root))
}

fn load_recipe(path: &Path) -> Result<SyntheticRecipe> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("reading recipe {}", path.display()))?,
    )
    .with_context(|| format!("parsing recipe {}", path.display()))
}

fn suite_report(suite_id: String, cases: Vec<CaseEvaluation>) -> SuiteEvaluation {
    SuiteEvaluation {
        schema_version: 1,
        suite_id,
        passed: cases.iter().all(|case| case.passed),
        cases,
    }
}

fn required_model_path(pack: &VerifiedModelPack, role: ModelArtifactRole) -> Result<PathBuf> {
    pack.path_for(role)
        .with_context(|| format!("verified model pack is missing role {role:?}"))
}

fn metric_delta(oracle: &CaseEvaluation, end_to_end: &CaseEvaluation) -> CaseMetricDelta {
    CaseMetricDelta {
        beat_f1: end_to_end.metrics.beats.f1 - oracle.metrics.beats.f1,
        tempo_median_error_percent: subtract_options(
            end_to_end.metrics.tempo.median_absolute_error_percent,
            oracle.metrics.tempo.median_absolute_error_percent,
        ),
        tempo_p95_error_percent: subtract_options(
            end_to_end.metrics.tempo.p95_absolute_error_percent,
            oracle.metrics.tempo.p95_absolute_error_percent,
        ),
        change_recall: end_to_end.metrics.changes.recall - oracle.metrics.changes.recall,
    }
}

fn subtract_options(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    left.zip(right).map(|(left, right)| left - right)
}
