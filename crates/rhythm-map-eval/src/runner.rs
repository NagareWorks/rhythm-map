use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use rhythm_map_core::{Analysis, TempoMapEstimator};
use serde::{Deserialize, Serialize};

use crate::{
    CaseEvaluation, CaseInput, EvaluationCase, EvaluationSuite, GeneratedTruth, SyntheticRecipe,
    evaluate_analysis, generate_truth, wav::render_click_track,
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
        render_click_track(&recipe, &truth, &wav_path)
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
