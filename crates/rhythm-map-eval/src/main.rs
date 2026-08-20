//! Command-line orchestration for reproducible Rhythm Map evaluation.

use std::{fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rhythm_map_eval::{
    evaluate_backend_suite, evaluate_core_suite, render_suite, score_prediction_directory,
};
use rhythm_map_models::verify_model_pack;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Reproducible evaluation and fixture tooling for Rhythm Map")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Gate the core estimator with ideal observations from synthetic recipes.
    Eval {
        /// Evaluation suite manifest.
        #[arg(long, default_value = "evaluation/suites/generated-v1.json")]
        suite: PathBuf,
        /// Optional JSON report destination.
        #[arg(long)]
        report: Option<PathBuf>,
        /// Emit failures without returning a non-zero exit code.
        #[arg(long)]
        no_fail: bool,
    },
    /// Compare ideal observations with the Beat This end-to-end audio path.
    EvalBackend {
        /// Evaluation suite manifest.
        #[arg(long, default_value = "evaluation/suites/generated-v1.json")]
        suite: PathBuf,
        /// Versioned model-pack manifest.
        #[arg(long, default_value = "models/beat-this-full-v1.json")]
        model_pack: PathBuf,
        /// Directory containing the model files named by the manifest.
        #[arg(long)]
        model_dir: PathBuf,
        /// Optional JSON report destination.
        #[arg(long)]
        report: Option<PathBuf>,
        /// Emit failures without returning a non-zero exit code.
        #[arg(long)]
        no_fail: bool,
    },
    /// Verify model-pack provenance, sizes, and SHA-256 digests.
    ModelVerify {
        /// Versioned model-pack manifest.
        #[arg(long, default_value = "models/beat-this-full-v1.json")]
        model_pack: PathBuf,
        /// Directory containing the model files named by the manifest.
        #[arg(long)]
        model_dir: PathBuf,
    },
    /// Render deterministic synthetic WAVs and truth JSON for backend evaluation.
    Render {
        /// Evaluation suite manifest.
        #[arg(long, default_value = "evaluation/suites/generated-v1.json")]
        suite: PathBuf,
        /// Explicit output directory, normally outside the Git checkout.
        #[arg(long)]
        output: PathBuf,
    },
    /// Score end-to-end Analysis JSON files produced by any product surface.
    Score {
        /// Evaluation suite manifest.
        #[arg(long, default_value = "evaluation/suites/generated-v1.json")]
        suite: PathBuf,
        /// Directory containing one `<case-id>.json` Analysis per case.
        #[arg(long)]
        predictions: PathBuf,
        /// Optional JSON report destination.
        #[arg(long)]
        report: Option<PathBuf>,
        /// Emit failures without returning a non-zero exit code.
        #[arg(long)]
        no_fail: bool,
    },
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Eval {
            suite,
            report,
            no_fail,
        } => emit_report(&evaluate_core_suite(&suite)?, report, no_fail),
        Command::EvalBackend {
            suite,
            model_pack,
            model_dir,
            report,
            no_fail,
        } => emit_report(
            &evaluate_backend_suite(&suite, &model_pack, &model_dir)?,
            report,
            no_fail,
        ),
        Command::ModelVerify {
            model_pack,
            model_dir,
        } => {
            let verified = verify_model_pack(&model_pack, &model_dir)?;
            let output = serde_json::json!({
                "schema_version": 1,
                "verified": true,
                "id": verified.manifest().id,
                "version": verified.manifest().version,
                "backend": verified.manifest().backend,
                "manifest_sha256": verified.manifest_sha256(),
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
            Ok(())
        }
        Command::Render { suite, output } => {
            for path in render_suite(&suite, &output)? {
                println!("{}", path.display());
            }
            Ok(())
        }
        Command::Score {
            suite,
            predictions,
            report,
            no_fail,
        } => emit_report(
            &score_prediction_directory(&suite, &predictions)?,
            report,
            no_fail,
        ),
    }
}

fn emit_report<T>(report: &T, destination: Option<PathBuf>, no_fail: bool) -> Result<()>
where
    T: EvaluationOutcome + Serialize,
{
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = destination {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(&path, format!("{json}\n"))
            .with_context(|| format!("writing {}", path.display()))?;
    }
    println!("{json}");
    if !report.passed() && !no_fail {
        bail!("evaluation acceptance budgets failed");
    }
    Ok(())
}

trait EvaluationOutcome {
    fn passed(&self) -> bool;
}

impl EvaluationOutcome for rhythm_map_eval::SuiteEvaluation {
    fn passed(&self) -> bool {
        self.passed
    }
}

impl EvaluationOutcome for rhythm_map_eval::BottleneckEvaluation {
    fn passed(&self) -> bool {
        self.passed
    }
}
