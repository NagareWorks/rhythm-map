//! Command-line orchestration for reproducible Rhythm Map evaluation.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rhythm_map_eval::{
    evaluate_backend_suite, evaluate_backend_suite_with_audio_directory, evaluate_core_suite,
    evaluate_decoder_recoverability_with_audio_directory,
    evaluate_decoder_sweep_with_audio_directory, fetch_public_dataset, inspect_audio_asset,
    render_suite, score_prediction_directory, standard_decoder_policies,
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
    /// Gate the core estimator with ideal observations from generated or external truth.
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
        /// Directory containing content-addressed external evaluation audio.
        #[arg(long)]
        audio_dir: Option<PathBuf>,
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
    /// Print the immutable digest and decoded shape needed by an external case.
    AudioInspect {
        /// Local audio file to hash and decode; the path is not written to the report.
        #[arg(long)]
        input: PathBuf,
    },
    /// Fetch and verify a public evaluation dataset outside the Git checkout.
    DatasetFetch {
        /// Versioned public-dataset lock manifest.
        #[arg(long, default_value = "evaluation/datasets/artbeat-v1.json")]
        manifest: PathBuf,
        /// Explicit destination directory, normally outside the Git checkout.
        #[arg(long)]
        output: PathBuf,
        /// Also fetch immutable annotation-source artifacts used to audit truth.
        #[arg(long)]
        with_annotations: bool,
    },
    /// Compare Beat This logit peak decoders without repeating model inference.
    DecoderSweep {
        /// Evaluation suite containing independent beat truth.
        #[arg(long, default_value = "evaluation/suites/artbeat-v1.json")]
        suite: PathBuf,
        /// Versioned model-pack manifest.
        #[arg(long, default_value = "models/beat-this-full-v1.json")]
        model_pack: PathBuf,
        /// Directory containing the model files named by the manifest.
        #[arg(long)]
        model_dir: PathBuf,
        /// Directory containing content-addressed external evaluation audio.
        #[arg(long)]
        audio_dir: PathBuf,
        /// Optional JSON report destination.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Inspect model evidence around truth beats missed by the upstream decoder.
    DecoderRecoverability {
        /// Evaluation suite containing independent beat truth.
        #[arg(long, default_value = "evaluation/suites/artbeat-v1.json")]
        suite: PathBuf,
        /// Versioned model-pack manifest.
        #[arg(long, default_value = "models/beat-this-full-v1.json")]
        model_pack: PathBuf,
        /// Directory containing the model files named by the manifest.
        #[arg(long)]
        model_dir: PathBuf,
        /// Directory containing content-addressed external evaluation audio.
        #[arg(long)]
        audio_dir: PathBuf,
        /// Optional JSON report destination.
        #[arg(long)]
        report: Option<PathBuf>,
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
            audio_dir,
            report,
            no_fail,
        } => {
            let result = if let Some(audio_dir) = audio_dir {
                evaluate_backend_suite_with_audio_directory(
                    &suite,
                    &model_pack,
                    &model_dir,
                    &audio_dir,
                )?
            } else {
                evaluate_backend_suite(&suite, &model_pack, &model_dir)?
            };
            emit_report(&result, report, no_fail)
        }
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
        Command::AudioInspect { input } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&inspect_audio_asset(input)?)?
            );
            Ok(())
        }
        Command::DatasetFetch {
            manifest,
            output,
            with_annotations,
        } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&fetch_public_dataset(
                    &manifest,
                    &output,
                    with_annotations,
                )?)?
            );
            Ok(())
        }
        Command::DecoderSweep {
            suite,
            model_pack,
            model_dir,
            audio_dir,
            report,
        } => run_decoder_sweep(&suite, &model_pack, &model_dir, &audio_dir, report),
        Command::DecoderRecoverability {
            suite,
            model_pack,
            model_dir,
            audio_dir,
            report,
        } => run_decoder_recoverability(&suite, &model_pack, &model_dir, &audio_dir, report),
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

fn run_decoder_sweep(
    suite: &Path,
    model_pack: &Path,
    model_dir: &Path,
    audio_dir: &Path,
    report: Option<PathBuf>,
) -> Result<()> {
    emit_json_report(
        &evaluate_decoder_sweep_with_audio_directory(
            suite,
            model_pack,
            model_dir,
            audio_dir,
            &standard_decoder_policies(),
        )?,
        report,
    )
}

fn run_decoder_recoverability(
    suite: &Path,
    model_pack: &Path,
    model_dir: &Path,
    audio_dir: &Path,
    report: Option<PathBuf>,
) -> Result<()> {
    emit_json_report(
        &evaluate_decoder_recoverability_with_audio_directory(
            suite, model_pack, model_dir, audio_dir,
        )?,
        report,
    )
}

fn emit_report<T>(report: &T, destination: Option<PathBuf>, no_fail: bool) -> Result<()>
where
    T: EvaluationOutcome + Serialize,
{
    emit_json_report(report, destination)?;
    if !report.passed() && !no_fail {
        bail!("evaluation acceptance budgets failed");
    }
    Ok(())
}

fn emit_json_report<T>(report: &T, destination: Option<PathBuf>) -> Result<()>
where
    T: Serialize,
{
    let json = serde_json::to_string_pretty(report)?;
    if let Some(path) = destination {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(&path, format!("{json}\n"))
            .with_context(|| format!("writing {}", path.display()))?;
    }
    println!("{json}");
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
